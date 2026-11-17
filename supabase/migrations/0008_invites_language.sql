-- Friend invites by QR: your QR opens the app with "<you> invited you to be friends"; once they sign up or sign
-- in, you're friends straight away (the QR is your consent). Plus: each device's language for notifications.

-- One invite code per player. Kept apart from profiles so it works before a username is picked.
create table if not exists public.friend_invites (
  code text primary key check (code ~ '^[a-z0-9]{10}$'),
  user_id uuid not null unique references auth.users (id) on delete cascade,
  name text not null default '' check (char_length(name) <= 40),
  number smallint check (number between 0 and 99),
  created_at timestamptz not null default now()
);
alter table public.friend_invites enable row level security; -- no policies: only through the functions below

-- My code (made on first use), with the name and number the invite shows.
create or replace function public.my_invite_code(p_name text, p_number integer)
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare c text;
begin
  if auth.uid() is null then raise exception 'not signed in'; end if;
  insert into public.friend_invites (code, user_id, name, number)
  values (substr(md5(random()::text || clock_timestamp()::text || auth.uid()::text), 1, 10), auth.uid(),
          left(coalesce(p_name, ''), 40), case when p_number between 0 and 99 then p_number end)
  on conflict (user_id) do update set name = excluded.name, number = excluded.number
  returning code into c;
  return c;
end;
$$;
revoke execute on function public.my_invite_code(text, integer) from public, anon;
grant execute on function public.my_invite_code(text, integer) to authenticated;

-- Who invited me: name and shirt number only (shown before the visitor has an account).
create or replace function public.invite_info(p_code text)
returns table (name text, number smallint)
language sql
stable
security definer
set search_path = ''
as $$
  select coalesce(nullif(p.display_name, ''), nullif(i.name, ''), p.username, ''), coalesce(p.number, i.number)
  from public.friend_invites i left join public.profiles p on p.user_id = i.user_id
  where i.code = lower(p_code)
$$;
revoke execute on function public.invite_info(text) from public;
grant execute on function public.invite_info(text) to anon, authenticated;

-- Take the invite: we're friends now. 'ok', 'self' (your own QR) or 'invalid'.
create or replace function public.accept_invite(p_code text)
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare owner uuid; me uuid := auth.uid();
begin
  if me is null then raise exception 'not signed in'; end if;
  select user_id into owner from public.friend_invites where code = lower(p_code);
  if owner is null then return 'invalid'; end if;
  if owner = me then return 'self'; end if;
  update public.friendships set status = 'accepted'
  where (requester = owner and addressee = me) or (requester = me and addressee = owner);
  if not found then
    insert into public.friendships (requester, addressee, status) values (owner, me, 'accepted');
  end if;
  return 'ok';
end;
$$;
revoke execute on function public.accept_invite(text) from public, anon;
grant execute on function public.accept_invite(text) to authenticated;

-- Notifications in the device's language.
alter table public.push_subscriptions add column if not exists lang text not null default 'en'
  check (lang in ('en', 'vi', 'es', 'pt', 'fr', 'id'));

create or replace function public.queue_weekly_pushes()
returns integer
language sql
security definer
set search_path = ''
as $$
  with due as (
    select distinct on (p.user_id) p.user_id, p.lang, (now() at time zone p.tz)::date as local_day
    from public.push_subscriptions p
    where p.weekly
      and extract(isodow from now() at time zone p.tz) = 7
      and extract(hour from now() at time zone p.tz) = 18
    order by p.user_id, p.created_at desc
  ), ins as (
    insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
    select d.user_id, now(), 'weekly',
      case d.lang when 'vi' then 'Lên kế hoạch tuần tới' when 'es' then 'Planifica la próxima semana'
        when 'pt' then 'Planeje a próxima semana' when 'fr' then 'Prépare la semaine prochaine'
        when 'id' then 'Rencanakan minggu depan' else 'Plan next week' end,
      case d.lang
        when 'vi' then 'Ngày thi đấu và ngày tập gym tuần tới đã sẵn sàng. Hai phút bây giờ giúp cả tuần nhẹ nhàng.'
        when 'es' then 'Tus días de partido y de gimnasio de la próxima semana están listos. Dos minutos ahora salvan la semana.'
        when 'pt' then 'Seus dias de jogo e de academia da próxima semana estão prontos. Dois minutos agora salvam a semana.'
        when 'fr' then 'Tes jours de match et de salle pour la semaine prochaine sont prêts. Deux minutes maintenant sauvent la semaine.'
        when 'id' then 'Hari pertandingan dan hari gym minggu depan sudah siap dicek. Dua menit sekarang menyelamatkan seminggu.'
        else 'Your game days and gym days for next week are ready to check. Two minutes now saves the week.' end,
      '/?tab=week', 'weekly-' || d.local_day
    from due d
    on conflict (user_id, tag) where tag is not null do nothing
    returning 1
  )
  select count(*)::integer from ins
$$;
revoke execute on function public.queue_weekly_pushes() from public, anon, authenticated;
grant execute on function public.queue_weekly_pushes() to service_role;
