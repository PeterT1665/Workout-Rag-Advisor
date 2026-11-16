-- Team, phase 1: public profiles (seen by friends only), friend requests, friends leaderboard, avatars.
-- See docs/system-design.md. Streak and rating are published by the app; the database keeps them sane.

-- ---------- profiles ----------
create table if not exists public.profiles (
  user_id uuid primary key references auth.users (id) on delete cascade,
  username text not null unique check (username ~ '^[a-z0-9_.]{3,20}$'),
  display_name text not null default '' check (char_length(display_name) <= 40),
  number smallint not null check (number between 0 and 99),
  positions text[] not null default '{}' check (cardinality(positions) <= 4),
  foot text not null default 'right' check (foot in ('left', 'right', 'both')),
  accent text not null default 'red' check (char_length(accent) <= 20),
  avatar_path text check (avatar_path is null or char_length(avatar_path) <= 200),
  card jsonb check (card is null or (jsonb_typeof(card) = 'object' and pg_column_size(card) < 2000)), -- card stats
  rating smallint check (rating between 1 and 99),
  streak integer not null default 0 check (streak between 0 and 3660),
  best_streak integer not null default 0 check (best_streak between 0 and 3660),
  trained_today boolean not null default false,
  last_active timestamptz,
  username_changed_at timestamptz not null default '1970-01-01', -- never changed: the first change is free
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
alter table public.profiles enable row level security;

-- ---------- friendships ----------
create table if not exists public.friendships (
  requester uuid not null references auth.users (id) on delete cascade,
  addressee uuid not null references auth.users (id) on delete cascade,
  status text not null default 'pending' check (status in ('pending', 'accepted')),
  created_at timestamptz not null default now(),
  primary key (requester, addressee),
  check (requester <> addressee)
);
-- one friendship per pair, whoever asked
create unique index if not exists friendships_pair on public.friendships (least(requester, addressee), greatest(requester, addressee));
alter table public.friendships enable row level security;

create or replace function public.are_friends(a uuid, b uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.friendships f
    where f.status = 'accepted'
      and ((f.requester = a and f.addressee = b) or (f.requester = b and f.addressee = a))
  )
$$;
revoke execute on function public.are_friends(uuid, uuid) from public, anon;
grant execute on function public.are_friends(uuid, uuid) to authenticated;

-- ---------- profile rules ----------
-- Username: lower case; the first change is free, then once every 30 days. Keeps updated_at current.
create or replace function public.profiles_before_write()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.username := lower(new.username);
  if tg_op = 'UPDATE' then
    if new.username <> old.username then
      if old.username_changed_at > now() - interval '30 days' then
        raise exception 'username_recently_changed' using hint = 'Usernames can be changed once every 30 days.';
      end if;
      new.username_changed_at := now();
    else
      new.username_changed_at := old.username_changed_at;
    end if;
    new.created_at := old.created_at;
  end if;
  new.best_streak := greatest(new.best_streak, new.streak, coalesce(case when tg_op = 'UPDATE' then old.best_streak end, 0));
  new.updated_at := now();
  return new;
end;
$$;
drop trigger if exists profiles_before_write on public.profiles;
create trigger profiles_before_write before insert or update on public.profiles
  for each row execute function public.profiles_before_write();

drop policy if exists "profiles: me and my friends read" on public.profiles;
drop policy if exists "profiles: I create mine" on public.profiles;
drop policy if exists "profiles: I update mine" on public.profiles;
drop policy if exists "profiles: I delete mine" on public.profiles;
create policy "profiles: me and my friends read" on public.profiles for select to authenticated
  using (user_id = (select auth.uid()) or public.are_friends((select auth.uid()), user_id));
create policy "profiles: I create mine" on public.profiles for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy "profiles: I update mine" on public.profiles for update to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy "profiles: I delete mine" on public.profiles for delete to authenticated
  using (user_id = (select auth.uid()));
grant select, insert, update, delete on public.profiles to authenticated;

-- ---------- friendship rules ----------
drop policy if exists "friendships: both sides read" on public.friendships;
drop policy if exists "friendships: I ask" on public.friendships;
drop policy if exists "friendships: they accept" on public.friendships;
drop policy if exists "friendships: either side removes" on public.friendships;
create policy "friendships: both sides read" on public.friendships for select to authenticated
  using ((select auth.uid()) in (requester, addressee));
create policy "friendships: I ask" on public.friendships for insert to authenticated
  with check (requester = (select auth.uid()) and status = 'pending');
create policy "friendships: they accept" on public.friendships for update to authenticated
  using (addressee = (select auth.uid())) with check (addressee = (select auth.uid()) and status = 'accepted');
create policy "friendships: either side removes" on public.friendships for delete to authenticated
  using ((select auth.uid()) in (requester, addressee));
grant select, insert, update, delete on public.friendships to authenticated;

-- ---------- search and requests ----------
-- Find a player by exact username AND shirt number (no browsing). Returns only what a stranger may see.
create or replace function public.find_player(p_username text, p_number integer)
returns table (user_id uuid, username text, display_name text, number smallint, avatar_path text, friendship text)
language sql
stable
security definer
set search_path = ''
as $$
  select p.user_id, p.username, p.display_name, p.number, p.avatar_path,
    (select case when f.status = 'accepted' then 'friends' when f.requester = auth.uid() then 'requested' else 'asked-you' end
       from public.friendships f
      where least(f.requester, f.addressee) = least(auth.uid(), p.user_id)
        and greatest(f.requester, f.addressee) = greatest(auth.uid(), p.user_id))
  from public.profiles p
  where auth.uid() is not null
    and p.username = lower(trim(p_username))
    and p.number = p_number
    and p.user_id <> auth.uid()
$$;
revoke execute on function public.find_player(text, integer) from public, anon;
grant execute on function public.find_player(text, integer) to authenticated;

-- Pending requests to and from me, with the basics of the other player.
create or replace function public.friend_requests()
returns table (user_id uuid, username text, display_name text, number smallint, avatar_path text, incoming boolean, created_at timestamptz)
language sql
stable
security definer
set search_path = ''
as $$
  select p.user_id, p.username, p.display_name, p.number, p.avatar_path, f.addressee = auth.uid(), f.created_at
  from public.friendships f
  join public.profiles p on p.user_id = case when f.requester = auth.uid() then f.addressee else f.requester end
  where f.status = 'pending' and auth.uid() in (f.requester, f.addressee)
  order by f.created_at desc
$$;
revoke execute on function public.friend_requests() from public, anon;
grant execute on function public.friend_requests() to authenticated;

-- ---------- avatars ----------
-- avatars/<user id>/avatar.jpg: any signed-in player can see avatars (search shows them); only the owner changes theirs.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('avatars', 'avatars', false, 1048576, array['image/jpeg', 'image/png', 'image/webp'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit, allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "avatars: signed-in players read" on storage.objects;
drop policy if exists "avatars: owner uploads" on storage.objects;
drop policy if exists "avatars: owner replaces" on storage.objects;
drop policy if exists "avatars: owner deletes" on storage.objects;
create policy "avatars: signed-in players read" on storage.objects for select to authenticated
  using (bucket_id = 'avatars');
create policy "avatars: owner uploads" on storage.objects for insert to authenticated
  with check (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "avatars: owner replaces" on storage.objects for update to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
create policy "avatars: owner deletes" on storage.objects for delete to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);
