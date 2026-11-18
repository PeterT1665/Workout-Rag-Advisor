-- Match day: a review after every game (journal + the app's match rating), and reminders around it.
--   9:00 on a game day        "When do you play?"   (/?match=times)
--   full time + 15 min        "Rate your match"     (booked by the app when you enter your times)
--   21:00 if no times entered "Rate your match"     (only if nothing was booked and no review saved yet)
-- Game days come from the player's synced 'games' (day: 0 = Monday), reviews from 'matches'.

-- sync the reviews and the kick-off/full-time times
alter table public.user_state drop constraint if exists user_state_key_check;
alter table public.user_state add constraint user_state_key_check
  check (key in ('history', 'games', 'settings', 'quotes', 'mySessions', 'profile', 'weekPlan', 'videos', 'matches', 'matchTimes'));

-- its own switch in Settings → Notifications
alter table public.push_subscriptions add column if not exists matches boolean not null default true;

-- a third kind of reminder
alter table public.scheduled_pushes drop constraint if exists scheduled_pushes_kind_check;
alter table public.scheduled_pushes add constraint scheduled_pushes_kind_check check (kind in ('rest', 'weekly', 'match'));
-- a rest is minutes away; a full-time reminder can be booked in the morning for the evening
alter table public.scheduled_pushes drop constraint if exists scheduled_pushes_check;
alter table public.scheduled_pushes add constraint scheduled_pushes_check
  check (send_at < created_at + case when kind = 'rest' then interval '1 hour' else interval '3 days' end);

-- The app books the full-time reminder (replaces an earlier one for that day). Runs as the owner so it can
-- replace the row, but only ever for the signed-in player, with a fixed link and a time within 3 days.
create or replace function public.schedule_match_push(p_at timestamptz, p_day date, p_title text, p_body text)
returns void
language plpgsql
security definer
set search_path = ''
as $$
begin
  if auth.uid() is null then raise exception 'not signed in'; end if;
  if p_at < now() - interval '1 hour' or p_at > now() + interval '3 days' then raise exception 'bad time'; end if;
  delete from public.scheduled_pushes where user_id = auth.uid() and tag = 'match-ft-' || p_day and sent_at is null;
  insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
  values (auth.uid(), greatest(p_at, now()), 'match', left(p_title, 80), left(p_body, 200), '/?match=review', 'match-ft-' || p_day)
  on conflict (user_id, tag) where tag is not null do update set send_at = excluded.send_at, title = excluded.title, body = excluded.body, sent_at = null;
end;
$$;
revoke execute on function public.schedule_match_push(timestamptz, date, text, text) from public, anon;
grant execute on function public.schedule_match_push(timestamptz, date, text, text) to authenticated;

-- Run every hour (from queue_weekly_pushes, so the existing cron job picks it up).
create or replace function public.queue_match_pushes()
returns integer
language sql
security definer
set search_path = ''
as $$
  with dev as (
    select distinct on (p.user_id) p.user_id, p.lang, p.tz,
      (now() at time zone p.tz)::date as local_day,
      extract(hour from now() at time zone p.tz)::int as local_hour,
      extract(isodow from now() at time zone p.tz)::int - 1 as weekday -- 0 = Monday, like the app
    from public.push_subscriptions p
    where p.matches
    order by p.user_id, p.created_at desc
  ), game_today as (
    select d.* from dev d
    join public.user_state g on g.user_id = d.user_id and g.key = 'games' and jsonb_typeof(g.value) = 'array'
    where exists (select 1 from jsonb_array_elements(g.value) x where (x->>'day')::int = d.weekday)
  ), morning as (
    select gt.user_id, gt.lang, 'match-am-' || gt.local_day as tag, '/?match=times' as url,
      case gt.lang when 'vi' then 'Hôm nay có trận' when 'es' then 'Hoy hay partido' when 'pt' then 'Hoje tem jogo'
        when 'fr' then 'Jour de match' when 'id' then 'Hari ini ada pertandingan' else 'Game day' end as title,
      case gt.lang
        when 'vi' then 'Mấy giờ bạn đá? Nhập giờ bắt đầu và kết thúc để app nhắc bạn đánh giá trận đấu khi còn nhớ rõ.'
        when 'es' then '¿A qué hora juegas? Pon la hora de inicio y de final y te recordamos valorar el partido en caliente.'
        when 'pt' then 'Que horas você joga? Coloque o horário de início e fim e lembramos você de avaliar o jogo enquanto está fresco.'
        when 'fr' then 'Tu joues à quelle heure ? Indique le début et la fin, on te rappellera de noter ton match à chaud.'
        when 'id' then 'Main jam berapa? Isi jam mulai dan selesai, nanti kami ingatkan untuk menilai pertandinganmu selagi masih segar.'
        else 'When do you play? Add kick-off and full-time and we''ll remind you to review the match while it''s fresh.' end as body
    from game_today gt where gt.local_hour = 9
  ), evening as (
    select gt.user_id, gt.lang, 'match-pm-' || gt.local_day as tag, '/?match=review' as url,
      case gt.lang when 'vi' then 'Trận đấu của bạn thế nào?' when 'es' then '¿Qué tal el partido?' when 'pt' then 'Como foi o jogo?'
        when 'fr' then 'Alors, ce match ?' when 'id' then 'Bagaimana pertandinganmu?' else 'How did the game go?' end as title,
      case gt.lang
        when 'vi' then 'Hai phút đánh giá trận đấu: bàn thắng, kiến tạo, những khoảnh khắc chính. App sẽ chấm điểm cho bạn.'
        when 'es' then 'Dos minutos para valorar el partido: goles, asistencias, momentos clave. La app te pone nota.'
        when 'pt' then 'Dois minutos para avaliar o jogo: gols, assistências, momentos-chave. O app te dá uma nota.'
        when 'fr' then 'Deux minutes pour noter ton match : buts, passes décisives, moments clés. L''app te donne une note.'
        when 'id' then 'Dua menit untuk menilai pertandingan: gol, assist, momen penting. Aplikasi akan memberi rating.'
        else 'Two minutes to review it: goals, assists, key moments. The app rates your match.' end as body
    from game_today gt
    where gt.local_hour = 21
      -- not if the app booked a full-time reminder, or the review is already saved
      and not exists (select 1 from public.scheduled_pushes s where s.user_id = gt.user_id and s.tag = 'match-ft-' || gt.local_day)
      and not exists (
        select 1 from public.user_state m, jsonb_array_elements(case when jsonb_typeof(m.value) = 'array' then m.value else '[]'::jsonb end) r
        where m.user_id = gt.user_id and m.key = 'matches' and r->>'date' = gt.local_day::text)
  ), ins as (
    insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
    select x.user_id, now(), 'match', x.title, x.body, x.url, x.tag from (select * from morning union all select * from evening) x
    on conflict (user_id, tag) where tag is not null do nothing
    returning 1
  )
  select count(*)::integer from ins
$$;
revoke execute on function public.queue_match_pushes() from public, anon, authenticated;
grant execute on function public.queue_match_pushes() to service_role;

-- The hourly cron already runs queue_weekly_pushes(): have it queue match-day reminders too.
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
  select (select count(*)::integer from ins) + public.queue_match_pushes()
$$;
revoke execute on function public.queue_weekly_pushes() from public, anon, authenticated;
grant execute on function public.queue_weekly_pushes() to service_role;
