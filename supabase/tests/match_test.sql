-- Match day (0009), after stub_supabase.sql and every migration. Lines marked ERROR must fail.
\set u '11111111-1111-1111-1111-111111111111'
set role authenticated;
set request.jwt.claim.sub = :'u';
select public.put_state('matches', '[]'::jsonb, 1);                                    -- new sync key accepted
select public.put_state('matchTimes', '{}'::jsonb, 1);
select public.schedule_match_push(now() + interval '2 hours', current_date, 'Full time. Rate your match', 'Two minutes');
select public.schedule_match_push(now() + interval '3 hours', current_date, 'Full time. Rate your match', 'Moved');  -- replaces
select count(*) as booked, max(body) as body from public.scheduled_pushes where kind = 'match'; -- 1 | Moved
select public.schedule_match_push(now() + interval '9 days', current_date, 'x', 'y');   -- ERROR: bad time
reset role;
delete from public.scheduled_pushes;

-- reminders: pick time zones where it's 9 am / 9 pm right now, with a game on that local weekday
create temp table tz9 as select name from pg_timezone_names where extract(hour from now() at time zone name) = 9 and extract(minute from now() at time zone name) = extract(minute from now() at time zone 'UTC') limit 1;
create temp table tz21 as select name from pg_timezone_names where extract(hour from now() at time zone name) = 21 and extract(minute from now() at time zone name) = extract(minute from now() at time zone 'UTC') limit 1;
select (select count(*) from tz9) = 1 as have_9am_zone, (select count(*) from tz21) = 1 as have_9pm_zone;
-- morning player (A), evening player with no times (B), evening player who already reviewed (C), evening player with a booked full time (D)
insert into auth.users (id) values ('aaaaaaaa-0000-0000-0000-000000000001'), ('aaaaaaaa-0000-0000-0000-000000000002'), ('aaaaaaaa-0000-0000-0000-000000000003'), ('aaaaaaaa-0000-0000-0000-000000000004') on conflict do nothing;
insert into public.push_subscriptions (endpoint, user_id, p256dh, auth, tz, lang) values
  ('https://push.example/A', 'aaaaaaaa-0000-0000-0000-000000000001', 'k', 'a', (select name from tz9), 'vi'),
  ('https://push.example/B', 'aaaaaaaa-0000-0000-0000-000000000002', 'k', 'a', (select name from tz21), 'en'),
  ('https://push.example/C', 'aaaaaaaa-0000-0000-0000-000000000003', 'k', 'a', (select name from tz21), 'en'),
  ('https://push.example/D', 'aaaaaaaa-0000-0000-0000-000000000004', 'k', 'a', (select name from tz21), 'en');
insert into public.user_state (user_id, key, value, updated_at)
select id, 'games', jsonb_build_array(jsonb_build_object('day', extract(isodow from now() at time zone tz)::int - 1, 'kind', 'competitive')), 1
from (values ('aaaaaaaa-0000-0000-0000-000000000001'::uuid, (select name from tz9)), ('aaaaaaaa-0000-0000-0000-000000000002'::uuid, (select name from tz21)),
             ('aaaaaaaa-0000-0000-0000-000000000003'::uuid, (select name from tz21)), ('aaaaaaaa-0000-0000-0000-000000000004'::uuid, (select name from tz21))) v(id, tz);
insert into public.user_state (user_id, key, value, updated_at)
values ('aaaaaaaa-0000-0000-0000-000000000003', 'matches', jsonb_build_array(jsonb_build_object('date', ((now() at time zone (select name from tz21))::date)::text, 'rating', 7.1)), 1);
insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
values ('aaaaaaaa-0000-0000-0000-000000000004', now() - interval '1 hour', 'match', 'x', 'y', '/?match=review', 'match-ft-' || (now() at time zone (select name from tz21))::date);
select public.queue_weekly_pushes() >= 2 as queued;                                    -- t (weekly runs the match ones too)
select right(user_id::text, 1) as who, split_part(tag, '-', 2) as which, title, url from public.scheduled_pushes where tag not like 'match-ft-%' order by 1;
-- expect: 1 | am | Hôm nay có trận | /?match=times   and   2 | pm | How did the game go? | /?match=review   (no row for 3 or 4)
select public.queue_match_pushes() as again;                                           -- 0: once per day
