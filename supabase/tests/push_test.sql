-- Push rules (after stub_supabase.sql and every migration). Lines marked ERROR must fail.
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
insert into public.push_subscriptions (endpoint, user_id, p256dh, auth, tz) values ('https://push.example/a', '11111111-1111-1111-1111-111111111111', 'k', 'a', 'Australia/Sydney');
insert into public.push_subscriptions (endpoint, user_id, p256dh, auth) values ('https://push.example/b', '22222222-2222-2222-2222-222222222222', 'k', 'a'); -- ERROR: not mine
select public.schedule_rest_push(90, 'Rest is over', 'Lock in');
select public.schedule_rest_push(60, 'Rest is over', 'Lets roll');                      -- replaces the first
select count(*) as pending, max(body) as body from public.scheduled_pushes;            -- 1 | Lets roll
insert into public.scheduled_pushes (user_id, send_at, kind, title, body) values ('11111111-1111-1111-1111-111111111111', now(), 'weekly', 'x', 'y'); -- ERROR: only rest
insert into public.scheduled_pushes (user_id, send_at, kind, title, body) values ('22222222-2222-2222-2222-222222222222', now(), 'rest', 'x', 'y');   -- ERROR: not mine
select public.claim_due_pushes();                                                      -- ERROR: sender only

set request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
select count(*) as user2_sees from public.scheduled_pushes;                            -- 0
select count(*) as user2_devices from public.push_subscriptions;                       -- 0
delete from public.scheduled_pushes;                                                   -- DELETE 0

set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
delete from public.scheduled_pushes where kind = 'rest';                               -- DELETE 1: back in the app, cancel

reset role;
-- the sender: a due reminder is claimed exactly once
insert into public.scheduled_pushes (user_id, send_at, kind, title, body) values ('11111111-1111-1111-1111-111111111111', now() - interval '1 second', 'rest', 'Rest is over', 'Go');
select count(*) as first_claim from public.claim_due_pushes();                         -- 1
select count(*) as second_claim from public.claim_due_pushes();                        -- 0
select public.queue_weekly_pushes() >= 0 as weekly_runs;                               -- t
select public.prune_pushes();

-- 0007: a rest call-back names its device; someone else's device is ignored
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select public.schedule_rest_push(60, 'Rest is over', 'Go', 'https://push.example/a');
select endpoint from public.scheduled_pushes where kind = 'rest' and sent_at is null;          -- https://push.example/a
select public.schedule_rest_push(60, 'Rest is over', 'Go', 'https://push.example/not-mine');
select count(*) as pending, max(endpoint) as endpoint from public.scheduled_pushes where kind = 'rest' and sent_at is null; -- 1 | (null)
select public.schedule_rest_push(60, 'Rest is over', 'Go');                                    -- old 3-argument call still works
reset role;
