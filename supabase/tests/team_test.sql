-- Team rules (run after stub_supabase.sql and every migration). \set ON_ERROR_STOP is off: lines marked ERROR must fail.
-- Users: 1 = peter, 2 = sam, 3 = stranger.
insert into auth.users values ('33333333-3333-3333-3333-333333333333') on conflict do nothing;
set role authenticated;

set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
insert into public.profiles (user_id, username, number, streak) values ('11111111-1111-1111-1111-111111111111', 'Peter.7', 7, 9);
insert into public.profiles (user_id, username, number) values ('22222222-2222-2222-2222-222222222222', 'fake', 1);   -- ERROR: not mine
insert into public.profiles (user_id, username, number) values ('11111111-1111-1111-1111-111111111111', 'x', 1);      -- ERROR: duplicate / bad username
select username, best_streak from public.profiles;                                                                     -- peter.7 | 9

set request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
insert into public.profiles (user_id, username, number, streak) values ('22222222-2222-2222-2222-222222222222', 'sam', 11, 4);
insert into public.profiles (user_id, username, number) values ('22222222-2222-2222-2222-222222222222', 'peter.7', 3); -- ERROR: one profile each
select count(*) as sam_sees_profiles from public.profiles;                                                             -- 1 (only own)
select username, friendship from public.find_player('PETER.7', 10);                                                    -- 0 rows: wrong number
select username, friendship from public.find_player(' Peter.7 ', 7);                                                   -- peter.7 | null
insert into public.friendships (requester, addressee) values ('22222222-2222-2222-2222-222222222222', '11111111-1111-1111-1111-111111111111');
insert into public.friendships (requester, addressee, status) values ('22222222-2222-2222-2222-222222222222', '33333333-3333-3333-3333-333333333333', 'accepted'); -- ERROR: can't self-accept
select username, friendship from public.find_player('peter.7', 7);                                                     -- requested
update public.friendships set status = 'accepted' where requester = '22222222-2222-2222-2222-222222222222';           -- 0 rows: only peter can accept
select count(*) as sam_sees_after_request from public.profiles;                                                        -- 1

set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
insert into public.friendships (requester, addressee) values ('11111111-1111-1111-1111-111111111111', '22222222-2222-2222-2222-222222222222'); -- ERROR: pair exists
select username, incoming from public.friend_requests();                                                               -- sam | t
update public.friendships set status = 'accepted' where requester = '22222222-2222-2222-2222-222222222222';           -- UPDATE 1
select username, streak from public.profiles order by username;                                                        -- peter.7 9, sam 4
update public.profiles set username = 'peter' where user_id = '11111111-1111-1111-1111-111111111111';                  -- UPDATE 1: first change is free
update public.profiles set username = 'peter.7' where user_id = '11111111-1111-1111-1111-111111111111';                -- ERROR: changed < 30 days ago
update public.profiles set streak = 2 where user_id = '11111111-1111-1111-1111-111111111111';
select streak, best_streak from public.profiles where username = 'peter';                                            -- 2 | 9
update public.profiles set streak = 99999 where user_id = '11111111-1111-1111-1111-111111111111';                      -- ERROR: check
update public.profiles set streak = 50 where user_id = '22222222-2222-2222-2222-222222222222';                         -- 0 rows: not mine

set request.jwt.claim.sub = '33333333-3333-3333-3333-333333333333';
select count(*) as stranger_sees from public.profiles;                                                                 -- 0
select count(*) as stranger_sees_friendships from public.friendships;                                                  -- 0
delete from public.friendships;                                                                                        -- DELETE 0

set request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
delete from public.friendships where addressee = '11111111-1111-1111-1111-111111111111';                               -- DELETE 1: unfriend
select count(*) as sam_after_unfriend from public.profiles;                                                            -- 1

reset role; set role anon;
select public.find_player('peter', 7);                                                                                 -- ERROR: permission denied
select count(*) from public.profiles;                                                                                  -- ERROR: permission denied
