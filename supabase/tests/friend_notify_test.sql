-- Friend notifications (0011), after stub_supabase.sql, every migration and team_test.sql's profiles.
\set a '11111111-1111-1111-1111-111111111111'
\set b '22222222-2222-2222-2222-222222222222'
delete from public.friendships; delete from public.scheduled_pushes; delete from public.push_subscriptions;
insert into public.push_subscriptions (endpoint, user_id, p256dh, auth, tz, lang) values
  ('https://push.example/a', :'a', 'k', 'x', 'UTC', 'en'), ('https://push.example/b', :'b', 'k', 'x', 'UTC', 'vi');
update public.profiles set display_name = 'Peter' where user_id = :'a';

-- A asks B: B hears it, in Vietnamese
set role authenticated; set request.jwt.claim.sub = :'a';
insert into public.friendships (requester, addressee) values (:'a', :'b');
reset role;
select user_id = :'b' as to_b, kind, title, body, url from public.scheduled_pushes;     -- t | friend | Lời mời kết bạn | Peter muốn kết bạn... | /?tab=team
delete from public.scheduled_pushes;

-- B accepts: A hears it, in English
set role authenticated; set request.jwt.claim.sub = :'b';
update public.friendships set status = 'accepted' where requester = :'a' and addressee = :'b';
reset role;
select user_id = :'a' as to_a, title, body from public.scheduled_pushes;                -- t | New friend | sam accepted your friend request.
delete from public.scheduled_pushes; delete from public.friendships;

-- B scans A's QR: A hears it
set role authenticated; set request.jwt.claim.sub = :'a';
select public.my_invite_code('Peter', 7) is not null as have_code;
reset role;
select code as invite from public.friend_invites where user_id = :'a' \gset
set role authenticated; set request.jwt.claim.sub = :'b';
select public.accept_invite(:'invite');                                                  -- ok
reset role;
select user_id = :'a' as to_a, body from public.scheduled_pushes;                       -- t | sam scanned your QR. You're friends now.
delete from public.scheduled_pushes; delete from public.friendships;

-- friend notifications off (or no device): nothing queued, the request still goes through
update public.push_subscriptions set friends = false where user_id = :'b';
set role authenticated; set request.jwt.claim.sub = :'a';
insert into public.friendships (requester, addressee) values (:'a', :'b');
reset role;
select (select count(*) from public.scheduled_pushes) as queued, (select count(*) from public.friendships) as friendships;  -- 0 | 1
