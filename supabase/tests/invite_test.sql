-- Invites (after stub_supabase.sql and every migration). Lines marked ERROR must fail.
insert into auth.users (id) values ('33333333-3333-3333-3333-333333333333') on conflict do nothing;
set role anon;
select public.my_invite_code('X', 1);                                   -- ERROR: guests have no code
reset role;
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select length(public.my_invite_code('Peter', 7)) as code_len;           -- 10
select public.my_invite_code('Peter', 7) = public.my_invite_code('Peter', 7) as same_code; -- t
select count(*) as can_read_table from public.friend_invites;          -- ERROR or 0: no direct access
reset role;
create temp table c as select code from public.friend_invites where user_id = '11111111-1111-1111-1111-111111111111';
grant select on c to anon, authenticated;
set role anon;
select name, number from public.invite_info((select code from c));      -- Peter | 7
select public.accept_invite((select code from c));                      -- ERROR: sign in first
reset role;
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select public.accept_invite((select code from c)) as own;               -- self
set request.jwt.claim.sub = '33333333-3333-3333-3333-333333333333';
select public.accept_invite('zzzzzzzzzz') as bad;                       -- invalid
select public.accept_invite((select code from c)) as first;             -- ok
select public.accept_invite((select code from c)) as again;             -- ok (still friends, no duplicate)
select status from public.friendships where '33333333-3333-3333-3333-333333333333' in (requester, addressee); -- accepted (1 row)
select public.are_friends('11111111-1111-1111-1111-111111111111', '33333333-3333-3333-3333-333333333333') as friends; -- t
reset role;
