-- Expected: user 1 can write and merge; user 2 sees nothing of user 1; anon is refused; unknown keys are refused.
set role authenticated;
set request.jwt.claim.sub = '11111111-1111-1111-1111-111111111111';
select key, updated_at from public.put_state('games', '[{"day":2}]', 100);
select key, updated_at from public.put_state('games', '[{"day":9}]', 50);  -- older: stored copy kept (100)
select key from public.put_state('weekPlan', '{"2026-11-13":{"rest":true}}', 1);
select public.put_state('bogus', '1', 1);                                  -- ERROR: check constraint
set request.jwt.claim.sub = '22222222-2222-2222-2222-222222222222';
select count(*) as user2_sees from public.user_state;                      -- 0
reset role; set role anon;
select count(*) from public.user_state;                                    -- ERROR: permission denied
