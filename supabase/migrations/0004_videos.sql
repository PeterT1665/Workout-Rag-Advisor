-- Allow syncing each player's own video links per exercise ('videos': exercise name -> link). Private, like every key.
alter table public.user_state drop constraint if exists user_state_key_check;
alter table public.user_state add constraint user_state_key_check
  check (key in ('history', 'games', 'settings', 'quotes', 'mySessions', 'profile', 'weekPlan', 'videos'));
