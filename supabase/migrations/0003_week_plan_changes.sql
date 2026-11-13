-- Allow syncing manual week changes ('weekPlan': which session or rest a player put on a given date).
alter table public.user_state drop constraint if exists user_state_key_check;
alter table public.user_state add constraint user_state_key_check
  check (key in ('history', 'games', 'settings', 'quotes', 'mySessions', 'profile', 'weekPlan'));
