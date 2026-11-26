-- Showcase stats on the public profile: the 3 to 6 stats a player picks to show friends (goals, assists, 20 m sprint, gym PRs).
-- Each item is { k: key, v: value as text }, worked out by the app from what it logged. Same rule as the rest of the profile:
-- only you and accepted friends read it (policy "profiles: me and my friends read", 0005). The database only keeps it small.
alter table public.profiles add column if not exists showcase jsonb not null default '[]'::jsonb;
alter table public.profiles drop constraint if exists profiles_showcase_small;
alter table public.profiles add constraint profiles_showcase_small
  check (jsonb_typeof(showcase) = 'array' and jsonb_array_length(showcase) <= 6 and pg_column_size(showcase) < 1500);
