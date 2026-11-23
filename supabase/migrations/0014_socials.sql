-- Social links on the public profile (Instagram, TikTok, Snapchat, Facebook, WhatsApp, Zalo), so friends can message
-- each other and make group chats there. Same rule as the rest of the profile: only you and accepted friends read it
-- (policy "profiles: me and my friends read", 0005). The app cleans the values; the database only keeps them small.
alter table public.profiles add column if not exists socials jsonb not null default '{}'::jsonb;
alter table public.profiles drop constraint if exists profiles_socials_small;
alter table public.profiles add constraint profiles_socials_small
  check (jsonb_typeof(socials) = 'object' and pg_column_size(socials) < 1200);
