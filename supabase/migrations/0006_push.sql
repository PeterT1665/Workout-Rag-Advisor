-- Web push notifications: "rest is over" during a session, and "plan next week" on Sunday evening.
-- The app schedules a rest notification when it goes to the background mid-rest (iPhone pauses the app, so it can't
-- remind itself); the send-push Edge Function delivers due ones every few seconds (see supabase/setup/push_cron.sql).

-- One row per device that allowed notifications.
create table if not exists public.push_subscriptions (
  endpoint text primary key check (endpoint ~ '^https://' and char_length(endpoint) <= 1000),
  user_id uuid not null references auth.users (id) on delete cascade,
  p256dh text not null check (char_length(p256dh) <= 200),
  auth text not null check (char_length(auth) <= 100),
  rest_alerts boolean not null default true,
  weekly boolean not null default true,       -- Sunday "plan next week"
  tz text not null default 'UTC' check (char_length(tz) <= 64), -- for Sunday 18:00 local time
  created_at timestamptz not null default now()
);
create index if not exists push_subscriptions_user on public.push_subscriptions (user_id);
alter table public.push_subscriptions enable row level security;

drop policy if exists "push: my devices" on public.push_subscriptions;
create policy "push: my devices" on public.push_subscriptions for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
grant select, insert, update, delete on public.push_subscriptions to authenticated;
grant select, delete on public.push_subscriptions to service_role; -- the sender drops devices that unsubscribed

-- Notifications waiting to go out. `tag` dedupes (one weekly reminder per week; a new rest replaces the last).
create table if not exists public.scheduled_pushes (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  send_at timestamptz not null,
  kind text not null check (kind in ('rest', 'weekly')),
  title text not null check (char_length(title) <= 80),
  body text not null check (char_length(body) <= 200),
  url text not null default '/' check (url ~ '^/' and char_length(url) <= 100),
  tag text check (char_length(tag) <= 80),
  sent_at timestamptz,
  created_at timestamptz not null default now(),
  check (send_at < created_at + interval '1 hour') -- a rest is minutes, not days
);
create unique index if not exists scheduled_pushes_tag on public.scheduled_pushes (user_id, tag) where tag is not null;
create index if not exists scheduled_pushes_due on public.scheduled_pushes (send_at) where sent_at is null;
alter table public.scheduled_pushes enable row level security;

-- Players schedule and cancel their own rest reminders only; weekly ones are made by the server.
drop policy if exists "push queue: I see mine" on public.scheduled_pushes;
drop policy if exists "push queue: I schedule my rest" on public.scheduled_pushes;
drop policy if exists "push queue: I cancel mine" on public.scheduled_pushes;
create policy "push queue: I see mine" on public.scheduled_pushes for select to authenticated
  using (user_id = (select auth.uid()));
create policy "push queue: I schedule my rest" on public.scheduled_pushes for insert to authenticated
  with check (user_id = (select auth.uid()) and kind = 'rest' and sent_at is null);
create policy "push queue: I cancel mine" on public.scheduled_pushes for delete to authenticated
  using (user_id = (select auth.uid()) and sent_at is null);
grant select, insert, delete on public.scheduled_pushes to authenticated;

-- Schedule (or move) this player's rest reminder: one pending rest reminder at a time.
create or replace function public.schedule_rest_push(p_seconds integer, p_title text, p_body text)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if auth.uid() is null then raise exception 'not signed in'; end if;
  delete from public.scheduled_pushes where user_id = auth.uid() and kind = 'rest' and sent_at is null;
  insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
  values (auth.uid(), now() + make_interval(secs => greatest(1, least(p_seconds, 1800))), 'rest', p_title, p_body, '/', null);
end;
$$;
revoke execute on function public.schedule_rest_push(integer, text, text) from public, anon;
grant execute on function public.schedule_rest_push(integer, text, text) to authenticated;

-- For the sender (service role only): take the due notifications, marking them sent so no two runs send one twice.
create or replace function public.claim_due_pushes(p_limit integer default 200)
returns table (id bigint, user_id uuid, title text, body text, url text, kind text)
language sql
security definer
set search_path = ''
as $$
  update public.scheduled_pushes q set sent_at = now()
  where q.id in (
    select s.id from public.scheduled_pushes s
    where s.sent_at is null and s.send_at <= now()
    order by s.send_at
    limit p_limit
    for update skip locked
  )
  returning q.id, q.user_id, q.title, q.body, q.url, q.kind
$$;
revoke execute on function public.claim_due_pushes(integer) from public, anon, authenticated;
grant execute on function public.claim_due_pushes(integer) to service_role;

-- Sunday 18:00 in each player's time zone: queue "plan next week" (once per week). Run hourly by cron.
create or replace function public.queue_weekly_pushes()
returns integer
language sql
security definer
set search_path = ''
as $$
  with due as (
    select distinct on (p.user_id) p.user_id, (now() at time zone p.tz)::date as local_day
    from public.push_subscriptions p
    where p.weekly
      and extract(isodow from now() at time zone p.tz) = 7
      and extract(hour from now() at time zone p.tz) = 18
  ), ins as (
    insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag)
    select d.user_id, now(), 'weekly', 'Plan next week', 'Your game days and gym days for next week are ready to check. Two minutes now saves the week.', '/?tab=week', 'weekly-' || d.local_day
    from due d
    on conflict (user_id, tag) where tag is not null do nothing
    returning 1
  )
  select count(*)::integer from ins
$$;
revoke execute on function public.queue_weekly_pushes() from public, anon, authenticated;
grant execute on function public.queue_weekly_pushes() to service_role;

-- Old queue rows go after a day.
create or replace function public.prune_pushes()
returns void
language sql
security definer
set search_path = ''
as $$
  delete from public.scheduled_pushes where created_at < now() - interval '1 day' and (sent_at is not null or send_at < now() - interval '1 hour')
$$;
revoke execute on function public.prune_pushes() from public, anon, authenticated;
grant execute on function public.prune_pushes() to service_role;
