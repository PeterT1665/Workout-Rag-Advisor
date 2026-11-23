-- Exercises players ask for: a name, the body parts it works and a short how-to. The player can use it in their own
-- sessions straight away; the team builds its 3D demo and adds it to the shared library later (status).
-- Players can add and read their own requests only. Reading them all, and changing the status, is for the team
-- (service role / dashboard).
create table if not exists public.exercise_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  local_id text not null, -- the id the app gave it (my-ex-...), so a resend doesn't make a copy
  name text not null check (char_length(name) between 2 and 60),
  parts text[] not null default '{}' check (cardinality(parts) <= 14),
  how text not null default '' check (char_length(how) <= 600),
  sport text not null default 'gym' check (sport in ('football', 'gym', 'home', 'mobility')),
  lang text, -- the app's language when it was written
  status text not null default 'requested' check (status in ('requested', 'building', 'added', 'declined')),
  created_at timestamptz not null default now(),
  unique (user_id, local_id)
);

alter table public.exercise_requests enable row level security;

create policy "players add their own requests" on public.exercise_requests
  for insert to authenticated with check (user_id = (select auth.uid()));
create policy "players read their own requests" on public.exercise_requests
  for select to authenticated using (user_id = (select auth.uid()));

-- No more than 20 requests a day per player (a typo-fixing resend is the same row, so it doesn't count twice).
create or replace function public.exercise_requests_limit()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if (select count(*) from public.exercise_requests r
      where r.user_id = new.user_id and r.created_at > now() - interval '1 day') >= 20 then
    raise exception 'too many exercise requests today';
  end if;
  return new;
end;
$$;
drop trigger if exists exercise_requests_limit on public.exercise_requests;
create trigger exercise_requests_limit before insert on public.exercise_requests
  for each row execute function public.exercise_requests_limit();
