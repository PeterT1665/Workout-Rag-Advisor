-- Per-user app state, mirroring the app's local-first keys (one JSON value per key).
-- Every row belongs to one user and row-level security lets each user see and change only their own rows.
-- Run once in the Supabase SQL editor (or `supabase db push`).

create table if not exists public.user_state (
  user_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  key text not null check (key in ('history', 'games', 'settings', 'quotes', 'mySessions', 'profile')),
  value jsonb not null,
  updated_at bigint not null, -- client clock, ms since epoch; newer wins (history is merged instead)
  primary key (user_id, key)
);

alter table public.user_state enable row level security;

drop policy if exists "users manage their own state" on public.user_state;
create policy "users manage their own state" on public.user_state
  for all to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));

-- New tables are not exposed to the API automatically (project setting), so grant explicitly. Never to anon.
revoke all on public.user_state from anon;
grant select, insert, update, delete on public.user_state to authenticated;

-- Union of sessions from two devices. Same session (date|id|minutes|title): the copy with a check-in (load) wins,
-- then the one with more fields, then the stored one. Same rule as the old FastAPI backend.
create or replace function public.merge_history(stored jsonb, incoming jsonb)
returns jsonb
language sql
immutable
set search_path = ''
as $$
  select coalesce(jsonb_agg(e order by e ->> 'date', ord, n), '[]'::jsonb)
  from (
    select distinct on (k) e, ord, n
    from (
      select e, ord, n,
        concat_ws('|', e ->> 'date', e ->> 'id', e ->> 'minutes', e ->> 'title') as k,
        (e ->> 'load') is not null as checked,
        (select count(*) from jsonb_object_keys(e)) as fields
      from (
        select e, 0 as ord, n from jsonb_array_elements(coalesce(stored, '[]'::jsonb)) with ordinality as t(e, n)
        union all
        select e, 1 as ord, n from jsonb_array_elements(coalesce(incoming, '[]'::jsonb)) with ordinality as t(e, n)
      ) x
    ) y
    order by k, checked desc, fields desc, ord
  ) z
$$;

-- Write one key for the signed-in user. Newer timestamp wins; history is merged, never overwritten.
-- Runs with the caller's rights, so row-level security still applies.
create or replace function public.put_state(p_key text, p_value jsonb, p_updated_at bigint)
returns public.user_state
language plpgsql
security invoker
set search_path = ''
as $$
declare
  cur public.user_state;
  v jsonb := p_value;
  u bigint := p_updated_at;
begin
  if auth.uid() is null then
    raise exception 'not signed in';
  end if;
  select * into cur from public.user_state where user_id = auth.uid() and key = p_key for update;
  if found then
    if p_key = 'history' then
      v := public.merge_history(cur.value, p_value);
      u := greatest(p_updated_at, cur.updated_at, (extract(epoch from now()) * 1000)::bigint);
    elsif cur.updated_at >= p_updated_at then
      return cur; -- stored copy is newer: keep it
    end if;
  end if;
  insert into public.user_state (user_id, key, value, updated_at)
  values (auth.uid(), p_key, v, u)
  on conflict (user_id, key) do update set value = excluded.value, updated_at = excluded.updated_at
  returning * into cur;
  return cur;
end;
$$;

revoke execute on function public.put_state(text, jsonb, bigint) from public, anon;
grant execute on function public.put_state(text, jsonb, bigint) to authenticated;
revoke execute on function public.merge_history(jsonb, jsonb) from public, anon;
grant execute on function public.merge_history(jsonb, jsonb) to authenticated;
