-- A rest call-back goes only to the device whose session it is (it used to go to every device on the account,
-- so the same phone with two installs, or two phones, buzzed twice). Weekly reminders go to the newest device.

alter table public.scheduled_pushes add column if not exists endpoint text check (char_length(endpoint) <= 1000);

drop function if exists public.schedule_rest_push(integer, text, text);
create or replace function public.schedule_rest_push(p_seconds integer, p_title text, p_body text, p_endpoint text default null)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if auth.uid() is null then raise exception 'not signed in'; end if;
  -- only one of my own devices; anything else falls back to all my devices
  if p_endpoint is not null and not exists (select 1 from public.push_subscriptions where endpoint = p_endpoint and user_id = auth.uid()) then
    p_endpoint := null;
  end if;
  delete from public.scheduled_pushes where user_id = auth.uid() and kind = 'rest' and sent_at is null;
  insert into public.scheduled_pushes (user_id, send_at, kind, title, body, url, tag, endpoint)
  values (auth.uid(), now() + make_interval(secs => greatest(1, least(p_seconds, 1800))), 'rest', p_title, p_body, '/', null, p_endpoint);
end;
$$;
revoke execute on function public.schedule_rest_push(integer, text, text, text) from public, anon;
grant execute on function public.schedule_rest_push(integer, text, text, text) to authenticated;

drop function if exists public.claim_due_pushes(integer);
create or replace function public.claim_due_pushes(p_limit integer default 200)
returns table (id bigint, user_id uuid, title text, body text, url text, kind text, endpoint text)
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
  returning q.id, q.user_id, q.title, q.body, q.url, q.kind, q.endpoint
$$;
revoke execute on function public.claim_due_pushes(integer) from public, anon, authenticated;
grant execute on function public.claim_due_pushes(integer) to service_role;
