-- Read-only database login for debugging (notifications, friends, sync) from the developer's laptop.
-- It can read every table and the push/cron logs, and can't change anything: every transaction is read-only,
-- and it has no insert/update/delete rights. Run once in the SQL Editor after replacing the password.
-- To remove it later:  drop owned by claude_ro; drop role claude_ro;

create role claude_ro login password 'REPLACE-WITH-A-LONG-RANDOM-PASSWORD';
alter role claude_ro set default_transaction_read_only = on;
alter role claude_ro set statement_timeout = '15s';

grant usage on schema public to claude_ro;
grant select on all tables in schema public to claude_ro;

-- Row level security hides every row from a new role: let this one read (only read) all rows.
do $$
declare t text;
begin
  for t in select tablename from pg_tables where schemaname = 'public' loop
    execute format('drop policy if exists "debug: read only" on public.%I', t);
    execute format('create policy "debug: read only" on public.%I for select to claude_ro using (true)', t);
  end loop;
end $$;

-- The cron runs and the send-push replies (who got a notification, and why one failed).
do $$ begin
  grant usage on schema cron to claude_ro;
  grant select on cron.job, cron.job_run_details to claude_ro;
exception when others then raise notice 'cron logs not shared: %', sqlerrm;
end $$;
do $$ begin
  grant usage on schema net to claude_ro;
  grant select on net._http_response to claude_ro;
exception when others then raise notice 'send-push replies not shared: %', sqlerrm;
end $$;
