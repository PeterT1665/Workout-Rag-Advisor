-- Push notifications: the schedule. Run ONCE in the SQL editor, after:
--   1. Database -> Extensions: enable pg_cron and pg_net
--   2. migration 0006_push.sql
--   3. Edge Functions: deploy send-push (Verify JWT off) and set its secrets
-- Replace the two placeholders below before running. Never commit the real values.

select vault.create_secret('https://YOUR-PROJECT-REF.supabase.co', 'project_url');
select vault.create_secret('PASTE-CRON_SECRET-HERE', 'push_cron_secret');

-- Every 5 seconds: if a notification is due, ask send-push to deliver it (no call at all when nothing is due).
select cron.schedule('send-push', '5 seconds', $job$
  select net.http_post(
    url := (select decrypted_secret from vault.decrypted_secrets where name = 'project_url') || '/functions/v1/send-push',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'x-cron-secret', (select decrypted_secret from vault.decrypted_secrets where name = 'push_cron_secret')
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 8000
  )
  where exists (select 1 from public.scheduled_pushes where sent_at is null and send_at <= now())
$job$);

-- Every hour: queue Sunday 18:00 "plan next week" reminders (each player's own time zone), tidy the queue.
select cron.schedule('weekly-pushes', '0 * * * *', $job$ select public.queue_weekly_pushes(); select public.prune_pushes(); $job$);

-- Check: select jobname, schedule, active from cron.job;
-- Undo:  select cron.unschedule('send-push'); select cron.unschedule('weekly-pushes');
