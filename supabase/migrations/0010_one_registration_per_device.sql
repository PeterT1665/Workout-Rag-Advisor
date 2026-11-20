-- One notification registration per device for each app (staging and production count as two apps).
-- A reinstall or a renewed push address used to leave the old registration behind; notifications then went out
-- to dead ones too. The app now saves which app and device a registration is for, and removes that device's
-- older ones whenever it saves a new one (web/src/lib/push.ts).
alter table public.push_subscriptions add column if not exists app text;
alter table public.push_subscriptions add column if not exists device text;
create index if not exists push_subscriptions_device on public.push_subscriptions (user_id, app, device);

-- One-time clean-up: registrations from before this can't tell their device apart, so they go.
-- Every phone with notifications on registers again (once) the next time it opens the app.
delete from public.push_subscriptions where app is null;
