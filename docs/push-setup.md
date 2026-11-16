# Push notifications: one-time setup (Supabase dashboard)

What it does:
- **"Rest is over":** leave the app mid-rest and it calls you back when the rest ends.
- **"Plan next week":** a reminder on Sunday at 18:00, in your time zone.

How it works:
1. The app schedules a reminder in the database when it goes to the background during a rest, and cancels it when you come back.
2. A database job checks every 5 seconds. Only when something is due does it call the `send-push` function, which delivers to every device of that player.

Secrets live in Supabase only: the VAPID private key and the cron secret. They are never in this repo. The VAPID public key is in `web/.env.production` (public by design).

## Steps (about 15 minutes, in this order)

1. **Extensions:** Database → Extensions: enable **pg_cron** and **pg_net**.
2. **Tables:** SQL editor → run `supabase/migrations/0006_push.sql`.
3. **Function:**
   1. Edge Functions → Deploy a new function → Via editor.
   2. Name it `send-push` and paste `supabase/functions/send-push/index.ts`. Deploy.
   3. In the function's settings, turn **Verify JWT** (Enforce JWT verification) **off**. The function checks its own secret instead.
4. **Function secrets:** Edge Functions → Secrets → add `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY` and `CRON_SECRET`. The values are in the secrets file Claude generated (never commit or paste them in chat).
5. **Schedule:**
   1. SQL editor → open `supabase/setup/push_cron.sql`.
   2. Replace the two placeholders: your project URL (`https://<ref>.supabase.co`) and the same `CRON_SECRET`.
   3. Run it.
   4. Check with `select jobname, schedule, active from cron.job;`.
6. **Phone:**
   1. Open the staging app from your Home Screen.
   2. Sign in, and tap **Turn on** on the Home card (or Settings → Notifications). Allow.
   3. Start a session, wait for a rest, swipe to another app. The notification should arrive when the rest ends.

## If it doesn't arrive

- **Function logs:** Edge Functions → send-push → Logs. A `401` means `CRON_SECRET` differs between the function secret and the vault secret.
- **Queue:** `select * from scheduled_pushes order by id desc limit 5;`
  - A row with `sent_at` set means it was handed to Apple.
  - A row with no `sent_at` past its time means the cron job isn't calling the function: check step 5.
- **Phone:** iPhone needs the app opened from the Home Screen (iOS 16.4+), and notifications allowed in iPhone Settings.
