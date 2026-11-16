// send-push: delivers due notifications (rest is over, plan next week) to every device of the player.
// Called by pg_cron (supabase/setup/push_cron.sql) only when something is due, with the x-cron-secret header.
// Secrets (Edge Function secrets): VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY, CRON_SECRET.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase. Deploy with "Verify JWT" OFF.
import webpush from 'npm:web-push@3.6.7'
import { createClient } from 'npm:@supabase/supabase-js@2'

const env = (k: string) => Deno.env.get(k) ?? ''

webpush.setVapidDetails('https://football-personal-coach.ducphu1665.workers.dev', env('VAPID_PUBLIC_KEY'), env('VAPID_PRIVATE_KEY'))

type Due = { id: number; user_id: string; title: string; body: string; url: string; kind: string }
type Sub = { endpoint: string; p256dh: string; auth: string; rest_alerts: boolean; weekly: boolean }

Deno.serve(async (req) => {
  if (!env('CRON_SECRET') || req.headers.get('x-cron-secret') !== env('CRON_SECRET')) return new Response('no', { status: 401 })
  const sb = createClient(env('SUPABASE_URL'), env('SUPABASE_SERVICE_ROLE_KEY'), { auth: { persistSession: false } })

  const { data: due, error } = await sb.rpc('claim_due_pushes', { p_limit: 200 })
  if (error) return new Response(error.message, { status: 500 })
  const list = (due ?? []) as Due[]
  if (!list.length) return Response.json({ sent: 0 })

  const users = [...new Set(list.map((d) => d.user_id))]
  const { data: subs } = await sb.from('push_subscriptions').select('endpoint, p256dh, auth, rest_alerts, weekly, user_id').in('user_id', users)
  const byUser = new Map<string, Sub[]>()
  for (const s of (subs ?? []) as (Sub & { user_id: string })[]) byUser.set(s.user_id, [...(byUser.get(s.user_id) ?? []), s])

  let sent = 0
  const gone: string[] = []
  await Promise.all(list.flatMap((d) => (byUser.get(d.user_id) ?? [])
    .filter((s) => (d.kind === 'rest' ? s.rest_alerts : s.weekly))
    .map(async (s) => {
      try {
        await webpush.sendNotification(
          { endpoint: s.endpoint, keys: { p256dh: s.p256dh, auth: s.auth } },
          JSON.stringify({ title: d.title, body: d.body, url: d.url, tag: d.kind }),
          { TTL: d.kind === 'rest' ? 120 : 6 * 3600, urgency: d.kind === 'rest' ? 'high' : 'normal' },
        )
        sent++
      } catch (e) {
        const code = (e as { statusCode?: number }).statusCode
        if (code === 404 || code === 410) gone.push(s.endpoint) // the device unsubscribed
      }
    })))
  if (gone.length) await sb.from('push_subscriptions').delete().in('endpoint', gone)
  return Response.json({ sent, gone: gone.length })
})
