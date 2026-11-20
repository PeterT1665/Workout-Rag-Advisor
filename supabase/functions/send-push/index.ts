// send-push: delivers due notifications (rest is over, plan next week, match day, friends) to every device of the player.
// Called by pg_cron (supabase/setup/push_cron.sql) only when something is due, with the x-cron-secret header.
// Secrets (Edge Function secrets): VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY, CRON_SECRET.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase. Deploy with "Verify JWT" OFF.
import webpush from 'npm:web-push@3.6.7'
import { createClient } from 'npm:@supabase/supabase-js@2'

const env = (k: string) => Deno.env.get(k) ?? ''

webpush.setVapidDetails('https://football-personal-coach.ducphu1665.workers.dev', env('VAPID_PUBLIC_KEY'), env('VAPID_PRIVATE_KEY'))

type Due = { id: number; user_id: string; title: string; body: string; url: string; kind: string; endpoint: string | null }
type Sub = { endpoint: string; p256dh: string; auth: string; rest_alerts: boolean; weekly: boolean; matches?: boolean; friends?: boolean; created_at: string }

Deno.serve(async (req) => {
  if (!env('CRON_SECRET') || req.headers.get('x-cron-secret') !== env('CRON_SECRET')) return new Response('no', { status: 401 })
  const sb = createClient(env('SUPABASE_URL'), env('SUPABASE_SERVICE_ROLE_KEY'), { auth: { persistSession: false } })

  const { data: due, error } = await sb.rpc('claim_due_pushes', { p_limit: 200 })
  if (error) return new Response(error.message, { status: 500 })
  const list = (due ?? []) as Due[]
  if (!list.length) return Response.json({ sent: 0 })

  const users = [...new Set(list.map((d) => d.user_id))]
  const { data: subs } = await sb.from('push_subscriptions').select('*').in('user_id', users)
  const byUser = new Map<string, Sub[]>()
  for (const s of (subs ?? []) as (Sub & { user_id: string })[]) byUser.set(s.user_id, [...(byUser.get(s.user_id) ?? []), s])

  let sent = 0
  const gone: string[] = []
  const failed: string[] = []
  const noDevice = list.filter((d) => !byUser.get(d.user_id)?.length).length
  // A rest goes to the device whose session it is (all devices if it didn't say); "plan next week" and match-day
  // reminders to the newest one.
  const targets = (d: Due) => {
    const mine = byUser.get(d.user_id) ?? []
    if (d.kind === 'rest') {
      const one = d.endpoint ? mine.filter((s) => s.endpoint === d.endpoint) : []
      return (one.length ? one : mine).filter((s) => s.rest_alerts)
    }
    // friend requests: every device that wants them (one per app since 0010), so whichever you pick up shows it
    if (d.kind === 'friend') return mine.filter((s) => s.friends !== false)
    const wants = (s: Sub) => (d.kind === 'match' ? s.matches !== false : s.weekly)
    return mine.filter(wants).sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 1)
  }
  await Promise.all(list.flatMap((d) => targets(d)
    .map(async (s) => {
      try {
        await webpush.sendNotification(
          { endpoint: s.endpoint, keys: { p256dh: s.p256dh, auth: s.auth } },
          JSON.stringify({ title: d.title, body: d.body, url: d.url, kind: d.kind, tag: `${d.kind}-${d.id}` }),
          { TTL: d.kind === 'rest' ? 120 : d.kind === 'match' ? 3 * 3600 : d.kind === 'friend' ? 24 * 3600 : 6 * 3600, urgency: d.kind === 'weekly' ? 'normal' : 'high' },
        )
        sent++
      } catch (e) {
        const err = e as { statusCode?: number; body?: string; message?: string }
        if (err.statusCode === 404 || err.statusCode === 410) gone.push(s.endpoint) // the device unsubscribed
        const why = `${new URL(s.endpoint).host} ${err.statusCode ?? ''} ${String(err.body ?? err.message ?? e).slice(0, 200)}`
        failed.push(why)
        console.error('push failed:', why)
      }
    })))
  if (gone.length) await sb.from('push_subscriptions').delete().in('endpoint', gone)
  const result = { due: list.length, sent, gone: gone.length, noDevice, failed }
  console.log('send-push', JSON.stringify(result))
  return Response.json(result)
})
