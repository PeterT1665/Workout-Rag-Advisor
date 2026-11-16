import { useEffect, useState } from 'react'
import { Button, Panel, Toggle } from './ui'
import { disablePush, enablePush, getPrefs, pushState, setPrefs, type Prefs, type PushState } from '../lib/push'

const WHY: Partial<Record<PushState, string>> = {
  'need-account': 'Notifications need an account (they come from our server).',
  'need-install': 'On iPhone, add the app to your Home Screen first (Share → Add to Home Screen), open it from there, then turn notifications on here.',
  unsupported: "This browser can't receive notifications from the app.",
  denied: 'Notifications are blocked. Turn them on in iPhone Settings → Notifications → Football Personal Coach.',
}

// Settings: rest-over call-backs and the Sunday reminder.
export function NotificationSettings() {
  const [state, setState] = useState<PushState>(pushState)
  const [prefs, setP] = useState<Prefs>({ rest_alerts: true, weekly: true })
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (state === 'on') void getPrefs().then((p) => p && setP(p)) }, [state])

  const change = async (p: Partial<Prefs>) => {
    const next = { ...prefs, ...p }
    setP(next)
    if (state === 'on') {
      if (!next.rest_alerts && !next.weekly) { await disablePush(); setState(pushState()) }
      else await setPrefs(p)
      return
    }
    setBusy(true)
    setState(await enablePush(next))
    setBusy(false)
  }

  return (
    <Panel className="flex flex-col gap-4">
      {WHY[state]
        ? <p className="m-0 text-sm text-muted">{WHY[state]}</p>
        : <>
            <Toggle id="n-rest" label="Rest is over: call me back if I leave the app" checked={state === 'on' && prefs.rest_alerts} onChange={(v) => void change({ rest_alerts: v })} />
            <Toggle id="n-week" label="Sunday evening: plan next week" checked={state === 'on' && prefs.weekly} onChange={(v) => void change({ weekly: v })} />
            {busy && <p className="m-0 text-sm text-muted">Asking your phone…</p>}
          </>}
    </Panel>
  )
}

// Home, once: ask before the first session that needs it. Hidden when it can't work or was answered.
const ASKED = 'push:asked'
export function NotifyPrompt() {
  const [state, setState] = useState<PushState>(pushState)
  const [asked, setAsked] = useState(() => { try { return localStorage.getItem(ASKED) === '1' } catch { return true } })
  if (asked || state !== 'off') return null
  const done = () => { try { localStorage.setItem(ASKED, '1') } catch { /* ignore */ } setAsked(true) }
  return (
    <section aria-label="Notifications" className="rise mt-6 rounded-[22px] bg-surface p-4">
      <p className="m-0 font-semibold">Don't lose your rest to TikTok</p>
      <p className="m-0 mt-1 text-sm text-muted">If you leave the app during a rest, we'll call you back the moment it's over. Plus a Sunday nudge to plan your week.</p>
      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" onClick={async () => { const s = await enablePush(); setState(s); if (s === 'on' || s === 'denied') done() }}>Turn on</Button>
        <button onClick={done} className="min-h-10 px-3 text-sm font-semibold text-muted">Not now</button>
      </div>
    </section>
  )
}
