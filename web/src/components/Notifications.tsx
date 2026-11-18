import { useEffect, useState } from 'react'
import { Button, Panel, Toggle } from './ui'
import { disablePush, enablePush, getPrefs, pushState, sendTest, setPrefs, type Prefs, type PushState } from '../lib/push'
import { t } from '../i18n'

const WHY: Partial<Record<PushState, string>> = {
  'need-account': t('Notifications need an account (they come from our server).'),
  'need-install': t('On iPhone, add the app to your Home Screen first (Share → Add to Home Screen), open it from there, then turn notifications on here.'),
  unsupported: t("This browser can't receive notifications from the app."),
  denied: t('Notifications are blocked. Turn them on in iPhone Settings → Notifications → Football Personal Coach.'),
}

// Settings: rest-over call-backs and the Sunday reminder.
export function NotificationSettings() {
  const [state, setState] = useState<PushState>(pushState)
  const [prefs, setP] = useState<Prefs>({ rest_alerts: true, weekly: true, matches: true })
  const [busy, setBusy] = useState(false)
  useEffect(() => { if (state === 'on') void getPrefs().then((p) => p && setP({ ...p, matches: p.matches ?? true })) }, [state])

  const change = async (p: Partial<Prefs>) => {
    const next = { ...prefs, ...p }
    setP(next)
    if (state === 'on') {
      if (!next.rest_alerts && !next.weekly && next.matches === false) { await disablePush(); setState(pushState()) }
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
            <Toggle id="n-rest" label={t('Rest is over: call me back if I leave the app')} checked={state === 'on' && prefs.rest_alerts} onChange={(v) => void change({ rest_alerts: v })} />
            <Toggle id="n-week" label={t('Sunday evening: plan next week')} checked={state === 'on' && prefs.weekly} onChange={(v) => void change({ weekly: v })} />
            <Toggle id="n-match" label={t('Game days: when do you play, then rate your match')} checked={state === 'on' && prefs.matches !== false} onChange={(v) => void change({ matches: v })} />
            {busy && <p className="m-0 text-sm text-muted">{t('Asking your phone…')}</p>}
            {state === 'on' && __TEST_TOOLS__ && <TestButton />}
          </>}
    </Panel>
  )
}

function TestButton() {
  const [msg, setMsg] = useState('')
  return (
    <div className="flex items-center gap-3">
      <Button size="sm" variant="ghost" onClick={async () => setMsg((await sendTest()) ? t('Sent. Lock your phone: it arrives in about 10 seconds.') : t("Couldn't reach the server. Check your connection."))}>{t('Send a test')}</Button>
      {msg && <p className="m-0 text-sm text-muted">{msg}</p>}
    </div>
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
    <section aria-label={t('Notifications')} className="rise mt-6 rounded-[22px] bg-surface p-4">
      <p className="m-0 font-semibold">{t("Don't lose your rest to TikTok")}</p>
      <p className="m-0 mt-1 text-sm text-muted">{t("If you leave the app during a rest, we'll call you back the moment it's over. Plus a Sunday nudge to plan your week.")}</p>
      <div className="mt-3 flex items-center gap-2">
        <Button size="sm" onClick={async () => { const s = await enablePush(); setState(s); if (s === 'on' || s === 'denied') done() }}>{t('Turn on')}</Button>
        <button onClick={done} className="min-h-10 px-3 text-sm font-semibold text-muted">{t('Not now')}</button>
      </div>
    </section>
  )
}
