import { Button, Panel, Screen, Slider, Title, Toggle, TopBar } from '../components/ui'
import { say } from '../lib/audio'
import type { Settings } from '../lib/types'
import { deleteAccount, signOut, syncEnabled } from '../lib/sync'
import { safeAreaInfo } from '../lib/safeArea'
import { DEMO, seedDemo, startOverDemo } from '../lib/demo'
import { positionTag, useProfile } from '../lib/profile'
import credits from '../data/credits.json'

export function SettingsView({ settings: s, clipCount, onChange, onMedia, onProfile, onDone }: {
  settings: Settings; clipCount: number; onChange: (p: Partial<Settings>) => void; onMedia: () => void; onProfile: () => void; onDone: () => void
}) {
  const me = useProfile()
  return (
    <Screen>
      <TopBar left={<Title>Settings</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>Done</Button>} />
      <Panel className="mb-6 flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="truncate font-display text-lg uppercase">{me.name} <span className="text-muted">#{me.number}</span></div>
          <div className="text-sm text-muted">{positionTag(me)} · {me.foot} foot{me.idol ? ` · ${me.idol}` : ''}</div>
        </div>
        <Button variant="ghost" size="sm" className="shrink-0 bg-surface-2" onClick={onProfile}>Edit player</Button>
      </Panel>
      <Panel className="flex flex-col gap-4">
        <Toggle id="voice" label="Spoken announcements and cues" checked={s.voice} onChange={(v) => onChange({ voice: v })} />
        <Slider id="vol" label="Voice volume" min={0.2} max={1} step={0.1} value={s.volume} onChange={(v) => onChange({ volume: v })} />
        <Slider id="cmin" label="Shortest gap between cues (s)" min={2} max={10} step={1} value={s.cueMin} onChange={(v) => onChange({ cueMin: v })} />
        <Slider id="cmax" label="Longest gap between cues (s)" min={3} max={15} step={1} value={s.cueMax} onChange={(v) => onChange({ cueMax: v })} />
        <Button variant="ghost" className="bg-surface-2" onClick={() => { say(s, 'Left', true); setTimeout(() => say(s, 'Right', true), 900) }}>Test voice</Button>
      </Panel>
      <p className="text-sm text-muted">Cues come at random times, only in the 1v1 drills. Left/right means the defender is committing that way: go the other way.</p>

      <h2 className="mt-8 mb-3 font-display text-lg uppercase">Mindset</h2>
      <Panel className="flex flex-col gap-4">
        <Toggle id="cs" label="Mindset clip before the session" checked={s.clipStart} onChange={(v) => onChange({ clipStart: v })} />
        <Toggle id="cr" label="Clips during rests (never during drills)" checked={s.clipRest} onChange={(v) => onChange({ clipRest: v })} />
        <Slider id="rc" label="Chance of a clip on each rest" min={0.1} max={1} step={0.1} value={s.restClipChance} onChange={(v) => onChange({ restClipChance: v })} />
        <Toggle id="se" label="SIUUU at the end" checked={s.siuuEnd} onChange={(v) => onChange({ siuuEnd: v })} />
        <Toggle id="cd" label="Motivational cards" checked={s.cards} onChange={(v) => onChange({ cards: v })} />
        <Slider id="cv" label="Clip volume" min={0.2} max={1} step={0.1} value={s.clipVolume} onChange={(v) => onChange({ clipVolume: v })} />
        <Button variant="ghost" className="bg-surface-2" onClick={onMedia}>Make it yours: photos, colour, clips ({clipCount})</Button>
      </Panel>
      {!syncEnabled() && (
          <Panel className="mt-8 flex items-center justify-between gap-4">
            <p className="m-0 text-sm text-muted">{me.sample ? 'Sample player: not a real log.' : 'Guest mode: saved in this browser.'}</p>
            <div className="flex shrink-0 flex-col gap-2">
              {me.sample && <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { seedDemo(new Date(), true); location.reload() }}>Reset sample</Button>}
              <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { startOverDemo(); location.reload() }}>Start over</Button>
            </div>
          </Panel>
      )}
      {DEMO && (
        <>
          {credits.length > 0 && (
            <>
              <h2 className="mt-8 mb-3 font-display text-lg uppercase">Photo credits</h2>
              <ul className="m-0 flex list-none flex-col gap-1 p-0 text-sm text-muted">
                {(credits as { photographer: string; source: string; url: string }[]).map((c) => (
                  <li key={c.url}><a href={c.url} target="_blank" rel="noreferrer" className="underline">{c.photographer}</a> on {c.source}</li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
      <Panel className="mt-8">
        <h3 className="m-0 mb-2 font-display text-base uppercase">Privacy</h3>
        <p className="m-0 text-sm text-muted">
          {syncEnabled()
            ? 'Your data is private to your account. Clips stay on this phone.'
            : 'Everything stays in this browser.'}
        </p>
      </Panel>
      {syncEnabled() && (
        <Panel className="mt-4 flex items-center justify-between gap-4">
          <p className="m-0 text-sm text-muted">Signed in and synced.</p>
          <Button variant="ghost" size="sm" className="bg-surface-2" onClick={async () => { await signOut(); location.reload() }}>Sign out</Button>
        </Panel>
      )}
      {syncEnabled() && (
        <button className="mt-6 min-h-11 w-full text-sm font-semibold text-red-ink"
          onClick={async () => {
            if (!confirm('Delete your account and all its data for good?')) return
            if (await deleteAccount()) location.reload()
            else alert("Couldn't delete it. Check your connection.")
          }}>
          Delete my account
        </button>
      )}
      {/* tiny diagnostic: lets a screenshot show how the phone reports its screen edges */}
      <p className="mt-8 text-center text-[11px] text-muted/60">
        {safeAreaInfo ? `display ${safeAreaInfo.standalone ? 'app' : safeAreaInfo.fullScreen ? 'full screen' : 'browser'} · top ${Math.round(safeAreaInfo.reported)}px${safeAreaInfo.applied !== null ? ` → ${safeAreaInfo.applied}px` : ''}` : ''}
      </p>
    </Screen>
  )
}
