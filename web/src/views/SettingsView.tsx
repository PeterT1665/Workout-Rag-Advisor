import { Button, Panel, Screen, Slider, Title, Toggle, TopBar } from '../components/ui'
import { say } from '../lib/audio'
import type { Settings } from '../lib/types'

export function SettingsView({ settings: s, clipCount, onChange, onMedia, onDone }: {
  settings: Settings; clipCount: number; onChange: (p: Partial<Settings>) => void; onMedia: () => void; onDone: () => void
}) {
  return (
    <Screen>
      <TopBar left={<Title>Settings</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>Done</Button>} />
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
        <Button variant="ghost" className="bg-surface-2" onClick={onMedia}>Clips & quotes ({clipCount} clips)</Button>
      </Panel>
    </Screen>
  )
}
