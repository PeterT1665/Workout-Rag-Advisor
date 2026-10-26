import { useState } from 'react'
import { Button, Panel, Screen, Title, TopBar } from '../components/ui'
import { addClip, clipsOf, removeClip } from '../lib/clips'
import { playBlob, stopClip } from '../lib/audio'
import { DEFAULT_QUOTES, loadQuotes, save } from '../lib/storage'
import type { ClipKind, Settings } from '../lib/types'

function ClipList({ kind, settings }: { kind: ClipKind; settings: Settings }) {
  const items = clipsOf(kind)
  if (!items.length) return <p className="text-sm text-muted">None yet.</p>
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0">
      {items.map((c) => (
        <li key={c.id} className="flex items-center justify-between gap-3 rounded-md bg-surface px-4 py-2">
          <span className="truncate">{c.name}</span>
          <span className="flex shrink-0 gap-2">
            <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => playBlob(settings, c.blob)}>Play</Button>
            <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { stopClip(); void removeClip(c.id) }}>Delete</Button>
          </span>
        </li>
      ))}
    </ul>
  )
}

export function MediaView({ settings, onDone }: { settings: Settings; onDone: () => void }) {
  const [kind, setKind] = useState<ClipKind>('mindset')
  const [quotes, setQuotes] = useState(() => loadQuotes().join('\n'))
  const field = 'mt-1 block w-full rounded-md bg-surface-2 p-3 text-ink'
  return (
    <Screen>
      <TopBar left={<Title>Clips & quotes</Title>} right={<Button variant="ghost" size="sm" onClick={() => { stopClip(); onDone() }}>Done</Button>} />
      <p className="text-sm text-muted">Clips are stored only on this phone and never uploaded. Keep them private: don't publish them.</p>
      <Panel className="flex flex-col gap-3">
        <h3 className="m-0 font-display text-xl font-bold uppercase">Add clips</h3>
        <label className="text-sm text-muted">Type
          <select id="kind" value={kind} onChange={(e) => setKind(e.target.value as ClipKind)} className={field}>
            <option value="mindset">Mindset (commentary, interviews)</option>
            <option value="siuu">SIUUU (end of session)</option>
          </select>
        </label>
        <label className="text-sm text-muted">Audio files
          <input id="files" type="file" accept="audio/*" multiple className={field}
            onChange={async (e) => {
              for (const f of Array.from(e.target.files ?? [])) {
                await addClip({ id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, kind, name: f.name.replace(/\.[^.]+$/, ''), blob: f })
              }
              e.target.value = ''
            }} />
        </label>
      </Panel>
      <h2 className="mt-8 mb-3 font-display text-lg uppercase">Mindset clips</h2>
      <ClipList kind="mindset" settings={settings} />
      <h2 className="mt-8 mb-3 font-display text-lg uppercase">SIUUU</h2>
      <ClipList kind="siuu" settings={settings} />
      <Panel className="mt-8 flex flex-col gap-3">
        <h3 className="m-0 font-display text-xl font-bold uppercase">Motivational cards</h3>
        <label className="text-sm text-muted">One quote per line
          <textarea rows={7} value={quotes} onChange={(e) => setQuotes(e.target.value)} className={field} />
        </label>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => save('quotes', quotes.split('\n').map((q) => q.trim()).filter(Boolean))}>Save quotes</Button>
          <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { save('quotes', DEFAULT_QUOTES); setQuotes(DEFAULT_QUOTES.join('\n')) }}>Reset</Button>
        </div>
      </Panel>
    </Screen>
  )
}
