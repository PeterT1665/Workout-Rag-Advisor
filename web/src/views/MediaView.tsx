import { useState } from 'react'
import { Button, Panel, Screen, Title, TopBar } from '../components/ui'
import { addClip, clipsOf, removeClip } from '../lib/clips'
import { playBlob, stopClip } from '../lib/audio'
import { DEFAULT_QUOTES, loadQuotes, save } from '../lib/storage'
import type { ClipKind, Settings } from '../lib/types'
import { PHOTO_SLOTS, removePhoto, setPhoto, usePhotos } from '../lib/photos'
import { syncEnabled } from '../lib/sync'

// One tile per image slot: tap to choose a photo, replace or remove it.
function PhotoSlots() {
  const photos = usePhotos()
  return (
    <div className="grid grid-cols-2 gap-3">
      {PHOTO_SLOTS.map((s) => {
        const url = photos[s.id]
        return (
          <div key={s.id} className="relative overflow-hidden rounded-2xl bg-surface-2">
            <label className="block cursor-pointer">
              <div className="aspect-[4/3] w-full">
                {url ? <img src={url} alt="" className="h-full w-full object-cover" /> : <div className="grid h-full place-items-center text-3xl text-muted" aria-hidden="true">+</div>}
              </div>
              <div className="px-3 py-2">
                <span className="block text-sm font-semibold">{s.label}</span>
                <span className="block text-xs text-muted">{s.hint}</span>
              </div>
              <input type="file" accept="image/*" className="sr-only" aria-label={`${url ? 'Replace' : 'Add'} ${s.label} photo`}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) void setPhoto(s.id, f); e.target.value = '' }} />
            </label>
            {url && (
              <button onClick={() => void removePhoto(s.id)} aria-label={`Remove ${s.label} photo`}
                className="absolute top-2 right-2 grid h-9 w-9 place-items-center rounded-full bg-black/60 text-white">✕</button>
            )}
          </div>
        )
      })}
    </div>
  )
}

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
      <TopBar left={<Title>Media</Title>} right={<Button variant="ghost" size="sm" onClick={() => { stopClip(); onDone() }}>Done</Button>} />
      <Panel className="mb-6 flex flex-col gap-3">
        <h3 className="m-0 font-display text-xl font-bold uppercase">Photos</h3>
        <p className="m-0 text-sm text-muted">Your role model, your team, your gear: your photos replace the app's. {syncEnabled() ? 'Saved privately to your account: only you can see them.' : 'Saved in this browser only.'} Use photos you're allowed to use, and keep them to yourself.</p>
        <PhotoSlots />
      </Panel>
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
