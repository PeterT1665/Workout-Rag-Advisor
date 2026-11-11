import { useState } from 'react'
import { Button, Panel, Screen, Title, TopBar } from '../components/ui'
import { addClip, clipsOf, removeClip } from '../lib/clips'
import { playBlob, stopClip } from '../lib/audio'
import { DEFAULT_QUOTES, loadQuotes, save } from '../lib/storage'
import type { ClipKind, Settings } from '../lib/types'
import { PHOTO_SLOTS, removePhoto, setPhoto, usePhotos, type Slot } from '../lib/photos'
import { usePersonalImage } from '../lib/personal'
import { syncEnabled } from '../lib/sync'
import { ACCENTS, setProfile, useProfile } from '../lib/profile'

// One tile per image slot: shows the photo in use now (the app's default until replaced); tap to change.
function PhotoSlots() {
  return <div className="grid grid-cols-2 gap-3">{PHOTO_SLOTS.map((s) => <PhotoTile key={s.id} id={s.id} label={s.label} hint={s.hint} />)}</div>
}

function PhotoTile({ id, label, hint }: { id: Slot; label: string; hint: string }) {
  const own = usePhotos()[id]
  const shown = usePersonalImage(`${id}.jpg`) // own photo, else the app's default
  return (
    <div className="relative overflow-hidden rounded-2xl bg-surface-2">
      <label className="block cursor-pointer">
        <div className="relative aspect-[4/3] w-full">
          {shown ? <img src={shown} alt="" className={`h-full w-full object-cover ${own ? '' : 'opacity-60'}`} /> : <div className="grid h-full place-items-center text-3xl text-muted" aria-hidden="true">+</div>}
          {!own && shown && <span className="absolute bottom-2 left-2 rounded-full bg-black/60 px-2 py-0.5 text-[11px] text-white">Default · tap to change</span>}
        </div>
        <div className="px-3 py-2">
          <span className="block text-sm font-semibold">{label}</span>
          <span className="block text-xs text-muted">{hint}</span>
        </div>
        <input type="file" accept="image/*" className="sr-only" aria-label={`${own ? 'Replace' : 'Add'} ${label} photo`}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void setPhoto(id, f); e.target.value = '' }} />
      </label>
      {own && (
        <button onClick={() => void removePhoto(id)} aria-label={`Remove ${label} photo`}
          className="absolute top-2 right-2 grid h-9 w-9 place-items-center rounded-full bg-black/60 text-white">✕</button>
      )}
    </div>
  )
}

function ClipList({ kind, settings }: { kind: ClipKind; settings: Settings }) {
  const items = clipsOf(kind)
  if (!items.length) return <p className="text-sm text-muted">None yet.</p>
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0">
      {items.map((c) => (
        <li key={c.id} className="flex items-center justify-between gap-3 rounded-2xl bg-surface-2 px-4 py-2">
          <span className="truncate">{c.name}</span>
          <span className="flex shrink-0 gap-2">
            <Button variant="ghost" size="sm" className="bg-surface" onClick={() => playBlob(settings, c.blob)}>Play</Button>
            <Button variant="ghost" size="sm" className="bg-surface" onClick={() => { stopClip(); void removeClip(c.id) }}>Delete</Button>
          </span>
        </li>
      ))}
    </ul>
  )
}

export function MediaView({ settings, onDone }: { settings: Settings; onDone: () => void }) {
  const [kind, setKind] = useState<ClipKind>('mindset')
  const [quotes, setQuotes] = useState(() => loadQuotes().join('\n'))
  const me = useProfile()
  const field = 'mt-1 block w-full rounded-2xl bg-surface-2 p-3 text-ink'
  return (
    <Screen>
      <TopBar left={<Title>Make it yours</Title>} right={<Button variant="ghost" size="sm" onClick={() => { stopClip(); onDone() }}>Done</Button>} />
      <div className="mb-4" />

      <Section title="Team colour">
        <div className="grid grid-cols-9 gap-1.5">
          {ACCENTS.map((a) => (
            <button key={a.id} aria-pressed={me.accent === a.id} aria-label={a.label} onClick={() => setProfile({ ...me, accent: a.id })}
              className={`aspect-square rounded-full transition-transform duration-150 ease-snap active:scale-90 ${me.accent === a.id ? 'scale-110 ring-2 ring-ink ring-offset-2 ring-offset-surface' : ''}`}
              style={{ background: a.hex }}><span className="sr-only">{a.label}</span></button>
          ))}
        </div>
      </Section>

      <Section title="Photos" hint={syncEnabled() ? 'Private to your account.' : 'Saved in this browser.'}>
        <PhotoSlots />
      </Section>

      <Section title="Role model & quotes" hint="Shown before sessions and in rests.">
        <label htmlFor="idol" className="text-sm text-muted">Role model</label>
        <input id="idol" value={me.idol} placeholder="Your favourite player" onChange={(e) => setProfile({ ...me, idol: e.target.value.slice(0, 40) })} className={field} />
        <label htmlFor="quotes" className="mt-4 block text-sm text-muted">Quotes, one per line</label>
        <textarea id="quotes" rows={6} value={quotes} onChange={(e) => setQuotes(e.target.value)} className={field} />
        <div className="mt-3 flex gap-2">
          <Button size="sm" onClick={() => save('quotes', quotes.split('\n').map((q) => q.trim()).filter(Boolean))}>Save quotes</Button>
          <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { save('quotes', DEFAULT_QUOTES); setQuotes(DEFAULT_QUOTES.join('\n')) }}>Reset</Button>
        </div>
      </Section>

      <Section title="Audio clips" hint="Play before sessions and at the end. Stay on this phone.">
        <div className="flex gap-2">
          <select id="kind" aria-label="Clip type" value={kind} onChange={(e) => setKind(e.target.value as ClipKind)} className="min-h-11 flex-1 rounded-full bg-surface-2 px-4 text-ink">
            <option value="mindset">Mindset clip</option>
            <option value="siuu">End-of-session clip</option>
          </select>
          <label className="grid min-h-11 cursor-pointer place-items-center rounded-full bg-red px-5 font-display text-xs uppercase text-on-red">
            + Add
            <input id="files" type="file" accept="audio/*" multiple className="sr-only" aria-label="Audio files"
              onChange={async (e) => {
                for (const f of Array.from(e.target.files ?? [])) {
                  await addClip({ id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, kind, name: f.name.replace(/\.[^.]+$/, ''), blob: f })
                }
                e.target.value = ''
              }} />
          </label>
        </div>
        <h4 className="mt-5 mb-2 font-display text-sm uppercase text-muted">Mindset</h4>
        <ClipList kind="mindset" settings={settings} />
        <h4 className="mt-5 mb-2 font-display text-sm uppercase text-muted">End of session</h4>
        <ClipList kind="siuu" settings={settings} />
      </Section>
    </Screen>
  )
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <Panel className="mb-4">
      <h3 className="m-0 font-display text-lg uppercase">{title}</h3>
      {hint && <p className="m-0 mt-1 mb-4 text-sm text-muted">{hint}</p>}
      {!hint && <div className="mb-4" />}
      {children}
    </Panel>
  )
}
