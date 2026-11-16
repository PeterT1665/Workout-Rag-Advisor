import { useState } from 'react'
import { Button, Panel, Screen, Title, TopBar } from '../components/ui'
import { DEFAULT_QUOTES, loadQuotes, save } from '../lib/storage'
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
          {shown ? <img src={shown} alt="" data-fade className={`h-full w-full object-cover ${own ? '' : 'opacity-60'}`} /> : <div className="grid h-full place-items-center text-3xl text-muted" aria-hidden="true">+</div>}
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

export function MediaView({ onDone }: { onDone: () => void }) {
  const [quotes, setQuotes] = useState(() => loadQuotes().join('\n'))
  const me = useProfile()
  const field = 'mt-1 block w-full rounded-2xl bg-surface-2 p-3 text-ink'
  return (
    <Screen>
      <TopBar left={<Title>Make it yours</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>Done</Button>} />
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
