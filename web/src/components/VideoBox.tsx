import { useState } from 'react'
import { Button } from './ui'
import { t } from '../i18n'
import { cleanLink, embedFor, searchLink, setVideo, useVideo } from '../lib/videos'
import { Demo3D } from './Demo3D'
import { demoFor } from '../lib/demo3d/catalog'

// The exercise's demo: a 3D figure for every library exercise, plus your own video link if you saved one.
export function VideoBox({ name }: { name: string }) {
  const has3d = !!demoFor(name)
  return (
    <>
      {has3d && <Demo3D name={name} className="mt-4" />}
      <LinkBox name={name} folded={has3d} />
    </>
  )
}

// Your own video: plays your saved link, or lets you paste one (find it on TikTok, copy link, paste).
function LinkBox({ name, folded }: { name: string; folded: boolean }) {
  const url = useVideo(name)
  const [open, setOpen] = useState(!folded)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState(false)
  const embed = url ? embedFor(url) : null

  if (url && !editing) {
    return (
      <div className="mt-4">
        {embed
          ? <div className={`overflow-hidden rounded-2xl bg-black ${embed.tall ? 'mx-auto aspect-[9/16] max-h-[60dvh]' : 'aspect-video'}`}>
              <iframe src={embed.src} title={t('{name} video', { name })} className="h-full w-full" allow="encrypted-media; picture-in-picture" allowFullScreen loading="lazy" />
            </div>
          : <a href={url} target="_blank" rel="noreferrer" className="flex min-h-12 items-center justify-center rounded-2xl bg-surface-2 font-semibold">▶ {t('Watch the video')}</a>}
        <div className="mt-2 flex justify-end gap-3 text-sm">
          <a href={url} target="_blank" rel="noreferrer" className="text-muted underline underline-offset-4">{t('Open')}</a>
          <button onClick={() => { setDraft(url); setEditing(true) }} className="text-muted underline underline-offset-4">{t('Change')}</button>
          <button onClick={() => setVideo(name, null)} className="text-muted underline underline-offset-4">{t('Remove')}</button>
        </div>
      </div>
    )
  }

  if (!open) {
    return <button onClick={() => setOpen(true)} className="mt-2 min-h-10 w-full text-sm font-semibold text-muted underline underline-offset-4">{t('Add your own video')}</button>
  }
  return (
    <div className="mt-4 rounded-2xl bg-surface-2 p-3">
      <p className="m-0 text-sm font-semibold">{folded ? t('Your own video') : t('Video')}</p>
      <p className="m-0 mt-0.5 text-sm text-muted">{t('Paste a TikTok, YouTube or Instagram link. Only you see it.')}</p>
      <div className="mt-2 flex gap-2">
        <input value={draft} onChange={(e) => { setDraft(e.target.value); setError(false) }} placeholder="https://www.tiktok.com/@…/video/…"
          aria-label={t('Video link for {name}', { name })} inputMode="url" className="min-h-11 min-w-0 flex-1 rounded-xl bg-surface px-3 text-[16px] text-ink placeholder:text-muted" />
        <Button size="sm" onClick={() => { const c = cleanLink(draft); if (!c) { setError(true); return } setVideo(name, c); setEditing(false); setDraft('') }}>{t('Save')}</Button>
      </div>
      {error && <p className="m-0 mt-1 text-sm text-red-ink">{t("That doesn't look like a link.")}</p>}
      <div className="mt-2 flex justify-between text-sm">
        <a href={searchLink(name)} target="_blank" rel="noreferrer" className="text-muted underline underline-offset-4">{t('Find one on TikTok')}</a>
        {(editing || folded) && <button onClick={() => { setEditing(false); if (folded) setOpen(false) }} className="text-muted underline underline-offset-4">{t('Cancel')}</button>}

      </div>
    </div>
  )
}
