import { useEffect, useRef, useState } from 'react'
import { Button } from './ui'
import { useSheet } from '../lib/useSheet'
import { t } from '../i18n'

// Crop a photo: drag to move, pinch (or the slider) to zoom. Square inside a circle guide (profile and card pictures), or any
// shape with `aspect` (width / height: 9/16 for a story). Saves an image of what's in the frame, `size` px wide.

export function CropSheet({ file, title = t('Move and scale'), size = 900, aspect = 1, onDone, onCancel }: {
  file: Blob; title?: string; size?: number; aspect?: number; onDone: (b: Blob) => void; onCancel: () => void
}) {
  // the frame on screen: 280 px for a square; a tall one fits the screen height
  const FW = aspect >= 1 ? 280 : Math.round(Math.min(280, (typeof innerHeight === 'number' ? innerHeight - 300 : 500) * aspect)), FH = Math.round(FW / aspect)
  useSheet(onCancel)
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [url, setUrl] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pos, setPos] = useState({ x: 0, y: 0 }) // image top-left inside the frame
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const gesture = useRef<{ dist: number; zoom: number } | null>(null)

  useEffect(() => {
    const u = URL.createObjectURL(file)
    const i = new Image()
    i.onload = () => {
      setUrl(u)
      setImg(i)
      const s = Math.max(FW / i.width, FH / i.height)
      setPos({ x: (FW - i.width * s) / 2, y: (FH - i.height * s) / 2 })
    }
    i.src = u
    return () => URL.revokeObjectURL(u)
  }, [file])

  const base = img ? Math.max(FW / img.width, FH / img.height) : 1
  const scale = base * zoom
  const w = (img?.width ?? 0) * scale
  const h = (img?.height ?? 0) * scale
  // the photo always covers the frame
  const clamp = (p: { x: number; y: number }, ww = w, hh = h) => ({ x: Math.min(0, Math.max(FW - ww, p.x)), y: Math.min(0, Math.max(FH - hh, p.y)) })

  // zoom around the frame's centre
  const zoomTo = (z: number) => {
    if (!img) return
    const nz = Math.min(4, Math.max(1, z))
    const ns = base * nz
    const cx = (FW / 2 - pos.x) / scale
    const cy = (FH / 2 - pos.y) / scale
    setZoom(nz)
    setPos(clamp({ x: FW / 2 - cx * ns, y: FH / 2 - cy * ns }, img.width * ns, img.height * ns))
  }

  const down = (e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()]
      gesture.current = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom }
    }
  }
  const move = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId)
    if (!prev) return
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.current.size === 2 && gesture.current) {
      const [a, b] = [...pointers.current.values()]
      zoomTo(gesture.current.zoom * (Math.hypot(a.x - b.x, a.y - b.y) / gesture.current.dist))
    } else if (pointers.current.size === 1) {
      setPos((p) => clamp({ x: p.x + e.clientX - prev.x, y: p.y + e.clientY - prev.y }))
    }
  }
  const up = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2) gesture.current = null
  }

  const save = () => {
    if (!img) return
    const canvas = document.createElement('canvas')
    canvas.width = size; canvas.height = Math.round(size / aspect)
    const ctx = canvas.getContext('2d')
    if (!ctx) { onDone(file); return }
    ctx.drawImage(img, -pos.x / scale, -pos.y / scale, FW / scale, FH / scale, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((b) => onDone(b ?? file), 'image/jpeg', 0.88)
  }

  return (
    <div className="fixed inset-0 z-[70] flex flex-col items-center justify-center bg-black px-5 text-white" role="dialog" aria-modal="true" aria-label={title}>
      <h2 className="m-0 font-display text-lg uppercase">{title}</h2>
      <p className="m-0 mt-1 text-sm text-white/60">{t('Drag to move · pinch or slide to zoom')}</p>
      <div className="relative mt-6 touch-none select-none overflow-hidden rounded-2xl" style={{ width: FW, height: FH }}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
        onWheel={(e) => zoomTo(zoom * (e.deltaY < 0 ? 1.08 : 0.92))}>
        {img && url && <img src={url} alt="" draggable={false} className="pointer-events-none absolute max-w-none"
          style={{ left: pos.x, top: pos.y, width: w, height: h }} />}
        {/* square: a circle guide, dim outside it, like a profile picture; other shapes: the frame's edge */}
        {aspect === 1 ? <>
          <div className="pointer-events-none absolute inset-0 z-10" style={{ background: 'radial-gradient(circle closest-side, transparent calc(100% - 1px), rgba(0,0,0,.72) 100%)' }} />
          <div className="pointer-events-none absolute inset-0 z-10 rounded-full border-2 border-white/85" />
        </> : <div className="pointer-events-none absolute inset-0 z-10 rounded-2xl border-2 border-white/85" />}
      </div>
      <label className="mt-6 flex w-full max-w-[280px] items-center gap-3 text-sm text-white/70">
        <span aria-hidden="true">−</span>
        <input type="range" min={1} max={4} step={0.01} value={zoom} onChange={(e) => zoomTo(Number(e.target.value))} aria-label={t('Zoom')} className="flex-1 accent-[var(--red)]" />
        <span aria-hidden="true">+</span>
      </label>
      <div className="mt-8 flex w-full max-w-[320px] gap-3 pb-[env(safe-area-inset-bottom)]">
        <Button variant="ghost" className="flex-1 bg-white/10 text-white" onClick={onCancel}>{t('Cancel')}</Button>
        <Button className="flex-1" disabled={!img} onClick={save}>{t('Use photo')}</Button>
      </div>
    </div>
  )
}
