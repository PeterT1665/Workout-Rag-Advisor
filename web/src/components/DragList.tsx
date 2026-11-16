import { useEffect, useRef, useState, type ReactNode } from 'react'

// Drag-to-reorder for lists where some items travel together (a superset moves as one).
// Hold the grip and move: only the dragged group follows the finger, the page stays still, and the list scrolls
// only when you hold the group near the top or bottom edge. Keyboard: arrow keys on the grip.
//
// `units` are the groups in order; `onMove(from, to)` gets unit indexes. Each unit is rendered by `render`,
// which receives the grip's props (spread them on the grip button) and whether it's being dragged.

export type GripProps = {
  onPointerDown: (e: React.PointerEvent) => void
  onKeyDown: (e: React.KeyboardEvent) => void
  style: React.CSSProperties
}

type Drag = { from: number; to: number; dy: number; heights: number[]; gap: number }

const EDGE = 72 // px from the edge where auto-scroll starts
const SPEED = 14 // px per frame at the very edge

export function DragList<U>({ units, keyOf, render, onMove, scroller, className = 'flex flex-col gap-2' }: {
  units: U[]
  keyOf: (u: U) => string
  render: (u: U, index: number, grip: GripProps, dragging: boolean) => ReactNode
  onMove: (from: number, to: number) => void
  scroller?: React.RefObject<HTMLElement | null> // scrolling container; the page when absent
  className?: string
}) {
  const list = useRef<HTMLDivElement>(null)
  const [drag, setDragState] = useState<Drag | null>(null)
  const latest = useRef<Drag | null>(null)
  const setDrag = (d: Drag | null) => { latest.current = d; setDragState(d) }
  const live = useRef<{ from: number; startY: number; startScroll: number; y: number; tops: number[]; heights: number[]; gap: number; raf: number } | null>(null)

  const scrollEl = () => scroller?.current ?? null
  const getScroll = () => scrollEl()?.scrollTop ?? window.scrollY
  const setScroll = (v: number) => { const el = scrollEl(); if (el) el.scrollTop = v; else window.scrollTo(0, v) }
  const viewport = () => {
    const el = scrollEl()
    if (!el) return { top: 0, bottom: window.innerHeight }
    const r = el.getBoundingClientRect()
    return { top: r.top, bottom: r.bottom }
  }

  // where the dragged unit would land, from its centre against the others' original centres
  const targetFor = (from: number, dy: number, tops: number[], heights: number[]) => {
    const centre = tops[from] + heights[from] / 2 + dy
    let to = 0
    for (let i = 0; i < tops.length; i++) {
      if (i === from) continue
      if (tops[i] + heights[i] / 2 < centre) to++
    }
    return to
  }

  const update = () => {
    const l = live.current
    if (!l) return
    const dy = l.y - l.startY + (getScroll() - l.startScroll)
    setDrag({ from: l.from, to: targetFor(l.from, dy, l.tops, l.heights), dy, heights: l.heights, gap: l.gap })
  }

  const stop = (commit: boolean) => {
    const l = live.current
    if (!l) return
    cancelAnimationFrame(l.raf)
    live.current = null
    document.documentElement.style.removeProperty('overscroll-behavior')
    const d = latest.current
    setDrag(null)
    if (commit && d && d.to !== d.from) onMove(d.from, d.to)
  }

  useEffect(() => {
    if (!drag) return
    // while dragging, a finger move never scrolls the page (iOS ignores touch-action on its own)
    const block = (e: TouchEvent) => e.preventDefault()
    const move = (e: PointerEvent) => { if (live.current) { live.current.y = e.clientY; update() } }
    const up = () => stop(true)
    const cancel = () => stop(false)
    window.addEventListener('touchmove', block, { passive: false })
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', cancel)
    return () => {
      window.removeEventListener('touchmove', block)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', cancel)
    }
  }, [drag !== null]) // eslint-disable-line react-hooks/exhaustive-deps

  const start = (index: number) => (e: React.PointerEvent) => {
    if (e.button !== 0 || !list.current) return
    e.preventDefault()
    const kids = [...list.current.children] as HTMLElement[]
    const rects = kids.map((k) => k.getBoundingClientRect())
    const scroll = getScroll()
    const gap = rects.length > 1 ? Math.max(0, rects[1].top - rects[0].bottom) : 8
    live.current = {
      from: index, startY: e.clientY, startScroll: scroll, y: e.clientY,
      tops: rects.map((r) => r.top + scroll), heights: rects.map((r) => r.height), gap, raf: 0,
    }
    document.documentElement.style.setProperty('overscroll-behavior', 'none')
    // auto-scroll near the edges
    const tick = () => {
      const l = live.current
      if (!l) return
      const v = viewport()
      const push = l.y < v.top + EDGE ? -(1 - (l.y - v.top) / EDGE) : l.y > v.bottom - EDGE ? 1 - (v.bottom - l.y) / EDGE : 0
      if (push) { setScroll(getScroll() + Math.sign(push) * Math.ceil(Math.min(1, Math.abs(push)) * SPEED)); update() }
      l.raf = requestAnimationFrame(tick)
    }
    live.current.raf = requestAnimationFrame(tick)
    update()
  }

  const keys = (index: number) => (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowUp' && index > 0) { e.preventDefault(); onMove(index, index - 1) }
    if (e.key === 'ArrowDown' && index < units.length - 1) { e.preventDefault(); onMove(index, index + 1) }
  }

  // others slide out of the way by the dragged unit's height
  const shift = (i: number) => {
    if (!drag || i === drag.from) return 0
    const h = drag.heights[drag.from] + drag.gap
    if (drag.from < drag.to && i > drag.from && i <= drag.to) return -h
    if (drag.from > drag.to && i >= drag.to && i < drag.from) return h
    return 0
  }

  return (
    <div ref={list} className={className}>
      {units.map((u, i) => {
        const dragging = drag?.from === i
        return (
          <div key={keyOf(u)} className="relative"
            style={{
              transform: `translateY(${dragging ? drag!.dy : shift(i)}px)${dragging ? ' scale(1.02)' : ''}`,
              transition: dragging ? 'none' : 'transform 180ms cubic-bezier(.2,.8,.2,1)',
              zIndex: dragging ? 20 : undefined,
              boxShadow: dragging ? '0 18px 40px rgba(0,0,0,.55)' : undefined,
              borderRadius: dragging ? 18 : undefined,
            }}>
            {render(u, i, { onPointerDown: start(i), onKeyDown: keys(i), style: { touchAction: 'none', cursor: drag ? 'grabbing' : 'grab' } }, dragging)}
          </div>
        )
      })}
    </div>
  )
}
