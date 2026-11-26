import { partLabel, type Part } from '../lib/injuries'
import { t } from '../i18n'

// A small front and back figure with the muscles and joints an exercise works (red) and also loads (lighter red).
// The regions are the library's own words (works / also): hamstrings, quads, calves, ankles, hips, groin, knees, glutes, back,
// chest, shoulder, arms, core.
type Shape = { region: string; d: { x: number; y: number; w: number; h: number; r: number } | { cx: number; cy: number; rx: number; ry: number } }
const rect = (region: string, x: number, y: number, w: number, h: number, r = 5): Shape => ({ region, d: { x, y, w, h, r } })
const oval = (region: string, cx: number, cy: number, rx: number, ry = rx): Shape => ({ region, d: { cx, cy, rx, ry } })
const legs: Shape[] = [
  oval('knees', 22, 146, 7), oval('knees', 48, 146, 7),
  rect('calves', 15, 154, 14, 26, 6), rect('calves', 41, 154, 14, 26, 6),
  oval('ankles', 22, 185, 6, 4), oval('ankles', 48, 185, 6, 4),
]
const upper: Shape[] = [oval('shoulder', 14, 34, 9, 7), oval('shoulder', 56, 34, 9, 7), rect('arms', 3, 42, 10, 44, 5), rect('arms', 57, 42, 10, 44, 5)]
const FRONT: Shape[] = [
  ...upper, rect('chest', 22, 32, 26, 22, 7), rect('core', 24, 56, 22, 28, 5),
  rect('hips', 15, 86, 9, 14, 4), rect('hips', 46, 86, 9, 14, 4), rect('groin', 28, 88, 14, 16, 4),
  rect('quads', 14, 102, 17, 40, 7), rect('quads', 39, 102, 17, 40, 7), ...legs,
]
const BACK: Shape[] = [
  ...upper, rect('back', 20, 32, 30, 50, 8), rect('glutes', 16, 84, 38, 18, 8),
  rect('hamstrings', 14, 104, 17, 38, 7), rect('hamstrings', 39, 104, 17, 38, 7), ...legs,
]

export function MuscleMap({ works, also, className = '' }: { works?: string[]; also?: string[]; className?: string }) {
  const fill = (region: string) => (works?.includes(region) ? 'var(--red)' : also?.includes(region) ? 'color-mix(in srgb, var(--red) 40%, transparent)' : 'color-mix(in srgb, var(--text) 13%, transparent)')
  const label = (xs?: string[]) => (xs ?? []).map((x) => partLabel(x as Part)).join(', ')
  const draw = (shapes: Shape[], dx: number) => (
    <g transform={`translate(${dx} 0)`}>
      <circle cx="35" cy="12" r="9" fill="color-mix(in srgb, var(--text) 13%, transparent)" />
      {shapes.map((s, i) => ('x' in s.d
        ? <rect key={i} x={s.d.x} y={s.d.y} width={s.d.w} height={s.d.h} rx={s.d.r} fill={fill(s.region)} />
        : <ellipse key={i} cx={s.d.cx} cy={s.d.cy} rx={s.d.rx} ry={s.d.ry} fill={fill(s.region)} />))}
    </g>
  )
  return (
    <svg viewBox="0 0 150 192" role="img" className={className} aria-label={`${t('Works')}: ${label(works)}${also?.length ? `. ${t('Also loads')}: ${label(also)}` : ''}`}>
      {draw(FRONT, 0)}{draw(BACK, 78)}
    </svg>
  )
}
