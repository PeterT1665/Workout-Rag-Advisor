// Share pictures, Strava-style: the stats stacked in the middle, one graphic under them, the app's name and your streak.
// Three looks: a sticker (transparent, to drop on your own Instagram story photo), black, or on the day's photo.
// Drawn on the phone; nothing is uploaded unless you share the picture.
import type { Graphic, ShareItem } from './shareStats'
import { FIRE, type FlameTone } from '../components/icons'

export type { ShareItem } from './shareStats'
export type Look = 'sticker' | 'black' | 'photo'
export type ShareOpts = { look: Look; photo?: HTMLImageElement | null; streak: number; games?: { wins: number; unbeaten: number }; accent: string; dateLabel: string }

export const W = 1080, H = 1920 // a phone story
const FLAME = 'M12 1.8c.6 3.1-1 4.9-2.6 6.6C7.8 10.1 6 12 6 15a6 6 0 0 0 12 0c0-2.4-1.1-4.3-2.3-5.8-.2 1.3-.8 2.3-1.8 2.9.3-3.9-.9-7.6-1.9-10.3Z'
const FLAME_IN = 'M12 12.2c-.3 1.4-1.1 2.1-1.8 2.8-.6.6-1 1.2-1 2.1a2.8 2.8 0 0 0 5.6 0c0-1.2-.6-2.1-1.3-2.8-.2.6-.6 1-1.1 1.2.1-1.2 0-2.3-.4-3.3Z'
// A social-app type (close to what Instagram uses), loaded only when a share picture is drawn.
const FONT = '"Plus Jakarta Sans Variable", system-ui, sans-serif'
let fontReady: Promise<unknown> | null = null
function loadFont() {
  fontReady ??= import('@fontsource-variable/plus-jakarta-sans').then(() => Promise.all([document.fonts?.load(`800 100px ${FONT}`), document.fonts?.load(`600 40px ${FONT}`)])).catch(() => null)
  return fontReady
}
export function flame(x: CanvasRenderingContext2D, cx: number, top: number, size: number, tone: FlameTone | 'off' = 'plan') {
  const [a, b, c] = FIRE[tone]
  x.save(); x.translate(cx - size / 2, top); x.scale(size / 24, size / 24)
  const g = x.createLinearGradient(0, 24, 0, 0); g.addColorStop(0, a); g.addColorStop(1, b)
  x.fillStyle = g; x.fill(new Path2D(FLAME)); x.fillStyle = c; x.fill(new Path2D(FLAME_IN)); x.restore()
}

function fit(x: CanvasRenderingContext2D, text: string, max: number, px: number, weight: number, font: string) {
  let size = px
  x.font = `${weight} ${size}px ${font}`
  while (x.measureText(text).width > max && size > 40) { size -= 4; x.font = `${weight} ${size}px ${font}` }
  return size
}
const round = (x: CanvasRenderingContext2D, rx: number, ry: number, w: number, h: number, r: number) => { x.beginPath(); x.roundRect(rx, ry, w, h, Math.min(r, w / 2, h / 2)); x.fill() }

function drawGraphic(x: CanvasRenderingContext2D, g: Graphic, cx: number, top: number, accent: string, ink: string) {
  const w = 620, h = 300
  if (g.type === 'sets') {
    // every set of the session, in order: bar height is the weight moved
    const bars = g.bars.slice(0, 40), max = Math.max(...bars)
    const gap = 10, bw = Math.min(46, (w - gap * (bars.length - 1)) / bars.length)
    const total = bars.length * bw + (bars.length - 1) * gap
    x.fillStyle = accent
    bars.forEach((v, i) => { const bh = Math.max(14, (v / max) * h); round(x, cx - total / 2 + i * (bw + gap), top + h - bh, bw, bh, bw / 2) })
  } else if (g.type === 'shots') {
    // the goal: goals in the net (filled), saved shots on target (rings), misses outside the frame (faded)
    const gw = 560, gh = 220, gx = cx - gw / 2, gy = top + 40
    x.strokeStyle = ink; x.lineWidth = 12; x.lineCap = 'round'
    x.beginPath(); x.moveTo(gx, gy + gh); x.lineTo(gx, gy); x.lineTo(gx + gw, gy); x.lineTo(gx + gw, gy + gh); x.stroke()
    const inGoal = Math.min(g.goals, 24), saved = Math.min(Math.max(0, g.onTarget - g.goals), 12), off = Math.min(Math.max(0, g.shots - g.onTarget), 16)
    const r = 17, perRow = 8
    for (let i = 0; i < inGoal; i++) { x.fillStyle = accent; x.beginPath(); x.arc(gx + 50 + (i % perRow) * ((gw - 100) / (perRow - 1)), gy + 50 + Math.floor(i / perRow) * 58, r, 0, Math.PI * 2); x.fill() }
    x.strokeStyle = ink; x.lineWidth = 5
    for (let i = 0; i < saved; i++) { x.beginPath(); x.arc(gx + 70 + (i % 6) * ((gw - 140) / 5), gy + gh + 50 + Math.floor(i / 6) * 50, r - 3, 0, Math.PI * 2); x.stroke() }
    x.save(); x.globalAlpha = 0.35; x.fillStyle = ink
    for (let i = 0; i < off; i++) { const left = i % 2 === 0, k = Math.floor(i / 2); x.beginPath(); x.arc(left ? gx - 50 - (k % 2) * 40 : gx + gw + 50 + (k % 2) * 40, gy + 20 + Math.floor(k / 2) * 60, r - 4, 0, Math.PI * 2); x.fill() }
    x.restore()
  } else if (g.type === 'card') {
    // the six card stats as bars out of 99
    const rows = g.rows.slice(0, 6), rh = 46
    rows.forEach((row, i) => {
      const y = top + i * rh
      x.fillStyle = ink; x.font = `800 30px ${FONT}`; x.textAlign = 'left'; x.fillText(row.label, cx - 300, y + 30)
      x.save(); x.globalAlpha = 0.18; round(x, cx - 200, y + 10, 420, 22, 11); x.restore()
      x.fillStyle = accent; round(x, cx - 200, y + 10, (420 * row.value) / 99, 22, 11)
      x.fillStyle = ink; x.textAlign = 'right'; x.fillText(String(row.value), cx + 300, y + 30)
    })
  } else flame(x, cx, top, 300, g.tone ?? 'plan')
}

export async function drawShare(c: HTMLCanvasElement, item: ShareItem, o: ShareOpts) {
  await loadFont()
  c.width = W; c.height = H
  const x = c.getContext('2d')!
  x.clearRect(0, 0, W, H)
  const ink = o.look === 'black' ? '#f3ede4' : '#ffffff'
  if (o.look === 'black') { x.fillStyle = '#000'; x.fillRect(0, 0, W, H) }
  if (o.look === 'photo') {
    if (o.photo) {
      const s = Math.max(W / o.photo.width, H / o.photo.height), w = o.photo.width * s, h = o.photo.height * s
      x.drawImage(o.photo, (W - w) / 2, (H - h) / 2, w, h)
    } else { x.fillStyle = '#111'; x.fillRect(0, 0, W, H) }
    x.fillStyle = 'rgba(0,0,0,.38)'; x.fillRect(0, 0, W, H)
  }
  if (o.look !== 'black') { x.shadowColor = 'rgba(0,0,0,.35)'; x.shadowBlur = 16 }
  const cx = W / 2
  x.textAlign = 'center'; x.textBaseline = 'alphabetic'; x.fillStyle = ink

  // three stats, label over number, like a Strava card
  let y = 360
  for (const s of item.stats.slice(0, 3)) {
    x.font = `600 44px ${FONT}`; x.fillText(s.label, cx, y)
    const size = fit(x, s.value, W - 140, 128, 800, FONT)
    x.fillText(s.value, cx, y + size + 4)
    y += size + 120
  }
  // one graphic, then the name
  x.shadowBlur = o.look === 'black' ? 0 : 8
  drawGraphic(x, item.graphic, cx, Math.max(y + 20, 1010), o.accent, ink)
  x.shadowBlur = o.look === 'black' ? 0 : 16
  brand(x, cx, 1640, ink, item.kind === 'match' ? o.games : undefined, o.streak)
}

// FOOTBALL COACH, with the streak flame and number in front: the only branding.
function brand(x: CanvasRenderingContext2D, cx: number, y: number, ink: string, games: ShareOpts['games'], streak: number) {
  const g = games ?? { wins: 0, unbeaten: 0 }
  const [n, tone] = games ? (g.wins > 1 ? [g.wins, 'wins' as const] : [g.unbeaten, 'unbeaten' as const]) : [streak, 'plan' as const]
  x.font = `800 58px ${FONT}`
  const word = 'FOOTBALL COACH', ww = x.measureText(word).width
  const num = n > 0 ? String(n) : ''
  x.font = `800 50px ${FONT}`
  const nw = num ? x.measureText(num).width + 56 + 26 : 0
  let left = cx - (ww + nw) / 2
  if (num) {
    flame(x, left + 24, y - 54, 52, tone)
    x.fillStyle = ink; x.textAlign = 'left'; x.fillText(num, left + 54, y - 4)
    left += nw
  }
  x.font = `800 58px ${FONT}`; x.textAlign = 'left'; x.fillStyle = ink
  x.fillText(word, left, y)
  x.textAlign = 'center'
}

// Share the picture (phone share sheet: Instagram, WhatsApp...), or save it where sharing files isn't possible.
export async function sharePicture(c: HTMLCanvasElement, name: string) {
  const blob = await new Promise<Blob | null>((ok) => c.toBlob(ok, 'image/png'))
  if (!blob) return 'failed' as const
  const file = new File([blob], `${name}.png`, { type: 'image/png' })
  try {
    if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file] }); return 'shared' as const }
  } catch (e) { if ((e as Error).name === 'AbortError') return 'cancelled' as const }
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob); a.download = file.name
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 4000)
  return 'saved' as const
}
