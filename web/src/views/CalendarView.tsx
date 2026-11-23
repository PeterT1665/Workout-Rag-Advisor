import { useEffect, useMemo, useRef, useState } from 'react'
import { FlameIcon } from '../components/icons'
import { useSheet } from '../lib/useSheet'
import { counts, type Streak } from '../lib/streak'
import { resultOf, resultStreaks, type MatchReview } from '../lib/match'
import { ACCENTS, useProfile } from '../lib/profile'
import { latestBody } from '../lib/body'
import { drawShare, sharePicture, type Look } from '../lib/share'
import { matchShare, ratingShare, sessionShare, type ShareItem } from '../lib/shareStats'
import { loadImage, setDayPhoto, useDayPhotos } from '../lib/dayPhotos'
import type { HistoryEntry } from '../lib/storage'
import { locale, t } from '../i18n'

const ymd = (d: Date) => d.toLocaleDateString('en-CA')
const parse = (iso: string) => { const [y, m, d] = iso.split('-').map(Number); return new Date(y, m - 1, d) }
const longDate = (iso: string) => parse(iso).toLocaleDateString(locale(), { weekday: 'long', day: 'numeric', month: 'long' })
const STREAK_GOLD = '#ffc21a'
// Results, Valorant-style: green win, red loss, grey draw.
const WIN = '#2fd6a0', LOSS = '#ff4655', DRAW = '#9aa0a6'
const RESULT_TILE: Record<'W' | 'D' | 'L', string> = { W: 'bg-[#2fd6a0]/25 text-[#2fd6a0]', D: 'bg-white/15 text-[#c4c8cc]', L: 'bg-[#ff4655]/25 text-[#ff6b77]' }

// Recent games as a results list: the result word in its colour, the score, your rating and goals/assists, and the
// streak flame each game was part of (purple: a win streak, blue: unbeaten). Tap one to share it.
function GameList({ matches, onShare }: { matches: MatchReview[]; onShare: (m: MatchReview) => void }) {
  const recent = matches.slice().sort((a, b) => b.date.localeCompare(a.date)).slice(0, 12)
  if (!recent.length) return <p className="m-0 mt-4 rounded-[18px] bg-white/[.06] p-4 text-center text-sm text-white/60">{t('No games reviewed yet. After a game, review it from Home and it shows here.')}</p>
  // the streak each game belonged to, counted from the oldest
  const old = recent.slice().reverse()
  const runs = new Map<MatchReview, { wins: number; unbeaten: number }>()
  let w = 0, u = 0
  for (const m of old) { const r = resultOf(m); w = r === 'W' ? w + 1 : 0; u = r === 'L' ? 0 : u + 1; runs.set(m, { wins: w, unbeaten: u }) }
  return (
    <ul className="m-0 mt-4 flex list-none flex-col gap-1.5 p-0">
      {recent.map((m) => {
        const r = resultOf(m), run = runs.get(m)!
        const color = r === 'W' ? WIN : r === 'L' ? LOSS : DRAW
        return (
          <li key={m.date}>
            <button onClick={() => onShare(m)} className="flex w-full items-center gap-3 overflow-hidden rounded-[14px] py-2.5 pr-3 pl-0 text-left active:scale-[0.99]"
              style={{ background: `linear-gradient(90deg, ${color}38, ${color}12 60%, transparent)` }}>
              <span className="h-11 w-1.5 shrink-0 rounded-r-full" style={{ background: color }} />
              <span className="w-[5.9rem] shrink-0">
                <span className="block whitespace-nowrap font-display text-[14px] uppercase leading-none tracking-wide" style={{ color }}>{r === 'W' ? t('Victory') : r === 'L' ? t('Defeat') : t('Draw')}</span>
                <span className="mt-1 block font-display text-lg tabular-nums leading-none">{m.us} – {m.them}</span>
              </span>
              <span className="min-w-0 flex-1 text-xs text-white/65">
                {m.rating >= 8.5 && <span className="block truncate font-display text-[10px] uppercase leading-tight text-[#ffc21a]">{t('Match MVP')}</span>}
                <span className="block"><span className="font-semibold text-white">{m.rating.toFixed(1)}</span> {t('rating')}</span>
                <span className="block truncate">{t('G/A {g}/{a}', { g: m.goals ?? 0, a: m.assists ?? 0 })} · {parse(m.date).toLocaleDateString(locale(), { day: 'numeric', month: 'short' })}</span>
              </span>
              {(run.wins > 1 || run.unbeaten > 1) && (
                <span className="flex shrink-0 items-center gap-0.5 font-display text-sm tabular-nums">
                  <FlameIcon tone={run.wins > 1 ? 'wins' : 'unbeaten'} className="h-5 w-5" />{run.wins > 1 ? run.wins : run.unbeaten}
                </span>
              )}
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function TypeIcon({ type, className = 'h-6 w-6' }: { type?: string; className?: string }) {
  if (type === 'gym' || type === 'home') return <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><path d="M6.5 7v10M17.5 7v10M3.5 9.5v5M20.5 9.5v5M6.5 12h11" /></svg>
  if (type === 'recovery') return <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 19c0-8 6-13 14-14-1 8-6 14-14 14Zm0 0 7-7" /></svg>
  return <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="12" cy="12" r="8.5" /><path d="m12 8 3 2.2-1.1 3.6h-3.8L9 10.2 12 8Zm0 0V3.6m3 6.6 4.6-1.4m-5.7 5 2.8 3.9m-6.6-3.9-2.8 3.9M9 10.2 4.4 8.8" strokeLinejoin="round" /></svg>
}

// The streak calendar (Locket-style), from the streak flame on Home: a small window over the app. Every trained day is
// a tile with the day's photo; streak days get a gold frame, game days a flame. Switch to games to
// see results (win streak: purple fire, unbeaten: blue). Tap a day to see it and share it.
export function CalendarView({ history, matches, streak, onBack }: { history: HistoryEntry[]; matches: MatchReview[]; streak: Streak; onBack: () => void }) {
  useSheet(onBack)
  const p = useProfile()
  const photos = useDayPhotos()
  const todayKey = ymd(new Date())
  const [month, setMonth] = useState(() => { const d = new Date(); return new Date(d.getFullYear(), d.getMonth(), 1) })
  const [mode, setMode] = useState<'plan' | 'games'>('plan')
  const [picked, setPicked] = useState<string | null>(null)
  const [sharing, setSharing] = useState<ShareItem | null>(null)
  const games = resultStreaks(matches)
  const gameTone = games.wins > 1 ? 'wins' as const : 'unbeaten' as const // a win streak burns purple, unbeaten blue
  const byDay = useMemo(() => { const m = new Map<string, HistoryEntry[]>(); for (const h of history) if (counts(h)) m.set(h.date, [...(m.get(h.date) ?? []), h]); return m }, [history])
  const matchOn = useMemo(() => new Map(matches.map((m) => [m.date, m])), [matches])
  const first = (month.getDay() + 6) % 7
  const daysIn = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate()
  const cells = [...Array(first).fill(null), ...Array.from({ length: daysIn }, (_, i) => ymd(new Date(month.getFullYear(), month.getMonth(), i + 1)))] as (string | null)[]
  const atNow = month.getFullYear() === new Date().getFullYear() && month.getMonth() === new Date().getMonth()
  // the streak runs: how many kept days in a row end on each day (shown on the last day of a run)
  const st = streak.status ?? {}
  const add = (iso: string, n: number) => { const d = parse(iso); d.setDate(d.getDate() + n); return ymd(d) }
  const kept = (k: string) => st[k] === 'kept' || st[k] === 'made-up'
  const runEnd = (key: string) => {
    if (!kept(key) || (kept(add(key, 1)) && add(key, 1) <= todayKey)) return 0
    let n = 0
    for (let k = key; kept(k); k = add(k, -1)) n++
    return n
  }
  const kg = latestBody(p)?.kg ?? 70
  const accent = ACCENTS.find((a) => a.id === p.accent)?.hex ?? '#e01b35'
  const itemsFor = (day: string): ShareItem[] => [
    ...history.filter((h) => h.date === day && !h.deleted).map((h) => sessionShare(h, history, kg)),
    ...matches.filter((m) => m.date === day).map(matchShare),
    ...(p.ratings?.date === day ? [ratingShare(p.ratings)] : []),
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3" role="dialog" aria-modal="true" aria-label={t('Streak calendar')}>
      <button aria-label={t('Close')} className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onBack} />
      <div className="rise relative flex max-h-[92dvh] w-full max-w-md flex-col overflow-hidden rounded-[32px] bg-[#211e1d] text-white shadow-[0_30px_80px_rgba(0,0,0,.6)]">
        {picked === null ? (
          <>
            <div className="bg-white/[.06] px-5 pt-5 pb-4">
              <div className="flex items-center justify-between gap-2">
                <h2 className="m-0 whitespace-nowrap font-display text-xl leading-none [hyphens:none] first-letter:uppercase">{month.toLocaleDateString(locale(), { month: 'long', year: 'numeric' })}</h2>
                <div className="flex gap-1.5">
                  <button onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() - 1, 1))} aria-label={t('Previous month')} className="grid h-9 w-9 place-items-center rounded-full bg-white/10 text-lg active:scale-95">‹</button>
                  <button onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() + 1, 1))} disabled={atNow} aria-label={t('Next month')} className="grid h-9 w-9 place-items-center rounded-full bg-white/10 text-lg active:scale-95 disabled:opacity-30">›</button>
                </div>
              </div>
              <div role="tablist" aria-label={t('Streak')} className="mt-4 grid grid-cols-2 gap-1 rounded-full bg-black/30 p-1">
                {([['plan', 'plan', streak.days === 1 ? t('1-day streak') : t('{n}-day streak', { n: streak.days })], ['games', gameTone, t('Games')]] as const).map(([id, tone, label]) => (
                  <button key={id} role="tab" aria-selected={mode === id} onClick={() => setMode(id)}
                    className={`flex min-h-10 items-center justify-center gap-1.5 rounded-full text-sm font-semibold transition-colors ${mode === id ? 'bg-white text-black' : 'text-white/75'}`}>
                    <FlameIcon tone={tone} lit={id === 'plan' ? streak.days > 0 : games.unbeaten > 0} className="h-5 w-5" />{label}
                  </button>
                ))}
              </div>
              {mode === 'games' && (
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {([['wins', games.wins, games.wins === 1 ? t('win in a row') : t('wins in a row')], ['unbeaten', games.unbeaten, t('unbeaten')]] as const).map(([tone, n, label]) => (
                    <div key={tone} className="flex items-center gap-2 rounded-[16px] bg-black/25 px-3 py-2">
                      <FlameIcon tone={tone} lit={n > 0} className="h-8 w-8" />
                      <span><span className="block font-display text-2xl leading-none tabular-nums">{n}</span><span className="block text-[11px] text-white/60">{label}</span></span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="overflow-y-auto px-4 pt-4 pb-5">
              <div className="grid grid-cols-7 gap-1.5">
                {cells.map((key, i) => {
                  if (!key) return <span key={i} />
                  const future = key > todayKey
                  const done = byDay.get(key), game = matchOn.get(key), photo = photos[key]
                  const status = st[key]
                  const run = mode === 'plan' ? runEnd(key) : 0
                  const res = game ? resultOf(game) : null
                  const label = `${longDate(key)}${done ? `, ${t('trained')}` : ''}${game ? `, ${t('match')}` : ''}${status === 'missed' ? `, ${t('missed')}` : ''}`
                  if (mode === 'games') {
                    return (
                      <button key={i} onClick={() => setPicked(key)} disabled={future || !game} aria-label={label}
                        className={`relative grid aspect-square place-items-center rounded-[26%] text-sm font-display ${game ? RESULT_TILE[res!] : 'bg-white/[.05] text-white/25'}`}>
                        {game ? <>{res === 'W' ? t('W') : res === 'L' ? t('L') : t('D')}<span className="absolute bottom-1 text-[9px] font-sans font-semibold opacity-80">{game.us}–{game.them}</span></> : parse(key).getDate()}
                      </button>
                    )
                  }
                  return (
                    <button key={i} onClick={() => setPicked(key)} disabled={future} aria-label={label}
                      className={`relative aspect-square overflow-hidden rounded-[26%] active:scale-95 ${done ? 'bg-white/10' : 'bg-white/[.06]'} ${key === todayKey ? 'outline outline-2 outline-offset-2 outline-white/80' : ''}`}
                      style={done ? { boxShadow: `inset 0 0 0 3px ${STREAK_GOLD}` } : undefined}>
                      {done && photo && <img src={photo} alt="" className="absolute inset-[3px] h-[calc(100%-6px)] w-[calc(100%-6px)] rounded-[22%] object-cover" />}
                      {done && !photo && <span className="absolute inset-0 grid place-items-center text-white/80"><TypeIcon type={done[0].type} className="h-5 w-5" /></span>}
                      {!done && status === 'kept' && !future && <span className="absolute inset-0 grid place-items-center"><span className="h-2.5 w-2.5 rounded-full bg-white/30" /></span>}
                      {!done && (future || (status !== 'kept' && status !== 'missed')) && <span className="absolute inset-0 grid place-items-center text-[11px] text-white/25">{parse(key).getDate()}</span>}
                      {game && <span className="absolute -top-0.5 -right-0.5 grid h-5 w-5 place-items-center rounded-full bg-[#211e1d]"><FlameIcon tone={resultOf(game) === 'W' ? 'wins' : 'unbeaten'} lit={resultOf(game) !== 'L'} className="h-4 w-4" /></span>}
                      {run > 1 && <span className="absolute right-0.5 bottom-0.5 min-w-[18px] rounded-[7px] px-1 text-[11px] font-bold leading-[18px] text-black" style={{ background: STREAK_GOLD }}>{run}</span>}
                    </button>
                  )
                })}
              </div>
              {mode === 'plan'
                ? <p className="m-0 mt-4 text-center text-xs text-white/50">{t('Gold frame: trained. Dot: rest day kept. Flame: game day.')}</p>
                : <GameList matches={matches} onShare={(m) => setSharing(matchShare(m))} />}
            </div>
          </>
        ) : (
          <DayDetail day={picked} items={itemsFor(picked)} photo={photos[picked]} kept={st[picked] === 'kept'} trained={byDay.has(picked)} onBack={() => setPicked(null)} onShare={setSharing} />
        )}
      </div>
      {sharing && <ShareSheet item={sharing} photo={photos[sharing.date]} streak={streak.days} games={games} accent={accent} onClose={() => setSharing(null)} />}
    </div>
  )
}

function DayDetail({ day, items, photo, kept, trained, onBack, onShare }: { day: string; items: ShareItem[]; photo?: string; kept: boolean; trained: boolean; onBack: () => void; onShare: (i: ShareItem) => void }) {
  return (
    <div className="overflow-y-auto p-4">
      <div className="flex items-center gap-2">
        <button onClick={onBack} aria-label={t('Back')} className="grid h-9 w-9 place-items-center rounded-full bg-white/10 text-lg active:scale-95">‹</button>
        <p className="m-0 font-semibold first-letter:uppercase">{longDate(day)}</p>
      </div>
      {(trained || photo) && (
        <label className="relative mt-3 block aspect-[4/3] cursor-pointer overflow-hidden rounded-[24px] bg-white/[.06]">
          {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : <span className="grid h-full place-items-center text-sm font-semibold text-white/60">{t('Add a photo of this day')}</span>}
          {photo && <span className="absolute right-2.5 bottom-2.5 rounded-full bg-black/55 px-3 py-1.5 text-xs font-semibold backdrop-blur">{t('Change photo')}</span>}
          <input type="file" accept="image/*" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) void setDayPhoto(day, f) }} />
        </label>
      )}
      {!items.length && <p className="m-0 mt-3 rounded-[18px] bg-white/[.06] p-4 text-sm text-white/70">{kept ? t('Rest day: the plan said rest, so it counts.') : t('Nothing logged this day.')}</p>}
      <div className="mt-3 flex flex-col gap-2">
        {items.map((it, i) => (
          <div key={i} className="flex items-center gap-3 rounded-[18px] bg-white/[.06] p-3.5">
            <span className="min-w-0 flex-1">
              <span className="block truncate font-semibold">{it.title}</span>
              <span className="block truncate text-xs text-white/60">{it.stats.map((s) => `${s.label} ${s.value}`).join(' · ')}</span>
            </span>
            <button onClick={() => onShare(it)} className="min-h-10 shrink-0 rounded-full bg-white px-4 font-display text-xs uppercase tracking-wide text-black active:scale-95">{t('Share')}</button>
          </div>
        ))}
      </div>
    </div>
  )
}

// The picture to post, Strava-style. Sticker: transparent, to drop on your own story photo (previewed on the day's photo).
export function ShareSheet({ item, photo, streak, games, accent, onClose }: { item: ShareItem; photo?: string; streak: number; games?: { wins: number; unbeaten: number }; accent: string; onClose: () => void }) {
  useSheet(onClose)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [look, setLook] = useState<Look>('sticker')
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  useEffect(() => { let on = true; if (photo) void loadImage(photo).then((i) => on && setImg(i)); else setImg(null); return () => { on = false } }, [photo])
  useEffect(() => { if (canvas.current) void drawShare(canvas.current, item, { look, photo: img, streak, games, accent, dateLabel: longDate(item.date) }) }, [item, look, img, streak, games, accent])
  const looks: [Look, string][] = [['sticker', t('Sticker')], ['black', t('Black')], ['photo', t('On my photo')]]
  return (
    <div className="fixed inset-0 z-[60] flex flex-col bg-black text-white" role="dialog" aria-modal="true" aria-label={t('Share')}>
      <div className="flex items-center justify-between px-4 pt-[max(14px,var(--sat))] pb-2">
        <button onClick={onClose} className="min-h-10 px-2 text-sm font-semibold text-white/80">{t('Close')}</button>
        <p className="m-0 font-display text-sm uppercase">{t('Share')}</p>
        <span className="w-14" />
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center px-6">
        {/* the sticker is previewed on the day's photo (or a checker), the way it will sit on a story */}
        <div className="relative h-full max-h-full overflow-hidden rounded-[18px]" style={{ aspectRatio: '9 / 16', background: look === 'sticker' ? (photo ? `center/cover url(${photo})` : 'repeating-conic-gradient(#222 0 25%, #2c2c2c 0 50%) 0 0/28px 28px') : '#000' }}>
          <canvas ref={canvas} role="img" aria-label={t('Share picture: {title}', { title: item.title })} className="absolute inset-0 h-full w-full" />
        </div>
      </div>
      <div className="mx-auto flex w-full max-w-xl flex-col gap-2 px-5 pt-4 pb-[max(20px,env(safe-area-inset-bottom))]">
        <div role="radiogroup" aria-label={t('Look')} className="grid grid-cols-3 gap-1 rounded-full bg-white/10 p-1">
          {looks.map(([id, label]) => (
            <button key={id} role="radio" aria-checked={look === id} onClick={() => setLook(id)}
              className={`min-h-10 rounded-full text-xs font-semibold ${look === id ? 'bg-white text-black' : 'text-white/75'}`}>{label}</button>
          ))}
        </div>
        {look === 'photo' && !photo && (
          <label className="flex min-h-11 cursor-pointer items-center justify-center rounded-full bg-white/15 text-xs font-semibold">
            {t('Add a photo of this day')}
            <input type="file" accept="image/*" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) void setDayPhoto(item.date, f) }} />
          </label>
        )}
        <button disabled={busy} onClick={async () => { if (!canvas.current) return; setBusy(true); const r = await sharePicture(canvas.current, `football-coach-${item.date}`); setBusy(false); setNote(r === 'saved' ? t('Saved to your downloads.') : r === 'failed' ? t('Could not make the picture.') : '') }}
          className="min-h-14 rounded-full font-display text-sm uppercase tracking-wide text-white active:scale-[0.97] disabled:opacity-60" style={{ background: accent }}>{t('Share')}</button>
        {note && <p className="m-0 text-center text-xs text-white/70" role="status">{note}</p>}
        <p className="m-0 text-center text-[11px] text-white/50">{look === 'sticker' ? t('A see-through sticker: add it to your own story photo.') : t('Your photo stays on this phone unless you share the picture.')}</p>
      </div>
    </div>
  )
}
