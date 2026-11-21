// The AI coach, deterministic: it reads your question, works out what you're asking (intent), pulls the right
// facts from your own data and the app's knowledge (retrieval), and writes the answer from them. No model, no network:
// every number it says is computed by the same code the rest of the app uses.
import { t } from '../../i18n'
import { CATEGORY_LABELS, EXERCISES, type Exercise } from '../library'
import { sessionMinutes } from '../logic'
import { bestsFrom, e1rm, isBodyweight, moved } from '../lifts'
import { bmi, bmiBand, latestBody, latestFat, weightTrend } from '../body'
import { counts } from '../streak'
import { gameLength, resultOf, resultStreaks, type MatchReview } from '../match'
import { FOCUS_LABEL, kindLabel } from '../weekplan'
import type { Profile } from '../profile'
import type { HistoryEntry } from '../storage'
import type { Today } from '../today'
import type { Streak } from '../streak'
import type { Session } from '../types'
import { FACTS, type Fact } from './kb'
import { Index, fold, has, tokens } from './search'

export type CoachCtx = { profile: Profile; history: HistoryEntry[]; matches: MatchReview[]; today: Today; sessions: Session[]; streak: Streak; now?: Date }
export type Card =
  | { kind: 'exercise'; id: string; name: string; sub: string }
  | { kind: 'session'; id: string; title: string; sub: string }
  | { kind: 'go'; to: 'record' | 'week' | 'settings' | 'team'; label: string }
export type Reply = { text: string[]; cards?: Card[]; after?: string[]; chips?: string[] } // after: text under the cards

// ---------- words that give the question away (any of the app's languages, accents optional) ----------
const W = {
  hello: ['hi', 'hello', 'hey', 'yo', 'sup', 'chao', 'xin chao', 'hola', 'ola', 'oi', 'bonjour', 'salut', 'halo', 'hai', 'help', 'giup', 'ayuda', 'ajuda', 'aide', 'bantu', 'what can you do'],
  today: ['today', 'tonight', 'now', 'next session', 'what should i do', 'what do i do', 'train today', 'hom nay', 'hoy', 'hoje', 'aujourd hui', 'hari ini', 'kick off'],
  week: ['week', 'weekly', 'this week', 'tuan', 'tuan nay', 'semana', 'semaine', 'minggu', 'load', 'acwr', 'too much', 'overtraining', 'enough'],
  streak: ['streak', 'flame', 'chuoi', 'racha', 'sequencia', 'serie', 'beruntun'],
  last: ['last session', 'last workout', 'yesterday', 'last time', 'how did i do', 'previous session', 'buoi truoc', 'hom qua', 'ultima sesion', 'ultimo treino', 'derniere seance', 'sesi terakhir'],
  lifts: ['pr', 'prs', 'personal best', 'best', 'max', '1rm', 'strongest', 'how strong', 'strength', 'lift', 'lifts', 'lifting', 'bench', 'squat', 'deadlift', 'ky luc', 'record personal', 'recorde', 'maximo', 'rekor', 'progress'],
  body: ['weight', 'weigh', 'bmi', 'body fat', 'fat', 'kg', 'heavy', 'height', 'can nang', 'bmi', 'peso', 'poids', 'berat', 'lemak', 'gordura'],
  match: ['match', 'matches', 'game', 'games', 'rating', 'rated', 'played', 'goals', 'assists', 'win', 'won', 'lost', 'tran', 'tran dau', 'partido', 'jogo', 'partida', 'match', 'pertandingan', 'gol'],
  sore: ['sore', 'pain', 'hurt', 'hurts', 'injury', 'injured', 'ache', 'tight', 'tightness', 'strain', 'pulled', 'doms', 'dau', 'moi', 'chan thuong', 'dolor', 'duele', 'lesion', 'dor', 'lesao', 'douleur', 'mal', 'blessure', 'sakit', 'cedera', 'pegal'],
  how: ['how do i', 'how to', 'how should', 'form', 'technique', 'explain', 'show me', 'demo', 'video', 'lam sao', 'cach', 'como', 'comment', 'bagaimana', 'cara', 'what is', 'whats'],
  alt: ['instead', 'alternative', 'replace', 'swap', 'substitute', 'without', 'no barbell', 'no gym', 'no equipment', 'don t have', 'dont have', 'thay', 'thay the', 'en lugar', 'sin', 'em vez', 'sem', 'au lieu', 'sans', 'ganti', 'tanpa'],
  session: ['session', 'workout', 'routine', 'drill', 'drills', 'plan me', 'build me', 'give me', 'suggest', 'recommend', 'minutes', 'min', 'buoi tap', 'bai tap', 'sesion', 'entrenamiento', 'treino', 'seance', 'entrainement', 'sesi', 'latihan'],
  position: ['position', 'winger', 'striker', 'midfielder', 'defender', 'defence', 'defense', 'centre back', 'fullback', 'full back', 'forward', 'my role', 'vi tri', 'hau ve', 'tien dao', 'tien ve', 'posicion', 'defensa', 'delantero', 'posicao', 'zagueiro', 'poste', 'defenseur', 'posisi', 'bek'],
  why: ['why', 'tai sao', 'vi sao', 'por que', 'porque', 'pourquoi', 'kenapa', 'mengapa', 'help me', 'benefit', 'good for', 'do for me'],
}

// Body parts → stretches and easy work for them, and what to keep off for now.
const PARTS: { words: string[]; label: string; easy: string[] }[] = [
  { words: ['hamstring', 'hammy', 'hammies', 'back of thigh', 'gan kheo', 'isquio', 'ischio', 'paha belakang'], label: 'hamstrings', easy: ['hamstring-stretch', 'glute-bridge', 'foam-roll-quads-and-glutes'] },
  { words: ['hip', 'hips', 'groin', 'adductor', 'hong', 'bang', 'cadera', 'ingle', 'quadril', 'virilha', 'hanche', 'aine', 'pinggul', 'selangkangan'], label: 'hips', easy: ['90-90-hip-switches', 'figure-4-glute-stretch', 'couch-stretch-hip-flexor-quad'] },
  { words: ['quad', 'quads', 'thigh', 'knee', 'knees', 'dui', 'goi', 'cuadriceps', 'rodilla', 'joelho', 'genou', 'cuisse', 'lutut', 'paha'], label: 'quads and knees', easy: ['foam-roll-quads-and-glutes', 'couch-stretch-hip-flexor-quad', 'glute-bridge'] },
  { words: ['calf', 'calves', 'ankle', 'achilles', 'shin', 'bap chan', 'co chan', 'gemelo', 'tobillo', 'panturrilha', 'tornozelo', 'mollet', 'cheville', 'betis', 'pergelangan'], label: 'calves and ankles', easy: ['side-lying-leg-raise-clamshell', 'glute-bridge', 'hamstring-stretch'] },
  { words: ['back', 'lower back', 'spine', 'lung', 'espalda', 'costas', 'dos', 'punggung'], label: 'back', easy: ['dead-bug', 'glute-bridge', 'figure-4-glute-stretch'] },
  { words: ['glute', 'glutes', 'butt', 'mong', 'gluteo', 'fessier', 'bokong'], label: 'glutes', easy: ['figure-4-glute-stretch', 'foam-roll-quads-and-glutes', '90-90-hip-switches'] },
]
const PART_LABEL = (k: string) => ({ hamstrings: t('hamstrings'), hips: t('hips'), 'quads and knees': t('quads and knees'), 'calves and ankles': t('calves and ankles'), back: t('back'), glutes: t('glutes') })[k] ?? k

// Short names people use for exercises.
const ALIASES: Record<string, string> = {
  rdl: 'romanian-deadlift', rdls: 'romanian-deadlift', nordic: 'nordic-curl', nordics: 'nordic-curl', pullup: 'pull-up', 'pull up': 'pull-up', 'chin up': 'pull-up',
  bench: 'bench-press', squat: 'back-squat', squats: 'back-squat', deadlift: 'deadlift', ohp: 'db-overhead-press', 'overhead press': 'db-overhead-press', 'hip thrust': 'hip-thrust',
  lunge: 'home-reverse-lunge', lunges: 'home-reverse-lunge', bulgarian: 'bulgarian-split-squat', bss: 'bulgarian-split-squat', copenhagen: 'copenhagen-plank', pushup: 'push-up', 'push up': 'push-up',
  clean: 'power-clean', snatch: 'hang-snatch', jerk: 'push-jerk', 'box jump': 'box-jump', swing: 'kettlebell-swing', 'kb swing': 'kettlebell-swing', 'calf raise': 'calf-raise', row: 'bent-over-row', dips: 'home-chair-dip',
}

let exIndex: Index<Exercise> | null = null
let factIndex: Index<Fact> | null = null
const exercises = () => (exIndex ??= new Index(EXERCISES.map((e) => ({ id: e.id, item: e, boost: `${e.name} ${t(e.name)} ${e.id.replace(/-/g, ' ')}`, text: `${CATEGORY_LABELS[e.category] ?? e.category} ${e.category} ${e.how ?? ''} ${(e.needs ?? []).join(' ')} ${(e.positions ?? []).join(' ')}` }))))
const facts = () => (factIndex ??= new Index(FACTS().map((f) => ({ id: f.id, item: f, boost: `${f.title} ${f.tags}`, text: f.answer }))))
export const resetCoachIndex = () => { exIndex = null; factIndex = null } // after a language change

const exById = (id: string) => EXERCISES.find((e) => e.id === id)
const exCard = (e: Exercise): Card => ({ kind: 'exercise', id: e.id, name: e.name, sub: [CATEGORY_LABELS[e.category], e.reps ? t('{sets} × {reps}', { sets: e.rounds || 1, reps: e.reps }) : null].filter(Boolean).join(' · ') })
const sessionCard = (s: Session): Card => ({ kind: 'session', id: s.id, title: s.title, sub: t('{n} min · {d} drills', { n: sessionMinutes(s), d: s.blocks.length }) })
const fmtKg = (n: number) => String(Math.round(n * 10) / 10)
const fmtDate = (iso: string, now: Date) => {
  const [y, m, d] = iso.split('-').map(Number)
  const days = Math.round((new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() - new Date(y, m - 1, d).getTime()) / 864e5)
  return days === 0 ? t('today') : days === 1 ? t('yesterday') : t('{n} days ago', { n: days })
}

// The exercise a question is about, if it names one.
export function findExercise(q: string): Exercise | null {
  const f = ` ${fold(q).replace(/[^a-z0-9]+/g, ' ')} `
  // full names first, longest first ("single-leg romanian deadlift" before "deadlift")
  const byName = [...EXERCISES].sort((a, b) => b.name.length - a.name.length).find((e) => f.includes(` ${fold(e.name).replace(/[^a-z0-9]+/g, ' ').trim()} `) || f.includes(` ${fold(t(e.name)).replace(/[^a-z0-9]+/g, ' ').trim()} `))
  if (byName) return byName
  for (const [k, id] of Object.entries(ALIASES).sort((a, b) => b[0].length - a[0].length)) if (f.includes(` ${k} `)) return exById(id) ?? null
  const hit = exercises().search(q, 2)
  return hit[0] && hit[0].score > 6 && (!hit[1] || hit[0].score > hit[1].score * 1.25) ? hit[0].doc.item : null
}

// ---------- personal answers ----------
function todayReply(c: CoachCtx): Reply {
  const td = c.today
  const s = td.session
  const text: string[] = []
  if (td.slot?.kind === 'game') text.push(t('Game day{md}. Keep the legs fresh: no training today, just a good warm-up before kick-off.', { md: td.md ? ` (${td.md})` : '' }))
  else if (td.done) text.push(t("Today's training is done. Rest up: the work turns into fitness while you recover."))
  else if (td.scheduled) {
    const job = td.slot?.focus ? FOCUS_LABEL[td.slot.focus] : td.slot ? kindLabel(td.slot.kind) : ''
    text.push(t('Today{md}: {title}, {n} minutes.', { md: td.md ? ` (${td.md})` : '', title: s.title, n: sessionMinutes(s) }) + (job ? ' ' + t('The job today: {job}.', { job: job.toLowerCase() }) : ''))
    if (td.slot?.note) text.push(t('Why: {note}.', { note: td.slot.note }))
  } else text.push(t('Rest day on your plan. If you feel fresh, the next session is {title} ({n} min).', { title: s.title, n: sessionMinutes(s) }))
  if (!c.streak.today && c.streak.days > 0 && td.scheduled && !td.done) text.push(t('Train today to keep your {n}-day streak going.', { n: c.streak.days }))
  return { text, cards: td.slot?.kind === 'game' || td.done ? [] : [sessionCard(s)], chips: [t('Why is my week planned like this?'), t('How was my week?')] }
}

const mondayOf = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate() - ((d.getDay() + 6) % 7))
const ymd = (d: Date) => d.toLocaleDateString('en-CA')
function weekReply(c: CoachCtx): Reply {
  const now = c.now ?? new Date()
  const mon = mondayOf(now)
  const weekOf = (k: number) => { const a = ymd(new Date(mon.getTime() - k * 7 * 864e5)), b = ymd(new Date(mon.getTime() - (k - 1) * 7 * 864e5)); return c.history.filter((h) => h.date >= a && h.date < b) }
  const thisWeek = weekOf(0)
  const load = (hs: HistoryEntry[]) => hs.reduce((a, h) => a + (h.load ?? 0), 0)
  const minutes = thisWeek.reduce((a, h) => a + h.minutes, 0)
  const planned = c.today.plan.slots.filter((s) => s.kind !== 'game' && s.kind !== 'cheer').length
  const done = thisWeek.filter(counts).length
  const text = [t('This week: {done} of {planned} planned sessions, {min} minutes, load {load}.', { done, planned, min: minutes, load: load(thisWeek) })]
  // acute:chronic: this week against the 3 before (the ratio sports scientists watch for injury risk)
  const prev = [1, 2, 3].map((k) => load(weekOf(k)))
  const chronic = prev.reduce((a, b) => a + b, 0) / 3
  if (chronic > 0) {
    const days = ((now.getDay() + 6) % 7) + 1
    const pace = (load(thisWeek) / days) * 7
    const r = pace / chronic
    text.push(r > 1.4 ? t('You are on pace for {p}% of your usual week. Big jumps raise injury risk: keep the next sessions easy unless you feel great.', { p: Math.round(r * 100) })
      : r < 0.7 ? t('You are on pace for {p}% of your usual week. Fine for a lighter week; otherwise there is room to add a session.', { p: Math.round(r * 100) })
        : t('You are on pace for {p}% of your usual week: right in the safe zone.', { p: Math.round(r * 100) }))
  } else if (!thisWeek.length) text.push(t('Nothing logged yet this week. The first session is the hardest one to start.'))
  if (c.streak.days) text.push(c.streak.days === 1 ? t('Streak: 1 day.') : t('Streak: {n} days.', { n: c.streak.days }))
  return { text, cards: [{ kind: 'go', to: 'record', label: t('Open Record') }], chips: [t('What should I do today?'), t('How is the load calculated?')] }
}

function streakReply(c: CoachCtx): Reply {
  const s = c.streak
  const text = [s.days ? (s.days === 1 ? t('Your streak is 1 day.') : t('Your streak is {n} days.', { n: s.days })) + (s.today ? ' ' + t('Today already counts.') : '') : t('No streak yet: follow the plan for a day and it starts.')]
  if (s.makeup) text.push(t('You missed a session on {date}: an extra session of 20 minutes or more by {by} makes it up.', { date: s.makeup.date, by: s.makeup.by }))
  const f = FACTS().find((x) => x.id === 'streak')!
  text.push(f.answer)
  return { text, chips: [t('What should I do today?')] }
}

function lastReply(c: CoachCtx): Reply {
  const h = c.history[c.history.length - 1]
  if (!h) return { text: [t('No sessions in your log yet. Kick off your first one from Home.')] }
  const now = c.now ?? new Date()
  const text = [t('Last session: {title}, {when}. {min} minutes.', { title: t(h.title), when: fmtDate(h.date, now), min: h.minutes })]
  if (h.rpe) text.push(t('Effort {e} out of 10, load {load}.', { e: h.effort ?? h.rpe, load: h.load ?? 0 }) + (h.sore?.length ? ' ' + t('You noted soreness: {list}.', { list: h.sore.map((x) => t(x).toLowerCase()).join(', ') }) : ''))
  else text.push(t('No check-in yet for it: add the effort in Record so it counts in your load.'))
  if (h.lifts?.length) {
    const tops = h.lifts.map((l) => {
      const best = l.sets.reduce((a, s) => ((s.kg ?? 0) > (a.kg ?? 0) ? s : a), l.sets[0])
      return isBodyweight(l) ? t('{name}: {n} sets, best {reps} reps', { name: t(l.name), n: l.sets.length, reps: Math.max(...l.sets.map((s) => s.reps ?? 0)) }) : t('{name}: {n} sets, top {kg} kg × {reps}', { name: t(l.name), n: l.sets.length, kg: fmtKg(best.kg ?? 0), reps: best.reps ?? 0 })
    })
    text.push(tops.join('\n'))
  }
  return { text, cards: [{ kind: 'go', to: 'record', label: t('Open Record') }], chips: [t('What are my best lifts?'), t('How was my week?')] }
}

function liftsReply(c: CoachCtx, ex: Exercise | null): Reply {
  const logged = c.history.filter((h) => h.lifts?.length && h.rpe)
  if (!logged.length) return { text: [t('No sets logged yet. After a gym session, log weight and reps for each set and I can track your bests and progress.')], chips: [t('How are my sets scored?')] }
  const bests = bestsFrom(logged)
  const bw = latestBody(c.profile)?.kg
  if (ex) {
    const key = ex.id
    const rows = logged.flatMap((h) => (h.lifts ?? []).filter((l) => (l.ex ?? l.name) === key || l.name === ex.name).map((l) => ({ h, l })))
    if (!rows.length) return { text: [t('You haven\'t logged {name} yet.', { name: t(ex.name) })], cards: [exCard(ex)] }
    const top = (x: { l: NonNullable<HistoryEntry['lifts']>[number] }) => x.l.sets.reduce((a, s) => Math.max(a, e1rm(moved(x.l, s, bw) ?? 0, s.reps ?? 0)), 0)
    const first = top(rows[0]), last = top(rows[rows.length - 1])
    const heaviest = rows.flatMap((x) => x.l.sets).reduce((a, s) => ((s.kg ?? 0) > (a.kg ?? 0) || ((s.kg ?? 0) === (a.kg ?? 0) && (s.reps ?? 0) > (a.reps ?? 0)) ? s : a))
    const text = [t('{name}: logged in {n} sessions. Estimated max about {max} kg.', { name: t(ex.name), n: rows.length, max: fmtKg(bests.get(key) ?? last) })]
    if (!isBodyweight({ ex: ex.id })) text.push(t('Heaviest set: {kg} kg × {reps}.', { kg: fmtKg(heaviest.kg ?? 0), reps: heaviest.reps ?? 0 }))
    if (rows.length > 1 && first > 0) {
      const d = Math.round(((last - first) / first) * 100)
      text.push(d > 0 ? t('Up {d}% since your first logged session. Keep going.', { d }) : d < 0 ? t('Down {d}% from your first logged session: a lighter day, or time for a deload and a fresh start.', { d: -d }) : t('Holding steady since your first session. Add 2.5 kg when every set hits the top of the rep range.'))
    }
    if (bw && !isBodyweight({ ex: ex.id }) && bests.get(key)) text.push(t('That is {x}× your body weight.', { x: ((bests.get(key) ?? 0) / bw).toFixed(2) }))
    return { text, cards: [exCard(ex)], chips: [t('When should I add weight?')] }
  }
  const list = [...bests.entries()].map(([k, v]) => ({ e: exById(k), v })).filter((x) => x.e).sort((a, b) => b.v - a.v).slice(0, 6)
  return {
    text: [t('Your estimated maxes, from the sets you logged:'), list.map((x) => `• ${t(x.e!.name)}: ${fmtKg(x.v)} kg`).join('\n'), t('Ask about one lift for its history, e.g. "my bench".')],
    chips: list.slice(0, 2).map((x) => t('My {name}', { name: t(x.e!.name).toLowerCase() })),
  }
}

function bodyReply(c: CoachCtx): Reply {
  const b = latestBody(c.profile)
  if (!b) return { text: [t('No body weight logged yet. Add it in Record → Body: it makes bodyweight exercises count properly and tracks your trend.')], cards: [{ kind: 'go', to: 'record', label: t('Open Record') }] }
  const text = [t('Body weight {kg} kg (logged {when}).', { kg: b.kg, when: fmtDate(b.date, c.now ?? new Date()) })]
  const tr = weightTrend(c.profile)
  if (tr) text.push(tr.kg === 0 ? t('Steady over the last weeks.') : t('{d} kg since {date}.', { d: (tr.kg > 0 ? '+' : '') + tr.kg, date: tr.since }))
  if (c.profile.heightCm) {
    const v = bmi(b.kg, c.profile.heightCm)
    const band = { under: t('Below range'), healthy: t('Healthy range'), above: t('Above range'), 'well-above': t('Well above range') }[bmiBand(v)]
    text.push(t('BMI {v}: {band}. BMI can\'t tell muscle from fat, so a strong player can read high.', { v: v.toFixed(1), band: band.toLowerCase() }))
  }
  const fat = latestFat(c.profile)
  if (fat) text.push(t('Body fat {f}%.', { f: fat }))
  return { text, cards: [{ kind: 'go', to: 'record', label: t('Open Record') }] }
}

function matchReply(c: CoachCtx): Reply {
  const ms = c.matches
  if (!ms.length) return { text: [t('No matches reviewed yet. On a game day, Home asks when you play; after full time you review it and the app rates it.')], chips: [t('How does the match rating work?')] }
  const last = ms[ms.length - 1]
  const avg = ms.reduce((a, m) => a + m.rating, 0) / ms.length
  const best = ms.reduce((a, m) => (m.rating > a.rating ? m : a))
  const st = resultStreaks(ms)
  const r = resultOf(last)
  const word = r === 'W' ? t('a win') : r === 'L' ? t('a loss') : t('a draw')
  const text = [
    t('Last game: {us}–{them}, {result}. You played {min} of {len} minutes and the app rated you {rating}.', { us: last.us, them: last.them, result: word, min: last.minutes, len: gameLength(last), rating: last.rating.toFixed(1) }),
    t('{n} games reviewed, average {avg}, best {best}.', { n: ms.length, avg: avg.toFixed(1), best: best.rating.toFixed(1) }),
  ]
  if (st.unbeaten > 1) text.push(t('Unbeaten in {n}.', { n: st.unbeaten }))
  const extras = [last.goals ? t('{n} goals', { n: last.goals }) : '', last.assists ? t('{n} assists', { n: last.assists }) : '', last.dribbles ? t('{n} dribbles', { n: last.dribbles }) : ''].filter(Boolean)
  if (extras.length) text.push(t('Last game: {list}.', { list: extras.join(', ') }))
  return { text, cards: [{ kind: 'go', to: 'record', label: t('Open Record') }], chips: [t('How does the match rating work?')] }
}

function soreReply(q: string): Reply {
  const part = PARTS.find((p) => has(q, p.words))
  const general = FACTS().find((f) => f.id === 'sore')!
  if (!part) return { text: [general.answer, t('Tell me where it is (hamstring, hip, knee, calf, back) and I\'ll point you to the right stretches.')], chips: [t('My hamstrings are sore'), t('My hips feel tight')] }
  const easy = part.easy.map(exById).filter((e) => !!e) as Exercise[]
  return {
    text: [t('Sore {part}: keep moving gently, it recovers faster than full rest. These help:', { part: PART_LABEL(part.label) })],
    cards: easy.map(exCard),
    after: [t('Skip heavy or explosive work for the {part} until it feels normal. Sharp pain, swelling or a limp: see a physio. I\'m not a doctor.', { part: PART_LABEL(part.label) })],
  }
}

function exerciseReply(c: CoachCtx, ex: Exercise, q: string): Reply {
  const text = [`${t(ex.name)}${ex.reps ? ` · ${t('{sets} × {reps}', { sets: ex.rounds || 1, reps: ex.reps })}` : ''}`, ex.how ?? '']
  if (ex.needs?.length) text.push(t('You need: {list}.', { list: ex.needs.map((n) => t(n)).join(', ') }))
  const mine = c.profile.positions.filter((p) => ex.positions?.includes(p))
  if (has(q, W.why) || has(q, W.position) || mine.length) {
    const why = CATEGORY_WHY()[ex.category]
    if (why) text.push(why)
    if (mine.length) text.push(t('Picked out for your position too.'))
  }
  return { text: text.filter(Boolean), cards: [exCard(ex)], chips: [t('Alternatives to {name}', { name: t(ex.name).toLowerCase() })] }
}

// What each kind of exercise does for a player (the "why" behind it).
const CATEGORY_WHY = (): Record<string, string> => ({
  'lower-strength': t('Why: stronger legs mean faster first steps, higher jumps and fewer muscle injuries.'),
  'upper-strength': t('Why: upper-body strength wins shoulder-to-shoulder duels and protects the ball.'),
  'single-limb': t('Why: football is played on one leg at a time; single-leg strength fixes left/right gaps and protects knees.'),
  olympic: t('Why: Olympic lifts train producing force fast, the same quality as sprint starts and jumps.'),
  explosive: t('Why: explosive work turns strength into speed: sprints, jumps, shots.'),
  plyometrics: t('Why: jumps make your tendons springy, for acceleration and headers.'),
  sprints: t('Why: top speed only improves by sprinting at top speed, fully rested.'),
  speed: t('Why: most game sprints are 5–20 m with a turn: this is that.'),
  core: t('Why: a strong trunk passes force from legs to ball and protects your back and groin.'),
  isometrics: t('Why: holds build tendon strength with little fatigue, good in a game week.'),
  mobility: t('Why: mobility keeps hips and hamstrings moving freely, the usual football tight spots.'),
  'ball-mastery': t('Why: more touches, head up, so the ball obeys you under pressure.'),
  'first-touch': t('Why: a clean first touch buys time; this is the skill pros value most.'),
  passing: t('Why: passing with both feet keeps you useful in every position.'),
  finishing: t('Why: repetition under fatigue makes finishing automatic in a game.'),
  '1v1': t('Why: winning 1v1s, attacking or defending, decides games.'),
  conditioning: t('Why: the engine to repeat sprints in the last 15 minutes.'),
})

function altReply(ex: Exercise | null, q: string): Reply {
  if (!ex) return { text: [t('Which exercise do you want to swap? For example: "alternative to back squat".')] }
  const noKit = has(q, ['no gym', 'no equipment', 'home', 'at home', 'nha', 'casa', 'maison', 'rumah', 'barbell', 'without'])
  const alts = EXERCISES.filter((e) => e.category === ex.category && e.id !== ex.id && e.kind === ex.kind)
    .filter((e) => !noKit || e.id.startsWith('home-') || !e.needs?.length && e.sport !== 'gym')
  const pick = (alts.length ? alts : EXERCISES.filter((e) => e.category === ex.category && e.id !== ex.id)).slice(0, 4)
  if (!pick.length) return { text: [t('No close swap for {name} in the library yet.', { name: t(ex.name) })] }
  return { text: [t('Same job as {name}:', { name: t(ex.name) })], cards: pick.map(exCard), chips: [t('How do I do {name}?', { name: t(pick[0].name).toLowerCase() })] }
}

function sessionReply(c: CoachCtx, q: string): Reply {
  const min = Number(fold(q).match(/(\d{2,3})\s*(m|min|minute|phut|minuto)/)?.[1] ?? 0)
  const wantType = has(q, ['home', 'at home', 'nha', 'casa', 'maison', 'rumah']) ? 'home' : has(q, ['gym', 'weights', 'lift', 'phong gym', 'gimnasio', 'academia', 'salle']) ? 'gym' : has(q, ['recovery', 'recover', 'stretch', 'mobility', 'phuc hoi', 'recuper']) ? 'recovery' : has(q, ['ball', 'football', 'soccer', 'pitch', 'bong', 'balon', 'bola', 'ballon', 'weak foot', 'dribbl', 'finish', 'pass', 'shoot']) ? 'football' : null
  const idx = new Index(c.sessions.map((s) => ({ id: s.id, item: s, boost: s.title, text: `${s.summary} ${s.blocks.map((b) => `${b.name} ${t(b.name)} ${b.category ?? ''}`).join(' ')}` })))
  const hits = idx.search(q, 12).map((h) => h.doc.item)
  let pool = (hits.length ? hits : c.sessions).filter((s) => !wantType || s.type === wantType || (wantType === 'football' && s.type === 'football'))
  if (!pool.length) pool = c.sessions.filter((s) => !wantType || s.type === wantType)
  if (min) pool = [...pool].sort((a, b) => Math.abs(sessionMinutes(a) - min) - Math.abs(sessionMinutes(b) - min))
  const pick = pool.slice(0, 3)
  if (!pick.length) return { text: [t('I could not find a session for that. Try "a 30-minute home session" or "weak foot session".')] }
  const text = [t('Here are the closest sessions:')]
  if (min) text.push(t('Sorted by how close they are to {n} minutes. Open one to change drills or time in the builder.', { n: min }))
  return { text, cards: pick.map(sessionCard) }
}

function positionReply(c: CoachCtx, q: string): Reply {
  const POS_WORDS: Record<string, string[]> = {
    winger: ['winger', 'wing', 'tien ve canh', 'extremo', 'ponta', 'ailier', 'sayap'],
    striker: ['striker', 'forward', 'tien dao', 'delantero', 'atacante', 'attaquant', 'penyerang'],
    midfielder: ['midfielder', 'midfield', 'tien ve', 'centrocampista', 'meio campo', 'milieu', 'gelandang'],
    defender: ['defender', 'defence', 'defense', 'centre back', 'fullback', 'full back', 'hau ve', 'defensa', 'zagueiro', 'defenseur', 'bek'],
  }
  const asked = Object.keys(POS_WORDS).find((p) => has(q, POS_WORDS[p])) as Profile['positions'][number] | undefined
  const pos = asked ?? c.profile.positions[0] ?? 'winger'
  const f = FACTS().find((x) => x.id === `pos-${pos}`)!
  const picks = EXERCISES.filter((e) => e.positions?.includes(pos)).slice(0, 4)
  return { text: [f.answer, t('Exercises picked out for this position:')], cards: picks.map(exCard), chips: [t('Why is my week planned like this?')] }
}

const intro = (c: CoachCtx): Reply => ({
  text: [t('Hi {name}. I know your plan, your log, your lifts, your matches and every exercise in the app. Ask me anything about your training.', { name: c.profile.name || t('player') })],
  chips: [t('What should I do today?'), t('How was my week?'), t('What are my best lifts?'), t('How do the exercises help my position?')],
})

// ---------- the coach ----------
export function answer(q: string, c: CoachCtx): Reply {
  const query = q.trim()
  if (!query) return intro(c)
  const words = tokens(query)
  const ex = findExercise(query)

  if (words.length <= 2 && has(query, W.hello)) return intro(c)
  if (has(query, W.sore)) return soreReply(query)
  if (has(query, W.alt) && ex) return altReply(ex, query)
  if (has(query, W.position) && (has(query, W.why) || has(query, ['exercise', 'exercises', 'train', 'training', 'help', 'bai tap', 'ejercicio', 'exercicio', 'exercice', 'latihan']))) return positionReply(c, query)
  if (has(query, W.streak)) return streakReply(c)
  if (has(query, W.match) && !has(query, ['match day', 'matchday', 'md'])) {
    const f = facts().search(query, 1)[0]
    if (has(query, ['how does', 'how is', 'work', 'calculated', 'calculate']) && f?.doc.id === 'match-rating') return factReply(f.doc.item)
    return matchReply(c)
  }
  if (has(query, W.lifts) && (has(query, ['my', 'best', 'pr', 'max', 'strongest', 'progress', 'cua toi', 'mi', 'meu', 'mon', 'saya']) || !ex) && !has(query, ['how do i', 'how to', 'form', 'technique', 'show me', 'demo', 'video', 'instead', 'alternative'])) return liftsReply(c, ex)
  if (has(query, W.body) && !ex && !has(query, ['add weight', 'more weight', 'heavier', 'when'])) return bodyReply(c)
  if (has(query, W.last)) return lastReply(c)
  if (has(query, W.today) && !has(query, ['why'])) return todayReply(c)
  if (has(query, ['how was my week', 'my week', 'this week', 'week so far', 'tuan nay', 'mi semana', 'minha semana', 'ma semaine', 'minggu ini']) && !has(query, ['why', 'planned', 'plan'])) return weekReply(c)
  if (ex && (has(query, W.how) || words.length <= 4)) return exerciseReply(c, ex, query)
  if (has(query, W.session) && !has(query, ['why'])) return sessionReply(c, query)

  // anything else: the best matching facts, then exercises
  const f = facts().search(query, 2)
  if (f[0] && f[0].score > 2.5) return factReply(f[0].doc.item, f[1] && f[1].score > f[0].score * 0.8 ? f[1].doc.item : undefined)
  if (ex) return exerciseReply(c, ex, query)
  const e = exercises().search(query, 3)
  if (e[0] && e[0].score > 3) return { text: [t('These exercises match that best:')], cards: e.map((h) => exCard(h.doc.item)) }
  if (has(query, W.week)) return weekReply(c)
  return {
    text: [t("I'm not sure about that one yet. I can help with your plan, your log and lifts, your matches, any exercise, soreness and how the app works.")],
    chips: [t('What should I do today?'), t('How is the load calculated?'), t('My hamstrings are sore')],
  }
}

function factReply(f: Fact, also?: Fact): Reply {
  const text = [f.answer]
  if (also && also.id !== f.id) text.push(also.answer)
  const go = f.see ? [{ kind: 'go' as const, to: f.see, label: { record: t('Open Record'), week: t('Open Week'), settings: t('Open Settings'), team: t('Open Team') }[f.see] }] : []
  return { text, cards: go }
}

export const starterChips = () => [t('What should I do today?'), t('How was my week?'), t('What are my best lifts?'), t('How do the exercises help my position?'), t('My hamstrings are sore'), t('Give me a 30-minute home session')]
