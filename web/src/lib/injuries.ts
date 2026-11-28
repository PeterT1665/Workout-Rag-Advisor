// Injuries the player told the coach about, kept in the app's own terms: part, tissue, what happened, how bad.
// On the profile (synced to their account, never shown to friends). Severity decides what training is allowed:
//   light    a niggle: exercises that mainly work the part are swapped out for 2 days, then it clears by itself
//   moderate every exercise that loads the part is out, and no jumps, sprints or Olympic lifts for a leg injury
//   serious  no session at all until it's healed: rest and see a physio
import { t } from '../i18n'
import { EXERCISES, exerciseBlock, exerciseByName } from './library'
import type { Block, Session } from './types'
import { getProfile, setProfile, type Injury, type InjuryPart, type Profile } from './profile'
import { has } from './coach/search'

export type Part = InjuryPart
export type Severity = NonNullable<Injury['severity']>

// Body parts: the words people use for them (any of the app's languages, accents optional), and easy work for soreness there.
export const PARTS: { part: Part; words: string[]; easy: string[] }[] = [
  { part: 'hamstrings', words: ['hamstring', 'hamstrings', 'hammy', 'hammies', 'back of thigh', 'back of my thigh', 'gan kheo', 'dui sau', 'isquio', 'isquiotibial', 'isquiotibiais', 'ischio', 'ischios', 'paha belakang'], easy: ['hamstring-stretch', 'glute-bridge', 'foam-roll-quads-and-glutes'] },
  { part: 'groin', words: ['groin', 'adductor', 'adductors', 'inner thigh', 'dau hang', 'bi hang', 'dui trong', 'ingle', 'aductor', 'aductores', 'virilha', 'adutor', 'aine', 'adducteur', 'adducteurs', 'selangkangan'], easy: ['90-90-hip-switches', 'side-lying-leg-raise-clamshell', 'figure-4-glute-stretch'] },
  { part: 'hips', words: ['hip', 'hips', 'hip flexor', 'hip flexors', 'hong', 'khop hang', 'cadera', 'quadril', 'hanche', 'pinggul'], easy: ['90-90-hip-switches', 'figure-4-glute-stretch', 'couch-stretch-hip-flexor-quad'] },
  { part: 'knees', words: ['knee', 'knees', 'kneecap', 'patella', 'acl', 'mcl', 'pcl', 'meniscus', 'goi', 'dau goi', 'rodilla', 'menisco', 'joelho', 'genou', 'menisque', 'lutut'], easy: ['glute-bridge', 'side-lying-leg-raise-clamshell', 'foam-roll-quads-and-glutes'] },
  { part: 'quads', words: ['quad', 'quads', 'quadricep', 'quadriceps', 'thigh', 'front of thigh', 'dead leg', 'dui truoc', 'dui', 'cuadriceps', 'muslo', 'coxa', 'cuisse', 'paha depan', 'paha'], easy: ['foam-roll-quads-and-glutes', 'couch-stretch-hip-flexor-quad', 'glute-bridge'] },
  { part: 'calves', words: ['calf', 'calves', 'shin', 'shins', 'bap chan', 'cang chan', 'gemelo', 'gemelos', 'pantorrilla', 'panturrilha', 'canela', 'mollet', 'tibia', 'betis'], easy: ['side-lying-leg-raise-clamshell', 'glute-bridge', 'hamstring-stretch'] },
  { part: 'ankles', words: ['ankle', 'ankles', 'achilles', 'foot', 'feet', 'heel', 'mat ca', 'co chan', 'ban chan', 'got chan', 'tobillo', 'pie', 'talon', 'tornozelo', 'pe', 'calcanhar', 'cheville', 'pied', 'pergelangan', 'pergelangan kaki', 'tumit', 'telapak'], easy: ['side-lying-leg-raise-clamshell', 'glute-bridge', 'hamstring-stretch'] },
  { part: 'back', words: ['back', 'lower back', 'upper back', 'spine', 'lung', 'that lung', 'cot song', 'espalda', 'lumbar', 'costas', 'lombar', 'au dos', 'mal au dos', 'punggung', 'pinggang'], easy: ['dead-bug', 'glute-bridge', 'figure-4-glute-stretch'] },
  { part: 'glutes', words: ['glute', 'glutes', 'butt', 'bum', 'mong', 'gluteo', 'gluteos', 'fessier', 'fessiers', 'bokong'], easy: ['figure-4-glute-stretch', 'foam-roll-quads-and-glutes', '90-90-hip-switches'] },
  { part: 'core', words: ['abs', 'ab', 'stomach', 'oblique', 'obliques', 'ribs', 'bung', 'co bung', 'abdomen', 'abdominal', 'abdominales', 'abdomen', 'costela', 'abdos', 'perut'], easy: ['figure-4-glute-stretch', '90-90-hip-switches', 'glute-bridge'] },
  { part: 'chest', words: ['chest', 'pec', 'pecs', 'nguc', 'pecho', 'pectoral', 'peito', 'poitrine', 'pecs', 'dada'], easy: ['90-90-hip-switches', 'glute-bridge', 'dead-bug'] },
  { part: 'shoulder', words: ['shoulder', 'shoulders', 'rotator', 'rotator cuff', 'collarbone', 'vai', 'hombro', 'ombro', 'epaule', 'bahu'], easy: ['dead-bug', 'glute-bridge', '90-90-hip-switches'] },
  { part: 'arms', words: ['arm', 'arms', 'elbow', 'wrist', 'hand', 'finger', 'bicep', 'tricep', 'tay', 'khuyu tay', 'co tay', 'brazo', 'codo', 'muneca', 'braco', 'cotovelo', 'pulso', 'bras', 'coude', 'poignet', 'lengan', 'siku', 'pergelangan tangan'], easy: ['glute-bridge', '90-90-hip-switches', 'hamstring-stretch'] },
]
const LEG: Part[] = ['hamstrings', 'quads', 'groin', 'hips', 'glutes', 'knees', 'calves', 'ankles']
const JOINTS: Part[] = ['knees', 'ankles', 'hips', 'shoulder', 'back', 'arms']
export const isLeg = (p: Part) => LEG.includes(p)

export const partLabel = (p: Part) => ({
  hamstrings: t('hamstring'), quads: t('quad'), groin: t('groin'), hips: t('hip'), glutes: t('glute'), knees: t('knee'), calves: t('calf'), ankles: t('ankle'),
  back: t('back'), core: t('core'), chest: t('chest'), shoulder: t('shoulder'), arms: t('arm'), other: t('other'),
})[p]
// "muscle strain", "ligament sprain"...
export const injuryKind = (i: Pick<Injury, 'kind' | 'tissue'>) => ({
  strain: t('muscle strain'), tear: i.tissue === 'ligament' ? t('ligament tear') : t('muscle tear'), sprain: t('ligament sprain'), tendon: t('tendon pain'),
  bruise: t('bruise'), fracture: t('fracture'), pain: i.tissue === 'joint' ? t('joint pain') : t('muscle pain'),
})[i.kind]
export const severityLabel = (s?: Severity) => (s === 'light' ? t('Light') : s === 'serious' ? t('Serious') : s === 'moderate' ? t('Moderate') : t('Not rated yet'))
export const partOf = (q: string): Part | null => PARTS.find((p) => has(q, p.words))?.part ?? null

// What happened, from the words used: "pulled" is a strain, "rolled" a sprain...
const KIND_WORDS: [Injury['kind'], string[]][] = [
  ['fracture', ['broke', 'broken', 'fracture', 'fractured', 'cracked', 'gay', 'gay xuong', 'nut xuong', 'fractura', 'roto', 'fratura', 'quebrei', 'quebrado', 'casse', 'patah', 'retak']],
  ['tear', ['tear', 'torn', 'tore', 'rupture', 'ruptured', 'rach', 'dut', 'desgarro', 'rotura', 'ruptura', 'rompi', 'dechirure', 'dechire', 'robek', 'sobek']],
  ['sprain', ['sprain', 'sprained', 'rolled', 'twisted', 'turned my ankle', 'went over on', 'trat', 'bong gan', 'lat', 'esguince', 'torci', 'torcedura', 'torci o', 'entorse', 'foule', 'keseleo', 'terkilir']],
  ['tendon', ['tendon', 'tendinitis', 'tendonitis', 'tendinopathy', 'achilles', 'jumpers knee', 'patellar', 'gan', 'viem gan', 'tendinitis', 'tendinite', 'tendon', 'tendinite', 'tendon']],
  ['bruise', ['bruise', 'bruised', 'dead leg', 'knock', 'knocked', 'kicked', 'got kicked', 'bam', 'bam tim', 'contusion', 'golpe', 'moraton', 'pancada', 'hematoma', 'contusao', 'bleu', 'coup', 'memar', 'lebam']],
  ['strain', ['pulled', 'pull', 'strain', 'strained', 'tweaked', 'tight', 'cang co', 'keo co', 'gian co', 'distension', 'tiron', 'estiramento', 'distensao', 'claquage', 'elongation', 'contracture', 'tertarik', 'kram']],
]
const SEVERITY_WORDS: [Severity, string[]][] = [
  ['serious', ['serious', 'seriously', 'very bad', 'really bad', 'badly', 'severe', 'swollen', 'swelling', 'cant walk', 'can t walk', 'cannot walk', 'limp', 'limping', 'pop', 'popped', 'cant run', 'can t run', 'cant put weight', 'nang', 'rat nang', 'sung', 'khong di duoc', 'grave', 'muy mal', 'hinchado', 'no puedo caminar', 'inchado', 'nao consigo andar', 'gonfle', 'je ne peux pas marcher', 'parah', 'bengkak', 'tidak bisa jalan']],
  ['moderate', ['moderate', 'medium', 'quite bad', 'hurts to run', 'hurts when i run', 'hurts to jump', 'hurts when i sprint', 'vua', 'vua phai', 'trung binh', 'moderado', 'moderada', 'moyen', 'modere', 'sedang', 'lumayan']],
  ['light', ['light', 'slight', 'slightly', 'mild', 'minor', 'small', 'little', 'a bit', 'bit', 'niggle', 'tiny', 'nhe', 'hoi', 'leve', 'poco', 'un poco', 'leger', 'legere', 'un peu', 'ringan', 'sedikit']],
]
export const severityOf = (q: string): Severity | null => SEVERITY_WORDS.find(([, w]) => has(q, w))?.[0] ?? null

// A message about an injury, in the app's terms.
export function describe(q: string, part: Part): Pick<Injury, 'part' | 'tissue' | 'kind' | 'severity'> {
  const kind = KIND_WORDS.find(([, w]) => has(q, w))?.[0] ?? 'pain'
  const ligament = has(q, ['acl', 'mcl', 'pcl', 'ligament', 'day chang', 'ligamento', 'ligament', 'ligamen'])
  const tissue: Injury['tissue'] = kind === 'fracture' ? 'bone' : kind === 'sprain' || ligament ? 'ligament' : kind === 'tendon' ? 'tendon'
    : kind === 'pain' && JOINTS.includes(part) ? 'joint' : kind === 'tear' && part === 'knees' ? 'ligament' : 'muscle'
  const severity = severityOf(q) ?? (kind === 'fracture' || kind === 'tear' ? 'serious' : undefined)
  return { part, tissue, kind: kind === 'tear' && tissue === 'ligament' ? 'tear' : kind, ...(severity ? { severity } : {}) }
}

// Exercises: the library says what each works (main) and also loads. Light injuries only rule out the main ones.
const exOf = (name: string) => exerciseByName(name)
const OLD: Record<string, RegExp> = { // the player's own exercises (no library data): guess from the name
  hamstrings: /rdl|deadlift|nordic|sprint|hamstring/, quads: /squat|lunge|jump|sprint/, groin: /copenhagen|adductor|lateral/, hips: /lunge|hip/, glutes: /squat|thrust|bridge|lunge/,
  knees: /squat|lunge|jump|hop|bound/, calves: /calf|jump|hop|sprint/, ankles: /jump|hop|sprint|cut/, back: /deadlift|rdl|squat|row|clean|snatch/, core: /plank|crunch|twist/,
  chest: /bench|press|push/, shoulder: /press|push|pull|row|throw/, arms: /curl|press|pull|row|dip/, other: /$^/,
}
export function loadsPart(block: { name: string; works?: string[]; also?: string[] }, part: Part, sev: Severity = 'moderate') {
  const e = exOf(block.name)
  const works = block.works ?? e?.works, also = block.also ?? e?.also
  if (!works) return OLD[part].test(block.name.toLowerCase())
  return works.includes(part) || (sev !== 'light' && !!also?.includes(part))
}
const EXPLOSIVE = ['plyometrics', 'sprints', 'speed', 'explosive', 'olympic']

const today = () => new Date().toLocaleDateString('en-CA')
const daysSince = (iso: string, now = new Date()) => { const [y, m, d] = iso.split('-').map(Number); return Math.round((new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() - new Date(y, m - 1, d).getTime()) / 864e5) }
// Light injuries clear after 2 days by themselves (tell the coach if it still hurts and it comes back as moderate).
export const activeInjuries = (p: Profile = getProfile(), now = new Date()) => (p.injuries ?? []).filter((i) => !i.healed && !(i.severity === 'light' && daysSince(i.since, now) > 2))
export const seriousInjury = (list: Injury[] = activeInjuries()) => list.find((i) => i.severity === 'serious') ?? null

// A new injury (a second report for the same active part updates it).
export function addInjury(x: Pick<Injury, 'part' | 'tissue' | 'kind' | 'severity'>, p: Profile = getProfile()): Injury {
  const all = p.injuries ?? []
  const open = activeInjuries(p).find((i) => i.part === x.part)
  const entry: Injury = open ? { ...open, ...x, severity: x.severity ?? open.severity } : { id: `${Date.now().toString(36)}`, ...x, since: today() }
  setProfile({ ...p, injuries: [...all.filter((i) => i.id !== entry.id), entry].slice(-30) })
  return entry
}
export function rateInjury(id: string, severity: Severity, p: Profile = getProfile()) {
  setProfile({ ...p, injuries: (p.injuries ?? []).map((i) => (i.id === id ? { ...i, severity, ...(severity === 'light' ? { since: today() } : {}) } : i)) })
}
export function healInjury(id: string, p: Profile = getProfile()) {
  setProfile({ ...p, injuries: (p.injuries ?? []).map((i) => (i.id === id ? { ...i, healed: today() } : i)) })
}

// Kick off with an injury on file. Serious: no session (blocked). Otherwise exercises that load the injured part are swapped
// for one of the same family that doesn't (same category, same sport), or left out; with a moderate leg injury, jumps,
// sprints and Olympic lifts are left out too. Warm-ups and cool-downs stay.
export type InjurySwap = { from: string; to: string | null; part: Part }
export function injurySafe(s: Session, injuries: Injury[] = activeInjuries()): { session: Session; swaps: InjurySwap[]; blocked: Injury | null } {
  if (!injuries.length) return { session: s, swaps: [], blocked: null }
  const blocked = seriousInjury(injuries)
  const sev = (i: Injury) => i.severity ?? 'moderate'
  const hurts = (b: { name: string; works?: string[]; also?: string[]; category?: string }) =>
    injuries.find((i) => loadsPart(b, i.part, sev(i)) || (sev(i) !== 'light' && isLeg(i.part) && EXPLOSIVE.includes(b.category ?? exOf(b.name)?.category ?? '')))?.part
  const used = new Set(s.blocks.map((b) => b.name))
  const swaps: InjurySwap[] = []
  const blocks: Block[] = []
  let unpairNext = false // the exercise left out led a superset: its partner goes on alone
  for (const orig of s.blocks) {
    const b = unpairNext && orig.superset ? { ...orig, superset: undefined } : orig
    unpairNext = false
    const part = b.kind === 'drill' ? hurts(b) : undefined
    if (!part) { blocks.push(b); continue }
    const me = exOf(b.name)
    const pick = EXERCISES.find((e) => e.kind === 'drill' && !e.heading && e.category === (me?.category ?? b.category) && e.sport === (me?.sport ?? e.sport) && !used.has(e.name) && !hurts(e))
    if (pick) {
      const blk = exerciseBlock(pick.id)!
      used.add(pick.name)
      blocks.push({ ...blk, rounds: b.rounds ?? blk.rounds, rest: b.rest ?? blk.rest, superset: b.superset })
      swaps.push({ from: b.name, to: pick.name, part })
    } else {
      swaps.push({ from: b.name, to: null, part })
      if (!b.superset) unpairNext = true
    }
  }
  // tidy supersets: a superset flag needs an exercise right before it
  const tidy = blocks.map((b, k) => (b.superset && (k === 0 || blocks[k - 1].kind !== 'drill') ? { ...b, superset: undefined } : b))
  // an ankle or knee injury on file: the full 11+ style warm-up instead of the usual one (it cuts ankle injuries)
  const warm = injuries.find((i) => i.part === 'ankles' || i.part === 'knees')
  const w11 = warm ? exerciseBlock('warmup-11-1') : undefined
  if (warm && w11 && tidy.some((b) => b.kind === 'warmup' || b.category === 'warmup') && !tidy.some((b) => b.name.startsWith('Warm-up 11+'))) {
    let placed = false
    const withWarm = tidy.flatMap((b) => (b.kind === 'warmup' || b.category === 'warmup' ? (placed ? [] : ((placed = true), [w11])) : [b]))
    swaps.push({ from: t('Warm-up'), to: w11.name, part: warm.part })
    return { session: { ...s, blocks: withWarm }, swaps, blocked }
  }
  return { session: { ...s, blocks: tidy }, swaps, blocked }
}

// A day whose session the injury guts (a leg day with a moderate knee injury, a sprint session with a hamstring strain):
// the session of the same kind (gym, football) among `others` that loses least, so the player still trains what they
// safely can. Null: keep today's session (kick-off swaps what it must).
export function safestInstead(s: Session, others: Session[], injuries: Injury[] = activeInjuries()): Session | null {
  if (!injuries.length || seriousInjury(injuries) || (s.type !== 'gym' && s.type !== 'football')) return null
  const lost = (x: Session) => injurySafe(x, injuries).swaps.length / Math.max(1, x.blocks.filter((b) => b.kind === 'drill').length)
  if (lost(s) < 0.5) return null
  const best = others.filter((x) => x.type === s.type && x.id !== s.id && !x.id.includes('partner')).sort((a, b) => lost(a) - lost(b))[0]
  return best && lost(best) < lost(s) ? best : null
}
