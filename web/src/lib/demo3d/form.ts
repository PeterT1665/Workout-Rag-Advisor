// Form check for the 3D demos: the key cue for a move, and its most common mistake acted out (red), then the fix (green).
// A mistake is a small change added to the move's poses, so it stays in step with the real move.
import { t } from '../../i18n'
import type { Move, Pose, Side } from './pose'

export type Mark = 'knees' | 'front-knee' | 'elbows' | 'back' | 'lowback' | 'hips' | 'chest'
type Delta = Partial<Record<'pitch' | 'roll' | 'yaw' | 'spine' | 'side' | 'twist' | 'neck', number>> & { L?: Side; R?: Side; both?: Side }
type On = 'work' | 'all' | 'start' // the keys it changes: all but the first (the start pose), every key, or only the first
export type Mistake = { label: string; fix: string; marks: Mark[]; delta: Delta; on?: On }
export type Form = { cue: string; mistakes: Mistake[] }

const addSide = (s: Side = {}, ...ds: (Side | undefined)[]): Side => {
  const out: Side = { ...s }
  for (const d of ds) if (d) for (const k of Object.keys(d) as (keyof Side)[]) out[k] = (out[k] ?? 0) + (d[k] ?? 0)
  return out
}
function addPose(p: Pose, d: Delta, k = 1): Pose {
  const sc = (x?: Side) => x && Object.fromEntries(Object.entries(x).map(([a, v]) => [a, (v ?? 0) * k])) as Side
  const out: Pose = { ...p, L: addSide(p.L, sc(d.both), sc(d.L)), R: addSide(p.R, sc(d.both), sc(d.R)) }
  for (const key of ['pitch', 'roll', 'yaw', 'spine', 'side', 'twist', 'neck'] as const) if (d[key]) out[key] = (p[key] ?? 0) + d[key]! * k
  return out
}

// How far a pose is from the start pose (joint angles summed): the mistake grows with it, so a back rounds as you
// hinge and knees cave as you go down, not while standing.
const JOINTS = ['hip', 'knee', 'abd', 'sh', 'shAbd', 'elb'] as const
function dist(a: Pose, b: Pose) {
  let d = Math.abs((a.pitch ?? 0) - (b.pitch ?? 0)) + Math.abs((a.spine ?? 0) - (b.spine ?? 0)) + Math.abs((a.air ?? 0) - (b.air ?? 0)) * 100
  for (const s of ['L', 'R'] as const) for (const j of JOINTS) d += Math.abs((a[s]?.[j] ?? 0) - (b[s]?.[j] ?? 0))
  return d
}

// The move done with this mistake.
export function withMistake(move: Move, m: Mistake): Move {
  const on = m.on ?? (move.keys.length === 1 ? 'all' : 'work')
  // depth is measured from the most neutral key (a squat's standing pose, a deadlift's lockout), wherever it is in the loop
  const zero: Pose = {}
  const start = move.keys.map(([, p]) => p).reduce((a, b) => (dist(b, zero) < dist(a, zero) ? b : a))
  const far = Math.max(1, ...move.keys.map(([, p]) => dist(p, start)))
  return {
    ...move,
    keys: move.keys.map(([time, p], i) => {
      const k = on === 'all' ? 1 : on === 'start' ? (i === 0 ? 1 : 0) : dist(p, start) / far
      return [time, k ? addPose(p, m.delta, k) : p]
    }),
  }
}

// Cues and mistakes, shared by moves that are the same pattern. Built on call so they're in the app's language.
function forms(): Record<string, Form> {
  const roundedBack = (cue: string, spine = 40): Form => ({ cue, mistakes: [{ label: t('Rounded back'), fix: t('Chest proud, flat back, hinge from the hips'), marks: ['back'], delta: { spine } }] })
  const squat: Form = { cue: t('Knees out over your toes, chest up, sit down between your heels.'), mistakes: [{ label: t('Knees caving in'), fix: t('Push your knees out over your toes'), marks: ['knees'], delta: { L: { abd: -14 }, R: { abd: -14 } } }] }
  const hinge = roundedBack(t('Push your hips back, keep the weight close to your legs and your back flat.'))
  const bench: Form = { cue: t('Elbows at about 45 degrees, feet planted, lower to your lower chest.'), mistakes: [{ label: t('Elbows flared out wide'), fix: t('Tuck your elbows to about 45 degrees'), marks: ['elbows'], delta: { both: { shAbd: 32 } } }] }
  const pushUp: Form = { cue: t('Body in one straight line, elbows back at about 45 degrees.'), mistakes: [{ label: t('Hips sagging'), fix: t('Squeeze your glutes and brace: one straight line'), marks: ['lowback'], delta: { spine: -22 }, on: 'all' }] }
  const plank: Form = { cue: t('Elbows under your shoulders, glutes squeezed, one straight line.'), mistakes: [{ label: t('Hips sagging'), fix: t('Squeeze your glutes and brace: one straight line'), marks: ['lowback'], delta: { spine: -22 } }] }
  // hips dropping isn't shown yet: a side bend on the rig lifts the shoulder instead (cue only for now)
  const sidePlank: Form = { cue: t('Hips high: one straight line from your head to your feet.'), mistakes: [] }
  const split: Form = { cue: t('Front knee over your toes, chest tall, back knee drops straight down.'), mistakes: [{ label: t('Front knee caving in'), fix: t('Push the front knee out over your toes'), marks: ['front-knee'], delta: { L: { abd: -14 } } }] }
  const landing: Form = { cue: t('Swing your arms, jump, and land soft and quiet with your knees over your toes.'), mistakes: [{ label: t('Knees caving in on the landing'), fix: t('Land with your knees over your toes, hips back'), marks: ['knees'], delta: { L: { abd: -14 }, R: { abd: -14 } } }] }
  const thrust: Form = { cue: t('Chin tucked, ribs down, squeeze your glutes hard at the top.'), mistakes: [{ label: t('Arching the lower back'), fix: t('Ribs down: finish with your glutes, not your back'), marks: ['lowback'], delta: { spine: -22 } }] }
  const press: Form = { cue: t('Ribs down, glutes tight, press straight up.'), mistakes: [{ label: t('Leaning back'), fix: t('Squeeze your glutes, ribs down, stand tall'), marks: ['lowback'], delta: { spine: -20 } }] }
  const row = roundedBack(t('Flat back, pull to your lower ribs, squeeze your shoulder blades.'), 35)
  return {
    'squat-back': squat, 'squat-front': squat, 'squat-bw': squat, 'squat-half': squat, 'db-goblet-squat': squat, 'landmine-squat': squat,
    rdl: hinge, 'db-rdl': hinge, deadlift: hinge, 'db-deadlift': hinge, 'trap-bar-deadlift': hinge,
    'sl-rdl': { cue: t('Hips square to the floor, one long line from your head to your back heel.'), mistakes: [{ label: t('Hips opening up'), fix: t('Point the back toes at the floor, hips level'), marks: ['hips'], delta: { roll: 16 } }] },
    'bench-press': bench, 'db-bench-press': bench,
    'pull-up': { cue: t('Start from a full hang and pull your chest to the bar. No kicking.'), mistakes: [{ label: t('Half reps'), fix: t('Straighten your arms all the way at the bottom'), marks: ['elbows'], delta: { both: { elb: 70, sh: -40 } }, on: 'start' }] },
    'push-up': pushUp, 'plyo-push-up': pushUp, plank,
    'side-plank': sidePlank, copenhagen: { ...sidePlank, cue: t('Top leg on the bench, hips high, one straight line.') },
    nordic: { cue: t('One straight line from your knees to your head. Lower as slowly as you can.'), mistakes: [{ label: t('Bending at the hips'), fix: t('Squeeze your glutes and keep your hips forward'), marks: ['hips'], delta: { both: { hip: 45, knee: 45 } } }] },
    'split-squat-rear': split, 'iso-split': split, 'db-split-squat': split,
    'box-jump': landing, 'jump-squat': landing, cmj: landing, 'broad-jump': landing, 'db-jump-squat': landing,
    pogo: { cue: t('Stiff ankles, quick bounces off the balls of your feet.'), mistakes: [{ label: t('Bending the knees too much'), fix: t('Knees nearly straight: the bounce comes from your ankles'), marks: ['knees'], delta: { both: { hip: 25, knee: 45 } } }] },
    'hip-thrust': thrust, bridge: thrust, 'db-hip-thrust': thrust,
    ohp: press, 'ohp-barbell': press, 'db-push-press': press,
    row, 'db-row': row,
    'kb-swing': { cue: t('Snap your hips forward. Your arms just guide the bell.'), mistakes: [{ label: t('Squatting the swing'), fix: t('Hinge: hips back, shins nearly upright'), marks: ['knees'], delta: { pitch: -25, both: { knee: 55, hip: -5 } }, on: 'start' }] },
    'dead-bug': { cue: t('Press your lower back into the floor and reach slowly.'), mistakes: [{ label: t('Lower back arching off the floor'), fix: t('Breathe out and keep your back pressed down'), marks: ['lowback'], delta: { spine: -18 } }] },
    pallof: { cue: t("Press straight out and don't let the cable turn you."), mistakes: [{ label: t('Twisting toward the cable'), fix: t('Brace: hips and shoulders stay square'), marks: ['chest'], delta: { twist: 28 } }] },
    'step-up': { cue: t("Drive up through the front foot. Don't push off the back one."), mistakes: [{ label: t('Knee caving in'), fix: t('Keep the knee over your toes'), marks: ['front-knee'], delta: { L: { abd: -14 } } }] },
  }
}

let cache: Record<string, Form> | null = null
export const formFor = (move: string): Form | null => (cache ??= forms())[move] ?? null
export const FORM_MOVES = () => Object.keys(cache ??= forms())
