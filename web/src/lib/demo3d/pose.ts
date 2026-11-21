// Poses and moves for the 3D exercise demos: plain data, no three.js (unit-tested, cheap to import).
// Angles in degrees, from a neutral standing pose, figure facing the camera's side (+Z).
//   hip / knee / sh (shoulder) / elb: flexion, + = forward (knee: shin back, elbow: forearm up)
//   abd: away from the body; rot: twist about the limb; ankle: + toes up (feet are kept level unless `flatFeet` is off)
//   pitch: whole body tipped forward (90 = face down), roll: onto the side, yaw: turned
//   spine: bend forward (split over lower and upper back), side: side bend, twist; neck: chin down
export type Side = { hip?: number; abd?: number; rot?: number; knee?: number; ankle?: number; sh?: number; shAbd?: number; shRot?: number; elb?: number }
export type Vec = [number, number, number]
export type Pose = {
  x?: number; z?: number; air?: number // root offsets (m); air = lifted off the floor (jumps)
  pitch?: number; roll?: number; yaw?: number
  spine?: number; side?: number; twist?: number; neck?: number
  L?: Side; R?: Side
  ball?: Vec | { at: 'L' | 'R' | 'hands'; off?: Vec } // ball position in the world, or held on a foot / in the hands
}
export type Key = [t: number, pose: Pose]

export type PropKind =
  | 'barbell' | 'barbell-back' | 'barbell-front' | 'barbell-hips' | 'dumbbells' | 'dumbbell-R' | 'kettlebell' | 'trapbar' | 'medball' | 'landmine'
  | 'bench' | 'box' | 'bar' | 'wall' | 'cones' | 'hurdles' | 'sofa' | 'table' | 'bike' | 'sled' | 'goal' | 'roller' | 'band' | 'step' | 'towel'
export type Prop = { kind: PropKind; at?: Vec; size?: Vec; yaw?: number }

export type Move = {
  dur: number // seconds per loop
  keys: Key[]
  props?: Prop[]
  ball?: boolean // a football
  anchor?: 'feet' | 'L' | 'R' | 'hands' | 'none' // what stays put on the floor (default: both feet)
  contacts?: 'all' | 'feet' | 'back' | 'hands-feet' // which points touch the floor/support (default all)
  base?: number // height of what the contacts rest on (bench top...)
  hang?: number // hanging from a bar: hands at this height
  seat?: number // sitting on a saddle: hips at this height
  flatFeet?: boolean // keep feet level (default true)
  run?: number // the floor slides under the figure at this speed (m/s): running on the spot reads as running
  cam?: { yaw?: number; tilt?: number; zoom?: number } // view angle (default: three-quarter from the front-left)
  hold?: boolean // a static hold: only a gentle breathing motion
}

// Smooth step between keyframes, looping.
export const ease = (u: number) => u * u * (3 - 2 * u)
const lerp = (a: number, b: number, u: number) => a + (b - a) * u

function mixSide(a: Side = {}, b: Side = {}, u: number): Side {
  const out: Side = {}
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof Side>) out[k] = lerp(a[k] ?? 0, b[k] ?? 0, u)
  return out
}
const NUM = ['x', 'z', 'air', 'pitch', 'roll', 'yaw', 'spine', 'side', 'twist', 'neck'] as const
export function mixPose(a: Pose, b: Pose, u: number): Pose {
  const out: Pose = { L: mixSide(a.L, b.L, u), R: mixSide(a.R, b.R, u) }
  for (const k of NUM) out[k] = lerp(a[k] ?? 0, b[k] ?? 0, u)
  return out
}

// The pose at loop time t (0..1): keys are sorted by time; the last wraps to the first.
export function poseAt(keys: Key[], t: number): { pose: Pose; a: number; b: number; u: number } {
  const n = keys.length
  if (n === 1) return { pose: keys[0][1], a: 0, b: 0, u: 0 }
  t = ((t % 1) + 1) % 1
  let i = n - 1
  for (let k = 0; k < n; k++) if (keys[k][0] <= t) i = k
  const j = (i + 1) % n
  const t0 = keys[i][0], t1 = j === 0 ? 1 + keys[0][0] : keys[j][0]
  const span = t1 - t0 || 1
  const tt = t < t0 ? t + 1 : t
  const u = ease(Math.min(1, Math.max(0, (tt - t0) / span)))
  return { pose: mixPose(keys[i][1], keys[j][1], u), a: i, b: j, u }
}

// Both sides the same.
export const both = (s: Side): Pick<Pose, 'L' | 'R'> => ({ L: { ...s }, R: { ...s } })
// Mirror a pose left <-> right (for alternating sides).
export const mirror = (p: Pose): Pose => ({ ...p, L: p.R, R: p.L, x: p.x !== undefined ? -p.x : undefined, side: p.side !== undefined ? -p.side : undefined, twist: p.twist !== undefined ? -p.twist : undefined, yaw: p.yaw !== undefined ? -p.yaw : undefined, roll: p.roll !== undefined ? -p.roll : undefined, ball: Array.isArray(p.ball) ? [-p.ball[0], p.ball[1], p.ball[2]] : p.ball && { ...p.ball, at: p.ball.at === 'L' ? 'R' : p.ball.at === 'R' ? 'L' : 'hands' } })
// A loop that does one side, then the other: keys for 0..0.5, mirrored for 0.5..1.
export const alternate = (keys: Key[]): Key[] => [...keys.map(([t, p]): Key => [t / 2, p]), ...keys.map(([t, p]): Key => [0.5 + t / 2, mirror(p)])]
