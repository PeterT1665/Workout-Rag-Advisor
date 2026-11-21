// The movement library for the 3D demos. Each exercise maps onto one of these (catalog.ts).
// Angles follow pose.ts. Poses are kept simple: enough to see what the exercise is, not a biomechanics lab.
import { alternate, both, mirror, type Key, type Move, type Pose, type Side } from './pose'

const ARMS_FWD: Side = { sh: 80, shAbd: 10 }
const RUN_A: Pose = { spine: 8, L: { hip: 55, knee: 80, sh: -35, elb: 85 }, R: { hip: -15, knee: 35, sh: 40, elb: 85 } }
const RUN_B = mirror(RUN_A)
const SPRINT_A: Pose = { pitch: 10, spine: 5, L: { hip: 75, knee: 95, sh: -45, elb: 85 }, R: { hip: -25, knee: 60, sh: 60, elb: 80 } }
const SPRINT_B = mirror(SPRINT_A)
const ATHLETIC: Pose = { spine: 20, ...both({ hip: 35, knee: 40, abd: 10, sh: 20, elb: 40 }) }

// ---------- squats ----------
function squat(depth: number, arms: Side, extra: Partial<Move> = {}, dur = 3): Move {
  const leg = (d: number): Side => ({ hip: 100 * d, knee: 118 * d, abd: 8 })
  return {
    dur, cam: { yaw: 55 },
    keys: [[0, { ...both({ ...leg(0), ...arms }) }], [0.45, { spine: 28 * depth, ...both({ ...leg(depth), ...arms }) }], [0.6, { spine: 28 * depth, ...both({ ...leg(depth), ...arms }) }]],
    ...extra,
  }
}
const BACK_RACK: Side = { shAbd: 75, shRot: 80, sh: -10, elb: 105 }
const FRONT_RACK: Side = { sh: 95, shAbd: 15, elb: 150 }
const OVERHEAD: Side = { sh: 175, shAbd: 18 }

function jump(arms: boolean, extra: Partial<Move> = {}, holdArms: Side = {}): Move {
  const up: Side = arms ? { sh: 170, shAbd: 15 } : holdArms
  const load: Side = arms ? { sh: -40, shAbd: 10 } : holdArms
  return {
    dur: 1.8, cam: { yaw: 50 },
    keys: [
      [0, both({ ...holdArms })],
      [0.25, { spine: 25, ...both({ hip: 75, knee: 90, ...load }) }],
      [0.42, { air: 0.38, ...both({ ankle: -35, ...up }) }],
      [0.58, { spine: 15, ...both({ hip: 45, knee: 55, ...holdArms }) }],
      [0.75, both({ ...holdArms })],
    ],
    ...extra,
  }
}

// ---------- hinges ----------
const hinge = (p: number, hip: number, knee: number, arms: Side = {}): Pose => ({ pitch: p, spine: 5, ...both({ hip, knee, sh: p + 5, ...arms }) })
function deadlift(bottom: Pose, prop: 'barbell' | 'trapbar' | 'kettlebell' = 'barbell', dur = 3): Move {
  return { dur, cam: { yaw: 70 }, props: [{ kind: prop }], keys: [[0, bottom], [0.4, both({})], [0.55, both({})]] }
}

// ---------- olympic ----------
const HANG = hinge(25, 40, 20)
const RACK_DIP: Pose = { spine: 8, ...both({ hip: 45, knee: 60, ...FRONT_RACK }) }
const RACK: Pose = both({ ...FRONT_RACK })
const EXTEND: Pose = { air: 0.06, spine: -5, ...both({ ankle: -30, sh: 15, shAbd: 25, elb: 30 }) }

// ---------- lying ----------
const SUPINE = -90
// shoulders stay down, hips lift: the trunk tips up by `up` (0..1) and the hips open
const bridge = (up: number): Pose => ({ pitch: SUPINE - 26 * up, ...both({ hip: 70 - 62 * up, knee: 100 + 10 * up, sh: -10 }) })

// ---------- ball ----------
const touch = (side: 'L' | 'R', x: number, z = 0.38): Key[1] => ({ ...ATHLETIC, [side]: { hip: 35, knee: 30, abd: 0, sh: side === 'L' ? -20 : 20 }, ball: [x, 0.11, z] })
function passWall(dur = 2.2): Move {
  return {
    dur, ball: true, cam: { yaw: 60, zoom: 1 }, props: [{ kind: 'wall', at: [0, 0, 3.2], size: [3, 1.6, 0.1] }],
    keys: [
      [0, { ...ATHLETIC, R: { hip: -25, knee: 30, rot: -50, sh: 20 }, ball: [0.12, 0.11, 0.35] }],
      [0.18, { ...ATHLETIC, R: { hip: 35, knee: 10, rot: -50, sh: -20 }, L: { hip: 30, knee: 30, sh: 30 }, ball: [0.12, 0.11, 0.6] }],
      [0.48, { ...ATHLETIC, ball: [0.05, 0.11, 3.05] }],
      [0.78, { ...ATHLETIC, L: { hip: 20, knee: 25, rot: 40 }, ball: [-0.12, 0.11, 0.4] }],
    ],
  }
}
function strike(target: [number, number, number], props: Move['props'], dur = 2.4, volley = false): Move {
  return {
    dur, ball: true, cam: { yaw: 70, zoom: 1.05 }, props,
    keys: [
      [0, { spine: 5, L: { hip: 20, knee: 20, sh: 20, shAbd: 40 }, R: { hip: 0, knee: 10, shAbd: 40 }, ball: volley ? { at: 'hands', off: [0, 0, 0.1] } : [0.1, 0.11, 0.55] }],
      [0.25, { spine: 15, side: -10, L: { hip: 30, knee: 30, shAbd: 60 }, R: { hip: -45, knee: 95, shAbd: 30, sh: 30 }, ball: volley ? [0.1, 0.6, 0.55] : [0.1, 0.11, 0.55] }],
      [0.38, { spine: 15, L: { hip: 30, knee: 25, shAbd: 60 }, R: { hip: 60, knee: 10, ankle: -20, shAbd: 20, sh: -20 }, ball: volley ? [0.1, 0.45, 0.65] : [0.1, 0.11, 0.6] }],
      [0.7, { spine: 5, L: { hip: 15, knee: 15 }, R: { hip: 20, knee: 15 }, ball: target }],
    ],
  }
}

export const MOVES: Record<string, Move> = {
  // squats and jumps
  'squat-bw': squat(1, ARMS_FWD, {}, 3.6),
  'squat-back': squat(1, BACK_RACK, { props: [{ kind: 'barbell-back' }] }),
  'squat-front': squat(1, FRONT_RACK, { props: [{ kind: 'barbell-front' }] }),
  'squat-half': squat(0.7, BACK_RACK, { props: [{ kind: 'barbell-back' }] }),
  'squat-quarter': squat(0.42, BACK_RACK, { props: [{ kind: 'barbell-back' }] }, 2.2),
  'wall-sit': { dur: 4, hold: true, cam: { yaw: 70 }, props: [{ kind: 'wall', at: [0, 0, -0.68] }], keys: [[0, both({ hip: 90, knee: 90, sh: 0 })]] },
  'jump-squat': jump(false, {}, { shAbd: 12 }),
  'cmj': jump(true),
  'trap-bar-jump': jump(false, { props: [{ kind: 'trapbar' }] }, { shAbd: 8 }),
  'pogo': { dur: 0.9, cam: { yaw: 40 }, keys: [[0, both({ ankle: 0, sh: 10 })], [0.25, { air: 0.14, ...both({ ankle: -35, sh: 15 }) }], [0.5, both({ ankle: 0, knee: 8, sh: 10 })]] },
  'box-jump': {
    dur: 3, anchor: 'none', cam: { yaw: 75 }, props: [{ kind: 'box', at: [0, 0, 0.85], size: [0.6, 0.5, 0.5] }],
    keys: [
      [0, { z: 0, ...both({}) }],
      [0.18, { z: 0, spine: 25, ...both({ hip: 75, knee: 90, sh: -40 }) }],
      [0.32, { z: 0.45, air: 0.75, ...both({ hip: 40, knee: 60, ankle: -20, sh: 150 }) }],
      [0.45, { z: 0.82, air: 0.5, spine: 15, ...both({ hip: 60, knee: 70, sh: 60 }) }],
      [0.62, { z: 0.82, air: 0.5, ...both({}) }],
      [0.8, { z: 0.4, air: 0.25, ...both({ hip: 20, knee: 30 }) }],
    ],
  },
  'depth-jump': {
    dur: 3, anchor: 'none', cam: { yaw: 75 }, props: [{ kind: 'box', at: [0, 0, -0.75], size: [0.6, 0.45, 0.5] }],
    keys: [
      [0, { z: -0.72, air: 0.45, ...both({}) }],
      [0.2, { z: -0.3, air: 0.35, L: { hip: 30, knee: 20 }, R: {} }],
      [0.35, { z: 0, spine: 22, ...both({ hip: 60, knee: 75, sh: -30 }) }],
      [0.5, { z: 0, air: 0.4, ...both({ ankle: -35, sh: 165 }) }],
      [0.65, { z: 0, spine: 10, ...both({ hip: 35, knee: 45 }) }],
    ],
  },
  'broad-jump': {
    dur: 3, anchor: 'none', cam: { yaw: 85 },
    keys: [
      [0, { z: -0.6, ...both({}) }],
      [0.2, { z: -0.6, spine: 30, ...both({ hip: 80, knee: 90, sh: -50 }) }],
      [0.36, { z: 0.1, air: 0.35, pitch: 15, ...both({ hip: 30, knee: 40, ankle: -25, sh: 150 }) }],
      [0.5, { z: 0.75, spine: 25, ...both({ hip: 80, knee: 95, sh: 70 }) }],
      [0.7, { z: 0.75, ...both({}) }],
    ],
  },
  'hurdle-hops': {
    dur: 1.6, anchor: 'none', run: 0, cam: { yaw: 80 }, props: [{ kind: 'hurdles', at: [0, 0, 0.45] }],
    keys: [
      [0, { z: 0, spine: 15, ...both({ hip: 35, knee: 45, sh: -20 }) }],
      [0.3, { z: 0.45, air: 0.35, ...both({ hip: 80, knee: 110, sh: 60 }) }],
      [0.55, { z: 0.9, spine: 15, ...both({ hip: 35, knee: 45, sh: -20 }) }],
      [0.8, { z: 0.45, air: 0.05, ...both({ hip: 10, knee: 15 }) }],
    ],
  },
  'split-jump': {
    dur: 1.8, anchor: 'none', cam: { yaw: 80 },
    keys: [
      [0, { L: { hip: 70, knee: 80, sh: -25 }, R: { hip: -20, knee: 90, sh: 25 } }],
      [0.25, { air: 0.3, L: { hip: 20, knee: 20, ankle: -25, sh: 10 }, R: { hip: 10, knee: 20, ankle: -25, sh: 10 } }],
      [0.5, { R: { hip: 70, knee: 80, sh: -25 }, L: { hip: -20, knee: 90, sh: 25 } }],
      [0.75, { air: 0.3, R: { hip: 20, knee: 20, ankle: -25 }, L: { hip: 10, knee: 20, ankle: -25 } }],
    ],
  },
  'lateral-bound': {
    dur: 1.8, anchor: 'none', cam: { yaw: 10 },
    keys: [
      [0, { x: -0.55, spine: 25, roll: -8, R: { hip: 60, knee: 70, sh: 30, elb: 40 }, L: { hip: -10, knee: 70, abd: -10, sh: -20 } }],
      [0.25, { x: 0, air: 0.3, ...both({ hip: 20, knee: 30, ankle: -20 }) }],
      [0.5, { x: 0.55, spine: 25, roll: 8, L: { hip: 60, knee: 70, sh: 30, elb: 40 }, R: { hip: -10, knee: 70, abd: -10, sh: -20 } }],
      [0.75, { x: 0, air: 0.3, ...both({ hip: 20, knee: 30, ankle: -20 }) }],
    ],
  },
  'bounds': { dur: 1.1, anchor: 'none', run: 4, contacts: 'feet', cam: { yaw: 80 }, keys: [[0, { ...SPRINT_A, L: { hip: 85, knee: 80, sh: -50, elb: 80 } }], [0.25, { ...SPRINT_A, air: 0.2 }], [0.5, { ...SPRINT_B, R: { hip: 85, knee: 80, sh: -50, elb: 80 } }], [0.75, { ...SPRINT_B, air: 0.2 }]] },

  // hinges
  'deadlift': deadlift(hinge(45, 105, 80)),
  'rdl': deadlift(hinge(70, 80, 18), 'barbell', 3.4),
  'trap-bar-deadlift': deadlift({ pitch: 35, spine: 5, ...both({ hip: 100, knee: 85, sh: 40, shAbd: 15 }) }, 'trapbar'),
  'kb-swing': {
    dur: 1.6, cam: { yaw: 75 }, props: [{ kind: 'kettlebell' }],
    keys: [[0, hinge(65, 85, 30, { sh: 35 })], [0.45, both({ sh: 95, shAbd: 5 })], [0.6, both({ sh: 90, shAbd: 5 })]],
  },
  'iso-pull': { dur: 4, hold: true, cam: { yaw: 70 }, props: [{ kind: 'barbell' }], keys: [[0, hinge(22, 40, 30)]] },
  'sl-rdl': {
    dur: 3.4, anchor: 'L', cam: { yaw: 80 },
    keys: [[0, { L: { knee: 5 }, R: { hip: 10, knee: 20, sh: 10 } }], [0.45, { pitch: 75, spine: 5, L: { hip: 80, knee: 15, sh: 80 }, R: { hip: -5, knee: 0, sh: 80 } }], [0.6, { pitch: 75, spine: 5, L: { hip: 80, knee: 15, sh: 80 }, R: { hip: -5, knee: 0, sh: 80 } }]],
  },
  'clean': {
    dur: 3, cam: { yaw: 70 }, props: [{ kind: 'barbell' }],
    keys: [[0, HANG], [0.22, EXTEND], [0.4, RACK_DIP], [0.62, RACK], [0.85, RACK]],
  },
  'power-clean': {
    dur: 3.4, cam: { yaw: 70 }, props: [{ kind: 'barbell' }],
    keys: [[0, hinge(40, 100, 80)], [0.2, HANG], [0.32, EXTEND], [0.48, RACK_DIP], [0.66, RACK], [0.85, RACK]],
  },
  'snatch': {
    dur: 3.2, cam: { yaw: 60 }, props: [{ kind: 'barbell' }],
    keys: [[0, hinge(30, 45, 25, { shAbd: 25 })], [0.22, { ...EXTEND, ...both({ ankle: -30, sh: 40, shAbd: 35, elb: 60 }) }], [0.42, { spine: 10, ...both({ hip: 90, knee: 105, sh: 175, shAbd: 35 }) }], [0.66, both({ sh: 175, shAbd: 35 })], [0.85, both({ sh: 175, shAbd: 35 })]],
  },
  'push-jerk': {
    dur: 2.6, cam: { yaw: 55 }, props: [{ kind: 'barbell' }],
    keys: [[0, RACK], [0.2, { ...both({ hip: 25, knee: 30, ...FRONT_RACK }) }], [0.35, { air: 0.05, ...both({ ankle: -25, sh: 150, shAbd: 15, elb: 40 }) }], [0.5, { ...both({ hip: 25, knee: 30, ...OVERHEAD }) }], [0.7, both({ ...OVERHEAD })]],
  },
  'split-jerk': {
    dur: 2.8, anchor: 'none', cam: { yaw: 80 }, props: [{ kind: 'barbell' }],
    keys: [[0, RACK], [0.2, both({ hip: 25, knee: 30, ...FRONT_RACK })], [0.36, { air: 0.05, ...both({ ankle: -25, sh: 150, shAbd: 15, elb: 40 }) }], [0.5, { L: { hip: 45, knee: 50, ...OVERHEAD }, R: { hip: -30, knee: 35, ...OVERHEAD } }], [0.72, both({ ...OVERHEAD })]],
  },
  'db-snatch': {
    dur: 2.6, cam: { yaw: 50 }, props: [{ kind: 'dumbbell-R' }],
    keys: [[0, { pitch: 40, spine: 5, L: { hip: 95, knee: 75, sh: 30, elb: 30 }, R: { hip: 95, knee: 75, sh: 45 } }], [0.3, { air: 0.05, L: { ankle: -25 }, R: { ankle: -25, sh: 90, elb: 70 } }], [0.5, { L: { hip: 25, knee: 30 }, R: { hip: 25, knee: 30, sh: 175, shAbd: 10 } }], [0.7, { R: { sh: 175, shAbd: 10 } }]],
  },

  // lunges, split squats, single leg
  'reverse-lunge': { dur: 4.4, anchor: 'L', cam: { yaw: 80 }, keys: alternate([[0, {}], [0.45, { spine: 5, L: { hip: 85, knee: 90 }, R: { hip: -20, knee: 95 } }], [0.6, { spine: 5, L: { hip: 85, knee: 90 }, R: { hip: -20, knee: 95 } }]]) },
  'split-squat-rear': {
    dur: 3, anchor: 'L', contacts: 'feet', cam: { yaw: 80 }, props: [{ kind: 'bench', at: [-0.1, 0, -0.78], size: [0.4, 0.42, 0.4] }],
    keys: [[0, { spine: 5, L: { hip: 25, knee: 15 }, R: { hip: -25, knee: 95, ankle: -40 } }], [0.45, { spine: 15, L: { hip: 85, knee: 95 }, R: { hip: -5, knee: 120, ankle: -40 } }], [0.6, { spine: 15, L: { hip: 85, knee: 95 }, R: { hip: -5, knee: 120, ankle: -40 } }]],
  },
  'split-squat-sofa': {
    dur: 3, anchor: 'L', contacts: 'feet', cam: { yaw: 80 }, props: [{ kind: 'sofa', at: [-0.1, 0, -1.08], size: [1.2, 0.42, 0.6] }],
    keys: [[0, { spine: 5, L: { hip: 25, knee: 15 }, R: { hip: -25, knee: 95, ankle: -40 } }], [0.45, { spine: 15, L: { hip: 85, knee: 95 }, R: { hip: -5, knee: 120, ankle: -40 } }], [0.6, { spine: 15, L: { hip: 85, knee: 95 }, R: { hip: -5, knee: 120, ankle: -40 } }]],
  },
  'iso-split': { dur: 4, hold: true, anchor: 'L', cam: { yaw: 80 }, keys: [[0, { spine: 5, L: { hip: 85, knee: 90 }, R: { hip: -15, knee: 95 } }]] },
  'step-up': {
    dur: 3.2, anchor: 'L', cam: { yaw: 75 }, props: [{ kind: 'box', at: [0.1, 0, 0.2], size: [0.6, 0.45, 0.5] }],
    keys: [[0, { spine: 10, L: { hip: 80, knee: 95 }, R: { hip: -5, knee: 5 } }], [0.45, { air: 0.45, L: {}, R: { hip: 95, knee: 95, sh: -30, elb: 80 } }], [0.6, { air: 0.45, L: {}, R: { hip: 95, knee: 95, sh: -30, elb: 80 } }], [0.85, { spine: 10, L: { hip: 80, knee: 95 }, R: { hip: -5, knee: 5 } }]],
  },
  'calf-raise': {
    dur: 2, anchor: 'L', cam: { yaw: 70 }, props: [{ kind: 'step', at: [0.1, 0, 0.05], size: [0.5, 0.18, 0.4] }],
    keys: [[0, { air: 0.18, L: { ankle: 0 }, R: { hip: 10, knee: 80 } }], [0.45, { air: 0.18, L: { ankle: -40 }, R: { hip: 10, knee: 80 } }], [0.6, { air: 0.18, L: { ankle: -40 }, R: { hip: 10, knee: 80 } }]],
  },
  'skater-squat': {
    dur: 3.4, anchor: 'L', cam: { yaw: 80 },
    keys: [[0, { L: {}, R: { hip: 5, knee: 70, ...ARMS_FWD } }], [0.45, { spine: 35, L: { hip: 95, knee: 105, ...ARMS_FWD }, R: { hip: -15, knee: 95, ...ARMS_FWD } }], [0.6, { spine: 35, L: { hip: 95, knee: 105, ...ARMS_FWD }, R: { hip: -15, knee: 95, ...ARMS_FWD } }]],
  },
  'cossack': { dur: 4.4, anchor: 'feet', cam: { yaw: 10 }, keys: alternate([[0, both({ abd: 30, ...ARMS_FWD })], [0.45, { spine: 20, L: { hip: 100, knee: 115, abd: 30, ...ARMS_FWD }, R: { hip: 10, abd: 45, ankle: 25, ...ARMS_FWD } }], [0.6, { spine: 20, L: { hip: 100, knee: 115, abd: 30, ...ARMS_FWD }, R: { hip: 10, abd: 45, ankle: 25, ...ARMS_FWD } }]]) },

  // pushes
  'push-up': {
    dur: 2.6, flatFeet: false, contacts: 'hands-feet', cam: { yaw: 80, tilt: 15 },
    keys: [[0, { pitch: 73, ...both({ sh: 73, shAbd: 12 }) }], [0.45, { pitch: 84, ...both({ sh: 30, shAbd: 35, elb: 85 }) }], [0.6, { pitch: 84, ...both({ sh: 30, shAbd: 35, elb: 85 }) }]],
  },
  'plyo-push-up': {
    dur: 1.8, flatFeet: false, contacts: 'hands-feet', cam: { yaw: 80, tilt: 15 },
    keys: [[0, { pitch: 82, ...both({ sh: 30, shAbd: 35, elb: 85 }) }], [0.3, { pitch: 60, air: 0.12, ...both({ sh: 75, shAbd: 15 }) }], [0.55, { pitch: 82, ...both({ sh: 30, shAbd: 35, elb: 85 }) }]],
  },
  'pike-push-up': {
    dur: 2.6, flatFeet: false, contacts: 'hands-feet', cam: { yaw: 85 },
    keys: [[0, { pitch: 125, ...both({ hip: 100, sh: 175 }) }], [0.45, { pitch: 140, ...both({ hip: 105, sh: 145, elb: 70 }) }], [0.6, { pitch: 140, ...both({ hip: 105, sh: 145, elb: 70 }) }]],
  },
  'dip': {
    dur: 2.6, anchor: 'none', contacts: 'feet', cam: { yaw: 80 }, props: [{ kind: 'sofa', at: [0, 0, -0.45], size: [0.8, 0.45, 0.45] }],
    keys: [[0, { air: 0.1, z: 0.15, ...both({ hip: 80, knee: 30, sh: -35, shAbd: 10 }) }], [0.45, { z: 0.15, ...both({ hip: 80, knee: 50, sh: -70, shAbd: 10, elb: 85 }) }], [0.6, { z: 0.15, ...both({ hip: 80, knee: 50, sh: -70, shAbd: 10, elb: 85 }) }]],
  },
  'bench-press': {
    dur: 2.6, anchor: 'none', contacts: 'back', base: 0.45, cam: { yaw: 75, tilt: 20 }, props: [{ kind: 'barbell' }, { kind: 'bench', at: [0, 0, -0.35], size: [0.32, 0.45, 1.25] }],
    keys: [[0, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 90, shAbd: 12 }) }], [0.45, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 25, shAbd: 55, elb: 95 }) }], [0.6, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 25, shAbd: 55, elb: 95 }) }]],
  },
  'db-bench-1': {
    dur: 2.8, anchor: 'none', contacts: 'back', base: 0.45, cam: { yaw: 75, tilt: 20 }, props: [{ kind: 'dumbbell-R' }, { kind: 'bench', at: [0, 0, -0.35], size: [0.32, 0.45, 1.25] }],
    keys: [[0, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 0, shAbd: 25 }), R: { hip: 12, knee: 95, abd: 15, sh: 90, shAbd: 12 } }], [0.45, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 0, shAbd: 25 }), R: { hip: 12, knee: 95, abd: 15, sh: 25, shAbd: 55, elb: 95 } }], [0.6, { pitch: SUPINE, ...both({ hip: 12, knee: 95, abd: 15, sh: 0, shAbd: 25 }), R: { hip: 12, knee: 95, abd: 15, sh: 25, shAbd: 55, elb: 95 } }]],
  },
  'ohp': {
    dur: 2.8, cam: { yaw: 35 }, props: [{ kind: 'dumbbells' }],
    keys: [[0, both({ shAbd: 85, shRot: 90, elb: 95 })], [0.45, both({ shAbd: 170, shRot: 90, elb: 5 })], [0.6, both({ shAbd: 170, shRot: 90, elb: 5 })]],
  },
  'landmine-press': {
    dur: 2.4, anchor: 'L', cam: { yaw: 70 }, props: [{ kind: 'landmine' }],
    keys: [[0, { spine: 10, L: { hip: 30, knee: 35, sh: -20 }, R: { hip: -15, knee: 20, sh: 40, elb: 120, shAbd: 10 } }], [0.2, { spine: 10, L: { hip: 45, knee: 50, sh: -20 }, R: { hip: -10, knee: 35, sh: 40, elb: 120, shAbd: 10 } }], [0.45, { spine: 15, L: { hip: 20, knee: 10, ankle: -10 }, R: { hip: -15, knee: 10, sh: 130, elb: 5 } }], [0.65, { spine: 15, L: { hip: 20, knee: 10 }, R: { hip: -15, knee: 10, sh: 130, elb: 5 } }]],
  },

  // pulls
  'pull-up': {
    dur: 2.8, hang: 2.35, cam: { yaw: 30, tilt: 5 }, props: [{ kind: 'bar', at: [0, 2.35, 0] }],
    keys: [[0, both({ sh: 175, shAbd: 15, knee: 40, hip: 10 })], [0.45, { spine: -8, ...both({ sh: 75, shAbd: 20, elb: 140, knee: 40, hip: 10 }) }], [0.6, { spine: -8, ...both({ sh: 75, shAbd: 20, elb: 140, knee: 40, hip: 10 }) }]],
  },
  'row': { dur: 2.4, cam: { yaw: 75 }, props: [{ kind: 'barbell' }], keys: [[0, hinge(50, 70, 25)], [0.45, { ...hinge(50, 70, 25), ...both({ hip: 70, knee: 25, sh: 0, elb: 95, shAbd: 20 }) }], [0.6, { ...hinge(50, 70, 25), ...both({ hip: 70, knee: 25, sh: 0, elb: 95, shAbd: 20 }) }]] },
  'row-1': {
    dur: 2.6, anchor: 'R', cam: { yaw: 60 }, props: [{ kind: 'dumbbell-R' }, { kind: 'bench', at: [0.4, 0, 0.25], size: [0.4, 0.45, 1.1], yaw: 0 }],
    keys: [[0, { pitch: 75, L: { hip: 90, knee: 90, sh: 75 }, R: { hip: 80, knee: 15, sh: 75 } }], [0.45, { pitch: 75, L: { hip: 90, knee: 90, sh: 75 }, R: { hip: 80, knee: 15, sh: 20, elb: 95 } }], [0.6, { pitch: 75, L: { hip: 90, knee: 90, sh: 75 }, R: { hip: 80, knee: 15, sh: 20, elb: 95 } }]],
  },
  'table-row': {
    dur: 2.6, anchor: 'none', hang: 0.8, flatFeet: false, cam: { yaw: 80 }, props: [{ kind: 'table', at: [0, 0, -0.1], size: [1.1, 0.78, 0.6] }],
    keys: [[0, { pitch: -80, ...both({ sh: 100, shAbd: 12, ankle: 20 }) }], [0.45, { pitch: -66, ...both({ sh: 45, elb: 110, shAbd: 30, ankle: 20 }) }], [0.6, { pitch: -66, ...both({ sh: 45, elb: 110, shAbd: 30, ankle: 20 }) }]],
  },

  // core
  'plank': { dur: 4, hold: true, flatFeet: false, cam: { yaw: 80, tilt: 15 }, keys: [[0, { pitch: 82, ...both({ sh: 82, elb: 90, shAbd: 8 }) }]] },
  'plank-taps': {
    dur: 2.2, flatFeet: false, contacts: 'hands-feet', cam: { yaw: 70, tilt: 15 },
    keys: alternate([[0, { pitch: 73, ...both({ sh: 73, shAbd: 12, abd: 12 }) }], [0.45, { pitch: 73, roll: 8, L: { sh: 73, shAbd: 12, abd: 12 }, R: { sh: 140, shAbd: -10, elb: 140, abd: 12 } }], [0.6, { pitch: 73, roll: 8, L: { sh: 73, shAbd: 12, abd: 12 }, R: { sh: 140, shAbd: -10, elb: 140, abd: 12 } }]]),
  },
  'side-plank': { dur: 4, hold: true, flatFeet: false, cam: { yaw: 0, tilt: 10 }, keys: [[0, { roll: 72, R: { shAbd: 72, elb: 90, sh: 90 }, L: { shAbd: 0 } }]] },
  'copenhagen': { dur: 4, hold: true, flatFeet: false, contacts: 'all', cam: { yaw: 0, tilt: 10 }, props: [{ kind: 'bench', at: [0.95, 0, 0], size: [0.45, 0.36, 0.6] }], keys: [[0, { roll: 72, R: { shAbd: 72, sh: 90, elb: 90, abd: -12 }, L: { shAbd: 0 } }]] },
  'copenhagen-sofa': { dur: 4, hold: true, flatFeet: false, contacts: 'all', cam: { yaw: 0, tilt: 10 }, props: [{ kind: 'sofa', at: [1.05, 0, 0], size: [0.6, 0.36, 0.8] }], keys: [[0, { roll: 72, R: { shAbd: 72, sh: 90, elb: 90, abd: -12 }, L: { shAbd: 0 } }]] },
  'dead-bug': {
    dur: 3, anchor: 'none', flatFeet: false, cam: { yaw: 70, tilt: 25 },
    keys: alternate([[0, { pitch: SUPINE, ...both({ hip: 90, knee: 90, sh: 90 }) }], [0.45, { pitch: SUPINE, L: { hip: 20, knee: 5, sh: 90 }, R: { hip: 90, knee: 90, sh: 175 } }], [0.6, { pitch: SUPINE, L: { hip: 20, knee: 5, sh: 90 }, R: { hip: 90, knee: 90, sh: 175 } }]]),
  },
  'hollow': { dur: 4, hold: true, anchor: 'none', flatFeet: false, contacts: 'back', cam: { yaw: 75, tilt: 20 }, keys: [[0, { pitch: SUPINE + 12, spine: 18, ...both({ hip: 35, sh: 160, ankle: -20 }) }]] },
  'pallof': {
    dur: 2.6, cam: { yaw: 20 }, props: [{ kind: 'band', at: [0.9, 1.25, 0.35] }],
    keys: [[0, { ...ATHLETIC, ...both({ hip: 25, knee: 25, abd: 12, sh: 25, elb: 110, shAbd: 5 }) }], [0.45, { ...ATHLETIC, ...both({ hip: 25, knee: 25, abd: 12, sh: 88, elb: 0, shAbd: 5 }) }], [0.6, { ...ATHLETIC, ...both({ hip: 25, knee: 25, abd: 12, sh: 88, elb: 0, shAbd: 5 }) }]],
  },

  // bridges, hamstrings
  'bridge': { dur: 2.6, anchor: 'feet', flatFeet: true, cam: { yaw: 80, tilt: 15 }, keys: [[0, bridge(0)], [0.45, bridge(1)], [0.6, bridge(1)]] },
  'bridge-1': { dur: 2.6, anchor: 'L', cam: { yaw: 80, tilt: 15 }, keys: [[0, { ...bridge(0), R: { hip: 70, knee: 0, sh: -10 } }], [0.45, { ...bridge(1), R: { hip: 8, knee: 0, sh: -10 } }], [0.6, { ...bridge(1), R: { hip: 8, knee: 0, sh: -10 } }]] },
  'hip-thrust': {
    dur: 2.6, anchor: 'feet', contacts: 'feet', cam: { yaw: 80, tilt: 12 }, props: [{ kind: 'barbell-hips' }, { kind: 'bench', at: [0, 0, -0.75], size: [1.0, 0.42, 0.3] }],
    keys: [[0, { pitch: -35, ...both({ hip: 105, knee: 85, sh: 30, shAbd: 35 }) }], [0.45, { pitch: -85, ...both({ hip: 5, knee: 95, sh: 0, shAbd: 35 }) }], [0.6, { pitch: -85, ...both({ hip: 5, knee: 95, sh: 0, shAbd: 35 }) }]],
  },
  'iso-ham-bridge': { dur: 4, hold: true, anchor: 'none', flatFeet: false, cam: { yaw: 80, tilt: 12 }, props: [{ kind: 'bench', at: [0, 0, 1.0], size: [0.9, 0.42, 0.35] }], keys: [[0, { pitch: -75, ...both({ hip: 10, knee: 15, sh: -10 }) }]] },
  'ham-slide': { dur: 3, anchor: 'none', flatFeet: false, cam: { yaw: 80, tilt: 15 }, props: [{ kind: 'towel', at: [0, 0, 0.85] }], keys: [[0, { pitch: -70, ...both({ hip: 20, knee: 90, sh: -10 }) }], [0.45, { pitch: -78, ...both({ hip: 5, knee: 10, sh: -10 }) }], [0.6, { pitch: -78, ...both({ hip: 5, knee: 10, sh: -10 }) }]] },
  'nordic': {
    dur: 3.6, anchor: 'L', flatFeet: false, cam: { yaw: 85 },
    keys: [[0, { pitch: 0, ...both({ knee: 90, ankle: -60, sh: 10 }) }], [0.55, { pitch: 55, ...both({ knee: 35, ankle: -60, sh: 70, elb: 40 }) }], [0.75, { pitch: 75, ...both({ knee: 15, ankle: -60, sh: 100, elb: 60 }) }]],
  },

  // med ball
  'mb-chest-pass': {
    dur: 2.2, ball: false, cam: { yaw: 60 }, props: [{ kind: 'medball' }, { kind: 'wall', at: [0, 0, 2.6], size: [3, 2, 0.1] }],
    keys: [[0, { ...ATHLETIC, ...both({ hip: 25, knee: 30, abd: 10, sh: 40, elb: 120 }) }], [0.3, { spine: 5, ...both({ hip: 10, knee: 10, abd: 10, sh: 85, elb: 0 }) }], [0.55, { ...ATHLETIC, ...both({ hip: 25, knee: 30, abd: 10, sh: 70, elb: 30 }) }]],
  },
  'mb-slam': {
    dur: 1.8, cam: { yaw: 70 }, props: [{ kind: 'medball' }],
    keys: [[0, { ...both({ sh: 175, shAbd: 10, ankle: -15 }) }], [0.35, { pitch: 40, spine: 15, ...both({ hip: 80, knee: 70, sh: 60 }) }], [0.6, { pitch: 40, spine: 15, ...both({ hip: 80, knee: 70, sh: 60 }) }]],
  },
  'mb-rotate': {
    dur: 2, cam: { yaw: 0 }, props: [{ kind: 'medball' }, { kind: 'wall', at: [2, 0, 0], size: [0.1, 2, 3], yaw: 0 }],
    keys: [[0, { ...ATHLETIC, twist: -60, ...both({ hip: 30, knee: 35, abd: 10, sh: 60, elb: 30 }) }], [0.35, { spine: 10, twist: 50, ...both({ hip: 20, knee: 20, abd: 10, sh: 70, elb: 10 }) }], [0.6, { ...ATHLETIC, twist: 0, ...both({ hip: 30, knee: 35, abd: 10, sh: 60, elb: 30 }) }]],
  },

  // running
  'jog': { dur: 0.8, anchor: 'none', contacts: 'feet', run: 2.4, cam: { yaw: 75 }, keys: [[0, RUN_A], [0.25, { ...RUN_A, air: 0.05 }], [0.5, RUN_B], [0.75, { ...RUN_B, air: 0.05 }]] },
  'sprint': { dur: 0.6, anchor: 'none', contacts: 'feet', run: 7, cam: { yaw: 80 }, keys: [[0, SPRINT_A], [0.25, { ...SPRINT_A, air: 0.1 }], [0.5, SPRINT_B], [0.75, { ...SPRINT_B, air: 0.1 }]] },
  'accel': { dur: 0.75, anchor: 'none', contacts: 'feet', run: 4, cam: { yaw: 80 }, keys: [[0, { ...SPRINT_A, pitch: 35 }], [0.25, { ...SPRINT_A, pitch: 35, air: 0.06 }], [0.5, { ...SPRINT_B, pitch: 35 }], [0.75, { ...SPRINT_B, pitch: 35, air: 0.06 }]] },
  'sled': { dur: 1, anchor: 'none', contacts: 'feet', run: 2, cam: { yaw: 75 }, props: [{ kind: 'sled', at: [0, 0, 0.95] }], keys: [[0, { ...SPRINT_A, pitch: 45, L: { hip: 80, knee: 95 }, R: { hip: -15, knee: 30 }, ...{} }], [0.5, { ...SPRINT_B, pitch: 45, R: { hip: 80, knee: 95 }, L: { hip: -15, knee: 30 } }]].map(([t, p]) => [t, { ...(p as Pose), L: { ...(p as Pose).L, sh: 95, elb: 30 }, R: { ...(p as Pose).R, sh: 95, elb: 30 } }] as Key) },
  'high-knees': { dur: 0.6, anchor: 'none', contacts: 'feet', cam: { yaw: 60 }, keys: [[0, { L: { hip: 95, knee: 100, sh: -30, elb: 85 }, R: { sh: 40, elb: 85, ankle: -20 } }], [0.25, { air: 0.06, ...both({ elb: 85 }) }], [0.5, { R: { hip: 95, knee: 100, sh: -30, elb: 85 }, L: { sh: 40, elb: 85, ankle: -20 } }], [0.75, { air: 0.06, ...both({ elb: 85 }) }]] },
  'march': { dur: 1.6, anchor: 'none', contacts: 'feet', cam: { yaw: 55 }, keys: [[0, { L: { hip: 75, knee: 85, sh: -25, elb: 70 }, R: { sh: 30, elb: 70 } }], [0.5, { R: { hip: 75, knee: 85, sh: -25, elb: 70 }, L: { sh: 30, elb: 70 } }]] },
  'shuffle': {
    dur: 1.6, anchor: 'none', contacts: 'feet', cam: { yaw: 10 }, props: [{ kind: 'cones', at: [-1.1, 0, 0.6], yaw: 90 }],
    keys: [[0, { ...ATHLETIC, x: -0.45, L: { hip: 35, knee: 40, abd: 25, sh: 20, elb: 40 } }], [0.25, { ...ATHLETIC, x: -0.1, air: 0.05 }], [0.5, { ...ATHLETIC, x: 0.45, R: { hip: 35, knee: 40, abd: 25, sh: 20, elb: 40 } }], [0.75, { ...ATHLETIC, x: 0.1, air: 0.05 }]],
  },
  'cut': {
    dur: 2.4, anchor: 'none', contacts: 'feet', cam: { yaw: 40, zoom: 1.1 }, props: [{ kind: 'cones', at: [0, 0, -1.2] }],
    keys: [[0, { ...RUN_A, z: -0.8 }], [0.15, { ...RUN_B, z: -0.3 }], [0.3, { ...RUN_A, z: 0.2 }], [0.42, { z: 0.5, spine: 25, roll: -20, yaw: 20, L: { hip: 60, knee: 80, abd: 25 }, R: { hip: 20, knee: 30, abd: 10 } }], [0.55, { ...RUN_B, x: 0.4, z: 0.6, yaw: 90 }], [0.7, { ...RUN_A, x: 0.9, z: 0.6, yaw: 90 }], [0.85, { ...RUN_B, x: 0.3, z: -0.3, yaw: 200 }]],
  },
  'jockey': {
    dur: 1.6, anchor: 'none', contacts: 'feet', cam: { yaw: 30 },
    keys: [[0, { ...ATHLETIC, yaw: 35, z: 0.2, L: { hip: 45, knee: 50, abd: 10 }, R: { hip: 10, knee: 40 } }], [0.25, { ...ATHLETIC, yaw: 35, z: 0.05, air: 0.04 }], [0.5, { ...ATHLETIC, yaw: 35, z: -0.15, L: { hip: 45, knee: 50, abd: 10 }, R: { hip: 10, knee: 40 } }], [0.75, { ...ATHLETIC, yaw: 35, z: 0.05, air: 0.04 }]],
  },
  'burpee': {
    dur: 3, anchor: 'none', flatFeet: true, cam: { yaw: 80 },
    keys: [
      [0, both({})],
      [0.15, { spine: 40, ...both({ hip: 110, knee: 125, sh: 60 }) }],
      [0.32, { pitch: 66, z: -0.45, ...both({ sh: 66, ankle: -45 }) }],
      [0.5, { spine: 40, ...both({ hip: 110, knee: 125, sh: 60 }) }],
      [0.65, { air: 0.3, ...both({ sh: 170, ankle: -30 }) }],
      [0.8, both({})],
    ],
  },
  'bike': {
    dur: 1, seat: 0.98, flatFeet: false, cam: { yaw: 75 }, props: [{ kind: 'bike', at: [0, 0, 0] }],
    keys: [[0, { pitch: 25, L: { hip: 105, knee: 115, sh: 60, elb: 20 }, R: { hip: 55, knee: 35, sh: 60, elb: 20 } }], [0.5, { pitch: 25, R: { hip: 105, knee: 115, sh: 60, elb: 20 }, L: { hip: 55, knee: 35, sh: 60, elb: 20 } }]],
  },

  // ball at the feet
  'toe-taps': { dur: 0.9, anchor: 'none', ball: true, contacts: 'feet', cam: { yaw: 40, zoom: 0.85 }, keys: [[0, { L: { hip: 40, knee: 45, ankle: 15, sh: -20, elb: 40 }, R: { sh: 20, elb: 40 }, ball: [0, 0.11, 0.42] }], [0.25, { air: 0.05, ...both({ elb: 40 }), ball: [0, 0.11, 0.42] }], [0.5, { R: { hip: 40, knee: 45, ankle: 15, sh: -20, elb: 40 }, L: { sh: 20, elb: 40 }, ball: [0, 0.11, 0.42] }], [0.75, { air: 0.05, ...both({ elb: 40 }), ball: [0, 0.11, 0.42] }]] },
  'sole-roll': { dur: 1.6, anchor: 'none', ball: true, cam: { yaw: 20, zoom: 0.85 }, keys: [[0, { L: { sh: 10, shAbd: 25 }, R: { hip: 30, knee: 35, abd: 5, ankle: 10, shAbd: 25 }, ball: [-0.12, 0.11, 0.32] }], [0.5, { L: { shAbd: 25 }, R: { hip: 30, knee: 35, abd: 25, ankle: 10, shAbd: 25 }, ball: [-0.3, 0.11, 0.32] }]] },
  'inside-outside': { dur: 1, anchor: 'none', ball: true, cam: { yaw: 25, zoom: 0.85 }, keys: [[0, { ...touch('R', -0.05), R: { hip: 25, knee: 30, abd: 15, rot: -30 } }], [0.5, { ...touch('R', -0.3), R: { hip: 25, knee: 30, abd: -5, rot: 25 } }]] },
  'v-pull': { dur: 1.8, anchor: 'none', ball: true, cam: { yaw: 35, zoom: 0.85 }, keys: alternate([[0, { ...ATHLETIC, R: { hip: 40, knee: 40, ankle: 10 }, ball: [-0.1, 0.11, 0.45] }], [0.35, { ...ATHLETIC, R: { hip: 10, knee: 40 }, ball: [-0.1, 0.11, 0.15] }], [0.7, { ...ATHLETIC, R: { hip: 25, knee: 25, rot: -45 }, ball: [0.15, 0.11, 0.35] }]]) },
  'juggle': { dur: 1.3, anchor: 'none', ball: true, cam: { yaw: 50, zoom: 0.95 }, keys: alternate([[0, { spine: 10, L: { hip: 10, knee: 15, shAbd: 25 }, R: { hip: 45, knee: 35, ankle: 10, shAbd: 25 }, ball: [-0.1, 0.35, 0.42] }], [0.5, { spine: 10, ...both({ hip: 10, knee: 15, shAbd: 25 }), ball: [0, 1.05, 0.38] }]]) },
  'dribble': {
    dur: 0.9, anchor: 'none', ball: true, run: 2.2, contacts: 'feet', cam: { yaw: 65, zoom: 0.95 },
    keys: [[0, { ...RUN_A, spine: 15, ball: [0, 0.11, 0.45] }], [0.25, { ...RUN_A, spine: 15, air: 0.04, ball: [0.05, 0.11, 0.7] }], [0.5, { ...RUN_B, spine: 15, R: { hip: 40, knee: 50, sh: -30, elb: 80 }, ball: [0.05, 0.11, 0.45] }], [0.75, { ...RUN_B, spine: 15, air: 0.04, ball: [-0.05, 0.11, 0.7] }]],
  },
  'cone-weave': {
    dur: 3.2, anchor: 'none', ball: true, contacts: 'feet', cam: { yaw: 25, zoom: 0.8 }, props: [{ kind: 'cones', at: [0, 0, -1.8] }],
    keys: [0, 1, 2, 3, 4, 5, 6, 7].map((i): Key => {
      const z = -1.8 + i * 0.6, x = i % 2 ? 0.35 : -0.35
      return [i / 8, { ...(i % 2 ? RUN_A : RUN_B), spine: 15, x, z, yaw: i % 2 ? -25 : 25, ball: [x * 0.8, 0.11, z + 0.45] }]
    }),
  },
  'step-over': {
    dur: 2.4, anchor: 'none', ball: true, cam: { yaw: 20, zoom: 0.95 },
    keys: [[0, { ...ATHLETIC, ball: [0, 0.11, 0.42] }], [0.2, { ...ATHLETIC, side: 10, R: { hip: 50, knee: 70, abd: 30, rot: 20 }, ball: [0, 0.11, 0.45] }], [0.35, { ...ATHLETIC, side: 15, roll: -8, R: { hip: 25, knee: 30, abd: -5 }, ball: [0, 0.11, 0.45] }], [0.55, { ...ATHLETIC, x: 0.4, yaw: -25, L: { hip: 40, knee: 35, rot: 30 }, ball: [0.5, 0.11, 0.75] }], [0.8, { ...RUN_A, x: 0.6, z: 0.3, ball: [0.65, 0.11, 1.1] }]],
  },
  'pass-wall': passWall(),
  'pass-wall-quick': passWall(1.6),
  'pass-long': strike([0, 1.2, 6], undefined, 2.6),
  'shoot': strike([0.7, 0.7, 5.2], [{ kind: 'goal', at: [0, 0, 5.4] }]),
  'volley': strike([0.6, 0.9, 5.2], [{ kind: 'goal', at: [0, 0, 5.4] }], 2.6, true),
  'control-air': {
    dur: 2.6, anchor: 'none', ball: true, cam: { yaw: 55 },
    keys: [[0, { ...both({ shAbd: 25 }), ball: [0, 1.9, 0.4] }], [0.3, { L: { shAbd: 25 }, R: { hip: 75, knee: 85, shAbd: 25 }, ball: { at: 'R', off: [0, 0.32, -0.3] } }], [0.5, { L: { shAbd: 25 }, R: { hip: 30, knee: 30, ankle: 10, shAbd: 25 }, ball: [0, 0.3, 0.45] }], [0.75, { ...both({ shAbd: 25 }), ball: [0, 0.11, 0.45] }]],
  },
  'receive-turn': {
    dur: 2.8, anchor: 'none', ball: true, cam: { yaw: 40 }, props: [{ kind: 'wall', at: [0, 0, 3], size: [3, 1.6, 0.1] }],
    keys: [[0, { ...ATHLETIC, twist: -35, ball: [0.1, 0.11, 2.7] }], [0.3, { ...ATHLETIC, twist: 10, yaw: -20, R: { hip: 20, knee: 20, rot: -40 }, ball: [0.1, 0.11, 0.4] }], [0.5, { ...ATHLETIC, yaw: -90, ball: [0.5, 0.11, 0.1] }], [0.7, { ...RUN_A, yaw: -120, x: 0.4, ball: [0.9, 0.11, -0.2] }], [0.9, { ...ATHLETIC, yaw: 0, ball: [0.1, 0.11, 1.5] }]],
  },
  '1v1': {
    dur: 2.4, anchor: 'none', ball: true, contacts: 'feet', cam: { yaw: 25 }, props: [{ kind: 'cones', at: [-1.2, 0, 1.4], yaw: 90 }],
    keys: [[0, { ...ATHLETIC, ball: [0, 0.11, 0.42] }], [0.2, { ...ATHLETIC, roll: -12, x: -0.15, L: { hip: 45, knee: 55, abd: 25 }, ball: [0, 0.11, 0.42] }], [0.4, { ...ATHLETIC, roll: 12, x: 0.35, R: { hip: 45, knee: 50, rot: -30 }, ball: [0.55, 0.11, 0.6] }], [0.6, { ...RUN_A, x: 0.7, z: 0.5, yaw: 15, ball: [0.85, 0.11, 1.1] }], [0.85, { ...ATHLETIC, ball: [0, 0.11, 0.42] }]],
  },
  'ball-run': { dur: 0.7, anchor: 'none', ball: true, run: 5, contacts: 'feet', cam: { yaw: 70 }, keys: [[0, { ...SPRINT_A, ball: [0, 0.11, 0.7] }], [0.25, { ...SPRINT_A, air: 0.08, ball: [0, 0.11, 0.9] }], [0.5, { ...SPRINT_B, ball: [0, 0.11, 0.7] }], [0.75, { ...SPRINT_B, air: 0.08, ball: [0, 0.11, 0.9] }]] },

  // mobility and stretching
  'ham-stretch': { dur: 4, hold: true, cam: { yaw: 80 }, keys: [[0, { pitch: 70, spine: 10, ...both({ hip: 80, sh: 80 }) }]] },
  'couch': {
    dur: 4, hold: true, anchor: 'L', flatFeet: false, contacts: 'all', cam: { yaw: 80 }, props: [{ kind: 'sofa', at: [0, 0, -0.72], size: [1, 0.45, 0.5] }],
    keys: [[0, { L: { hip: 80, knee: 85, ankle: 0 }, R: { hip: -15, knee: 160, ankle: -60, sh: 10 } }]],
  },
  'figure-4': { dur: 4, hold: true, anchor: 'none', flatFeet: false, cam: { yaw: 60, tilt: 25 }, keys: [[0, { pitch: SUPINE, L: { hip: 100, knee: 90, sh: 90, elb: 60 }, R: { hip: 85, abd: 55, rot: 50, knee: 100, sh: 90, elb: 60 } }]] },
  '90-90': {
    dur: 4, anchor: 'none', flatFeet: false, cam: { yaw: 25, tilt: 20 },
    keys: [[0, { L: { hip: 90, rot: 60, knee: 90, abd: 40, sh: 30 }, R: { hip: 60, rot: -60, knee: 90, abd: 50, sh: 30 } }], [0.45, { R: { hip: 90, rot: 60, knee: 90, abd: 40, sh: 30 }, L: { hip: 60, rot: -60, knee: 90, abd: 50, sh: 30 } }], [0.5, { R: { hip: 90, rot: 60, knee: 90, abd: 40, sh: 30 }, L: { hip: 60, rot: -60, knee: 90, abd: 50, sh: 30 } }], [0.95, { L: { hip: 90, rot: 60, knee: 90, abd: 40, sh: 30 }, R: { hip: 60, rot: -60, knee: 90, abd: 50, sh: 30 } }]],
  },
  'foam-roll': {
    dur: 3, anchor: 'none', flatFeet: false, cam: { yaw: 80, tilt: 15 }, props: [{ kind: 'roller', at: [0, 0, 0.25] }],
    keys: [[0, { pitch: 82, z: 0, ...both({ sh: 82, elb: 90 }) }], [0.5, { pitch: 82, z: -0.35, ...both({ sh: 82, elb: 90 }) }]],
  },
  'clamshell': { dur: 2.4, anchor: 'none', flatFeet: false, cam: { yaw: 5, tilt: 25 }, keys: [[0, { roll: 90, L: { hip: 45, knee: 90, shAbd: 160, elb: 120 }, R: { hip: 45, knee: 90, shAbd: 20 } }], [0.45, { roll: 90, L: { hip: 45, knee: 90, shAbd: 160, elb: 120 }, R: { hip: 45, knee: 90, abd: 45, rot: 20, shAbd: 20 } }], [0.6, { roll: 90, L: { hip: 45, knee: 90, shAbd: 160, elb: 120 }, R: { hip: 45, knee: 90, abd: 45, rot: 20, shAbd: 20 } }]] },
  'leg-swing': { dur: 1.6, anchor: 'L', cam: { yaw: 80 }, keys: [[0, { L: { shAbd: 60 }, R: { hip: -35, knee: 10, shAbd: 60 } }], [0.5, { spine: -5, L: { shAbd: 60 }, R: { hip: 75, knee: 5, shAbd: 60 } }]] },
  'hip-opener': { dur: 2.4, anchor: 'L', cam: { yaw: 30 }, keys: alternate([[0, { L: { shAbd: 40 }, R: { hip: 90, knee: 95, shAbd: 40 } }], [0.5, { L: { shAbd: 40 }, R: { hip: 70, abd: 55, rot: 30, knee: 95, shAbd: 40 } }], [0.8, { L: { shAbd: 40 }, R: { hip: 10, abd: 20, knee: 60, shAbd: 40 } }]]) },
  'quad-stretch': { dur: 4, hold: true, anchor: 'L', cam: { yaw: 75 }, keys: [[0, { L: { shAbd: 50 }, R: { hip: -5, knee: 150, ankle: -40, sh: -40, elb: 40 } }]] },
  'cooldown': {
    dur: 12, anchor: 'L', cam: { yaw: 75 },
    keys: [[0, { L: { shAbd: 50 }, R: { hip: -5, knee: 150, ankle: -40, sh: -40, elb: 40 } }], [0.3, { L: { shAbd: 50 }, R: { hip: -5, knee: 150, ankle: -40, sh: -40, elb: 40 } }], [0.45, { pitch: 70, spine: 10, ...both({ hip: 80, sh: 80 }) }], [0.8, { pitch: 70, spine: 10, ...both({ hip: 80, sh: 80 }) }]],
  },
  'warmup': {
    dur: 6, anchor: 'none', contacts: 'feet', cam: { yaw: 60 },
    keys: [[0, RUN_A], [0.1, RUN_B], [0.2, RUN_A], [0.3, RUN_B], [0.45, { L: { shAbd: 60 }, R: { hip: -35, knee: 10, shAbd: 60 } }], [0.6, { L: { shAbd: 60 }, R: { hip: 75, knee: 5, shAbd: 60 } }], [0.75, { L: { shAbd: 60 }, R: { hip: -35, knee: 10, shAbd: 60 } }], [0.9, { L: { shAbd: 60 }, R: { hip: 75, knee: 5, shAbd: 60 } }]],
  },
}
