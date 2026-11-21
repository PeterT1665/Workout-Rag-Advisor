// Which 3D move shows each exercise (library.json id → moves.ts). Every exercise has one (catalog.test.ts).
import data from '../../data/library.json'

type Demo = { move: string; speed?: number }
const M = (move: string, speed?: number): Demo => ({ move, speed })

export const DEMOS: Record<string, Demo> = {
  // warm-ups and cool-downs
  'jog-dynamic-leg-swings': M('warmup'), 'gym-warmup': M('bike', 0.7), 'home-warmup': M('march'), 'hip-openers-glute-activation': M('hip-opener'),
  'cool-down': M('cooldown'), 'cool-down-hips-quads-hamstrings-it-band': M('cooldown'), 'gym-cooldown': M('couch'), 'home-cooldown': M('figure-4'),
  // ball mastery
  'cone-weave': M('cone-weave'), 'dribble-change-direction': M('1v1'), 'freestyle-dribbling': M('dribble'), 'home-keepy-uppy': M('juggle'),
  'home-toe-taps': M('toe-taps'), 'home-v-pulls': M('v-pull'), 'inside-outside-touches': M('inside-outside'), 'juggling': M('juggle'),
  'skill-moves': M('step-over'), 'sole-rolls': M('sole-roll'), 'tight-space-close-control': M('v-pull', 1.2), 'toe-taps': M('toe-taps'),
  // 1v1
  '1v1-reaction-voice-cues': M('1v1'), '1v1-then-finish': M('shoot'), '1v1-with-partner': M('1v1'), 'jockey-footwork': M('jockey'),
  // conditioning
  'ball-intervals': M('ball-run'), 'box-to-box': M('sprint', 0.8), 'home-burpees': M('burpee'), 'home-high-knees': M('high-knees'), 'repeated-sprints': M('sprint'),
  'bike-intervals': M('bike', 1.4),
  // core
  'copenhagen-plank': M('copenhagen'), 'dead-bug': M('dead-bug'), 'home-copenhagen-sofa': M('copenhagen-sofa'), 'home-hollow-hold': M('hollow'),
  'home-plank-taps': M('plank-taps'), 'home-side-plank': M('side-plank'), 'pallof-press': M('pallof'), 'plank': M('plank'),
  // explosive
  'jump-squat': M('jump-squat'), 'kettlebell-swing': M('kb-swing'), 'landmine-push-press': M('landmine-press'), 'med-ball-chest-pass': M('mb-chest-pass'),
  'med-ball-slam': M('mb-slam'), 'med-ball-throw': M('mb-rotate'), 'speed-bench': M('bench-press', 1.6), 'trap-bar-jump': M('trap-bar-jump'),
  // finishing
  'one-touch-finish': M('shoot'), 'shooting-on-the-move': M('shoot'), 'shooting-stationary': M('shoot'), 'volleys-self-feed': M('volley'), 'weak-foot-finishing': M('shoot'),
  // first touch
  'air-control-one-touch-back': M('control-air'), 'half-turn-receive': M('receive-turn'), 'home-wall-touch': M('pass-wall-quick'), 'scan-and-turn': M('receive-turn'),
  'self-toss-control': M('control-air'), 'wall-passes-in-the-air': M('control-air'), 'weak-foot-left-receive-and-pass': M('pass-wall'),
  // isometrics
  'home-wall-sit': M('wall-sit'), 'iso-hamstring-bridge': M('iso-ham-bridge'), 'iso-mid-thigh-pull': M('iso-pull'), 'iso-split-squat': M('iso-split'), 'wall-sit': M('wall-sit'),
  // lower strength
  'back-squat': M('squat-back'), 'deadlift': M('deadlift'), 'front-squat': M('squat-front'), 'half-squat': M('squat-half'), 'hip-thrust': M('hip-thrust'),
  'home-hamstring-slide': M('ham-slide'), 'home-tempo-squat': M('squat-bw', 0.7), 'nordic-curl': M('nordic'), 'quarter-squat': M('squat-quarter'),
  'romanian-deadlift': M('rdl'), 'trap-bar-deadlift': M('trap-bar-deadlift'),
  // mobility
  '90-90-hip-switches': M('90-90'), 'couch-stretch-hip-flexor-quad': M('couch'), 'figure-4-glute-stretch': M('figure-4'), 'foam-roll-quads-and-glutes': M('foam-roll'),
  'glute-bridge': M('bridge'), 'hamstring-stretch': M('ham-stretch'), 'side-lying-leg-raise-clamshell': M('clamshell'),
  // olympic
  'db-snatch': M('db-snatch'), 'hang-clean': M('clean'), 'hang-snatch': M('snatch'), 'power-clean': M('power-clean'), 'push-jerk': M('push-jerk'), 'split-jerk': M('split-jerk'),
  // passing
  'crossing-and-passing': M('pass-long'), 'ground-passing-one-touch': M('pass-wall-quick'), 'ground-passing-two-touch': M('pass-wall'), 'long-passing': M('pass-long'),
  'passing-while-moving': M('pass-wall'), 'wall-passes-ground-both-feet': M('pass-wall'),
  // plyometrics
  'box-jump': M('box-jump'), 'broad-jump': M('broad-jump'), 'countermovement-jump': M('cmj'), 'depth-jump': M('depth-jump'), 'home-skater-hops': M('lateral-bound', 1.2),
  'home-split-jump': M('split-jump'), 'home-squat-jump': M('jump-squat'), 'hurdle-hops': M('hurdle-hops'), 'lateral-bounds': M('lateral-bound'),
  'plyo-push-up': M('plyo-push-up'), 'pogo-hops': M('pogo'), 'single-leg-bounds': M('bounds'),
  // single limb
  'bulgarian-split-squat': M('split-squat-rear'), 'calf-raise': M('calf-raise'), 'home-cossack': M('cossack'), 'home-glute-bridge-sl': M('bridge-1'),
  'home-reverse-lunge': M('reverse-lunge'), 'home-sofa-split-squat': M('split-squat-sofa'), 'single-arm-db-press': M('db-bench-1'), 'single-arm-row': M('row-1'),
  'single-leg-rdl': M('sl-rdl'), 'skater-squat': M('skater-squat'), 'step-up': M('step-up'),
  // speed and sprints
  'acceleration-starts': M('accel'), 'build-up-sprints-no-ball': M('sprint', 0.8), 'lateral-shuffle-sprint': M('shuffle'), 'sprint-change-of-direction-no-ball': M('cut'),
  'flying-sprint': M('sprint', 1.1), 'resisted-sprint': M('sled'), 'sled-push': M('sled'), 'sprint-10-20': M('accel'),
  // upper strength
  'bench-press': M('bench-press'), 'bent-over-row': M('row'), 'db-overhead-press': M('ohp'), 'home-chair-dip': M('dip'), 'home-pike-push-up': M('pike-push-up'),
  'home-push-up': M('push-up', 0.7), 'home-table-row': M('table-row'), 'pull-up': M('pull-up'), 'push-up': M('push-up'),
}

const byName = new Map((data.exercises as { id: string; name: string }[]).map((e) => [e.name.toLowerCase(), e.id]))
// The demo for an exercise by its name (steps and blocks carry names; the user's own exercises may have none).
export function demoFor(name: string): Demo | null {
  const id = byName.get(name.replace(/\s*\([^)]*\)$/, '').trim().toLowerCase()) ?? byName.get(name.trim().toLowerCase())
  return id ? DEMOS[id] ?? null : null
}
