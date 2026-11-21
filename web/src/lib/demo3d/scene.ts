// The 3D demo: a faceless mannequin acting out a move, with its kit. Loaded only when a demo is shown.
import * as THREE from 'three'
import { poseAt, type Move, type Pose, type Prop, type Side, type Vec } from './pose'

const D = Math.PI / 180
// body sizes (m)
const THIGH = 0.44, SHIN = 0.42, ANKLE_H = 0.07, UPPER = 0.29, FORE = 0.26, ABS = 0.2, CHEST = 0.3

type Limb = { hip: THREE.Group; knee: THREE.Group; ankle: THREE.Group; toe: THREE.Object3D; heel: THREE.Object3D; sh: THREE.Group; elb: THREE.Group; hand: THREE.Object3D }
type Rig = { root: THREE.Group; pelvis: THREE.Group; spine: THREE.Group; chest: THREE.Group; neck: THREE.Group; head: THREE.Object3D; back: THREE.Object3D; front: THREE.Object3D; L: Limb; R: Limb; points: THREE.Object3D[] }

export type Palette = { body: string; joint: string; accent: string; floor: string; bg: string; kit: string; line: string }

function capsule(len: number, r: number, mat: THREE.Material, dir = -1) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.001, len - 2 * r), 6, 14), mat)
  m.position.y = (dir * len) / 2
  m.castShadow = true
  return m
}
const marker = (parent: THREE.Object3D, x: number, y: number, z: number) => { const o = new THREE.Object3D(); o.position.set(x, y, z); parent.add(o); return o }

function buildLimb(side: 1 | -1, pelvis: THREE.Group, chest: THREE.Group, mat: THREE.Material, shoe: THREE.Material, pts: THREE.Object3D[]): Limb {
  const hip = new THREE.Group(); hip.position.set(side * 0.1, -0.02, 0); hip.rotation.order = 'XZY'; pelvis.add(hip)
  hip.add(capsule(THIGH, 0.068, mat))
  const knee = new THREE.Group(); knee.position.y = -THIGH; hip.add(knee)
  knee.add(capsule(SHIN, 0.052, mat))
  const ankle = new THREE.Group(); ankle.position.y = -SHIN; knee.add(ankle)
  const foot = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.16, 4, 10), shoe)
  foot.rotation.x = Math.PI / 2; foot.position.set(0, -ANKLE_H + 0.045, 0.06); foot.castShadow = true; ankle.add(foot)
  const toe = marker(ankle, 0, -ANKLE_H, 0.17), heel = marker(ankle, 0, -ANKLE_H, -0.06)
  const sh = new THREE.Group(); sh.position.set(side * 0.19, CHEST - 0.04, 0); sh.rotation.order = 'XZY'; chest.add(sh)
  sh.add(capsule(UPPER, 0.045, mat))
  const elb = new THREE.Group(); elb.position.y = -UPPER; sh.add(elb)
  elb.add(capsule(FORE, 0.04, mat))
  const hand = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), mat); hand.position.y = -FORE; hand.castShadow = true; elb.add(hand)
  pts.push(toe, heel, knee, hand, elb)
  return { hip, knee, ankle, toe, heel, sh, elb, hand }
}

function buildRig(p: Palette): Rig {
  const mat = new THREE.MeshStandardMaterial({ color: p.body, roughness: 0.55, metalness: 0.05 })
  const shoe = new THREE.MeshStandardMaterial({ color: p.accent, roughness: 0.5 })
  const root = new THREE.Group()
  const pelvis = new THREE.Group(); root.add(pelvis)
  const hips = new THREE.Mesh(new THREE.CapsuleGeometry(0.1, 0.12, 4, 12), mat); hips.rotation.z = Math.PI / 2; hips.scale.set(1, 1, 0.8); hips.castShadow = true; pelvis.add(hips)
  const spine = new THREE.Group(); spine.position.y = 0.05; pelvis.add(spine)
  spine.add(capsule(ABS, 0.11, mat, 1)).scale.set(1, 1, 0.75)
  const chest = new THREE.Group(); chest.position.y = ABS; spine.add(chest)
  const torso = capsule(CHEST, 0.14, mat, 1); torso.scale.set(1.25, 1, 0.75); chest.add(torso)
  const neck = new THREE.Group(); neck.position.y = CHEST + 0.02; chest.add(neck)
  neck.add(capsule(0.1, 0.045, mat, 1))
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.105, 20, 16), mat); head.scale.set(0.92, 1.12, 1); head.position.y = 0.19; head.castShadow = true; neck.add(head)
  const back = marker(chest, 0, CHEST - 0.02, -0.12), front = marker(chest, 0, CHEST - 0.05, 0.13)
  const points: THREE.Object3D[] = [head, pelvis, marker(chest, 0, CHEST, -0.1), marker(spine, 0, 0.05, -0.1), marker(pelvis, 0, -0.05, -0.09)]
  const L = buildLimb(1, pelvis, chest, mat, shoe, points), R = buildLimb(-1, pelvis, chest, mat, shoe, points)
  return { root, pelvis, spine, chest, neck, head, back, front, L, R, points }
}

function applySide(l: Limb, s: Side = {}, side: 1 | -1, flatFeet: boolean, pitch: number, legDefault: boolean) {
  const abd = (s.abd ?? (legDefault ? 0 : 0)) * D
  l.hip.rotation.set(-(s.hip ?? 0) * D, side * (s.rot ?? 0) * D, side * abd)
  l.knee.rotation.x = (s.knee ?? 0) * D
  const legSum = pitch * D - (s.hip ?? 0) * D + (s.knee ?? 0) * D
  l.ankle.rotation.x = (flatFeet ? -legSum : 0) - (s.ankle ?? 0) * D
  l.sh.rotation.set(-(s.sh ?? 0) * D, side * (s.shRot ?? 0) * D, side * (s.shAbd ?? 8) * D)
  l.elb.rotation.x = -(s.elb ?? 0) * D
}

function applyPose(r: Rig, p: Pose, flatFeet: boolean) {
  r.root.rotation.set(0, (p.yaw ?? 0) * D, 0)
  r.pelvis.rotation.set((p.pitch ?? 0) * D, 0, (p.roll ?? 0) * D)
  const sp = (p.spine ?? 0) * D
  r.spine.rotation.set(sp * 0.5, (p.twist ?? 0) * D * 0.5, (p.side ?? 0) * D * 0.5)
  r.chest.rotation.set(sp * 0.5, (p.twist ?? 0) * D * 0.5, (p.side ?? 0) * D * 0.5)
  r.neck.rotation.x = (p.neck ?? 0) * D
  // feet level with the floor: account for the trunk pitch only (legs hang from the pelvis)
  applySide(r.L, p.L, 1, flatFeet, p.pitch ?? 0, true)
  applySide(r.R, p.R, -1, flatFeet, p.pitch ?? 0, true)
}

const tmp = new THREE.Vector3()
const world = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3())

// Put the posed figure on the floor (or its bench, or under its bar), with its anchor where it started.
function place(r: Rig, m: Move, p: Pose, anchor0: THREE.Vector3 | null) {
  r.root.position.set(0, 1, 0)
  r.root.updateMatrixWorld(true)
  if (m.hang !== undefined) {
    const h = world(r.L.hand).add(world(r.R.hand)).multiplyScalar(0.5)
    r.root.position.set(-h.x + (p.x ?? 0), 1 + m.hang - h.y, -h.z + (p.z ?? 0))
    r.root.updateMatrixWorld(true)
    return
  }
  if (m.seat !== undefined) {
    const h = world(r.pelvis)
    r.root.position.set(p.x ?? 0, 1 + m.seat - h.y, p.z ?? 0)
    r.root.updateMatrixWorld(true)
    return
  }
  const contacts = m.contacts ?? 'all'
  const pts = contacts === 'feet' ? [r.L.toe, r.L.heel, r.R.toe, r.R.heel]
    : contacts === 'back' ? [r.back, r.points[2], r.points[3], r.points[4]]
      : contacts === 'hands-feet' ? [r.L.toe, r.R.toe, r.L.hand, r.R.hand]
        : r.points
  let low = Infinity
  for (const o of pts) { o.getWorldPosition(tmp); low = Math.min(low, tmp.y) }
  let x = p.x ?? 0, z = p.z ?? 0
  const a = m.anchor ?? 'feet'
  if (a !== 'none' && anchor0) {
    const cur = a === 'feet' ? world(r.L.ankle).add(world(r.R.ankle)).multiplyScalar(0.5)
      : a === 'hands' ? world(r.L.hand).add(world(r.R.hand)).multiplyScalar(0.5)
        : world((a === 'L' ? r.L : r.R).ankle)
    x += anchor0.x - cur.x; z += anchor0.z - cur.z
  }
  r.root.position.set(x, 1 - low + (m.base ?? 0) + (p.air ?? 0), z)
  r.root.updateMatrixWorld(true)
}
function anchorPoint(r: Rig, m: Move): THREE.Vector3 | null {
  const a = m.anchor ?? 'feet'
  if (a === 'none' || m.hang !== undefined || m.seat !== undefined) return null
  if (a === 'feet') return world(r.L.ankle).add(world(r.R.ankle)).multiplyScalar(0.5).setY(0)
  if (a === 'hands') return world(r.L.hand).add(world(r.R.hand)).multiplyScalar(0.5).setY(0)
  return world((a === 'L' ? r.L : r.R).ankle).setY(0)
}

// ---------- kit ----------
function kitMesh(kind: Prop['kind'], p: Palette, size?: Vec): THREE.Object3D {
  const metal = new THREE.MeshStandardMaterial({ color: p.kit, roughness: 0.35, metalness: 0.6 })
  const dark = new THREE.MeshStandardMaterial({ color: p.line, roughness: 0.8 })
  const soft = new THREE.MeshStandardMaterial({ color: p.joint, roughness: 0.9 })
  const g = new THREE.Group()
  const add = (m: THREE.Mesh) => { m.castShadow = true; m.receiveShadow = true; g.add(m); return m }
  const cyl = (r: number, l: number, mat: THREE.Material) => new THREE.Mesh(new THREE.CylinderGeometry(r, r, l, 18), mat)
  const box = (x: number, y: number, z: number, mat: THREE.Material) => new THREE.Mesh(new THREE.BoxGeometry(x, y, z), mat)
  switch (kind) {
    case 'barbell': case 'barbell-back': case 'barbell-front': case 'barbell-hips': case 'landmine': {
      const bar = add(cyl(0.015, kind === 'landmine' ? 1.6 : 2.0, metal)); bar.rotation.z = Math.PI / 2
      if (kind === 'landmine') { bar.position.x = 0.8; const plate = add(cyl(0.2, 0.06, dark)); plate.rotation.z = Math.PI / 2; plate.position.x = 1.45; break }
      for (const s of [-1, 1]) { const plate = add(cyl(0.22, 0.07, dark)); plate.rotation.z = Math.PI / 2; plate.position.x = s * 0.78 }
      break
    }
    case 'dumbbells': case 'dumbbell-R': {
      const h = add(cyl(0.016, 0.16, metal)); h.rotation.z = Math.PI / 2
      for (const s of [-1, 1]) { const w = add(cyl(0.055, 0.06, dark)); w.rotation.z = Math.PI / 2; w.position.x = s * 0.1 }
      break
    }
    case 'kettlebell': { const b = add(new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12), dark)); b.position.y = -0.14; const t = add(new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.014, 8, 16), metal)); t.position.y = -0.02; break }
    case 'trapbar': {
      const frame = add(new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.016, 8, 6), metal)); frame.rotation.x = Math.PI / 2
      for (const s of [-1, 1]) { const plate = add(cyl(0.22, 0.07, dark)); plate.rotation.z = Math.PI / 2; plate.position.x = s * 0.72; const sleeve = add(cyl(0.015, 0.4, metal)); sleeve.rotation.z = Math.PI / 2; sleeve.position.x = s * 0.5 }
      break
    }
    case 'medball': add(new THREE.Mesh(new THREE.SphereGeometry(0.12, 18, 14), dark)); break
    case 'bench': { const [x, y, z] = size ?? [0.3, 0.45, 1.2]; const top = add(box(x, 0.06, z, dark)); top.position.y = y - 0.03; for (const s of [-1, 1]) { const leg = add(box(0.05, y - 0.06, 0.05, metal)); leg.position.set(0, (y - 0.06) / 2, s * (z / 2 - 0.1)) } break }
    case 'sofa': { const [x, y, z] = size ?? [0.9, 0.45, 0.8]; const seat = add(box(x, y, z, soft)); seat.position.y = y / 2; break }
    case 'box': case 'step': { const [x, y, z] = size ?? [0.6, kind === 'step' ? 0.2 : 0.5, 0.5]; const b = add(box(x, y, z, soft)); b.position.y = y / 2; break }
    case 'table': { const [x, y, z] = size ?? [1.0, 0.75, 0.8]; const top = add(box(x, 0.05, z, soft)); top.position.y = y; for (const sx of [-1, 1]) for (const sz of [-1, 1]) { const leg = add(box(0.05, y, 0.05, metal)); leg.position.set(sx * (x / 2 - 0.05), y / 2, sz * (z / 2 - 0.05)) } break }
    case 'bar': { const bar = add(cyl(0.018, 1.4, metal)); bar.rotation.z = Math.PI / 2; for (const s of [-1, 1]) { const post = add(cyl(0.025, 2.5, metal)); post.position.set(s * 0.7, -1.25 + 0.0, 0) } break }
    case 'wall': { const [x, y] = size ?? [3, 2.2]; const w = add(box(x, y, 0.1, soft)); w.position.y = y / 2; break }
    case 'cones': for (const z of [0, 1.2, 2.4, 3.6]) { const c = add(new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.2, 14), new THREE.MeshStandardMaterial({ color: p.accent }))); c.position.set(0, 0.1, z) } break
    case 'hurdles': for (const z of [0, 0.9, 1.8]) { const top = add(cyl(0.012, 0.6, new THREE.MeshStandardMaterial({ color: p.accent }))); top.rotation.z = Math.PI / 2; top.position.set(0, 0.22, z); for (const s of [-1, 1]) { const leg = add(cyl(0.01, 0.22, metal)); leg.position.set(s * 0.3, 0.11, z) } } break
    case 'goal': { const w = 2.4, h = 1.2; for (const s of [-1, 1]) { const post = add(cyl(0.03, h, soft)); post.position.set(s * w / 2, h / 2, 0) } const cross = add(cyl(0.03, w, soft)); cross.rotation.z = Math.PI / 2; cross.position.y = h; break }
    case 'bike': {
      const seat = add(box(0.12, 0.05, 0.25, dark)); seat.position.set(0, 0.95, -0.15)
      const post = add(cyl(0.03, 0.9, metal)); post.position.set(0, 0.45, -0.1); post.rotation.x = -0.2
      const front = add(cyl(0.03, 1.0, metal)); front.position.set(0, 0.55, 0.45); front.rotation.x = 0.25
      const bars = add(cyl(0.02, 0.45, metal)); bars.rotation.z = Math.PI / 2; bars.position.set(0, 1.05, 0.55)
      const wheel = add(new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.05, 10, 28), dark)); wheel.rotation.y = Math.PI / 2; wheel.position.set(0, 0.3, 0.35)
      const base = add(box(0.4, 0.04, 1.2, metal)); base.position.set(0, 0.02, 0.1)
      break
    }
    case 'sled': { const s = add(box(0.55, 0.12, 0.7, dark)); s.position.y = 0.06; for (const x of [-0.2, 0.2]) { const pole = add(cyl(0.02, 0.9, metal)); pole.position.set(x, 0.5, -0.3); pole.rotation.x = 0.3 } break }
    case 'roller': { const r = add(cyl(0.075, 0.45, soft)); r.rotation.z = Math.PI / 2; r.position.y = 0.075; break }
    case 'band': { const b = add(cyl(0.008, 1.2, new THREE.MeshStandardMaterial({ color: p.accent }))); b.rotation.z = Math.PI / 2; break }
    case 'towel': { const t = add(box(0.35, 0.01, 0.25, new THREE.MeshStandardMaterial({ color: p.accent }))); t.position.y = 0.005; break }
  }
  return g
}

// Footballs: white with dark patches (a canvas texture).
function football(p: Palette) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 128
  const x = c.getContext('2d')!
  x.fillStyle = '#f4f4f2'; x.fillRect(0, 0, 256, 128)
  x.fillStyle = p.line
  for (let i = 0; i < 12; i++) { x.beginPath(); const cx = (i % 6) * 44 + (i < 6 ? 10 : 32), cy = i < 6 ? 34 : 94; for (let k = 0; k < 5; k++) { const a = (k / 5) * Math.PI * 2; x.lineTo(cx + Math.cos(a) * 13, cy + Math.sin(a) * 13) } x.fill() }
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 18), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 }))
  m.castShadow = true
  return m
}

// ---------- the demo ----------
export type Demo = { dispose: () => void; setPlaying: (on: boolean) => void; resize: () => void }

export function createDemo(canvas: HTMLCanvasElement, move: Move, pal: Palette, opts: { speed?: number; at?: number } = {}): Demo {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'low-power' })
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.outputColorSpace = THREE.SRGBColorSpace
  const scene = new THREE.Scene()
  scene.background = new THREE.Color(pal.bg)
  scene.fog = new THREE.Fog(pal.bg, 7, 16)

  scene.add(new THREE.HemisphereLight('#ffffff', pal.floor, 1.3))
  const sun = new THREE.DirectionalLight('#ffffff', 2.1)
  sun.position.set(2.5, 5, 3.5); sun.castShadow = true
  sun.shadow.mapSize.set(1024, 1024)
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 15 })
  sun.shadow.radius = 4
  scene.add(sun)
  const rim = new THREE.DirectionalLight('#ffffff', 0.6); rim.position.set(-3, 2.5, -3); scene.add(rim)

  // floor: a soft disc with faint lines (they slide under the runner)
  const floorTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 512
    const x = c.getContext('2d')!
    x.fillStyle = pal.floor; x.fillRect(0, 0, 512, 512)
    x.strokeStyle = pal.line; x.globalAlpha = 0.35; x.lineWidth = 3
    for (let i = 0; i <= 8; i++) { x.beginPath(); x.moveTo(0, i * 64); x.lineTo(512, i * 64); x.stroke() }
    const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(4, 4); t.colorSpace = THREE.SRGBColorSpace
    return t
  })()
  const floor = new THREE.Mesh(new THREE.CircleGeometry(7, 48), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 1 }))
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor)

  const rig = buildRig(pal)
  scene.add(rig.root)
  const flat = move.flatFeet ?? true

  // kit in the world, and kit that follows the hands
  const held: { obj: THREE.Object3D; kind: Prop['kind'] }[] = []
  for (const pr of move.props ?? []) {
    const obj = kitMesh(pr.kind, pal, pr.size)
    if (['barbell', 'barbell-back', 'barbell-front', 'barbell-hips', 'dumbbells', 'dumbbell-R', 'kettlebell', 'trapbar', 'medball', 'landmine'].includes(pr.kind)) {
      if (pr.kind === 'dumbbells') { const other = kitMesh('dumbbells', pal); scene.add(other); held.push({ obj: other, kind: 'dumbbell-R' }); held.push({ obj, kind: 'dumbbell-L' as Prop['kind'] }) }
      else held.push({ obj, kind: pr.kind })
    } else {
      if (pr.at) obj.position.set(...pr.at)
      obj.rotation.y = (pr.yaw ?? 0) * D
    }
    scene.add(obj)
  }
  const ball = move.ball ? football(pal) : null
  if (ball) scene.add(ball)

  // where each key's ball is (keys hold it on a foot or in the hands, or put it in the world)
  applyPose(rig, move.keys[0][1], flat)
  place(rig, move, move.keys[0][1], null)
  const anchor0 = anchorPoint(rig, move)
  const ballKeys: THREE.Vector3[] = move.keys.map(([, p]) => {
    applyPose(rig, p, flat); place(rig, move, p, anchor0)
    const b = p.ball
    if (!b) return new THREE.Vector3(0.25, 0.11, 0.35)
    if (Array.isArray(b)) return new THREE.Vector3(...b)
    const base = b.at === 'hands' ? world(rig.L.hand).add(world(rig.R.hand)).multiplyScalar(0.5) : world((b.at === 'L' ? rig.L : rig.R).toe).add(new THREE.Vector3(0, 0.11, 0.08))
    return base.add(new THREE.Vector3(...(b.off ?? [0, 0, 0])))
  })

  // frame the camera on the whole loop (figure and kit)
  const box = new THREE.Box3()
  for (let i = 0; i < 12; i++) {
    const { pose } = poseAt(move.keys, i / 12)
    applyPose(rig, pose, flat); place(rig, move, pose, anchor0)
    for (const o of [...rig.points, rig.L.hand, rig.R.hand, rig.head]) box.expandByPoint(world(o))
  }
  for (const pr of move.props ?? []) if (pr.at && !['cones', 'hurdles', 'wall', 'goal'].includes(pr.kind)) box.expandByPoint(new THREE.Vector3(...pr.at))
  box.expandByPoint(new THREE.Vector3(box.min.x, 0, box.min.z))
  const center = box.getCenter(new THREE.Vector3())
  const sizeV = box.getSize(new THREE.Vector3())
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50)
  let yaw = (move.cam?.yaw ?? 35) * D
  const tilt = (move.cam?.tilt ?? 12) * D
  const fit = () => {
    const span = Math.max(sizeV.y * 1.25, Math.max(sizeV.x, sizeV.z) * 1.05 / Math.max(0.6, camera.aspect), 1.4)
    const dist = (span / 2 / Math.tan((camera.fov * D) / 2)) * (move.cam?.zoom ?? 1) + 0.6
    camera.position.set(center.x + Math.sin(yaw) * Math.cos(tilt) * dist, center.y + Math.sin(tilt) * dist + 0.1, center.z + Math.cos(yaw) * Math.cos(tilt) * dist)
    camera.lookAt(center.x, center.y, center.z)
  }

  const resize = () => {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 225
    renderer.setSize(w, h, false)
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    fit()
  }
  resize()

  // drag to turn the view
  let dragX: number | null = null
  const down = (e: PointerEvent) => { dragX = e.clientX; canvas.setPointerCapture(e.pointerId) }
  const moveP = (e: PointerEvent) => { if (dragX === null) return; yaw -= (e.clientX - dragX) * 0.01; dragX = e.clientX; fit(); if (!playing) frame(last) }
  const up = () => { dragX = null }
  canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', moveP); canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up)

  const speed = opts.speed ?? 1
  let playing = true, raf = 0, start = performance.now(), pausedAt = 0, last = 0
  const quat = new THREE.Quaternion()
  function frame(tSec: number) {
    last = tSec
    const loop = (tSec * speed) / move.dur
    const { pose, a, b, u } = poseAt(move.keys, move.hold ? 0 : loop)
    const breathe = move.hold ? Math.sin(tSec * 2) * 1.2 : 0
    applyPose(rig, { ...pose, spine: (pose.spine ?? 0) + breathe }, flat)
    place(rig, move, pose, anchor0)
    // held kit
    const lh = world(rig.L.hand), rh = world(rig.R.hand), mid = lh.clone().add(rh).multiplyScalar(0.5)
    for (const h of held) {
      if (h.kind === 'barbell-back' || h.kind === 'barbell-front') { h.obj.position.copy(world(h.kind === 'barbell-back' ? rig.back : rig.front)); rig.chest.getWorldQuaternion(quat); h.obj.quaternion.copy(quat); continue }
      if (h.kind === 'barbell-hips') { h.obj.position.copy(world(rig.pelvis)).add(new THREE.Vector3(0, 0.12, 0)); h.obj.rotation.set(0, 0, 0); continue }
      if (h.kind === 'dumbbell-R') { h.obj.position.copy(rh); rig.R.elb.getWorldQuaternion(quat); h.obj.quaternion.copy(quat); continue }
      if (h.kind === ('dumbbell-L' as Prop['kind'])) { h.obj.position.copy(lh); rig.L.elb.getWorldQuaternion(quat); h.obj.quaternion.copy(quat); continue }
      if (h.kind === 'landmine') { // pivots on the floor in front, the far end in the hands
        const pivot = new THREE.Vector3(mid.x, 0.03, mid.z + 1.7), dir = mid.clone().sub(pivot)
        h.obj.position.copy(pivot); h.obj.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir.clone().normalize()); h.obj.scale.set(dir.length() / 1.6, 1, 1)
        continue
      }
      h.obj.position.copy(mid)
      if (h.kind === 'kettlebell' || h.kind === 'medball') continue
      const dir = lh.clone().sub(rh)
      if (h.kind === 'trapbar') dir.setY(0)
      if (dir.lengthSq() > 1e-6) h.obj.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir.normalize())
    }
    if (ball) {
      const pa = ballKeys[a], pb = ballKeys[b]
      ball.position.lerpVectors(pa, pb, u)
      if (move.keys[a][1].ball && !Array.isArray(move.keys[a][1].ball) && a === b) ball.position.copy(pa)
      ball.rotation.x += 0.08 * speed
    }
    if (move.run) floorTex.offset.y = (tSec * move.run * speed) / 2
    renderer.render(scene, camera)
  }
  const loop = (now: number) => { frame((now - start) / 1000); raf = requestAnimationFrame(loop) }
  if (opts.at !== undefined) { playing = false; frame((opts.at * move.dur) / speed) } // one still frame (checks, reduced motion)
  else raf = requestAnimationFrame(loop)

  return {
    resize,
    setPlaying(on: boolean) {
      if (on === playing) return
      playing = on
      if (on) { start = performance.now() - pausedAt * 1000; raf = requestAnimationFrame(loop) }
      else { cancelAnimationFrame(raf); pausedAt = last }
    },
    dispose() {
      cancelAnimationFrame(raf)
      canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', moveP); canvas.removeEventListener('pointerup', up); canvas.removeEventListener('pointercancel', up)
      scene.traverse((o) => { if (o instanceof THREE.Mesh) { o.geometry.dispose(); const m = o.material as THREE.MeshStandardMaterial; m.map?.dispose(); m.dispose() } })
      floorTex.dispose()
      renderer.dispose()
    },
  }
}

