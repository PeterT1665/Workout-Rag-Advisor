// Clean, smooth-shaded football objects built in code (no downloaded models, no licensing).
// Style: soft 3D icon look: smooth surfaces, clearcoat, studio lighting, contact shadows.
import * as THREE from 'three'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'

export const COLORS = { red: 0xe8213a, redDark: 0xa3101f, white: 0xf5f5f2, black: 0x17181b, seam: 0xb9bcc2 }

const physical = (color, o = {}) =>
  new THREE.MeshPhysicalMaterial({ color, roughness: 0.38, clearcoat: 0.5, clearcoatRoughness: 0.25, ...o })

function shadowed(obj) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true } })
  return obj
}

// ---------- football: 12 black pentagons + 20 white hexagons, with recessed seams ----------
export function makeBall({ detail = 40, seam = 0.022, pentPenalty = 0.035 } = {}) {
  const ico = new THREE.IcosahedronGeometry(1, 0)
  const p = ico.attributes.position
  const pent = []
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i).normalize()
    if (!pent.some((q) => q.distanceTo(v) < 1e-4)) pent.push(v)
  }
  const hex = []
  for (let i = 0; i < p.count; i += 3) {
    const c = new THREE.Vector3()
    for (let k = 0; k < 3; k++) c.add(new THREE.Vector3().fromBufferAttribute(p, i + k))
    hex.push(c.normalize())
  }
  const centers = [...pent.map((v) => ({ v, pent: true })), ...hex.map((v) => ({ v, pent: false }))]

  const g = new THREE.IcosahedronGeometry(1, detail)
  const pos = g.attributes.position
  const colors = new Float32Array(pos.count * 3)
  const black = new THREE.Color(COLORS.black), white = new THREE.Color(COLORS.white), seamC = new THREE.Color(COLORS.seam)
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize()
    let best = Infinity, second = Infinity, bestPent = false
    for (const c of centers) {
      const a = Math.acos(Math.min(1, v.dot(c.v))) + (c.pent ? pentPenalty : 0)
      if (a < best) { second = best; best = a; bestPent = c.pent } else if (a < second) second = a
    }
    const edge = second - best // 0 on a seam
    const t = Math.min(1, edge / seam)
    const col = (bestPent ? black : white).clone().lerp(seamC, 1 - t)
    colors.set([col.r, col.g, col.b], i * 3)
    const r = 1 - 0.014 * (1 - t) * (1 - t) // recess along seams
    pos.setXYZ(i, v.x * r, v.y * r, v.z * r)
  }
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  g.computeVertexNormals()
  const m = new THREE.Mesh(g, physical(0xffffff, { vertexColors: true, roughness: 0.32, clearcoat: 0.6 }))
  return shadowed(m)
}

// ---------- training cone: smooth lathe profile, white band, rounded square base ----------
export function makeCone() {
  const grp = new THREE.Group()
  const prof = []
  const H = 1.25, R0 = 0.56, R1 = 0.1
  for (let i = 0; i <= 24; i++) { const t = i / 24; prof.push(new THREE.Vector2(R0 + (R1 - R0) * t, 0.08 + H * t)) }
  for (let i = 1; i <= 8; i++) { const a = (i / 8) * (Math.PI / 2); prof.push(new THREE.Vector2(R1 * Math.cos(a), 0.08 + H + R1 * 0.6 * Math.sin(a))) }
  const g = new THREE.LatheGeometry(prof, 64)
  const pos = g.attributes.position, colors = new Float32Array(pos.count * 3)
  const red = new THREE.Color(COLORS.red), white = new THREE.Color(COLORS.white)
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i)
    const c = y > 0.62 && y < 0.86 ? white : red
    colors.set([c.r, c.g, c.b], i * 3)
  }
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  grp.add(new THREE.Mesh(g, physical(0xffffff, { vertexColors: true })))
  const base = new THREE.Mesh(new RoundedBoxGeometry(1.5, 0.1, 1.5, 4, 0.05), physical(COLORS.red))
  base.position.y = 0.05
  grp.add(base)
  return shadowed(grp)
}

// ---------- goal: smooth posts, rounded joints, net as a fine alpha texture ----------
function netTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128
  const x = c.getContext('2d')
  x.strokeStyle = 'rgba(255,255,255,0.9)'; x.lineWidth = 3
  for (let i = 0; i <= 128; i += 32) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i, 128); x.stroke(); x.beginPath(); x.moveTo(0, i); x.lineTo(128, i); x.stroke() }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace
  return t
}
export function makeGoal(w = 6, h = 2.4, d = 1.8) {
  const grp = new THREE.Group(), r = 0.07, white = physical(COLORS.white, { roughness: 0.25, clearcoat: 0.8 })
  const cyl = (len) => new THREE.CylinderGeometry(r, r, len, 32)
  for (const x of [-w / 2, w / 2]) {
    const p = new THREE.Mesh(cyl(h), white); p.position.set(x, h / 2, 0); grp.add(p)
    const j = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), white); j.position.set(x, h, 0); grp.add(j)
  }
  const bar = new THREE.Mesh(cyl(w), white); bar.rotation.z = Math.PI / 2; bar.position.y = h; grp.add(bar)
  const tex = netTexture()
  const netMat = (rx, ry) => {
    const t = tex.clone(); t.needsUpdate = true; t.repeat.set(rx, ry)
    return new THREE.MeshBasicMaterial({ map: t, transparent: true, opacity: 0.75, side: THREE.DoubleSide, depthWrite: false })
  }
  const slope = Math.hypot(h, d)
  const back = new THREE.Mesh(new THREE.PlaneGeometry(w, slope), netMat(w / 0.35, slope / 0.35))
  back.position.set(0, h / 2, -d / 2); back.rotation.x = Math.atan2(d, h); grp.add(back)
  for (const s of [-1, 1]) {
    const shape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(0, h), new THREE.Vector2(d, 0)])
    const sg = new THREE.ShapeGeometry(shape)
    const uv = sg.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) / 0.35, uv.getY(i) / 0.35)
    const side = new THREE.Mesh(sg, netMat(1, 1)); side.rotation.y = Math.PI / 2; side.position.x = s * w / 2; grp.add(side)
  }
  return shadowed(grp)
}

// ---------- boot: extruded side profile with a soft bevel, laces, studs ----------
export function makeBoot() {
  const grp = new THREE.Group()
  // side view: x = heel(-) to toe(+), y = up
  const up = new THREE.Shape()
  up.moveTo(-1.05, 0.12)
  up.lineTo(0.95, 0.12)
  up.bezierCurveTo(1.25, 0.12, 1.3, 0.42, 1.0, 0.5)     // rounded toe
  up.bezierCurveTo(0.55, 0.6, 0.1, 0.7, -0.35, 0.92)    // throat rising to the ankle
  up.lineTo(-0.55, 1.02)
  up.bezierCurveTo(-0.85, 1.08, -1.1, 0.98, -1.12, 0.7) // collar and heel
  up.bezierCurveTo(-1.14, 0.45, -1.12, 0.2, -1.05, 0.12)
  const bevel = { depth: 0.5, bevelEnabled: true, bevelThickness: 0.16, bevelSize: 0.12, bevelSegments: 10, curveSegments: 32 }
  const upperG = new THREE.ExtrudeGeometry(up, bevel); upperG.translate(0, 0, -0.25)
  grp.add(new THREE.Mesh(upperG, physical(COLORS.red)))

  const so = new THREE.Shape()
  so.moveTo(-1.08, 0.0); so.lineTo(1.0, 0.0)
  so.bezierCurveTo(1.28, 0.0, 1.34, 0.16, 1.18, 0.2)
  so.lineTo(-1.1, 0.2); so.lineTo(-1.08, 0.0)
  const soleG = new THREE.ExtrudeGeometry(so, { depth: 0.66, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.05, bevelSegments: 6, curveSegments: 24 })
  soleG.translate(0, 0.02, -0.33)
  grp.add(new THREE.Mesh(soleG, physical(COLORS.black, { roughness: 0.5 })))

  // heel counter (white) and laces
  const heel = new THREE.Mesh(new THREE.SphereGeometry(0.3, 32, 24), physical(COLORS.white))
  heel.scale.set(0.55, 1.1, 1.35); heel.position.set(-1.0, 0.55, 0); grp.add(heel)
  for (let i = 0; i < 4; i++) {
    const t = i / 3, x = 0.55 - t * 0.75, y = 0.62 + t * 0.2
    const lace = new THREE.Mesh(new THREE.CapsuleGeometry(0.035, 0.42, 6, 16), physical(COLORS.white))
    lace.rotation.x = Math.PI / 2; lace.position.set(x, y + 0.14, 0); grp.add(lace)
  }
  for (const x of [-0.85, -0.5, 0.25, 0.6, 0.92]) for (const z of [-0.2, 0.2]) {
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 0.12, 20), physical(COLORS.white))
    s.position.set(x, -0.08, z); grp.add(s)
  }
  const opening = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.05, 40), physical(COLORS.black, { roughness: 0.7 }))
  opening.scale.set(1.35, 1, 1); opening.position.set(-0.78, 1.17, 0); opening.rotation.z = 0.12; grp.add(opening)
  grp.position.y = 0.14
  return shadowed(grp)
}

// ---------- water bottle: lathe body, red band, white cap ----------
export function makeBottle() {
  const grp = new THREE.Group()
  const prof = [[0, 0], [0.4, 0], [0.44, 0.04], [0.44, 1.2], [0.4, 1.32], [0.26, 1.42], [0.24, 1.5]].map(([x, y]) => new THREE.Vector2(x, y))
  const g = new THREE.LatheGeometry(prof, 64)
  const pos = g.attributes.position, colors = new Float32Array(pos.count * 3)
  const black = new THREE.Color(COLORS.black), red = new THREE.Color(COLORS.red)
  for (let i = 0; i < pos.count; i++) { const y = pos.getY(i); const c = y > 0.55 && y < 0.85 ? red : black; colors.set([c.r, c.g, c.b], i * 3) }
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  grp.add(new THREE.Mesh(g, physical(0xffffff, { vertexColors: true, roughness: 0.3 })))
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.25, 0.26, 40), physical(COLORS.white))
  cap.position.y = 1.62; grp.add(cap)
  const spout = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.14, 24), physical(COLORS.red))
  spout.position.y = 1.82; grp.add(spout)
  return shadowed(grp)
}

// ---------- lighting & helpers ----------
export function studio(renderer, scene, { warm = false } = {}) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.0
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  const pmrem = new THREE.PMREMGenerator(renderer)
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
  scene.environmentIntensity = 0.55
  const key = new THREE.DirectionalLight(warm ? 0xffd2a8 : 0xffffff, 2.2)
  key.position.set(-3, 6, 4); key.castShadow = true
  key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 6; key.shadow.bias = -0.0005
  Object.assign(key.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 0.5, far: 30 })
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xff3a52, warm ? 1.2 : 0.6); rim.position.set(4, 3, -5); scene.add(rim)
  return key
}

// Soft blob shadow under an object: cleaner than a hard shadow map for icons.
export function contactShadow(size = 2, opacity = 0.45) {
  const c = document.createElement('canvas'); c.width = c.height = 128
  const x = c.getContext('2d'), gr = x.createRadialGradient(64, 64, 0, 64, 64, 64)
  gr.addColorStop(0, `rgba(0,0,0,${opacity})`); gr.addColorStop(1, 'rgba(0,0,0,0)')
  x.fillStyle = gr; x.fillRect(0, 0, 128, 128)
  const m = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }))
  m.rotation.x = -Math.PI / 2; m.position.y = 0.002
  return m
}

// ---------- flat disc marker (the "saucer" cones), optionally stacked ----------
export function makeDisc({ stack = 1, color = COLORS.red } = {}) {
  const grp = new THREE.Group()
  const prof = []
  for (let i = 0; i <= 20; i++) { const t = i / 20; prof.push(new THREE.Vector2(0.62 - 0.44 * t, 0.26 * Math.sin(t * Math.PI / 2))) }
  prof.push(new THREE.Vector2(0.12, 0.26), new THREE.Vector2(0.1, 0.2)) // hole lip
  const g = new THREE.LatheGeometry(prof, 64)
  for (let i = 0; i < stack; i++) {
    const m = new THREE.Mesh(g, physical(color, { roughness: 0.45, side: THREE.DoubleSide }))
    m.position.y = i * 0.07
    grp.add(m)
  }
  return shadowed(grp)
}

// ---------- agility hurdle: one bent tube with rounded corners, small feet ----------
export function makeHurdle({ w = 1.4, h = 0.6, color = COLORS.red } = {}) {
  const grp = new THREE.Group(), r = 0.12
  const path = new THREE.CurvePath()
  const P = (x, y) => new THREE.Vector3(x, y, 0)
  path.add(new THREE.LineCurve3(P(-w / 2, 0), P(-w / 2, h - r)))
  path.add(new THREE.QuadraticBezierCurve3(P(-w / 2, h - r), P(-w / 2, h), P(-w / 2 + r, h)))
  path.add(new THREE.LineCurve3(P(-w / 2 + r, h), P(w / 2 - r, h)))
  path.add(new THREE.QuadraticBezierCurve3(P(w / 2 - r, h), P(w / 2, h), P(w / 2, h - r)))
  path.add(new THREE.LineCurve3(P(w / 2, h - r), P(w / 2, 0)))
  grp.add(new THREE.Mesh(new THREE.TubeGeometry(path, 120, 0.045, 20, false), physical(color)))
  for (const x of [-w / 2, w / 2]) {
    const foot = new THREE.Mesh(new RoundedBoxGeometry(0.12, 0.05, 0.42, 4, 0.02), physical(COLORS.black, { roughness: 0.5 }))
    foot.position.set(x, 0.025, 0); grp.add(foot)
  }
  return shadowed(grp)
}

// ---------- speed ring: flat torus lying on the grass ----------
export function makeRing({ color = COLORS.white } = {}) {
  const m = new THREE.Mesh(new THREE.TorusGeometry(0.75, 0.05, 16, 96), physical(color, { roughness: 0.4 }))
  m.rotation.x = Math.PI / 2; m.position.y = 0.05; m.scale.z = 0.6
  return shadowed(m)
}

// ---------- agility pole with a weighted base ----------
export function makePole({ color = COLORS.red } = {}) {
  const grp = new THREE.Group()
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.0, 24), physical(color))
  pole.position.y = 1.08; grp.add(pole)
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 12), physical(COLORS.white)); tip.position.y = 2.09; grp.add(tip)
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.1, 40), physical(COLORS.black, { roughness: 0.5 })); base.position.y = 0.05; grp.add(base)
  return shadowed(grp)
}

export { THREE }
