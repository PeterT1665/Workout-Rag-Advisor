// Card pictures from the 3D demos: one still per move (3:4) into public/demo-still/<move>.jpg, plus a contact sheet to check them.
//   npm run dev -- --port 8096   (in another terminal), then: npm run demo:stills -- <move,ids or all> <contact-sheet prefix, not inside the repo>
// The frame is the move's `still` (0–1 of the loop) if it has one, else its most bent key: usually the bottom of the rep.
import { chromium } from 'playwright-core'
import { mkdirSync, writeFileSync } from 'node:fs'
const only = process.argv[2] && process.argv[2] !== 'all' ? process.argv[2].split(',') : null
const sheet = process.argv[3] ?? '/tmp/stills'
const dir = new URL('../public/demo-still/', import.meta.url)
mkdirSync(dir, { recursive: true })
const b = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--use-angle=metal', '--enable-gpu'] })
const p = await (await b.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1 })).newPage()
const errs = []; p.on('pageerror', (e) => errs.push(e.message))
await p.goto('http://localhost:8096/?app=1'); await p.waitForTimeout(800); await p.evaluate(() => import('/src/lib/demo3d/scene.ts')).catch(() => {}); await p.waitForTimeout(3000); await p.goto('http://localhost:8096/?app=1'); await p.waitForTimeout(1200)
const rows = await p.evaluate(async ({ only }) => {
  const { createDemo } = await import('/src/lib/demo3d/scene.ts')
  const { MOVES } = await import('/src/lib/demo3d/moves.ts')
  // the app's dark-theme demo colours, on the card background
  const pal = { body: '#d9dcdf', joint: '#7d8288', accent: '#e01b35', floor: '#2b2b2f', bg: '#1c1c1f', kit: '#9aa3ab', line: '#3a3e43' }
  const out = []
  for (const [id, m] of Object.entries(MOVES)) {
    if (only && !only.includes(id)) continue
    // the most bent key (hips, knees, shoulders, back): the bottom of a squat, hinge or lunge, the reach of a throw
    const bend = (q) => Math.abs(q.spine ?? 0) + Math.abs(q.pitch ?? 0) + ['L', 'R'].reduce((n, k) => n + Math.abs(q[k]?.hip ?? 0) + Math.abs(q[k]?.knee ?? 0) + 0.5 * Math.abs(q[k]?.sh ?? 0), 0)
    const at = m.still ?? (m.hold ? 0 : m.keys.reduce((best, k) => (bend(k[1]) > bend(best[1]) + 1 ? k : best))[0])
    const c = document.createElement('canvas'); c.width = 360; c.height = 480; c.style.width = '360px'; c.style.height = '480px'; document.body.appendChild(c)
    const d = createDemo(c, m, pal, { at }); out.push({ id, url: c.toDataURL('image/jpeg', 0.72) }); d.dispose(); c.remove()
  }
  return out
}, { only })
for (const r of rows) writeFileSync(new URL(`${r.id}.jpg`, dir), Buffer.from(r.url.split(',')[1], 'base64'))
// contact sheet: small thumbs with the move id, 12 across
const per = 48
for (let i = 0; i * per < rows.length; i++) {
  const html = rows.slice(i * per, (i + 1) * per).map((r) => `<div style="display:inline-block;width:110px;margin:2px;font:10px sans-serif;color:#ff0;vertical-align:top"><img src="${r.url}" width=110 height=147><div>${r.id}</div></div>`).join('')
  await p.setContent(`<body style="margin:4px;background:#000;width:1370px">${html}</body>`); await p.waitForTimeout(200)
  await p.screenshot({ path: `${sheet}-${i}.jpg`, fullPage: true, type: 'jpeg', quality: 60 })
}
console.log(rows.length, 'stills', Math.ceil(rows.length / per), 'sheets', errs.slice(0, 3))
await b.close()
