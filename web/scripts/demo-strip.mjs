// Check 3D demos without opening the app: N frames per move in a row, labelled with the exercises that use it.
//   npm run dev -- --port 8096   (in another terminal), then: npm run demo:strip -- <move,ids or all> <frames=3> <out-prefix, e.g. /tmp/strip (not inside the repo)>
// Needs Google Chrome installed. Keep frames low (3) and look at one page at a time: images are expensive to review.
import { chromium } from 'playwright-core'
const only = process.argv[2] && process.argv[2] !== 'all' ? process.argv[2].split(',') : null
const N = Number(process.argv[3] ?? 3), out = process.argv[4] ?? '/tmp/strip'
const b = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--use-angle=metal', '--enable-gpu'] })
const p = await (await b.newContext({ viewport: { width: 1400, height: 900 } })).newPage()
const errs = []; p.on('pageerror', (e) => errs.push(e.message))
await p.goto('http://localhost:8096/?app=1'); await p.waitForTimeout(800); await p.evaluate(() => import('/src/lib/demo3d/scene.ts')).catch(() => {}); await p.waitForTimeout(3000); await p.goto('http://localhost:8096/?app=1'); await p.waitForTimeout(1200)
const rows = await p.evaluate(async ({ only, N }) => {
  const { createDemo } = await import('/src/lib/demo3d/scene.ts')
  const { MOVES } = await import('/src/lib/demo3d/moves.ts')
  const { DEMOS } = await import('/src/lib/demo3d/catalog.ts')
  const pal = { body: '#d9dcdf', joint: '#8b9096', accent: '#e01b35', floor: '#1c1f22', bg: '#121416', kit: '#9aa3ab', line: '#2e3338' }
  const users = {}; for (const [ex, d] of Object.entries(DEMOS)) (users[d.move] ??= []).push(ex)
  const out = []
  for (const id of Object.keys(MOVES)) {
    if (only && !only.includes(id)) continue
    const fr = []
    for (let i = 0; i < N; i++) {
      const c = document.createElement('canvas'); c.width = 360; c.height = 300; c.style.width = "360px"; c.style.height = "300px"; document.body.appendChild(c)
      const d = createDemo(c, MOVES[id], pal, { at: MOVES[id].hold ? 0 : i / N }); fr.push(c.toDataURL('image/jpeg', 0.8)); d.dispose(); c.remove()
    }
    out.push({ id, users: users[id] ?? [], fr })
  }
  return out
}, { only, N })
const per = 8
for (let i = 0; i * per < rows.length; i++) {
  const html = rows.slice(i * per, (i + 1) * per).map((r) => `<div style="display:flex;align-items:center;gap:3px;margin-bottom:3px"><div style="width:150px;font:bold 13px sans-serif;color:#ff0">${r.id}<div style="font:11px sans-serif;color:#ccc">${r.users.join('<br>')}</div></div>${r.fr.map((u) => `<img src="${u}" width=360 height=300>`).join('')}</div>`).join('')
  await p.setContent(`<body style="margin:4px;background:#000">${html}</body>`); await p.waitForTimeout(150)
  await p.screenshot({ path: `${out}-${i}.png`, fullPage: true })
}
console.log(rows.length, 'moves', Math.ceil(rows.length / per), 'pages', errs.slice(0, 3))
await b.close()
