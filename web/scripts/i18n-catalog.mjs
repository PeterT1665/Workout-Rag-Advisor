// Every English line the app can show: t('...'), tr('...') and mark('...') in src/, plus the exercise library
// (instructions, descriptive reps, equipment, session titles and summaries). Exercise names stay English.
// Used by src/i18n/i18n.test.ts (every language must have every line) and to write new lines for translators:
//   node scripts/i18n-catalog.mjs > /tmp/lines.json
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = fileURLToPath(new URL('../src/', import.meta.url))
const files = (d) => readdirSync(d).flatMap((f) => {
  const p = join(d, f)
  return statSync(p).isDirectory() ? files(p) : /\.tsx?$/.test(f) && !/\.test\./.test(f) ? [p] : []
})
const unescape = (s) => s.replace(/\\(['"\\])/g, '$1')

export function catalog() {
  const out = new Set()
  const call = /\b(?:t|tr|mark)\(\s*(?:'((?:\\.|[^'\\])*)'|"((?:\\.|[^"\\])*)")/g
  for (const f of files(SRC)) {
    const src = readFileSync(f, 'utf8')
    for (const m of src.matchAll(call)) out.add(unescape(m[1] ?? m[2]))
  }
  const lib = JSON.parse(readFileSync(join(SRC, 'data/library.json'), 'utf8'))
  for (const e of lib.exercises) {
    if (e.how) out.add(e.how)
    if (e.safe) out.add(e.safe)
    if (e.reps && !/^\d+(-\d+)?$/.test(e.reps)) out.add(e.reps)
    for (const n of e.needs ?? []) out.add(n)
  }
  for (const tpl of lib.templates) {
    out.add(tpl.title); out.add(tpl.summary)
    for (const n of tpl.needs ?? []) out.add(n)
    for (const it of tpl.items) if (it.how) out.add(it.how)
  }
  return [...out].filter(Boolean).sort()
}

if (process.argv[1] === fileURLToPath(import.meta.url)) console.log(JSON.stringify(catalog(), null, 1))
