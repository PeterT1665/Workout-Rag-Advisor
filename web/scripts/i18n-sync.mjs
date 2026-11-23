// Translation batch tool. While building, new lines fall back to English (the test only warns); before a staging push
// run a batch:
//   node scripts/i18n-sync.mjs missing > /tmp/missing.json   lines with no translation yet (English, as a JSON list)
//   node scripts/i18n-sync.mjs apply /tmp/tr.json            add translations: { "English": { vi, es, pt, fr, id } }
//   node scripts/i18n-sync.mjs prune                         drop lines the app no longer uses
//   npm run i18n:strict                                      must pass before staging
import { readFileSync, writeFileSync } from 'node:fs'
import { catalog } from './i18n-catalog.mjs'

const LANGS = ['vi', 'es', 'pt', 'fr', 'id']
const file = (l) => new URL(`../src/i18n/locales/${l}.json`, import.meta.url)
const read = (l) => JSON.parse(readFileSync(file(l), 'utf8'))
const write = (l, d) => writeFileSync(file(l), JSON.stringify(Object.fromEntries(Object.entries(d).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))), null, 1))
const lines = new Set(catalog())
const [cmd, arg] = process.argv.slice(2)

if (cmd === 'missing') {
  const miss = new Set()
  for (const l of LANGS) { const d = read(l); for (const k of lines) if (!d[k]?.trim()) miss.add(k) }
  console.log(JSON.stringify([...miss], null, 1))
} else if (cmd === 'apply') {
  const tr = JSON.parse(readFileSync(arg, 'utf8'))
  for (const l of LANGS) { const d = read(l); for (const [k, v] of Object.entries(tr)) if (v[l]) d[k] = v[l]; write(l, d) }
  console.log('applied', Object.keys(tr).length)
} else if (cmd === 'prune') {
  let n = 0
  for (const l of LANGS) { const d = read(l); for (const k of Object.keys(d)) if (!lines.has(k)) { delete d[k]; n++ } write(l, d) }
  console.log('pruned', n / LANGS.length, 'lines')
} else console.log('usage: node scripts/i18n-sync.mjs missing | apply <file.json> | prune')
