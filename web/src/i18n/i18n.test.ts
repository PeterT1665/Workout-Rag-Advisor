// @vitest-environment node
/// <reference types="node" />
// Languages can't miss a line: every language file has every line the app shows (scripts/i18n-catalog.mjs),
// with the same {placeholders} and <b>/<r> tags, and no screen shows text that skips t().
import { describe, expect, test } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
// @ts-expect-error plain JS build script
import { catalog } from '../../scripts/i18n-catalog.mjs'
import { LANGS } from '.'

const here = fileURLToPath(new URL('.', import.meta.url))
const src = join(here, '..')
const lines: string[] = catalog()
const tokens = (s: string) => [...s.matchAll(/\{\w+\}|<\/?[br]>/g)].map((m) => m[0]).sort().join(' ')

describe('languages', () => {
  for (const { id } of LANGS.filter((l) => l.id !== 'en')) {
    const file = join(here, 'locales', `${id}.json`)
    const dict = JSON.parse(readFileSync(file, 'utf8')) as Record<string, string>

    test(`${id}: every line is translated`, () => {
      expect(lines.filter((l) => !dict[l]?.trim())).toEqual([])
    })
    test(`${id}: placeholders and tags kept`, () => {
      expect(lines.filter((l) => dict[l] && tokens(dict[l]) !== tokens(l))).toEqual([])
    })
    test(`${id}: no lines the app no longer uses`, () => {
      expect(Object.keys(dict).filter((k) => !lines.includes(k))).toEqual([])
    })
  }
})

// Text in screens that doesn't go through t(): JSX text, text attributes, and text on its own line.
const files = (d: string): string[] => readdirSync(d).flatMap((f) => {
  const p = join(d, f)
  return statSync(p).isDirectory() ? (f === 'i18n' || f === 'data' ? [] : files(p)) : /\.tsx$/.test(f) && !/\.test\./.test(f) ? [p] : []
})

test('no screen text outside t()', () => {
  const found: string[] = []
  for (const f of files(src)) {
    readFileSync(f, 'utf8').split('\n').forEach((line, i) => {
      const s = line.trim()
      if (/^(\/\/|\*|import)/.test(s) || line.includes('i18n-ignore') || /\btr\(/.test(line)) return
      const code = line.replace(/className=("[^"]*"|\{`[^`]*`\})/g, '') // class names aren't text
      const at = `${f.slice(src.length)}:${i + 1}`
      // text between tags, or next to a {value}: >Text<  >Text {x}  {x} text<
      for (const m of code.matchAll(/(?<![=-])>([^<>{}]*[A-Za-z]{2,}[^<>{}]*?)[<{]|\}([^<>{}()'"`]*[A-Za-z]{2,}[^<>{}()'"`]*?)<\//g)) {
        const text = (m[1] ?? m[2]).trim()
        if (text && !/[=&(]/.test(text) && !/^(Promise|load|Set|Partial|Record|Array|Map)$/.test(text)) found.push(`${at} ${text}`)
      }
      for (const m of line.matchAll(/\b(aria-label|placeholder|title|label|alt|hint)="([^"]*[A-Za-z]{2,}[^"]*)"/g)) {
        if (!m[2].includes('://')) found.push(`${at} ${m[1]}="${m[2]}"`)
      }
      if (/^[A-Z][A-Za-z ,.'!?:;…-]+$/.test(s) && !s.endsWith(',')) found.push(`${at} ${s}`)
      if (/\b(confirm|alert)\(\s*['"`]/.test(line)) found.push(`${at} confirm/alert text`)
      // a single word picked in code: cond ? 'Now' : ..., name || 'You', sub="You"
      for (const m of code.matchAll(/(?:[?:]|\|\|)\s*'([A-Z][a-z]{2,})'(?=\s*[:)}])|\bsub="([A-Za-z][^"]*)"/g)) found.push(`${at} ${m[1] ?? m[2]}`)
    })
  }
  expect(found).toEqual([])
})
