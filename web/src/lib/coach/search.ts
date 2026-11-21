// Retrieval for the coach: a small BM25 index over the app's own knowledge (no network, no model).
// Text is folded (lower case, accents off, đ → d) so "dau goi" finds "đau gối", and words are trimmed to a rough stem.

const STOP = new Set(('a an and are as at be but by can do does for from how i if in into is it its me my of on or so that the this to was what when where which who why will with you your ' +
  'should could would am im ive get got have has had there their them they we our us just about any some more most much very really').split(' '))

export function fold(s: string) {
  return s.toLowerCase().replace(/đ/g, 'd').normalize('NFD').replace(/[̀-ͯ]/g, '')
}
const stem = (w: string) => (w.length > 5 && w.endsWith('ing') ? w.slice(0, -3) : w.length > 4 && w.endsWith('es') ? w.slice(0, -2) : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w)
export function tokens(s: string): string[] {
  return fold(s).split(/[^a-z0-9]+/).filter((w) => w && !STOP.has(w)).map(stem)
}

export type Doc<T> = { id: string; text: string; boost?: string; item: T }
export type Hit<T> = { doc: Doc<T>; score: number }

export class Index<T> {
  private docs: { doc: Doc<T>; tf: Map<string, number>; len: number }[] = []
  private df = new Map<string, number>()
  private avg = 1
  constructor(docs: Doc<T>[]) {
    for (const doc of docs) {
      // the title-ish part counts three times
      const words = [...tokens(doc.text), ...tokens(doc.boost ?? ''), ...tokens(doc.boost ?? ''), ...tokens(doc.boost ?? '')]
      const tf = new Map<string, number>()
      for (const w of words) tf.set(w, (tf.get(w) ?? 0) + 1)
      for (const w of tf.keys()) this.df.set(w, (this.df.get(w) ?? 0) + 1)
      this.docs.push({ doc, tf, len: words.length })
    }
    this.avg = this.docs.reduce((a, d) => a + d.len, 0) / Math.max(1, this.docs.length)
  }
  search(q: string, k = 5): Hit<T>[] {
    const qs = [...new Set(tokens(q))]
    const N = this.docs.length
    const out: Hit<T>[] = []
    for (const d of this.docs) {
      let score = 0
      for (const w of qs) {
        const f = d.tf.get(w)
        if (!f) continue
        const n = this.df.get(w) ?? 0
        const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5))
        score += idf * ((f * 2.2) / (f + 1.2 * (0.25 + 0.75 * (d.len / this.avg))))
      }
      if (score > 0) out.push({ doc: d.doc, score })
    }
    return out.sort((a, b) => b.score - a.score).slice(0, k)
  }
}

// Does the question contain any of these words or phrases (any language, folded)?
export function has(q: string, words: string[]) {
  const f = ` ${fold(q).replace(/[^a-z0-9]+/g, ' ')} `
  return words.some((w) => f.includes(` ${fold(w)} `) || (w.length > 4 && f.includes(fold(w))))
}
