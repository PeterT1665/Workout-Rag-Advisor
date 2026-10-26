import { pickNoRepeat } from './logic'
import { loadQuotes } from './storage'

let last: string | null = null
export function nextQuote() {
  const q = pickNoRepeat(loadQuotes(), last) ?? ''
  last = q
  return q
}
