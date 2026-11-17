import { pickNoRepeat } from './logic'
import { loadQuotes } from './storage'
import { t } from '../i18n'

let last: string | null = null
export function nextQuote() {
  const q = pickNoRepeat(loadQuotes(), last) ?? ''
  last = q
  return t(q) // the built-in quotes are translated; your own stay as you wrote them
}
