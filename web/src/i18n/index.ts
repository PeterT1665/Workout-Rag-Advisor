// Languages. The English text is the key: t('Rest is over') looks it up in the chosen language's file
// (src/i18n/locales/<lang>.json) and falls back to English when a line is missing.
// The language is loaded before the app starts (main.tsx), and changing it reloads the app, so every screen,
// the exercise library, the voice and the notifications are all in one language. Exercise names stay English.
// src/i18n/i18n.test.ts fails if any line is missing a translation or a screen has text outside t().

export type Lang = 'en' | 'vi' | 'es' | 'pt' | 'fr' | 'id'

export const LANGS: { id: Lang; name: string; voice: string }[] = [
  { id: 'en', name: 'English', voice: 'en-GB' },
  { id: 'vi', name: 'Tiếng Việt', voice: 'vi-VN' },
  { id: 'es', name: 'Español', voice: 'es-ES' },
  { id: 'pt', name: 'Português', voice: 'pt-BR' },
  { id: 'fr', name: 'Français', voice: 'fr-FR' },
  { id: 'id', name: 'Bahasa Indonesia', voice: 'id-ID' },
]
const isLang = (x: unknown): x is Lang => LANGS.some((l) => l.id === x)

let lang: Lang = 'en'
let dict: Record<string, string> = {}

// The phone's language on first open; the choice in Settings after that (kept on this phone).
const KEY = 'lang'
export function chosenLang(): Lang {
  try {
    const saved = localStorage.getItem(KEY)
    if (isLang(saved)) return saved
  } catch { /* ignore */ }
  const phone = typeof navigator !== 'undefined' ? (navigator.languages?.[0] ?? navigator.language ?? 'en') : 'en'
  const two = phone.slice(0, 2).toLowerCase()
  return isLang(two) ? two : two === 'in' ? 'id' : 'en'
}

const files = import.meta.glob<{ default: Record<string, string> }>('./locales/*.json')

export async function initLang(l: Lang = chosenLang()) {
  lang = l
  dict = {}
  if (l !== 'en') {
    const load = files[`./locales/${l}.json`]
    try { if (load) dict = (await load()).default } catch { /* offline and not cached: English */ }
  }
  if (typeof document !== 'undefined') document.documentElement.lang = l
}

export const currentLang = () => lang

// Switch language: save it and reload, so every screen, the library and the voice come back in it.
export function setLang(l: Lang) {
  try { localStorage.setItem(KEY, l) } catch { /* ignore */ }
  location.reload()
}
export const voiceLang = () => LANGS.find((l) => l.id === lang)!.voice
// For dates and numbers shown to the player (storage keys keep their fixed formats).
export const locale = () => (lang === 'en' ? 'en-GB' : voiceLang())

// t('Round {n} of {total}', { n: 1, total: 3 }). A '|note' after the text tells translators where it's used
// (t('Rest|strip') is the short label in the week strip); English drops the note.
export function t(en: string, vars?: Record<string, string | number>): string {
  const bar = en.indexOf('|')
  let s = dict[en] || (bar > 0 ? en.slice(0, bar) : en)
  if (vars) for (const k in vars) s = s.split(`{${k}}`).join(String(vars[k]))
  return s
}

// Marks text that is stored as written (so it stays matchable) and translated where it's shown with t().
export const mark = (en: string) => en
