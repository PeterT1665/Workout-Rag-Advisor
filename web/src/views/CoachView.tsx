import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Button } from '../components/ui'
import { DrillSheet } from '../components/DrillSheet'
import { answer, starterChips, type Card, type CoachCtx, type Reply, type Save } from '../lib/coach/coach'
import { addInjury, healInjury, rateInjury } from '../lib/injuries'
import { logBody } from '../lib/body'
import { getProfile, setProfile } from '../lib/profile'
import { exerciseBlock } from '../lib/library'
import { load, save } from '../lib/storage'
import { t } from '../i18n'

type Msg = { id: number; me?: string; reply?: Reply }
const KEY = 'coach:chat' // the chat itself stays on this phone; what it learns (injuries, weight) goes on the synced profile

// The coach: a chat that answers from your own data and the app's knowledge (lib/coach). Works offline.
export function CoachView({ ctx, ask: first, onBack, onSession, onGo }: {
  ctx: CoachCtx; ask?: string; onBack: () => void; onSession: (id: string) => void; onGo: (to: 'record' | 'week' | 'settings' | 'team') => void
}) {
  const [msgs, setMsgs] = useState<Msg[]>(() => load<Msg[]>(KEY, []))
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState<string | null>(null) // asked, answer on its way
  const typing = pending !== null
  const [drill, setDrill] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { save(KEY, msgs.slice(-60)) }, [msgs])
  useLayoutEffect(() => { end.current?.scrollIntoView({ block: 'end', behavior: msgs.length > 1 ? 'smooth' : 'auto' }) }, [msgs.length, typing])
  // a short pause reads as thought; the answer itself is instant
  useEffect(() => {
    if (pending === null) return
    const id = setTimeout(() => {
      const reply = answer(pending, ctx)
      reply.save?.forEach(remember)
      setMsgs((m) => [...m, { id: Date.now() + 1, reply: { ...reply, save: undefined } }]); setPending(null)
    }, 380 + Math.min(500, pending.length * 8))
    return () => clearTimeout(id)
  }, [pending]) // eslint-disable-line react-hooks/exhaustive-deps
  // opened with a question (Home's quick questions): ask it once
  const asked = useRef(false)
  useEffect(() => { if (first && !asked.current) { asked.current = true; ask(first) } }, [first]) // eslint-disable-line react-hooks/exhaustive-deps

  function ask(q: string) {
    const text = q.trim().slice(0, 300)
    if (!text || pending !== null) return
    setDraft('')
    setMsgs((m) => [...m, { id: Date.now(), me: text }])
    setPending(text)
  }
  const intro = answer('', ctx)
  const lastChips = [...msgs].reverse().find((m) => m.reply)?.reply?.chips
  const chips = (msgs.length ? lastChips : null) ?? starterChips()
  const block = drill ? exerciseBlock(drill) : undefined

  return (
    <main className="fixed inset-0 flex flex-col bg-bg text-ink">
      <header className="flex items-center gap-3 border-b border-white/5 px-4 pt-[max(14px,var(--sat))] pb-3">
        <Button variant="ghost" size="sm" onClick={onBack}>{t('Back')}</Button>
        <div className="min-w-0 flex-1">
          <h1 className="m-0 font-display text-xl uppercase leading-none">{t('AI coach')}</h1>
          <p className="m-0 mt-0.5 truncate text-xs text-muted">{t('Knows your plan, your log and every exercise. Works offline.')}</p>
        </div>
        {msgs.length > 0 && <button onClick={() => { if (confirm(t('Clear this chat?'))) setMsgs([]) }} className="min-h-10 px-2 text-sm font-semibold text-muted">{t('Clear')}</button>}
      </header>

      <div className="flex-1 overflow-y-auto overscroll-contain px-4 py-4" aria-live="polite">
        <div className="mx-auto flex max-w-xl flex-col gap-3">
          <CoachBubble reply={intro} onCard={(c) => open(c)} />
          {msgs.map((m) => (m.me
            ? <p key={m.id} className="rise m-0 max-w-[85%] self-end whitespace-pre-wrap rounded-[20px] rounded-br-md bg-red px-4 py-2.5 text-on-red">{m.me}</p>
            : m.reply && <CoachBubble key={m.id} reply={m.reply} onCard={(c) => open(c)} />))}
          {typing && (
            <div className="flex w-16 items-center justify-center gap-1 self-start rounded-[20px] rounded-bl-md bg-surface py-3.5" aria-label={t('Coach is typing')}>
              {[0, 1, 2].map((i) => <span key={i} className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted" style={{ animationDelay: `${i * 120}ms` }} />)}
            </div>
          )}
          <div ref={end} />
        </div>
      </div>

      <div className="border-t border-white/5 bg-bg pb-[max(10px,env(safe-area-inset-bottom))]">
        <div className="mx-auto max-w-xl">
          <div className="flex gap-2 overflow-x-auto px-4 pt-2.5 pb-1 [scrollbar-width:none]">
            {chips.map((c) => (
              <button key={c} onClick={() => ask(c)} disabled={typing} className="min-h-9 shrink-0 rounded-full bg-surface-2 px-3.5 text-sm font-semibold active:scale-95 disabled:opacity-50">{c}</button>
            ))}
          </div>
          <form className="flex items-end gap-2 px-4 pt-1.5" onSubmit={(e) => { e.preventDefault(); ask(draft) }}>
            <label htmlFor="coach-q" className="sr-only">{t('Ask your coach')}</label>
            <input id="coach-q" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={t('Ask about your training…')} autoComplete="off" enterKeyHint="send" maxLength={300}
              className="min-h-12 min-w-0 flex-1 rounded-full bg-surface px-5 text-[16px] text-ink placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-[var(--red)]" />
            <button type="submit" disabled={!draft.trim() || typing} aria-label={t('Send')}
              className="grid h-12 w-12 shrink-0 place-items-center rounded-full bg-red text-on-red transition-opacity active:scale-95 disabled:opacity-40">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
            </button>
          </form>
        </div>
      </div>
      {block && <DrillSheet block={block} onClose={() => setDrill(null)} />}
    </main>
  )

  function open(c: Card) {
    if (c.kind === 'exercise') setDrill(c.id)
    else if (c.kind === 'session') onSession(c.id)
    else onGo(c.to)
  }
}

// What the coach noted goes on the profile (synced to the account, private): injuries, weigh-ins, height.
function remember(s: Save) {
  if (s.kind === 'injury') addInjury(s.injury)
  else if (s.kind === 'rate') rateInjury(s.id, s.severity)
  else if (s.kind === 'healed') healInjury(s.id)
  else if (s.kind === 'weight') logBody({ kg: s.kg })
  else setProfile({ ...getProfile(), heightCm: s.cm })
}

function CoachBubble({ reply, onCard }: { reply: Reply; onCard: (c: Card) => void }) {
  return (
    <div className="rise flex max-w-[92%] flex-col gap-2 self-start">
      <div className="rounded-[20px] rounded-bl-md bg-surface px-4 py-3">
        {reply.text.map((p, i) => <p key={i} className={`m-0 whitespace-pre-wrap leading-relaxed ${i ? 'mt-2' : ''} ${i === 0 && reply.text.length > 1 && reply.text[0].length < 60 ? 'font-semibold' : ''}`}>{p}</p>)}
      </div>
      {!!reply.cards?.length && (
        <div className="flex flex-col gap-1.5">
          {reply.cards.map((c, i) => c.kind === 'go'
            ? <button key={i} onClick={() => onCard(c)} className="min-h-10 self-start rounded-full bg-surface-2 px-4 text-sm font-semibold active:scale-95">{c.label} →</button>
            : (
              <button key={i} onClick={() => onCard(c)} className="flex items-center gap-3 rounded-[16px] bg-surface-2 px-4 py-2.5 text-left active:scale-[0.98]">
                <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-full ${c.kind === 'session' ? 'bg-red text-on-red' : 'bg-surface'}`} aria-hidden="true">
                  {c.kind === 'session'
                    ? <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.4-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" /></svg>
                    : <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="5" r="2.2" /><path d="M12 8v6m0 0-3 6m3-6 3 6M7 11h10" strokeLinecap="round" /></svg>}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-semibold">{c.kind === 'session' ? t(c.title) : t(c.name)}</span>
                  <span className="block truncate text-xs text-muted">{c.sub}</span>
                </span>
                <span aria-hidden="true" className="text-muted">›</span>
              </button>
            ))}
        </div>
      )}
      {reply.after?.map((p, i) => <p key={i} className="m-0 rounded-[16px] bg-surface px-4 py-2.5 text-sm leading-relaxed text-muted">{p}</p>)}
    </div>
  )
}
