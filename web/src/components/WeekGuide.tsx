import { useState } from 'react'
import { Button, Pill } from './ui'
import { DAYS, FOCUS_LABEL, matchdayLabel, type Game, type WeekPlan } from '../lib/weekplan'

// The rules the planner follows, and where each came from (docs/training-references.md).
const RULES = [
  { title: 'Every gym day has a job', body: 'Set by how far it is from the game: legs at MD-4 or earlier, core & injury prevention at MD-3, upper body power at MD-2, a short activation at MD-1.', from: [1] },
  { title: 'Hard days hard, easy days easy', body: 'The hardest day (MD-4) puts high-intensity football, legs and speed together. The day after stays light.', from: [2] },
  { title: 'Fresh legs for the game', body: 'No heavy legs in the last three days. The day before is a 20-minute activation (isometric holds, a few fast jumps) or rest, never a hard session.', from: [1, 2] },
  { title: 'Recover first', body: 'The day after a game is for recovery (mobility, hips, hamstrings). Training starts again at MD+2.', from: [1, 2] },
  { title: 'Games carry the football load', body: '1 game: 2 football sessions. 2 games: 1. 3 games: none, just gym and recovery. Busy weeks drop heavy legs.', from: [] },
]
const REFERENCES = [
  { n: 1, who: 'A pro footballer (Sarmiento)', what: "His gym week around a Saturday game: a job for every day. Taught us MD-specific gym days and the match-eve activation." },
  { n: 2, who: '"Train like a Premier League player"', what: 'Team training + legs + speed on MD-4, a light MD-3. Taught us to stack the hard day and keep the next day easy.' },
]

const KIND_LABEL: Record<string, string> = { game: 'Game', football: 'Football', recovery: 'Recovery', cheer: 'Cheer' }

export function WeekGuide({ plan, games }: { plan: WeekPlan; games: Game[] }) {
  const [open, setOpen] = useState(false)
  const count = (k: string) => plan.slots.filter((s) => s.kind === k).length
  const summary = [
    `${count('game')} game${count('game') === 1 ? '' : 's'}`, `${count('football')} football`, `${count('gym')} gym`, `${count('recovery')} recovery`,
  ].join(' · ')
  const warnings = plan.reasons.slice(1)

  return (
    <>
      <button onClick={() => setOpen(true)} aria-haspopup="dialog"
        className="mt-4 flex w-full items-center gap-3 rounded-[16px] bg-surface px-4 py-3 text-left transition-transform duration-150 ease-snap active:scale-[0.98]">
        <span className="min-w-0 flex-1">
          <span className="block font-semibold">How your week works</span>
          <span className="block truncate text-sm text-muted">{summary}</span>
        </span>
        {warnings.length > 0 && <span className="h-2 w-2 shrink-0 rounded-full bg-red" aria-label="Has notes" />}
        <span className="shrink-0 text-red-ink" aria-hidden="true">→</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-end justify-center" role="dialog" aria-modal="true" aria-label="How your week works">
          <button aria-label="Close" className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <div className="rise relative max-h-[90dvh] w-full max-w-xl overflow-y-auto rounded-t-[28px] bg-surface px-5 pt-5 pb-[max(16px,env(safe-area-inset-bottom))]">
            <h2 className="m-0 font-display text-xl uppercase">How your week works</h2>
            <p className="m-0 mt-1 text-sm text-muted">{summary}. Built around your game{count('game') === 1 ? '' : 's'}, the way pros plan theirs.</p>

            <h3 className="m-0 mt-5 mb-2 font-display text-sm uppercase text-muted">Your week</h3>
            <ol className="m-0 flex list-none flex-col gap-1.5 p-0">
              {DAYS.map((name, day) => {
                const slots = plan.slots.filter((s) => s.day === day)
                const md = matchdayLabel(day, games)
                return (
                  <li key={day} className="flex gap-3 rounded-2xl bg-surface-2 px-3 py-2.5">
                    <span className="w-12 shrink-0 font-display text-xs uppercase text-muted">{name.slice(0, 3)}{md && <span className={`block text-[10px] ${md === 'MD' ? 'text-red-ink' : ''}`}>{md}</span>}</span>
                    <span className="min-w-0 flex-1 text-sm">
                      {slots.length === 0 && <span className="text-muted">Rest</span>}
                      {slots.map((s, i) => (
                        <span key={i} className="block">
                          <strong>{s.kind === 'gym' && s.focus ? `Gym: ${FOCUS_LABEL[s.focus]}` : KIND_LABEL[s.kind] ?? s.kind}</strong>
                          {s.note && <span className="text-muted">: {s.kind === 'game' ? `${s.note} game` : s.note}</span>}
                        </span>
                      ))}
                    </span>
                  </li>
                )
              })}
            </ol>

            {warnings.length > 0 && (
              <div className="mt-4 rounded-2xl border border-red/50 p-3 text-sm">
                {warnings.map((w, i) => <p key={i} className="m-0">{w}</p>)}
              </div>
            )}

            <h3 className="m-0 mt-6 mb-2 font-display text-sm uppercase text-muted">The rules</h3>
            <ul className="m-0 flex list-none flex-col gap-3 p-0">
              {RULES.map((r) => (
                <li key={r.title}>
                  <p className="m-0 font-semibold">{r.title} {r.from.map((n) => <Pill key={n} className="ml-1 align-middle">Ref {n}</Pill>)}</p>
                  <p className="m-0 mt-0.5 text-sm text-muted">{r.body}</p>
                </li>
              ))}
            </ul>

            <h3 className="m-0 mt-6 mb-2 font-display text-sm uppercase text-muted">Learned from</h3>
            <ul className="m-0 flex list-none flex-col gap-2 p-0">
              {REFERENCES.map((r) => (
                <li key={r.n} className="rounded-2xl bg-surface-2 px-3 py-2.5 text-sm">
                  <strong>Ref {r.n}: {r.who}</strong>
                  <span className="mt-0.5 block text-muted">{r.what}</span>
                </li>
              ))}
            </ul>

            <Button block className="mt-5" onClick={() => setOpen(false)}>Got it</Button>
          </div>
        </div>
      )}
    </>
  )
}
