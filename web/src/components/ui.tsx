import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { QUOTE_BY } from '../lib/demo'

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'ghost' | 'glass'
  size?: 'md' | 'sm'
  block?: boolean
}

const VARIANTS = {
  primary: 'bg-red text-on-red active:bg-red-press shadow-[0_6px_16px_-6px_rgba(0,0,0,.45)]',
  ghost: 'bg-surface text-ink hover:bg-surface-2 active:bg-surface-2',
  glass: 'bg-white/15 text-white backdrop-blur-md active:bg-white/25',
}

export function Button({ variant = 'primary', size = 'md', block, className = '', ...rest }: ButtonProps) {
  const s = size === 'sm' ? 'min-h-10 px-4 text-xs rounded-full' : 'min-h-14 px-7 text-sm rounded-full'
  return (
    <button
      {...rest}
      className={`font-display uppercase tracking-wide transition-[transform,background-color] duration-150 ease-snap active:scale-[0.97] disabled:opacity-50 ${VARIANTS[variant]} ${s} ${block ? 'w-full' : ''} ${className}`}
    />
  )
}

export function Screen({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <main className={`mx-auto min-h-dvh max-w-xl px-4 pt-[max(16px,env(safe-area-inset-top))] pb-[max(24px,env(safe-area-inset-bottom))] ${className}`}>
      {children}
    </main>
  )
}

export function TopBar({ left, right }: { left?: ReactNode; right?: ReactNode }) {
  return <div className="mb-5 flex min-h-10 items-center justify-between gap-3">{left}{right}</div>
}

export function Pill({ children, tone = 'muted', className = '' }: { children: ReactNode; tone?: 'red' | 'muted' | 'glass'; className?: string }) {
  const t = tone === 'red' ? 'bg-red text-on-red' : tone === 'glass' ? 'bg-white/15 text-white backdrop-blur-md' : 'bg-surface-2 text-muted'
  return <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1 font-display text-[11px] uppercase tracking-wider ${t} ${className}`}>{children}</span>
}

// kept for existing screens
export const Tag = ({ children, accent }: { children: ReactNode; accent?: boolean }) => <Pill tone={accent ? 'red' : 'muted'}>{children}</Pill>

export function Title({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <h1 className={`font-display text-3xl uppercase leading-none ${className}`}>{children}</h1>
}

export function SectionHead({ title, right }: { title: string; right?: ReactNode }) {
  return (
    <div className="mt-8 mb-3 flex items-baseline justify-between gap-3">
      <h2 className="m-0 font-display text-base uppercase">{title}</h2>
      {right && <span className="text-sm font-semibold text-muted">{right}</span>}
    </div>
  )
}

export function QuoteCard({ text }: { text: string }) {
  return (
    <figure className="m-0">
      <blockquote className="m-0 font-display text-xl leading-tight normal-case">“{text}”</blockquote>
      {QUOTE_BY && <figcaption className="mt-2 text-sm text-muted">{QUOTE_BY}</figcaption>}
    </figure>
  )
}

export function Toggle({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label htmlFor={id} className="flex min-h-11 items-center justify-between gap-4">
      <span>{label}</span>
      <input id={id} type="checkbox" className="h-5 w-5 accent-[var(--red)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  )
}

export function Slider({ id, label, min, max, step, value, onChange }: {
  id: string; label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void
}) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 flex justify-between text-sm text-muted">
        <span>{label}</span><span className="tabular-nums text-ink">{value}</span>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} className="w-full accent-[var(--red)]"
        onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-[20px] bg-surface p-4 ${className}`}>{children}</section>
}
