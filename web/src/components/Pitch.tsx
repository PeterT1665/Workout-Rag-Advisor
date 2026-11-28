import type { Kind, Scenario } from '../lib/daily'
import { t } from '../i18n'

// A top-down pitch for the daily play: you in red, teammates white, opponents dark. We attack to the right.
const ARROW: Record<Kind, string> = { pass: '', shoot: '', dribble: '', run: '3 2', hold: '' }
export type Mark = { to: [number, number]; kind: Kind; tone: 'best' | 'wrong' }

export function Pitch({ s, marks = [], className = '' }: { s: Scenario; marks?: Mark[]; className?: string }) {
  const line = { stroke: 'rgba(255,255,255,.4)', strokeWidth: 0.35, fill: 'none' }
  return (
    <svg viewBox="-2 -2 104 68" role="img" className={className} aria-label={t('Pitch: you in red, teammates in white, opponents in dark')}>
      <defs>
        {(['best', 'wrong'] as const).map((k) => (
          <marker key={k} id={`ah-${k}`} viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0 0L6 3L0 6z" fill={k === 'best' ? '#34d399' : '#f87171'} /></marker>
        ))}
      </defs>
      <rect x="-2" y="-2" width="104" height="68" rx="3" fill="#10241a" />
      {[0, 1, 2, 3, 4].map((i) => <rect key={i} x={i * 20} y="0" width="10" height="64" fill="rgba(255,255,255,.035)" />)}
      <rect x="0" y="0" width="100" height="64" {...line} />
      <path d="M50 0V64" {...line} /><circle cx="50" cy="32" r="7" {...line} />
      <rect x="84" y="14" width="16" height="36" {...line} /><rect x="94.5" y="25" width="5.5" height="14" {...line} />
      <rect x="0" y="14" width="16" height="36" {...line} /><rect x="0" y="25" width="5.5" height="14" {...line} />
      <rect x="100" y="28" width="1.8" height="8" fill="rgba(255,255,255,.6)" /><rect x="-1.8" y="28" width="1.8" height="8" fill="rgba(255,255,255,.6)" />
      {marks.map((m, i) => (
        <line key={i} x1={s.you[0]} y1={s.you[1]} x2={m.to[0]} y2={m.to[1]} strokeWidth={m.tone === 'best' ? 1 : 0.8} strokeDasharray={ARROW[m.kind]} strokeLinecap="round"
          stroke={m.tone === 'best' ? '#34d399' : '#f87171'} markerEnd={`url(#ah-${m.tone})`} opacity={0.95} />
      ))}
      {s.opps.map((p, i) => <circle key={`o${i}`} cx={p[0]} cy={p[1]} r="2.2" fill="#0b0c0d" stroke="#aab0b6" strokeWidth="0.6" />)}
      {s.mates.map((p, i) => <circle key={`m${i}`} cx={p[0]} cy={p[1]} r="2.2" fill="#f4f4f5" stroke="#0b0c0d" strokeWidth="0.4" />)}
      <circle cx={s.you[0]} cy={s.you[1]} r="3.1" fill="none" stroke="#fff" strokeWidth="0.6" />
      <circle cx={s.you[0]} cy={s.you[1]} r="2.3" fill="var(--red, #e01b35)" />
      {s.onBall && <circle cx={s.you[0] + 2.6} cy={s.you[1] + 2.2} r="1.15" fill="#fff" stroke="#0b0c0d" strokeWidth="0.4" />}
    </svg>
  )
}
