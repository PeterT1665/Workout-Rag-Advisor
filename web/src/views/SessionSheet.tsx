import { useState } from 'react'
import { Sheet } from '../components/Sheet'
import { LiftsEditor } from '../components/LiftsEditor'
import { LiftSummary, effortWord } from '../components/LiftSummary'
import { Button } from '../components/ui'
import { bestsFrom, planLifts, scoreLifts, type LiftLog } from '../lib/lifts'
import { editHistory, loadHistory, type HistoryEntry } from '../lib/storage'
import { templateFor, toSession } from '../lib/library'
import { bodyWeight } from '../lib/body'
import { t } from '../i18n'

// One logged session: minutes, effort, load, and every set. Sets and effort can be added or fixed later.
export function SessionSheet({ entry, date, onClose, onSaved }: { entry: HistoryEntry; date: string; onClose: () => void; onSaved: () => void }) {
  const all = loadHistory()
  const at = all.findIndex((x) => x === entry || (x.date === entry.date && x.id === entry.id && x.minutes === entry.minutes && x.title === entry.title))
  const prior = bestsFrom(all.slice(0, Math.max(0, at)), entry.bw)
  const scored = entry.rpe ? scoreLifts(entry.lifts ?? [], entry.rpe, prior, entry.bw) : { lifts: [], measured: null }
  // sets to add: what was logged, or the session's plan filled in with your earlier numbers
  const tpl = templateFor(entry.id)
  const planned = () => (entry.lifts?.length ? entry.lifts : tpl ? planLifts(toSession(tpl), all.slice(0, Math.max(0, at))) : [])
  const [editing, setEditing] = useState(false)
  const [lifts, setLifts] = useState<LiftLog[]>(planned)
  const [rpe, setRpe] = useState<number | null>(entry.rpe ?? null)
  const canLog = lifts.length > 0

  const save = () => {
    if (!rpe) return
    editHistory(entry, { rpe, lifts, bw: entry.bw ?? bodyWeight() })
    onSaved()
  }

  return (
    <Sheet label={t(entry.title)} onClose={onClose}>
      <p className="m-0 text-sm text-muted">{date}</p>
      <h2 className="m-0 mt-1 font-display text-2xl uppercase leading-tight">{t(entry.title)}</h2>

      {!editing ? (
        <>
          <dl className="m-0 mt-4 grid grid-cols-3 divide-x divide-white/10 rounded-[18px] bg-surface-2 py-3">
            {[[t('Minutes'), String(entry.minutes)], [t('Effort'), entry.effort ? entry.effort.toFixed(1) : entry.rpe ? String(entry.rpe) : '–'], [t('Load'), entry.load ? String(entry.load) : '–']].map(([k, v]) => (
              <div key={k} className="px-4">
                <dd className="m-0 font-display text-2xl tabular-nums">{v}</dd>
                <dt className="mt-0.5 text-xs text-muted">{k}</dt>
              </div>
            ))}
          </dl>
          {entry.rpe && scored.measured !== null && (
            <p className="m-0 mt-2 text-sm text-muted">{t('Effort {e}: you said {rpe}, your sets say {m}.', { e: entry.effort ?? entry.rpe, rpe: entry.rpe, m: scored.measured })}</p>
          )}
          {scored.lifts.length > 0 && <div className="mt-4"><LiftSummary lifts={scored.lifts} /></div>}
          {!entry.rpe && <p className="m-0 mt-4 text-sm text-muted">{t('No check-in for this session yet.')}</p>}
          {(canLog || !entry.rpe) && (
            <Button block className="mt-4" onClick={() => setEditing(true)}>
              {entry.lifts?.length ? t('Edit sets and effort') : canLog ? t('Add your sets') : t('Add your effort')}
            </Button>
          )}
          <Button block variant="ghost" className="mt-2" onClick={onClose}>{t('Close')}</Button>
        </>
      ) : (
        <>
          {canLog && <>
            <h3 className="m-0 mt-5 mb-2 font-semibold">{t('Your sets')}</h3>
            <LiftsEditor lifts={lifts} onChange={setLifts} />
          </>}
          <h3 className="m-0 mt-5 mb-2 font-semibold">{t('How hard was the whole session?')}{rpe ? <span className="ml-2 font-normal text-muted">{effortWord(rpe)}</span> : null}</h3>
          <div className="grid grid-cols-5 gap-2">
            {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
              <button key={n} onClick={() => setRpe(n)} aria-pressed={rpe === n} aria-label={t('Effort {n}', { n })}
                className={`h-12 rounded-2xl font-display text-lg tabular-nums active:scale-[0.94] ${rpe === n ? 'bg-[var(--red)] text-white' : 'bg-white/10'}`}>
                {n}
              </button>
            ))}
          </div>
          <Button block className="mt-5" disabled={!rpe} onClick={save}>{t('Save')}</Button>
          <Button block variant="ghost" className="mt-2" onClick={() => setEditing(false)}>{t('Cancel')}</Button>
        </>
      )}
    </Sheet>
  )
}
