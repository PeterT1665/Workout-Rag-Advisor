import { useMemo, useState } from 'react'
import { Sheet } from './Sheet'
import { Button } from './ui'
import { setProfile, useProfile } from '../lib/profile'
import { loadHistory } from '../lib/storage'
import { loadMatches } from '../lib/match'
import { SHOWCASE_MAX, SHOWCASE_MIN, resolveShowcase, showcaseOptions } from '../lib/showcase'
import { publishShowcase } from '../lib/team'
import { t } from '../i18n'

// Pick the 3 to 6 stats friends see on your card. Only stats the app has logged for you can be picked.
export function ShowcaseSheet({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const me = useProfile()
  const options = useMemo(() => showcaseOptions(loadHistory(), loadMatches(), me), [me])
  const [picked, setPicked] = useState<string[]>(() => (me.showcase ?? []).filter((k) => options.some((o) => o.k === k)))
  const toggle = (k: string) => setPicked((x) => (x.includes(k) ? x.filter((y) => y !== k) : x.length < SHOWCASE_MAX ? [...x, k] : x))
  const ok = picked.length === 0 || picked.length >= SHOWCASE_MIN
  const save = () => {
    setProfile({ ...me, showcase: picked })
    void publishShowcase(resolveShowcase(picked, loadHistory(), loadMatches(), me))
    onSaved()
  }
  return (
    <Sheet label={t('Your showcase')} onClose={onClose}>
      <h2 className="m-0 font-display text-xl uppercase">{t('Your showcase')}</h2>
      <p className="m-0 mt-1 text-sm text-muted">{t('Pick 3 to 6 stats friends see on your card.')}</p>
      {options.length === 0
        ? <p className="mt-4 text-sm text-muted">{t('Nothing to show yet. Log a match, a rating session or your sets in the gym.')}</p>
        : <ul className="m-0 mt-4 flex list-none flex-col gap-2 p-0">
          {options.map((o) => (
            <li key={o.k}>
              <button aria-pressed={picked.includes(o.k)} onClick={() => toggle(o.k)}
                className={`flex min-h-12 w-full items-center justify-between gap-3 rounded-2xl px-4 text-left active:scale-[0.98] ${picked.includes(o.k) ? 'bg-red text-on-red' : 'bg-surface-2'}`}>
                <span className="font-semibold">{o.label}</span>
                <span className="font-display tabular-nums">{o.value}</span>
              </button>
            </li>
          ))}
        </ul>}
      <div className="mt-5 flex gap-3">
        <Button variant="ghost" className="flex-1 bg-surface-2" onClick={onClose}>{t('Cancel')}</Button>
        <Button className="flex-[1.4]" disabled={!ok} onClick={save}>{picked.length === 0 || ok ? t('Save') : t('Pick {n} more', { n: SHOWCASE_MIN - picked.length })}</Button>
      </div>
    </Sheet>
  )
}
