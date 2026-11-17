import { Button, Panel, Screen, Slider, Title, Toggle, TopBar } from '../components/ui'
import { LanguagePicker } from '../components/LanguagePicker'
import { say } from '../lib/audio'
import type { Settings } from '../lib/types'
import { deleteAccount, signOut, syncEnabled } from '../lib/sync'
import { NotificationSettings } from '../components/Notifications'
import { safeAreaInfo } from '../lib/safeArea'
import { seedDemo, startOverDemo } from '../lib/demo'
import { positionTag, useProfile } from '../lib/profile'
import credits from '../data/credits.json'
import { t } from '../i18n'

export function SettingsView({ settings: s, onChange, onMedia, onProfile, onInvite, onDone }: {
  settings: Settings; onChange: (p: Partial<Settings>) => void; onMedia: () => void; onProfile: () => void; onInvite: () => void; onDone: () => void
}) {
  const me = useProfile()
  return (
    <Screen>
      <TopBar left={<Title>{t('Settings')}</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>{t('Done')}</Button>} />
      <Panel className="mb-6 flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="truncate font-display text-lg uppercase">{me.name} <span className="text-muted">#{me.number}</span></div>
          <div className="text-sm text-muted">{positionTag(me)} · {me.foot === 'both' ? t('Both feet') : me.foot === 'left' ? t('Left foot') : t('Right foot')}{me.idol ? ` · ${me.idol}` : ''}</div>
        </div>
        <Button variant="ghost" size="sm" className="shrink-0 bg-surface-2" onClick={onProfile}>{t('Edit player')}</Button>
      </Panel>
      <Button block className="mb-6" onClick={onInvite}>{t('Invite a friend')}</Button>
      <h2 className="mb-3 font-display text-lg uppercase">{t('Language')}</h2>
      <div className="mb-6"><LanguagePicker /></div>
      <Panel className="flex flex-col gap-4">
        <Toggle id="voice" label={t('Spoken announcements and cues')} checked={s.voice} onChange={(v) => onChange({ voice: v })} />
        <Toggle id="earbuds" label={t('AirPods control the session (press to pause, double-press to skip). Stops your music.')} checked={s.earbudControls} onChange={(v) => onChange({ earbudControls: v })} />
        <Slider id="vol" label={t('Voice volume')} min={0.2} max={1} step={0.1} value={s.volume} onChange={(v) => onChange({ volume: v })} />
        <Slider id="cmin" label={t('Shortest gap between cues (s)')} min={2} max={10} step={1} value={s.cueMin} onChange={(v) => onChange({ cueMin: v })} />
        <Slider id="cmax" label={t('Longest gap between cues (s)')} min={3} max={15} step={1} value={s.cueMax} onChange={(v) => onChange({ cueMax: v })} />
        <Button variant="ghost" className="bg-surface-2" onClick={() => { say(s, t('left'), true); setTimeout(() => say(s, t('right'), true), 900) }}>{t('Test voice')}</Button>
      </Panel>
      <p className="text-sm text-muted">{t('Cues come at random times, only in the 1v1 drills. Left/right means the defender is committing that way: go the other way.')}</p>

      <h2 className="mt-8 mb-3 font-display text-lg uppercase">{t('Notifications')}</h2>
      <NotificationSettings />

      <h2 className="mt-8 mb-3 font-display text-lg uppercase">{t('Look')}</h2>
      <Panel className="flex flex-col gap-4">
        <Toggle id="cd" label={t('Quote cards before sessions and in rests')} checked={s.cards} onChange={(v) => onChange({ cards: v })} />
        <Button variant="ghost" className="bg-surface-2" onClick={onMedia}>{t('Make it yours: photos, colour, quotes')}</Button>
      </Panel>
      {!syncEnabled() && (
          <Panel className="mt-8 flex items-center justify-between gap-4">
            <p className="m-0 text-sm text-muted">{me.sample ? t('Sample player: not a real log.') : t('Guest mode: saved in this browser.')}</p>
            <div className="flex shrink-0 flex-col gap-2">
              {me.sample && <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { seedDemo(new Date(), true); location.reload() }}>{t('Reset sample')}</Button>}
              <Button variant="ghost" size="sm" className="bg-surface-2" onClick={() => { startOverDemo(); location.reload() }}>{t('Start over')}</Button>
            </div>
          </Panel>
      )}
      {/* drill-card photos ship in every build, so credits always show */}
      {(
        <>
          {credits.length > 0 && (
            <>
              <h2 className="mt-8 mb-3 font-display text-lg uppercase">{t('Photo credits')}</h2>
              <ul className="m-0 flex list-none flex-col gap-1 p-0 text-sm text-muted">
                {(credits as { photographer: string; source: string; url: string }[]).map((c) => (
                  <li key={c.url}><a href={c.url} target="_blank" rel="noreferrer" className="underline">{c.photographer}</a> {t('on {source}', { source: c.source })}</li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
      <Panel className="mt-8">
        <h3 className="m-0 mb-2 font-display text-base uppercase">{t('Privacy')}</h3>
        <p className="m-0 text-sm text-muted">
          {syncEnabled()
            ? t('Your data is private to your account. Clips stay on this phone.')
            : t('Everything stays in this browser.')}
        </p>
      </Panel>
      {syncEnabled() && (
        <Panel className="mt-4 flex items-center justify-between gap-4">
          <p className="m-0 text-sm text-muted">{t('Signed in and synced.')}</p>
          <Button variant="ghost" size="sm" className="bg-surface-2" onClick={async () => { await signOut(); location.reload() }}>{t('Sign out')}</Button>
        </Panel>
      )}
      {syncEnabled() && (
        <button className="mt-6 min-h-11 w-full text-sm font-semibold text-red-ink"
          onClick={async () => {
            if (!confirm(t('Delete your account and all its data for good?'))) return
            if (await deleteAccount()) location.reload()
            else alert(t("Couldn't delete it. Check your connection."))
          }}>
          {t('Delete my account')}
        </button>
      )}
      {/* tiny diagnostic: lets a screenshot show how the phone reports its screen edges */}
      <p className="mt-8 text-center text-[11px] text-muted/60">
        {safeAreaInfo ? `display ${safeAreaInfo.standalone ? 'app' : safeAreaInfo.fullScreen ? 'full screen' : 'browser'} · top ${Math.round(safeAreaInfo.reported)}px${safeAreaInfo.applied !== null ? ` → ${safeAreaInfo.applied}px` : ''}` : ''}
      </p>
    </Screen>
  )
}
