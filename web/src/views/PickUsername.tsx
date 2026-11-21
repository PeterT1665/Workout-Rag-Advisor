import { UsernameForm } from './TeamView'
import { useProfile } from '../lib/profile'
import { t } from '../i18n'
import { peekNewFriend } from '../lib/invite'

// Right after an account is made (or the first open of one without a username): pick the username friends
// find you by, together with your shirt number. Asked once; Team keeps the same form if you skip.
export function PickUsername({ onDone }: { onDone: () => void }) {
  const me = useProfile()
  const friend = peekNewFriend() // joined from a friend's QR
  return (
    <div className="fixed inset-0 overflow-y-auto bg-bg text-ink">
      <div className="mx-auto flex min-h-full max-w-xl flex-col px-4 pt-[max(28px,var(--sat))] pb-[max(24px,env(safe-area-inset-bottom))]">
        {friend && <p className="rise m-0 mb-4 self-start rounded-full bg-surface px-4 py-2 text-sm font-semibold">{t('You and {name} are friends now.', { name: friend })}</p>}
        <h1 className="slam m-0 font-display text-[clamp(2.6rem,13vw,4rem)] uppercase leading-[.9]">{t('Your name on the team sheet')}</h1>
        <p className="m-0 mt-3 text-muted">{t('Friends add you with your username and shirt number, so set it once now. Change your number any time in Settings → Edit player.')}</p>
        <UsernameForm profile={me} onDone={onDone} />
        <button onClick={onDone} className="mt-4 min-h-11 text-sm font-semibold text-muted">{t('Skip for now')}</button>
      </div>
    </div>
  )
}
