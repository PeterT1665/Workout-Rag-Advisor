import { useEffect, useState } from 'react'
import { Button, Screen } from '../components/ui'
import { dropInvite, inviteInfo, laterInvite, pendingInvite, type Inviter } from '../lib/invite'
import { t } from '../i18n'
import { tr } from '../i18n/rich'

// Opened from a friend's QR: who invited you, then make an account (or sign in) and you're friends straight away.
export function FriendInvite({ onSignUp, onSignIn, onSkip }: { onSignUp: () => void; onSignIn: () => void; onSkip: () => void }) {
  const [who, setWho] = useState<Inviter | null | undefined>(undefined) // undefined: loading
  useEffect(() => { const c = pendingInvite(); if (c) void inviteInfo(c).then(setWho); else setWho(null) }, [])
  const skip = () => { dropInvite(); onSkip() } // invite not found: nothing to keep
  const later = () => { laterInvite(); onSkip() } // keep it: accepted once they have an account

  if (who === undefined) return <div className="fixed inset-0 bg-bg" aria-busy="true" />
  if (!who) {
    return (
      <Screen className="flex flex-col justify-center">
        <h1 className="m-0 font-display text-3xl uppercase leading-none">{t('Invite not found')}</h1>
        <p className="mt-3 text-muted">{t('Ask your friend to show their QR again.')}</p>
        <Button block className="mt-6" onClick={skip}>{t('Continue')}</Button>
      </Screen>
    )
  }
  const name = who.name || t('A friend')
  return (
    <Screen className="flex flex-col">
      <div className="mt-[18vh] grid h-24 w-24 place-items-center rounded-full bg-red font-display text-4xl text-on-red">{name.charAt(0).toUpperCase()}</div>
      <h1 className="rise m-0 mt-6 font-display text-4xl uppercase leading-[.95]">
        {tr('{name} invited you to be <r>friends</r>', {}, { name: `${name}${who.number !== null ? ` #${who.number}` : ''}` })}
      </h1>
      <p className="mt-3 text-muted">{t('Make a free account and you’re on each other’s team: streaks, ratings and who trained today.')}</p>
      <div className="mt-auto flex flex-col gap-2 pb-2">
        <Button block onClick={onSignUp}>{t('Create account')}</Button>
        <Button block variant="ghost" className="bg-surface-2" onClick={onSignIn}>{t('I have an account')}</Button>
        <button onClick={later} className="min-h-11 text-sm font-semibold text-muted">{t('Try the app first')}</button>
      </div>
    </Screen>
  )
}
