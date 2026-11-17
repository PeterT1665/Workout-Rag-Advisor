import { useEffect, useMemo, useState } from 'react'
import { qrPath } from '../lib/qr'
import { Button, Screen, Title, TopBar } from '../components/ui'
import { useProfile } from '../lib/profile'
import { syncEnabled } from '../lib/sync'
import { myInviteLink } from '../lib/invite'
import { t } from '../i18n'

// Invite a friend: a QR code to scan in person and a share link for messages. Signed in, it's your own friend QR:
// whoever scans it and makes an account (or signs in) becomes your friend straight away.
export const inviteLink = () => `${location.origin}/?invite=1`

export function Invite({ onDone, partner = false }: { onDone: () => void; partner?: boolean }) {
  const me = useProfile()
  const [friendLink, setFriendLink] = useState<string | null>(null)
  useEffect(() => { if (syncEnabled()) void myInviteLink(me.name, me.number).then(setFriendLink) }, [me.name, me.number])
  const link = friendLink ?? inviteLink()
  const [copied, setCopied] = useState(false)
  const qr = useMemo(() => qrPath(link), [link])

  const share = async () => {
    const text = friendLink ? t('Be my friend on Football Personal Coach') : partner ? t('Train with me on Football Personal Coach') : t('Try Football Personal Coach')
    try {
      if (navigator.share) { await navigator.share({ title: 'Football Personal Coach', text, url: link }); return }
    } catch { return } // share sheet closed
    try { await navigator.clipboard.writeText(link); setCopied(true) } catch { /* ignore */ }
  }

  return (
    <Screen>
      <TopBar left={<Title>{partner ? t('Train together') : t('Invite a friend')}</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>{t('Done')}</Button>} />
      <p className="mt-0 mb-5 text-muted">{partner ? t('Get your partner on the app, then run this session together.') : friendLink ? t('Your friend scans this, makes an account, and you’re on each other’s team straight away.') : t('Invite your friend to play and train football together.')}</p>
      <div className="mx-auto w-full max-w-[320px] rounded-[28px] bg-white p-5">
        <svg viewBox={`-1 -1 ${qr.n + 2} ${qr.n + 2}`} className="block w-full" role="img" aria-label={t('QR code for {link}', { link })}>
          <path d={qr.d} fill="#0b0b0c" shapeRendering="crispEdges" />
        </svg>
        <p className="m-0 mt-3 text-center font-display text-sm uppercase text-[#0b0b0c]">{friendLink ? t('Scan to be friends with {name}', { name: me.name || t('me') }) : t('Scan to join {name}', { name: me.name || t('me') })}</p>
      </div>
      <Button block className="mt-6" onClick={() => void share()}>{copied ? t('Link copied') : t('Share link')}</Button>
    </Screen>
  )
}
