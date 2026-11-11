import { useMemo, useState } from 'react'
import { qrPath } from '../lib/qr'
import { Button, Screen, Title, TopBar } from '../components/ui'
import { useProfile } from '../lib/profile'

// Invite a friend: a QR code to scan in person and a share link for messages. Same app, they sign up themselves.
export const inviteLink = () => `${location.origin}/?invite=1`

export function Invite({ onDone, partner = false }: { onDone: () => void; partner?: boolean }) {
  const me = useProfile()
  const link = inviteLink()
  const [copied, setCopied] = useState(false)
  const qr = useMemo(() => qrPath(link), [link])

  const share = async () => {
    const text = partner ? `Train with me on Football Personal Coach` : `Try Football Personal Coach`
    try {
      if (navigator.share) { await navigator.share({ title: 'Football Personal Coach', text, url: link }); return }
    } catch { return } // share sheet closed
    try { await navigator.clipboard.writeText(link); setCopied(true) } catch { /* ignore */ }
  }

  return (
    <Screen>
      <TopBar left={<Title>{partner ? 'Train together' : 'Invite a friend'}</Title>} right={<Button variant="ghost" size="sm" onClick={onDone}>Done</Button>} />
      <p className="mt-0 mb-5 text-muted">{partner ? 'Get your partner on the app, then run this session together.' : 'Invite your friend to play and train football together.'}</p>
      <div className="mx-auto w-full max-w-[320px] rounded-[28px] bg-white p-5">
        <svg viewBox={`-1 -1 ${qr.n + 2} ${qr.n + 2}`} className="block w-full" role="img" aria-label={`QR code for ${link}`}>
          <path d={qr.d} fill="#0b0b0c" shapeRendering="crispEdges" />
        </svg>
        <p className="m-0 mt-3 text-center font-display text-sm uppercase text-[#0b0b0c]">Scan to join {me.name || 'me'}</p>
      </div>
      <Button block className="mt-6" onClick={() => void share()}>{copied ? 'Link copied' : 'Share link'}</Button>
    </Screen>
  )
}
