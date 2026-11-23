// Social links on your Team profile: friends (only accepted friends can read it) tap one to message you there and make
// a group chat. Handles or phone numbers are cleaned here; the link is built from them, never typed in.
export type SocialId = 'instagram' | 'tiktok' | 'snapchat' | 'facebook' | 'whatsapp' | 'zalo'
export type Socials = Partial<Record<SocialId, string>>
type Social = { id: SocialId; label: string; kind: 'handle' | 'phone'; link: (v: string) => string }

export const SOCIALS: Social[] = [
  { id: 'instagram', label: 'Instagram', kind: 'handle', link: (h) => `https://ig.me/m/${h}` }, // opens the DM
  { id: 'tiktok', label: 'TikTok', kind: 'handle', link: (h) => `https://www.tiktok.com/@${h}` },
  { id: 'snapchat', label: 'Snapchat', kind: 'handle', link: (h) => `https://www.snapchat.com/add/${h}` },
  { id: 'facebook', label: 'Messenger', kind: 'handle', link: (h) => `https://m.me/${h}` },
  { id: 'whatsapp', label: 'WhatsApp', kind: 'phone', link: (n) => `https://wa.me/${n.replace(/\D/g, '')}` },
  { id: 'zalo', label: 'Zalo', kind: 'phone', link: (n) => `https://zalo.me/${n.replace(/\D/g, '')}` },
]

// A handle: letters, numbers, dot, underscore, dash (a pasted profile link or a leading @ is fine). A phone number:
// international, + and 7-15 digits. Anything else is refused (null).
export function cleanSocial(id: SocialId, raw: string): string | null {
  const s = raw.trim()
  if (!s) return ''
  const kind = SOCIALS.find((x) => x.id === id)!.kind
  if (kind === 'phone') {
    const digits = s.replace(/[\s().-]/g, '')
    return /^\+?\d{7,15}$/.test(digits) ? (digits.startsWith('+') ? digits : `+${digits}`) : null
  }
  const h = s.replace(/^https?:\/\/(www\.)?[^/]+\/(add\/|m\/)?@?/i, '').replace(/^@/, '').replace(/[/?#].*$/, '')
  return /^[A-Za-z0-9._-]{1,30}$/.test(h) ? h : null
}
export const socialLink = (id: SocialId, v: string) => SOCIALS.find((x) => x.id === id)!.link(v)
export const hasSocials = (s?: Socials | null) => !!s && Object.values(s).some(Boolean)
