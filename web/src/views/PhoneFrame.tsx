import { useEffect, useMemo, useState } from 'react'
import { qrPath } from '../lib/qr'

// Laptop and desktop: the app is a phone app, so the website shows the real app running inside a phone frame
// (an iframe of the same site), next to a QR code to open it on a phone. One layout to design, test and ship.
const W = 390
const H = 844
const BEZEL = 12

export function PhoneFrame() {
  const [scale, setScale] = useState(1)
  useEffect(() => {
    const fit = () => setScale(Math.min(1, (window.innerHeight - 48) / (H + BEZEL * 2), (window.innerWidth - 80 - 300 - 96) / (2 * (W + BEZEL * 2))))
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])
  // carry ?invite and sign-in links (#access_token...) into the app phone
  const frameSrc = (preview: string, hash = '') => {
    const params = new URLSearchParams(location.search)
    params.set('app', '1')
    params.set('preview', preview)
    return `/?${params.toString()}${hash}`
  }
  const qr = useMemo(() => qrPath(location.origin), [])

  const phone = (preview: string, label: string, hash = '') => (
    <div style={{ width: (W + BEZEL * 2) * scale, height: (H + BEZEL * 2) * scale }} className="shrink-0">
      <div style={{ width: W + BEZEL * 2, height: H + BEZEL * 2, transform: `scale(${scale})`, transformOrigin: 'top left', padding: BEZEL }}
        className="rounded-[60px] bg-[#1c1c1f] shadow-[0_30px_80px_-20px_rgba(0,0,0,.8),inset_0_0_0_2px_rgba(255,255,255,.06)]">
        <iframe src={frameSrc(preview, hash)} title={label} width={W} height={H} className="block rounded-[48px] border-0 bg-bg" allow="autoplay; screen-wake-lock" />
      </div>
    </div>
  )

  return (
    <main className="grid min-h-dvh place-items-center bg-bg px-10 py-6">
      <div className="flex items-center gap-12">
        {phone('welcome', 'Football Personal Coach: welcome')}
        {phone('home', 'Football Personal Coach', location.hash)}

        <section className="flex w-[280px] shrink-0 flex-col items-center text-center">
          <p className="m-0 mb-6 font-display text-lg uppercase leading-tight">Football<br /><span className="text-red-ink">Personal Coach</span></p>
          <div className="w-48 rounded-[24px] bg-white p-3">
            <svg viewBox={`-1 -1 ${qr.n + 2} ${qr.n + 2}`} className="block w-full" role="img" aria-label="QR code to open the app on your phone">
              <path d={qr.d} fill="#0b0b0c" shapeRendering="crispEdges" />
            </svg>
          </div>
          <p className="m-0 mt-4 font-display text-base uppercase">Get it on your phone</p>
          <p className="m-0 mt-1 text-sm text-muted">Scan with your camera, then Add to Home Screen.</p>
          <div className="mt-8 flex w-full flex-col gap-2.5">
            {['App Store', 'Google Play'].map((store) => (
              <div key={store} aria-disabled="true" className="flex min-h-12 items-center justify-between rounded-full border border-white/15 px-5 text-sm text-muted">
                <span className="font-semibold text-ink">{store}</span><span>Coming soon</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}

// The framed view is for laptops and desktops only; inside the frame (or on phones/tablets) the app runs directly.
export const showPhoneFrame = () =>
  typeof window !== 'undefined' && window.self === window.top &&
  !new URLSearchParams(location.search).has('app') &&
  typeof matchMedia === 'function' && matchMedia('(min-width: 1024px) and (pointer: fine)').matches
