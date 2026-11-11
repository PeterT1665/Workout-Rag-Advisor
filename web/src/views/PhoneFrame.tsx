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
    const fit = () => setScale(Math.min(1, (window.innerHeight - 48) / (H + BEZEL * 2)))
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])
  // carry ?invite and sign-in links (#access_token...) into the app
  const params = new URLSearchParams(location.search)
  params.set('app', '1')
  const src = `/?${params.toString()}${location.hash}`
  const qr = useMemo(() => qrPath(location.origin), [])

  return (
    <main className="grid min-h-dvh place-items-center bg-bg px-10 py-6">
      <div className="flex items-center gap-20">
        <section className="max-w-[420px]">
          <p className="m-0 font-display text-sm uppercase text-red-ink">Football Personal Coach</p>
          <h1 className="m-0 mt-3 font-display text-6xl uppercase leading-[.88]">Train like<br />a pro.</h1>
          <ul className="m-0 mt-8 flex list-none flex-col gap-3 p-0 text-lg">
            <li>🗓 A week planned around your games</li>
            <li>🔊 Drills coached out loud, LEFT / RIGHT cues</li>
            <li>🏋️ Gym sessions for speed and power</li>
          </ul>
          <div className="mt-10 flex items-center gap-5 rounded-[24px] bg-surface p-4">
            <div className="w-28 shrink-0 rounded-2xl bg-white p-2">
              <svg viewBox={`-1 -1 ${qr.n + 2} ${qr.n + 2}`} className="block w-full" role="img" aria-label="QR code to open the app on your phone">
                <path d={qr.d} fill="#0b0b0c" shapeRendering="crispEdges" />
              </svg>
            </div>
            <div>
              <p className="m-0 font-display text-base uppercase">Get it on your phone</p>
              <p className="m-0 mt-1 text-sm text-muted">Scan with your camera, then Add to Home Screen. Try it right here first.</p>
            </div>
          </div>
        </section>

        <div style={{ width: (W + BEZEL * 2) * scale, height: (H + BEZEL * 2) * scale }} className="shrink-0">
          <div style={{ width: W + BEZEL * 2, height: H + BEZEL * 2, transform: `scale(${scale})`, transformOrigin: 'top left', padding: BEZEL }}
            className="rounded-[60px] bg-[#1c1c1f] shadow-[0_30px_80px_-20px_rgba(0,0,0,.8),inset_0_0_0_2px_rgba(255,255,255,.06)]">
            <iframe src={src} title="Football Personal Coach" width={W} height={H} className="block rounded-[48px] border-0 bg-bg" allow="autoplay; screen-wake-lock" />
          </div>
        </div>
      </div>
    </main>
  )
}

// The framed view is for laptops and desktops only; inside the frame (or on phones/tablets) the app runs directly.
export const showPhoneFrame = () =>
  typeof window !== 'undefined' && window.self === window.top &&
  !new URLSearchParams(location.search).has('app') &&
  typeof matchMedia === 'function' && matchMedia('(min-width: 1024px) and (pointer: fine)').matches
