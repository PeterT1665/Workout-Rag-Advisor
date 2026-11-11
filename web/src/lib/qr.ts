import qrcode from 'qrcode-generator'

// QR code as one SVG path (module grid), for links people scan with their phone camera.
export function qrPath(text: string): { n: number; d: string } {
  const q = qrcode(0, 'M')
  q.addData(text)
  q.make()
  const n = q.getModuleCount()
  let d = ''
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (q.isDark(r, c)) d += `M${c} ${r}h1v1h-1z`
  return { n, d }
}
