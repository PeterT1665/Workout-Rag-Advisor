// Marks <img data-fade> elements as loaded so CSS can fade them in (one capture listener for the whole app).
export function fadeImagesIn() {
  if (typeof document === 'undefined') return
  const mark = (img: HTMLImageElement) => img.classList.add('is-loaded')
  document.addEventListener('load', (e) => { const t = e.target; if (t instanceof HTMLImageElement && t.dataset.fade !== undefined) mark(t) }, true)
  document.addEventListener('error', (e) => { const t = e.target; if (t instanceof HTMLImageElement && t.dataset.fade !== undefined) mark(t) }, true)
  // images that finished before React attached them (cache hits)
  new MutationObserver(() => {
    document.querySelectorAll<HTMLImageElement>('img[data-fade]:not(.is-loaded)').forEach((img) => { if (img.complete && img.naturalWidth) mark(img) })
  }).observe(document.documentElement, { childList: true, subtree: true })
}
