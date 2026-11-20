// Push notifications (loaded into the app's service worker; see vite.config.ts workbox.importScripts).
// "Rest is over" brings you back to the session; "Plan next week" opens the week.
self.addEventListener('push', (event) => {
  let d = {}
  try { d = event.data ? event.data.json() : {} } catch { d = { body: event.data && event.data.text() } }
  // Each notification gets its own tag: on iPhone, reusing a tag can swap the old one out silently, with no banner.
  // Older ones of the same kind are cleared first so Notification Centre doesn't fill up with rests.
  // Show it FIRST, then tidy: iPhone drops a push (and soon stops delivering) if showNotification isn't reached,
  // and waiting on getNotifications() first could stall there.
  const kind = d.kind || 'other'
  const tag = d.tag || `fpc-${Date.now()}`
  event.waitUntil(self.registration.showNotification(d.title || 'Football Personal Coach', {
    body: d.body || '', tag, icon: '/icon-192.png', badge: '/icon-192.png', data: { url: d.url || '/', kind },
  }).then(() => Promise.race([
    self.registration.getNotifications()
      .then((old) => old.forEach((n) => { if (n.tag !== tag && n.data && n.data.kind === kind) n.close() })),
    new Promise((r) => setTimeout(r, 1000)),
  ])).catch(() => {}))
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = (event.notification.data && event.notification.data.url) || '/'
  event.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((list) => {
    const open = list.find((c) => 'focus' in c)
    if (open) {
      // a session in progress keeps its page; only "plan next week" moves to the week
      if (url !== '/' && 'navigate' in open) open.navigate(url)
      return open.focus()
    }
    return self.clients.openWindow(url)
  }))
})
