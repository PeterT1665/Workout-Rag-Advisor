// Push notifications (loaded into the app's service worker; see vite.config.ts workbox.importScripts).
// "Rest is over" brings you back to the session; "Plan next week" opens the week.
self.addEventListener('push', (event) => {
  let d = {}
  try { d = event.data ? event.data.json() : {} } catch { d = { body: event.data && event.data.text() } }
  event.waitUntil(self.registration.showNotification(d.title || 'Football Personal Coach', {
    body: d.body || '', tag: d.tag || 'fpc', renotify: true,
    icon: '/icon-192.png', badge: '/icon-192.png', data: { url: d.url || '/' },
  }))
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
