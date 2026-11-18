/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import { defineConfig } from 'vite'

// Link previews (iMessage, WhatsApp, Messenger...) need the full address of the preview image.
// Cloudflare's build says which branch it is building: staging links preview from staging, the rest from production.
const SITE = process.env.WORKERS_CI_BRANCH === 'staging'
  ? 'https://staging-football-personal-coach.ducphu1665.workers.dev'
  : 'https://football-personal-coach.ducphu1665.workers.dev'

export default defineConfig({
  plugins: [
    { name: 'site-url', transformIndexHtml: (html) => html.replaceAll('%SITE%', SITE) },
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'prompt', // no automatic page reload; see src/lib/updates.ts
      includeAssets: ['icon-180.png', 'icon-192.png'],
      manifest: {
        name: 'Football Personal Coach',
        short_name: 'Football Coach',
        start_url: '/',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#0b0b0c',
        theme_color: '#0b0b0c',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
        ],
      },
      // never cache the API or private photos in the offline cache
      // skipWaiting + clientsClaim: a new version takes over at once, so a reload always shows it
      workbox: { importScripts: ['push-sw.js'], skipWaiting: true, clientsClaim: true, cleanupOutdatedCaches: true, globPatterns: ['**/*.{js,css,html,svg,png,jpg,woff2}'], globIgnores: ['personal/**', 'og.jpg'], navigateFallbackDenylist: [/^\/api\//, /^\/personal\//] },
    }),
  ],
  server: { host: '0.0.0.0', proxy: { '/api': 'http://localhost:8788' } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
