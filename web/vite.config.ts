/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
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
      workbox: { skipWaiting: true, clientsClaim: true, cleanupOutdatedCaches: true, globPatterns: ['**/*.{js,css,html,svg,png,jpg,woff2}'], globIgnores: ['personal/**'], navigateFallbackDenylist: [/^\/api\//, /^\/personal\//] },
    }),
  ],
  server: { host: '0.0.0.0', proxy: { '/api': 'http://localhost:8788' } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
