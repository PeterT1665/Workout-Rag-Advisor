/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import { defineConfig } from 'vite'

const demo = process.env.VITE_DEMO === '1'

export default defineConfig({
  plugins: [
    // the public demo gets its own browser-tab title
    { name: 'demo-title', transformIndexHtml: (html: string) => (demo ? html.replace('<title>Sport Personal Coach</title>', '<title>Sport Personal Coach (demo)</title>') : html) },
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon-180.png', 'icon-192.png'],
      manifest: {
        name: demo ? 'Sport Personal Coach (demo)' : 'Sport Personal Coach',
        short_name: demo ? 'Coach demo' : 'Sport Coach',
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
      workbox: { globPatterns: ['**/*.{js,css,html,svg,png,woff2}'], globIgnores: ['personal/**'], navigateFallbackDenylist: [/^\/api\//, /^\/personal\//] },
    }),
  ],
  server: { host: '0.0.0.0', proxy: { '/api': 'http://localhost:8788' } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
