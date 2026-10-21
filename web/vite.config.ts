import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' makes every asset + data request relative, so the built site works at any
// path on a static host (GitHub Pages project subpath included) with NO server of its own —
// it is plain files. The SPA fetches data/*.json as same-origin relative requests, which a
// static host (gh-pages) serves directly; no Python/Node server is needed at view time.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  test: { environment: 'jsdom', globals: true },
})
