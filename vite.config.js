import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  cacheDir: '/tmp/vite-cache',   // forces cache into Linux tmp, not C:\Windows
  server: {
    host: '0.0.0.0',             // expose to network so Windows browser can reach it
    port: 5173,
  },
  resolve: {
    alias: {
      '@': path.resolve('/home/sharon/skin-cancer-detection-ai/frontend/src'),
    },
  },
})
