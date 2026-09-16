/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the page runs on Vite's server, so API calls are forwarded
// to FastAPI. In production FastAPI serves the built page itself, from one origin.
const API_SERVER = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': API_SERVER,
      '/search': API_SERVER,
      '/answer': API_SERVER,
    },
  },
  test: {
    environment: 'jsdom',
    // Browser tests in e2e/ run under Playwright, not Vitest.
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['./src/setupTests.ts'],
  },
})
