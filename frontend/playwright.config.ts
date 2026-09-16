import { join } from 'node:path'

import { defineConfig } from '@playwright/test'

// Browser tests run the real system: the built page, the FastAPI server and the
// trained retriever. They need `npm run build` and the pipeline outputs first.
// Answers are off by default so the tests run without loading the 1.5B generator;
// set E2E_ANSWERS=1 to test real answers as well.
const PORT = 8766
// join() uses backslashes on Windows, where cmd.exe rejects `.venv/Scripts/...`.
const PYTHON = process.platform === 'win32' ? join('.venv', 'Scripts', 'python.exe') : join('.venv', 'bin', 'python')
const answersOn = process.env.E2E_ANSWERS === '1'

export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // The Chrome already installed on the machine, so no browser download is needed.
    channel: 'chrome',
  },
  webServer: {
    command: `${PYTHON} -m uvicorn docsearch.api:app --host 127.0.0.1 --port ${PORT}`,
    // Data and model paths are relative to the project root.
    cwd: '..',
    url: `http://127.0.0.1:${PORT}/health`,
    env: answersOn ? {} : { DOCSEARCH_GENERATOR: 'none' },
    timeout: 300_000,
    reuseExistingServer: false,
  },
})
