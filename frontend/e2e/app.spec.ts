import { expect, test, type Page } from '@playwright/test'

const answersOn = process.env.E2E_ANSWERS === '1'

async function ask(page: Page, question: string, mode: 'Ask for an answer' | 'Search the documentation') {
  await page.goto('/')
  await page.getByLabel(mode).check()
  await page.getByLabel('Your pandas question').fill(question)
  await page.getByRole('button', { name: mode === 'Ask for an answer' ? 'Ask' : 'Search' }).click()
}

test('search shows five documents from the real retriever, linked to the pinned source', async ({ page }) => {
  await ask(page, 'How do I delete rows that contain missing values?', 'Search the documentation')

  const results = page.locator('ol.results > li')
  await expect(results).toHaveCount(5)
  await expect(results.locator('code')).toContainText(['dropna'])
  await expect(results.first().getByRole('link')).toHaveAttribute(
    'href',
    /^https:\/\/github\.com\/pandas-dev\/pandas\/blob\/a183ef5779ecce1a5f3d6b766e130cf860afdfaf\/pandas\//,
  )
})

test('an empty question is stopped in the page', async ({ page }) => {
  let apiCalls = 0
  page.on('request', (request) => {
    if (request.method() === 'POST') apiCalls += 1
  })
  await page.goto('/')
  await page.getByRole('button', { name: 'Ask' }).click()

  await expect(page.getByRole('alert')).toHaveText('Type a question first.')
  expect(apiCalls).toBe(0)
})

test('asking when answers are off explains that search still works', async ({ page }) => {
  test.skip(answersOn, 'the server has answers enabled')
  await ask(page, 'How do I delete rows that contain missing values?', 'Ask for an answer')

  await expect(page.getByRole('alert')).toHaveText('Answers are turned off on this server. Search still works.')
})

test('asking returns an answer with its three sources', async ({ page }) => {
  test.skip(!answersOn, 'set E2E_ANSWERS=1 to load the generator')
  await ask(page, 'How do I delete rows that contain missing values?', 'Ask for an answer')

  await expect(page.getByRole('heading', { name: 'Answer' })).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('ol.sources > li')).toHaveCount(3)
  await expect(page.getByRole('alert')).toHaveCount(0)
})
