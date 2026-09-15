import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App, { preview, repositoryPath } from './App'
import type { AnswerResponse, SearchResponse } from './api'

/** Replace fetch with a fake server that returns one response, and record the requests. */
function fakeServer(status: number, body: unknown = {}) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function submit(question: string, mode: 'Ask for an answer' | 'Search the documentation' = 'Ask for an answer') {
  const user = userEvent.setup()
  render(<App />)
  await user.click(screen.getByLabelText(mode))
  if (question) {
    await user.type(screen.getByLabelText('Your pandas question'), question)
  }
  await user.click(screen.getByRole('button'))
}

const ANSWER: AnswerResponse = {
  question: 'delete empty rows',
  answer: 'Use DataFrame.dropna [1].',
  abstained: false,
  sources: [
    { number: 1, name: 'DataFrame.dropna', cited: true },
    { number: 2, name: 'DataFrame.fillna', cited: false },
    { number: 3, name: 'read_csv', cited: false },
  ],
  invalid_citations: [],
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('App', () => {
  it('does not call the server for an empty question', async () => {
    const fetchMock = fakeServer(200)
    await submit('')
    expect(await screen.findByRole('alert')).toHaveTextContent('Type a question first.')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('shows an answer and marks which sources it cited', async () => {
    const fetchMock = fakeServer(200, ANSWER)
    await submit('  delete empty rows  ')

    expect(await screen.findByText('Use DataFrame.dropna [1].')).toBeInTheDocument()
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/answer')
    expect(JSON.parse(init!.body as string)).toEqual({ question: 'delete empty rows' })

    const items = screen.getAllByRole('listitem')
    expect(items[0]).toHaveTextContent('DataFrame.dropna')
    expect(items[0]).toHaveTextContent('cited')
    expect(items[1]).toHaveTextContent('not cited')
    expect(screen.queryByText(/does not cite a source/)).not.toBeInTheDocument()
  })

  it('warns when an answer cites nothing or cites missing sources', async () => {
    fakeServer(200, {
      ...ANSWER,
      answer: 'Use dropna. See [5].',
      sources: ANSWER.sources.map((s) => ({ ...s, cited: false })),
      invalid_citations: [5],
    })
    await submit('delete empty rows')

    expect(await screen.findByText(/does not cite a source/)).toBeInTheDocument()
    expect(screen.getByText(/cites sources that do not exist: \[5\]/)).toBeInTheDocument()
  })

  it('says when the documentation did not answer the question', async () => {
    fakeServer(200, { ...ANSWER, answer: 'I could not find this in the pandas documentation.', abstained: true })
    await submit('plot with seaborn')

    expect(await screen.findByText(/did not answer this question/)).toBeInTheDocument()
    expect(screen.queryByText('I could not find this in the pandas documentation.')).not.toBeInTheDocument()
  })

  it('shows search results with scores and source links', async () => {
    const results: SearchResponse = {
      query: 'delete empty rows',
      results: [
        {
          rank: 1,
          name: 'DataFrame.dropna',
          score: 0.6149,
          excerpt: 'DataFrame.dropna: Remove missing values.',
          source_file: 'data/raw/pandas/pandas/core/frame.py',
        },
      ],
    }
    const fetchMock = fakeServer(200, results)
    await submit('delete empty rows', 'Search the documentation')

    expect(await screen.findByText('score 0.61')).toBeInTheDocument()
    expect(fetchMock.mock.calls[0][0]).toBe('/search')
    expect(screen.getByText('Remove missing values.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'pandas/core/frame.py' })).toHaveAttribute(
      'href',
      expect.stringMatching(/github\.com\/pandas-dev\/pandas\/blob\/[0-9a-f]{40}\/pandas\/core\/frame\.py$/),
    )
  })

  it.each([
    [503, 'Answers are turned off on this server. Search still works.'],
    [422, 'Please enter a question of 1 to 500 characters.'],
    [500, 'The server returned an error (HTTP 500).'],
  ])('explains an HTTP %i error', async (status, message) => {
    fakeServer(status)
    await submit('delete empty rows')
    expect(await screen.findByRole('alert')).toHaveTextContent(message)
  })

  it('explains when the server cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))))
    await submit('delete empty rows')
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not reach the server. Is it running?')
  })
})

describe('helpers', () => {
  it('previews the docstring without repeating the name and shortens long text', () => {
    const base = { rank: 1, name: 'f', score: 1, source_file: '' }
    expect(preview({ ...base, excerpt: 'f: short text' })).toBe('short text')
    expect(preview({ ...base, excerpt: `f: ${'x'.repeat(400)}` })).toBe(`${'x'.repeat(300)}...`)
  })

  it('turns a local source path into a repository path', () => {
    expect(repositoryPath('data/raw/pandas/pandas/core/frame.py')).toBe('pandas/core/frame.py')
    expect(repositoryPath('other/file.py')).toBe('other/file.py')
  })
})
