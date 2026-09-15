import { useState, type FormEvent } from 'react'

import {
  ApiError,
  MAX_QUERY_CHARS,
  answer,
  search,
  type AnswerResponse,
  type SearchResponse,
  type SearchResult,
} from './api'
import './App.css'

// Search results link to the exact pandas source the corpus was built from.
const PANDAS_SOURCE = 'https://github.com/pandas-dev/pandas/blob/a183ef5779ecce1a5f3d6b766e130cf860afdfaf/'
const EXCERPT_PREVIEW_CHARS = 300

type Mode = 'answer' | 'search'
type Result = { kind: 'answer'; data: AnswerResponse } | { kind: 'search'; data: SearchResponse }

/** The docstring part of an excerpt ("DataFrame.dropna: Remove ..."), shortened for a preview. */
export function preview(result: SearchResult): string {
  const prefix = `${result.name}: `
  const text = result.excerpt.startsWith(prefix) ? result.excerpt.slice(prefix.length) : result.excerpt
  return text.length > EXCERPT_PREVIEW_CHARS ? `${text.slice(0, EXCERPT_PREVIEW_CHARS)}...` : text
}

/** "data/raw/pandas/pandas/core/frame.py" -> "pandas/core/frame.py" */
export function repositoryPath(sourceFile: string): string {
  const marker = 'pandas/pandas/'
  const index = sourceFile.indexOf(marker)
  return index === -1 ? sourceFile : sourceFile.slice(index + 'pandas/'.length)
}

function AnswerView({ data }: { data: AnswerResponse }) {
  const citedAny = data.sources.some((source) => source.cited)
  return (
    <article className="answer">
      <h2>Answer</h2>
      {data.abstained ? (
        <p className="notice">The documentation excerpts below did not answer this question.</p>
      ) : (
        <p className="answer-text">{data.answer}</p>
      )}
      {data.invalid_citations.length > 0 && (
        <p className="warning">
          The answer cites sources that do not exist: {data.invalid_citations.map((n) => `[${n}]`).join(', ')}
        </p>
      )}
      {!data.abstained && !citedAny && (
        <p className="hint">This answer does not cite a source, so check it against the documentation.</p>
      )}
      <h3>Sources given to the model</h3>
      <ol className="sources">
        {data.sources.map((source) => (
          <li key={source.number} className={source.cited ? 'cited' : undefined}>
            <span className="source-number">[{source.number}]</span> <code>{source.name}</code>{' '}
            <span className={source.cited ? 'tag' : 'tag muted'}>{source.cited ? 'cited' : 'not cited'}</span>
          </li>
        ))}
      </ol>
    </article>
  )
}

function SearchView({ data }: { data: SearchResponse }) {
  return (
    <article>
      <h2>Documentation</h2>
      <ol className="results">
        {data.results.map((result) => (
          <li key={result.rank}>
            <div className="result-head">
              <code>{result.name}</code>
              <span className="score">score {result.score.toFixed(2)}</span>
            </div>
            <p className="excerpt">{preview(result)}</p>
            <a className="source" href={PANDAS_SOURCE + repositoryPath(result.source_file)} target="_blank" rel="noreferrer">
              {repositoryPath(result.source_file)}
            </a>
          </li>
        ))}
      </ol>
    </article>
  )
}

export default function App() {
  const [text, setText] = useState('')
  const [mode, setMode] = useState<Mode>('answer')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const query = text.trim()
    if (!query) {
      setError('Type a question first.')
      setResult(null)
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(mode === 'answer' ? { kind: 'answer', data: await answer(query) } : { kind: 'search', data: await search(query) })
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const busyLabel = mode === 'answer' ? 'Writing answer...' : 'Searching...'

  return (
    <main>
      <header>
        <h1>docsearch</h1>
        <p>Ask a pandas question in your own words. Answers cite the documentation they come from.</p>
      </header>

      <form onSubmit={handleSubmit}>
        <label htmlFor="question">Your pandas question</label>
        <textarea
          id="question"
          value={text}
          maxLength={MAX_QUERY_CHARS}
          rows={3}
          placeholder="How do I delete rows that contain missing values?"
          onChange={(event) => setText(event.target.value)}
        />
        <div className="form-row">
          <fieldset>
            <legend>Mode</legend>
            <label>
              <input type="radio" name="mode" checked={mode === 'answer'} onChange={() => setMode('answer')} /> Ask for an answer
            </label>
            <label>
              <input type="radio" name="mode" checked={mode === 'search'} onChange={() => setMode('search')} /> Search the documentation
            </label>
          </fieldset>
          <span className="count">
            {text.length}/{MAX_QUERY_CHARS}
          </span>
          <button type="submit" disabled={loading}>
            {loading ? busyLabel : mode === 'answer' ? 'Ask' : 'Search'}
          </button>
        </div>
      </form>

      <section aria-live="polite">
        {error && <p role="alert" className="error">{error}</p>}
        {result?.kind === 'answer' && <AnswerView data={result.data} />}
        {result?.kind === 'search' && <SearchView data={result.data} />}
      </section>
    </main>
  )
}
