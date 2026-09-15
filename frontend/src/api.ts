// Types mirror the response models in src/docsearch/api.py.

export const MAX_QUERY_CHARS = 500

export interface SearchResult {
  rank: number
  name: string
  score: number
  excerpt: string
  source_file: string
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
}

export interface AnswerSource {
  number: number
  name: string
  cited: boolean
}

export interface AnswerResponse {
  question: string
  answer: string
  abstained: boolean
  sources: AnswerSource[]
  invalid_citations: number[]
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** A message a person can act on, for each way a request can fail. */
export function messageFor(status: number): string {
  switch (status) {
    case 0:
      return 'Could not reach the server. Is it running?'
    case 422:
      return `Please enter a question of 1 to ${MAX_QUERY_CHARS} characters.`
    case 503:
      return 'Answers are turned off on this server. Search still works.'
    default:
      return `The server returned an error (HTTP ${status}).`
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    // fetch only throws when no response arrives at all, such as a stopped server.
    throw new ApiError(0, messageFor(0))
  }
  if (!response.ok) {
    throw new ApiError(response.status, messageFor(response.status))
  }
  return (await response.json()) as T
}

export function search(query: string, k = 5): Promise<SearchResponse> {
  return post<SearchResponse>('/search', { query, k })
}

export function answer(question: string): Promise<AnswerResponse> {
  return post<AnswerResponse>('/answer', { question })
}
