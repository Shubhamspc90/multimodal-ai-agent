
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

export class ApiError extends Error {
  readonly status: number

  constructor(
    message: string,
    status: number,
  ) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function getErrorMessage(response: Response): Promise<string> {
  const fallback = `API request failed (${response.status})`

  try {
    const body: unknown = await response.json()

    if (
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof body.detail === 'string'
    ) {
      return body.detail
    }

    if (
      typeof body === 'object' &&
      body !== null &&
      'message' in body &&
      typeof body.message === 'string'
    ) {
      return body.message
    }
  } catch {
    // The server may return an empty or non-JSON error response.
  }

  return fallback
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...init.headers,
    },
  })

  if (!response.ok) {
    throw new ApiError(
      await getErrorMessage(response),
      response.status,
    )
  }

  return (await response.json()) as T
}

export interface HealthResponse {
  status: string
}

export function checkApiHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>('/api/v1/health')
}
