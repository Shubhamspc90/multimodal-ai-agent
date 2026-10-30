
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
export interface ConversationMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  client_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ConversationMessage[];
}

export interface ConversationMessagePairResponse {
  conversation_id: string;
  user_message: ConversationMessage;
  assistant_message: ConversationMessage;
}

export function createConversation(
  clientId: string,
): Promise<ConversationSummary> {
  return apiRequest<ConversationSummary>("/api/v1/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client_id: clientId }),
  });
}

export function listConversations(
  clientId: string,
): Promise<ConversationSummary[]> {
  return apiRequest<ConversationSummary[]>(
    `/api/v1/conversations?client_id=${encodeURIComponent(clientId)}`,
  );
}

export function getConversation(
  conversationId: string,
  clientId: string,
): Promise<ConversationDetail> {
  return apiRequest<ConversationDetail>(
    `/api/v1/conversations/${encodeURIComponent(conversationId)}` +
      `?client_id=${encodeURIComponent(clientId)}`,
  );
}

export function sendConversationMessage(
  conversationId: string,
  clientId: string,
  message: string,
): Promise<ConversationMessagePairResponse> {
  return apiRequest<ConversationMessagePairResponse>(
    `/api/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: clientId,
        message,
      }),
    },
  );
}