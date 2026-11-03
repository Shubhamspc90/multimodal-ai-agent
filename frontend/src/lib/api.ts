
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(
  /\/+$/,
  "",
);

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

function getCookie(name: string): string | null {
  if (typeof document === "undefined") {
    return null;
  }

  const prefix = `${name}=`;
  const cookie = document.cookie
    .split(";")
    .map((item) => item.trim())
    .find((item) => item.startsWith(prefix));

  if (!cookie) {
    return null;
  }

  const value = cookie.slice(prefix.length);

  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

async function getErrorMessage(response: Response): Promise<string> {
  const fallback = `API request failed (${response.status})`;

  try {
    const body: unknown = await response.json();

    if (
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      return body.detail;
    }

    if (
      typeof body === "object" &&
      body !== null &&
      "message" in body &&
      typeof body.message === "string"
    ) {
      return body.message;
    }
  } catch {
    // The server may return an empty or non-JSON error response.
  }

  return fallback;
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);

  const isStateChangingMethod = ![
    "GET",
    "HEAD",
    "OPTIONS",
  ].includes(method);

  if (isStateChangingMethod) {
    const csrfToken = getCookie("multimodal_ai_csrf");

    if (csrfToken) {
      headers.set("X-CSRF-Token", csrfToken);
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    method,
    credentials: "same-origin",
    headers,
  });

  if (!response.ok) {
    throw new ApiError(
      await getErrorMessage(response),
      response.status,
    );
  }

  // Logout returns HTTP 204 with no JSON response body.
  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

// --------------------------------------------------
// Health API
// --------------------------------------------------

export interface HealthResponse {
  status: string;
}

export function checkApiHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/api/v1/health");
}

// --------------------------------------------------
// Authentication API
// --------------------------------------------------

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface LoginResponse {
  user: AuthUser;
  expires_at: string;
}

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export function registerUser(
  request: RegisterRequest,
): Promise<AuthUser> {
  return apiRequest<AuthUser>("/api/v1/auth/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export function loginUser(
  request: LoginRequest,
): Promise<LoginResponse> {
  return apiRequest<LoginResponse>("/api/v1/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export function getCurrentUser(): Promise<AuthUser> {
  return apiRequest<AuthUser>("/api/v1/auth/me");
}

export function logoutUser(): Promise<void> {
  return apiRequest<void>("/api/v1/auth/logout", {
    method: "POST",
  });
}

// --------------------------------------------------
// Conversation API
// --------------------------------------------------

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

export interface LegacyConversationClaimResponse {
  linked_conversations: number;
}

export function createConversation(
  clientId: string,
): Promise<ConversationSummary> {
  return apiRequest<ConversationSummary>("/api/v1/conversations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      client_id: clientId,
    }),
  });
}

export function claimLegacyConversations(
  clientId: string,
): Promise<LegacyConversationClaimResponse> {
  return apiRequest<LegacyConversationClaimResponse>(
    "/api/v1/conversations/claim-legacy",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        client_id: clientId,
      }),
    },
  );
}

export function listConversations(): Promise<ConversationSummary[]> {
  return apiRequest<ConversationSummary[]>("/api/v1/conversations");
}

export function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  return apiRequest<ConversationDetail>(
    `/api/v1/conversations/${encodeURIComponent(conversationId)}`,
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
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        client_id: clientId,
        message,
      }),
    },
  );
}
