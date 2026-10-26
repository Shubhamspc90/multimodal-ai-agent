
import { useCallback, useEffect, useState } from "react";
import { apiRequest, checkApiHealth } from "./lib/api"; import {
  AudioLines,
  Bot,
  ChevronDown,
  FileText,
  Image,
  Menu,
  MessageSquare,
  Plus,
  Search,
  Settings2,
  Sparkles,
  X,
} from "lucide-react";
import "./App.css";

type Workspace = "Chat" | "Documents" | "Images" | "Audio";
type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

type TextInputResponse = {
  received_message: string;
  status?: string;
};

const workspaces: {
  name: Workspace;
  icon: typeof MessageSquare;
  description: string;
}[] = [
    { name: "Chat", icon: MessageSquare, description: "Ask questions and explore ideas." },
    { name: "Documents", icon: FileText, description: "Extract, summarize, and ask questions about files." },
    { name: "Images", icon: Image, description: "Upload an image for visual analysis." },
    { name: "Audio", icon: AudioLines, description: "Transcribe supported audio files." },
  ];

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<Workspace>("Chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [messageDraft, setMessageDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const [apiStatus, setApiStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");

  const [apiError, setApiError] = useState("");

  const checkHealth = useCallback(async () => {
    setApiStatus("checking");
    setApiError("");

    try {
      const result = await checkApiHealth();

      if (result.status !== "ok") {
        throw new Error("The API returned an unexpected health status.");
      }

      setApiStatus("connected");
    } catch (error: unknown) {
      setApiStatus("disconnected");
      setApiError(
        error instanceof Error ? error.message : "Unable to reach the API.",
      );
    }
  }, []);

  const handleSendMessage = useCallback(async () => {
    const message = messageDraft.trim();

    if (!message || isSending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      content: message,
    };

    setChatMessages((previous) => [...previous, userMessage]);
    setMessageDraft("");
    setChatError("");
    setIsSending(true);

    try {
      const response = await apiRequest<TextInputResponse>("/api/v1/text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      setChatMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: response.received_message,
        },
      ]);
    } catch (error: unknown) {
      setChatError(
        error instanceof Error ? error.message : "Unable to send your message.",
      );
    } finally {
      setIsSending(false);
    }
  }, [messageDraft, isSending]);

  useEffect(() => {
    let cancelled = false;

    const loadHealth = async () => {
      try {
        const result = await checkApiHealth();

        if (cancelled) return;

        if (result.status !== "ok") {
          throw new Error("The API returned an unexpected health status.");
        }

        setApiError("");
        setApiStatus("connected");
      } catch (error: unknown) {
        if (cancelled) return;

        setApiStatus("disconnected");
        setApiError(
          error instanceof Error ? error.message : "Unable to reach the API.",
        );
      }
    };

    void loadHealth();

    return () => {
      cancelled = true;
    };
  }, []);
  const active = workspaces.find((item) => item.name === activeWorkspace)!;
  const ActiveIcon = active.icon;

  return (
    <main className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div>
            <strong>Multimodal AI</strong>
            <span>INTELLIGENCE WORKSPACE</span>
          </div>
          <button
            className="icon-button mobile-close"
            aria-label="Close sidebar"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <button className="new-chat" onClick={() => setActiveWorkspace("Chat")}>
          <Plus size={17} />
          <span>New conversation</span>
          <span className="shortcut">⌘ K</span>
        </button>

        <div className="nav-label">WORKSPACE</div>
        <nav className="workspace-nav" aria-label="Workspace">
          {workspaces.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${activeWorkspace === name ? "active" : ""}`}
              onClick={() => {
                setActiveWorkspace(name);
                setSidebarOpen(false);
              }}
            >
              <Icon size={18} strokeWidth={1.8} />
              <span>{name}</span>
              {name === "Chat" && <span className="nav-dot" />}
            </button>
          ))}
        </nav>

        <div className="nav-label recent-label">RECENT</div>
        <div className="empty-history">
          <MessageSquare size={17} />
          <p>Your conversations will appear here.</p>
        </div>

        <div className="sidebar-bottom">
          <div className="status-card">
            <div className="status-icon"><Bot size={19} /></div>
            <div>
              <strong>AI workspace</strong>
              <span>Local development</span>
            </div>
            <span className="status-dot" />
          </div>
          <button className="nav-item settings-button">
            <Settings2 size={18} />
            <span>Settings</span>
          </button>
          <div className="profile">
            <div className="avatar">AI</div>
            <div className="profile-copy">
              <strong>Workspace user</strong>
              <span>Personal workspace</span>
            </div>
            <ChevronDown size={17} />
          </div>
        </div>
      </aside>

      <section className="main-panel">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button menu-button"
              aria-label="Open navigation"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{activeWorkspace}</strong>
          </div>
          <div className="topbar-right">
            <span className="environment-badge">
              <span className="status-dot" /> Development
            </span>

            <div
              className={`api-status api-status-${apiStatus}`}
              role="status"
              aria-live="polite"
              title={apiError || "Backend API health status"}
            >
              <span className="status-dot" />
              <span>
                {apiStatus === "checking" && "API checking"}
                {apiStatus === "connected" && "API connected"}
                {apiStatus === "disconnected" && "API disconnected"}
              </span>

              {apiStatus === "disconnected" && (
                <button
                  type="button"
                  className="api-retry"
                  onClick={() => void checkHealth()}
                  aria-label="Retry API health check"
                >
                  Retry
                </button>
              )}
            </div>
            <button className="icon-button" aria-label="Search workspace">
              <Search size={18} />
            </button>
          </div>
        </header>

        <div className="content-area">
          <div className="welcome-section">
            <div className="eyebrow">
              <Sparkles size={14} />
              YOUR MULTIMODAL ASSISTANT
            </div>
            <h1>
              One workspace.
              <br />
              <span>Every kind of intelligence.</span>
            </h1>
            <p className="welcome-description">
              Explore ideas, understand documents, analyze images, and
              work with audio — all from one place.
            </p>
          </div>

          <div className="workspace-heading">
            <div className="section-icon"><ActiveIcon size={19} /></div>
            <div>
              <h2>{activeWorkspace}</h2>
              <p>{active.description}</p>
            </div>
          </div>

          <div className="workspace-grid">
            {workspaces.map(({ name, icon: Icon, description }) => (
              <button
                key={name}
                className={`feature-card ${activeWorkspace === name ? "feature-selected" : ""}`}
                onClick={() => setActiveWorkspace(name)}
              >
                <div className={`feature-icon feature-${name.toLowerCase()}`}>
                  <Icon size={20} strokeWidth={1.8} />
                </div>
                <div className="feature-title">
                  <strong>{name}</strong>
                  <span>Explore <span aria-hidden="true">↗</span></span>
                </div>
                <p>{description}</p>
              </button>
            ))}
          </div>
          {activeWorkspace === "Chat" && (
            <section className="chat-panel" aria-label="Chat">
              <div className="chat-panel-heading">
                <div>
                  <h2>Start a conversation</h2>
                  <p>Send a message to the connected backend.</p>
                </div>
                <span className="chat-mode-badge">API demo</span>
              </div>

              <div className="chat-messages" aria-live="polite">
                {chatMessages.length === 0 && (
                  <p className="chat-empty">
                    Your conversation starts here. Send a message below.
                  </p>
                )}

                {chatMessages.map((message) => (
                  <article
                    key={message.id}
                    className={`chat-message chat-message-${message.role}`}
                  >
                    <span className="chat-message-role">
                      {message.role === "user" ? "You" : "Backend response"}
                    </span>
                    <p>{message.content}</p>
                  </article>
                ))}

                {isSending && (
                  <p className="chat-loading" role="status">
                    Sending message…
                  </p>
                )}
              </div>

              {chatError && (
                <p className="chat-error" role="alert">
                  {chatError}
                </p>
              )}

              <form
                className="chat-composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  void handleSendMessage();
                }}
              >
                <textarea
                  aria-label="Your message"
                  placeholder="Ask something..."
                  value={messageDraft}
                  onChange={(event) => setMessageDraft(event.target.value)}
                  maxLength={10000}
                  rows={2}
                  disabled={isSending}
                />

                <button
                  type="submit"
                  disabled={isSending || !messageDraft.trim()}
                >
                  {isSending ? "Sending..." : "Send message"}
                </button>
              </form>

              <p className="chat-disclaimer">
                Current backend behavior: echoes your submitted message.
              </p>
            </section>
          )}

          <div className="notice">
            <div className="notice-icon"><Bot size={18} /></div>
            <div>
              <strong>Workspace foundation is ready</strong>
              <p>
                This is the initial interface. API integration, document
                workflows, image analysis, and audio transcription will be
                added and validated incrementally.
              </p>
            </div>
          </div>

          <footer className="content-footer">
            <span><span className="footer-dot" /> Built for multimodal workflows</span>
            <span>React · TypeScript · FastAPI</span>
          </footer>
        </div>
      </section>
    </main>
  );
}

export default App;
