
import { useCallback, useEffect, useState } from "react";
import { checkApiHealth } from "./lib/api";
import {
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
