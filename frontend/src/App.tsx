
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
type DocumentAction = "extract" | "summarize" | "ask";

type DocumentApiResponse = {
  filename: string;
  content_type: string;
  size_bytes: number;
  document_format: "PDF" | "TXT";
  page_count: number | null;
  text?: string;
  summary?: string;
  question?: string;
  answer?: string;
};

const MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_DOCUMENT_QUESTION_LENGTH = 1000;

type ImageUnderstandingResponse = {
  status: "completed";
  filename: string;
  content_type: string;
  size_bytes: number;
  image_format: "JPEG" | "PNG" | "WEBP";
  width: number;
  height: number;
  response: string;
  capability: "image_understanding";
};
const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024;
const DEFAULT_IMAGE_PROMPT = "Describe this image in detail.";

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

  const [selectedDocument, setSelectedDocument] = useState<File | null>(null);
  const [documentQuestion, setDocumentQuestion] = useState("");
  const [documentAction, setDocumentAction] = useState<DocumentAction | null>(null);
  const [documentResult, setDocumentResult] = useState<DocumentApiResponse | null>(null);
  const [documentError, setDocumentError] = useState("");

  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePrompt, setImagePrompt] = useState(DEFAULT_IMAGE_PROMPT);
  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false);
  const [imageResult, setImageResult] = useState<ImageUnderstandingResponse | null>(null);
  const [imageError, setImageError] = useState("");

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

  const handleDocumentAction = useCallback(
    async (action: DocumentAction) => {
      if (documentAction !== null) return;

      if (!selectedDocument) {
        setDocumentError("Please select a PDF or TXT document first.");
        return;
      }

      const extension = selectedDocument.name
        .split(".")
        .pop()
        ?.toLowerCase();

      if (extension !== "pdf" && extension !== "txt") {
        setDocumentError("Only PDF and TXT documents are supported.");
        return;
      }

      if (selectedDocument.size === 0) {
        setDocumentError("The selected document is empty.");
        return;
      }

      if (selectedDocument.size > MAX_DOCUMENT_SIZE_BYTES) {
        setDocumentError("The document must not exceed 10 MiB.");
        return;
      }

      const normalizedQuestion = documentQuestion.trim();

      if (action === "ask" && !normalizedQuestion) {
        setDocumentError("Please enter a question about the document.");
        return;
      }

      if (
        action === "ask" &&
        normalizedQuestion.length > MAX_DOCUMENT_QUESTION_LENGTH
      ) {
        setDocumentError("Your question must not exceed 1,000 characters.");
        return;
      }

      const endpointByAction: Record<DocumentAction, string> = {
        extract: "/api/v1/documents/extract",
        summarize: "/api/v1/documents/summarize",
        ask: "/api/v1/documents/ask",
      };

      const formData = new FormData();
      formData.append("file", selectedDocument);

      if (action === "ask") {
        formData.append("question", normalizedQuestion);
      }

      setDocumentAction(action);
      setDocumentError("");
      setDocumentResult(null);

      try {
        const response = await apiRequest<DocumentApiResponse>(
          endpointByAction[action],
          {
            method: "POST",
            body: formData,
          },
        );

        setDocumentResult(response);
      } catch (error: unknown) {
        setDocumentError(
          error instanceof Error
            ? error.message
            : "Unable to process the document. Please try again.",
        );
      } finally {
        setDocumentAction(null);
      }
    },
    [documentAction, selectedDocument, documentQuestion],
  );

  const handleImageAnalysis = useCallback(async () => {
    if (isAnalyzingImage) return;

    if (!selectedImage) {
      setImageError("Please select an image first.");
      return;
    }

    const supportedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!supportedTypes.includes(selectedImage.type)) {
      setImageError("Please select a JPEG, PNG, or WEBP image.");
      return;
    }

    if (selectedImage.size === 0) {
      setImageError("The selected image is empty.");
      return;
    }

    if (selectedImage.size > MAX_IMAGE_SIZE_BYTES) {
      setImageError("The image must not exceed 5 MiB.");
      return;
    }

    if (!imagePrompt.trim()) {
      setImageError("Please enter a prompt for the image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);
    formData.append("prompt", imagePrompt.trim());

    setIsAnalyzingImage(true);
    setImageError("");
    setImageResult(null);

    try {
      const response = await apiRequest<ImageUnderstandingResponse>(
        "/api/v1/images",
        {
          method: "POST",
          body: formData,
        },
      );

      setImageResult(response);
    } catch (error: unknown) {
      setImageError(
        error instanceof Error
          ? error.message
          : "Unable to analyze the image. Please try again.",
      );
    } finally {
      setIsAnalyzingImage(false);
    }
  }, [isAnalyzingImage, selectedImage, imagePrompt]);


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
              {activeWorkspace === name && <span className="nav-dot" />}
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

          {activeWorkspace === "Documents" && (
            <section className="document-panel" aria-label="Document tools">
              <div className="document-panel-heading">
                <div>
                  <h2>Understand your documents</h2>
                  <p>Upload a PDF or TXT file to get started.</p>
                </div>
                <span className="chat-mode-badge">Document AI</span>
              </div>

              <label className="document-upload">
                <FileText size={26} />
                <strong>
                  {selectedDocument
                    ? selectedDocument.name
                    : "Choose a document"}
                </strong>
                <span>
                  {selectedDocument
                    ? `${(selectedDocument.size / 1024).toFixed(1)} KB`
                    : "PDF or TXT · Maximum 10 MiB"}
                </span>
                <input
                  type="file"
                  accept=".pdf,.txt,application/pdf,text/plain"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setSelectedDocument(file);
                    setDocumentResult(null);
                    setDocumentError("");
                    event.target.value = "";
                  }}
                  disabled={documentAction !== null}
                />
              </label>

              <div className="document-actions">
                <button
                  type="button"
                  onClick={() => void handleDocumentAction("extract")}
                  disabled={documentAction !== null}
                >
                  {documentAction === "extract"
                    ? "Extracting..."
                    : "Extract text"}
                </button>

                <button
                  type="button"
                  onClick={() => void handleDocumentAction("summarize")}
                  disabled={documentAction !== null}
                >
                  {documentAction === "summarize"
                    ? "Summarizing..."
                    : "Summarize"}
                </button>
              </div>

              <form
                className="document-question-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  void handleDocumentAction("ask");
                }}
              >
                <label htmlFor="document-question">
                  Ask a question about this document
                </label>
                <textarea
                  id="document-question"
                  value={documentQuestion}
                  onChange={(event) => setDocumentQuestion(event.target.value)}
                  placeholder="e.g. What are the main points?"
                  rows={3}
                  maxLength={MAX_DOCUMENT_QUESTION_LENGTH}
                  disabled={documentAction !== null}
                />
                <div className="document-question-footer">
                  <span>{documentQuestion.length}/1000 characters</span>
                  <button
                    type="submit"
                    disabled={
                      documentAction !== null || !documentQuestion.trim()
                    }
                  >
                    {documentAction === "ask"
                      ? "Thinking..."
                      : "Ask question"}
                  </button>
                </div>
              </form>

              {documentError && (
                <p className="document-error" role="alert">
                  {documentError}
                </p>
              )}

              {documentAction !== null && (
                <p className="document-loading" role="status" aria-live="polite">
                  {documentAction === "extract" && "Extracting document text..."}
                  {documentAction === "summarize" && "Generating summary..."}
                  {documentAction === "ask" && "Finding an answer..."}
                </p>
              )}

              {documentResult && (
                <section
                  className="document-result"
                  aria-live="polite"
                  aria-label="Document result"
                >
                  <div className="document-result-heading">
                    <div>
                      <h3>
                        {documentResult.text !== undefined && "Extracted text"}
                        {documentResult.summary !== undefined && "Summary"}
                        {documentResult.answer !== undefined && "Answer"}
                      </h3>
                      <p>
                        {documentResult.filename} ·{" "}
                        {documentResult.document_format}
                        {documentResult.page_count != null &&
                          ` · ${documentResult.page_count} pages`}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        const content =
                          documentResult.text ??
                          documentResult.summary ??
                          documentResult.answer ??
                          "";
                        void navigator.clipboard.writeText(content);
                      }}
                    >
                      Copy result
                    </button>
                  </div>

                  {documentResult.question && (
                    <p className="document-result-question">
                      <strong>Question:</strong> {documentResult.question}
                    </p>
                  )}

                  <pre className="document-result-content">
                    {documentResult.text ??
                      documentResult.summary ??
                      documentResult.answer ??
                      "The API returned no result text."}
                  </pre>
                </section>
              )}
            </section>
          )}

          {activeWorkspace === "Images" && (
            <section className="image-panel" aria-label="Image analysis tools">
              <div className="image-panel-heading">
                <div>
                  <h2>Understand your images</h2>
                  <p>Upload an image and ask the vision model about it.</p>
                </div>
                <span className="chat-mode-badge">Image AI</span>
              </div>

              <label className="image-upload">
                <Image size={28} />
                <strong>
                  {selectedImage ? selectedImage.name : "Choose an image"}
                </strong>
                <span>
                  {selectedImage
                    ? `${(selectedImage.size / 1024).toFixed(1)} KB`
                    : "JPEG, PNG or WEBP · Maximum 5 MiB"}
                </span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
                  disabled={isAnalyzingImage}
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;
                    setSelectedImage(file);
                    setImageResult(null);
                    setImageError("");
                    event.target.value = "";
                  }}
                />
              </label>

              <form
                className="image-prompt-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  void handleImageAnalysis();
                }}
              >
                <label htmlFor="image-prompt">
                  What would you like to know about this image?
                </label>
                <textarea
                  id="image-prompt"
                  value={imagePrompt}
                  onChange={(event) => setImagePrompt(event.target.value)}
                  placeholder="Describe this image in detail."
                  rows={3}
                  maxLength={1000}
                  disabled={isAnalyzingImage}
                />
                <div className="image-prompt-footer">
                  <span>{imagePrompt.length}/1000 characters</span>
                  <button
                    type="submit"
                    disabled={
                      isAnalyzingImage ||
                      !selectedImage ||
                      !imagePrompt.trim()
                    }
                  >
                    {isAnalyzingImage ? "Analyzing..." : "Analyze image"}
                  </button>
                </div>
              </form>

              {imageError && (
                <p className="image-error" role="alert">
                  {imageError}
                </p>
              )}

              {isAnalyzingImage && (
                <p className="image-loading" role="status" aria-live="polite">
                  Analyzing your image. This may take a little while...
                </p>
              )}

              {imageResult && (
                <section
                  className="image-result"
                  aria-label="Image analysis result"
                  aria-live="polite"
                >
                  <div className="image-result-heading">
                    <div>
                      <h3>Analysis result</h3>
                      <p>
                        {imageResult.filename} · {imageResult.image_format} ·{" "}
                        {imageResult.width} × {imageResult.height}px
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        void navigator.clipboard
                          .writeText(imageResult.response)
                          .catch(() => {
                            setImageError(
                              "Unable to copy the result. Please select and copy the text manually.",
                            );
                          });
                      }}
                    >
                      Copy result
                    </button>
                  </div>

                  <pre className="image-result-content">
                    {imageResult.response}
                  </pre>
                </section>
              )}
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
