
import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiRequest,
  checkApiHealth,
  createConversation,
  getConversation,
  listConversations,
  sendConversationMessage,
} from "./lib/api";

import type {
  ConversationDetail,
  ConversationSummary,
} from "./lib/api";

import {
  AudioLines,
  Bot,
  ChevronDown,
  FileText,
  Image,
  Menu,
  MessageSquare,
  Mic,
  Plus,
  Search,
  Settings2,
  Sparkles,
  Square,
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
  status: "completed";
  input_type: "text";
  response: string;
  capability: "text";
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

type AudioTranscriptionResponse = {
  status: "transcribed";
  filename: string;
  transcript: string;
  language: string;
  language_probability: number;
  duration_seconds: number;
};

const MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024;
const SUPPORTED_AUDIO_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/vnd.wave",
];

type AudioMetadataResponse = {
  status: "accepted";
  filename: string;
  content_type: string;
  size_bytes: number;
  audio_format: "WAV";
  duration_seconds: number;
  sample_rate_hz: number;
  channels: number;
};

const VOICE_SAMPLE_RATE = 16_000;

function encodePcmChunksToWav(
  chunks: Float32Array[],
  inputSampleRate: number,
): File {
  if (!chunks.length || inputSampleRate <= 0) {
    throw new Error("No valid audio samples were captured.");
  }

  const totalSamples = chunks.reduce(
    (total, chunk) => total + chunk.length,
    0,
  );

  if (totalSamples === 0) {
    throw new Error("The voice recording is empty.");
  }

  const inputSamples = new Float32Array(totalSamples);
  let offset = 0;

  for (const chunk of chunks) {
    inputSamples.set(chunk, offset);
    offset += chunk.length;
  }

  const outputSampleCount = Math.floor(
    (totalSamples * VOICE_SAMPLE_RATE) / inputSampleRate,
  );

  const dataSize = outputSampleCount * 2;

  if (dataSize + 44 > MAX_AUDIO_SIZE_BYTES) {
    throw new Error("The voice recording exceeds the 10 MiB limit.");
  }

  if (outputSampleCount <= 0) {
    throw new Error("No valid audio samples were captured.");
  }

  const wavBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wavBuffer);

  const writeText = (position: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(position + i, value.charCodeAt(i));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, VOICE_SAMPLE_RATE, true);
  view.setUint32(28, VOICE_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataSize, true);

  const ratio = inputSampleRate / VOICE_SAMPLE_RATE;

  for (let i = 0; i < outputSampleCount; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, totalSamples - 1);
    const fraction = position - left;

    const sample =
      inputSamples[left] * (1 - fraction) +
      inputSamples[right] * fraction;

    const clamped = Math.max(-1, Math.min(1, sample));

    view.setInt16(
      44 + i * 2,
      clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff,
      true,
    );
  }

  return new File(
    [wavBuffer],
    "voice-message.wav",
    { type: "audio/wav" },
  );
}

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

const CHAT_CLIENT_ID_KEY = "multimodal-ai-agent-chat-client-id";

function getOrCreateChatClientId(): string {
  const existingId = window.localStorage.getItem(CHAT_CLIENT_ID_KEY);

  if (existingId) {
    return existingId;
  }

  const newId = window.crypto.randomUUID();
  window.localStorage.setItem(CHAT_CLIENT_ID_KEY, newId);

  return newId;
}

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<Workspace>("Chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [messageDraft, setMessageDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState("");

  const [chatClientId] = useState<string>(() => getOrCreateChatClientId());

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null,
  );

  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [isCreatingConversation, setIsCreatingConversation] = useState(false);

  const [historyError, setHistoryError] = useState("");

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

  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [isTranscribingAudio, setIsTranscribingAudio] = useState(false);
  const [audioResult, setAudioResult] = useState<AudioTranscriptionResponse | null>(null);
  const [audioError, setAudioError] = useState("");
  const [isPreviewingAudio, setIsPreviewingAudio] = useState(false);
  const [audioMetadata, setAudioMetadata] = useState<AudioMetadataResponse | null>(null);
  const [audioMetadataError, setAudioMetadataError] = useState("");

  const [voiceRecordingStatus, setVoiceRecordingStatus] = useState<
    "idle" | "recording" | "processing"
  >("idle");

  const [voiceError, setVoiceError] = useState("");
  const [voiceNotice, setVoiceNotice] = useState("");

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioProcessorRef = useRef<AudioWorkletNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Float32Array[]>([]);

  const flushResolveRef = useRef<(() => void) | null>(null);

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

  const refreshConversations = useCallback(async () => {
    const result = await listConversations(chatClientId);
    setConversations(result);
    return result;
  }, [chatClientId]);

  const handleNewConversation = useCallback(async () => {
    if (isCreatingConversation || isSending) return;

    setActiveWorkspace("Chat");
    setSidebarOpen(false);
    setChatError("");
    setHistoryError("");
    setMessageDraft("");
    setIsCreatingConversation(true);

    try {
      const conversation = await createConversation(chatClientId);

      setConversations((previous) => [
        conversation,
        ...previous.filter((item) => item.id !== conversation.id),
      ]);

      setActiveConversationId(conversation.id);
      setChatMessages([]);
    } catch (error: unknown) {
      setChatError(
        error instanceof Error
          ? error.message
          : "Unable to create a new conversation.",
      );
    } finally {
      setIsCreatingConversation(false);
    }
  }, [chatClientId, isCreatingConversation, isSending]);

  const handleSelectConversation = useCallback(
    async (conversationId: string) => {
      if (isLoadingConversation || isSending) return;

      setIsLoadingConversation(true);
      setHistoryError("");
      setChatError("");

      try {
        const conversation: ConversationDetail = await getConversation(
          conversationId,
          chatClientId,
        );

        setActiveConversationId(conversation.id);

        setChatMessages(
          conversation.messages.map((message) => ({
            id: message.id,
            role: message.role,
            content: message.content,
          })),
        );

        setActiveWorkspace("Chat");
        setSidebarOpen(false);
      } catch (error: unknown) {
        setHistoryError(
          error instanceof Error
            ? error.message
            : "Unable to load this conversation.",
        );
      } finally {
        setIsLoadingConversation(false);
      }
    },
    [chatClientId, isLoadingConversation, isSending],
  );

  const handleSendMessage = useCallback(async () => {
    const message = messageDraft.trim();

    if (
      !message ||
      isSending ||
      isLoadingHistory ||
      isLoadingConversation ||
      isCreatingConversation ||
      voiceRecordingStatus !== "idle"
    ) {
      return;
    }

    const optimisticId = Date.now();

    const userMessage: ChatMessage = {
      id: optimisticId,
      role: "user",
      content: message,
    };

    setChatMessages((previous) => [...previous, userMessage]);
    setMessageDraft("");
    setChatError("");
    setHistoryError("");
    setIsSending(true);

    let conversationId = activeConversationId;

    try {
      if (!conversationId) {
        const conversation = await createConversation(chatClientId);

        conversationId = conversation.id;
        setActiveConversationId(conversation.id);

        setConversations((previous) => [
          conversation,
          ...previous.filter((item) => item.id !== conversation.id),
        ]);
      }

      const result = await sendConversationMessage(
        conversationId,
        chatClientId,
        message,
      );

      setChatMessages((previous) => [
        ...previous.filter((item) => item.id !== optimisticId),
        {
          id: result.user_message.id,
          role: "user",
          content: result.user_message.content,
        },
        {
          id: result.assistant_message.id,
          role: "assistant",
          content: result.assistant_message.content,
        },
      ]);

      try {
        await refreshConversations();
      } catch {
        setHistoryError(
          "Your message was saved, but the conversation list could not refresh.",
        );
      }
    } catch (error: unknown) {
      if (!conversationId) {
        // Keep the existing AI chat usable if PostgreSQL is unavailable
        // before a new conversation can be created.
        try {
          const response = await apiRequest<TextInputResponse>(
            "/api/v1/text",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message }),
            },
          );

          setChatMessages((previous) => [
            ...previous.filter((item) => item.id !== optimisticId),
            userMessage,
            {
              id: Date.now() + 1,
              role: "assistant",
              content: response.response,
            },
          ]);

          setHistoryError(
            "Chat history is unavailable. This exchange was not saved.",
          );
        } catch (fallbackError: unknown) {
          setChatMessages((previous) =>
            previous.filter((item) => item.id !== optimisticId),
          );

          setChatError(
            fallbackError instanceof Error
              ? fallbackError.message
              : "Unable to send your message.",
          );
        }
      } else {
        // The backend saves the user message before calling the AI.
        // Recover it when AI generation fails.
        try {
          const conversation = await getConversation(
            conversationId,
            chatClientId,
          );

          setChatMessages(
            conversation.messages.map((item) => ({
              id: item.id,
              role: item.role,
              content: item.content,
            })),
          );

          await refreshConversations();
        } catch {
          // Preserve the currently displayed messages and original error.
        }

        setChatError(
          error instanceof Error
            ? error.message
            : "Unable to complete your message.",
        );
      }
    } finally {
      setIsSending(false);
    }
  }, [
    messageDraft,
    isSending,
    isLoadingHistory,
    isLoadingConversation,
    isCreatingConversation,
    activeConversationId,
    chatClientId,
    refreshConversations,
    voiceRecordingStatus,

  ]);

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

  const handleAudioTranscription = useCallback(async () => {
    if (isTranscribingAudio) return;

    if (!selectedAudio) {
      setAudioError("Please select a WAV audio file first.");
      return;
    }

    if (!selectedAudio.name.toLowerCase().endsWith(".wav")) {
      setAudioError("Only WAV audio files are supported.");
      return;
    }

    if (selectedAudio.size === 0) {
      setAudioError("The selected audio file is empty.");
      return;
    }

    if (selectedAudio.size > MAX_AUDIO_SIZE_BYTES) {
      setAudioError("The audio file must be 10 MiB or smaller.");
      return;
    }

    if (
      selectedAudio.type &&
      !SUPPORTED_AUDIO_TYPES.includes(selectedAudio.type.toLowerCase())
    ) {
      setAudioError("Please select a supported WAV audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedAudio);

    setAudioError("");
    setAudioResult(null);
    setIsTranscribingAudio(true);

    try {
      const response = await apiRequest<AudioTranscriptionResponse>(
        "/api/v1/audio/transcribe",
        {
          method: "POST",
          body: formData,
        },
      );

      setAudioResult(response);
    } catch (error: unknown) {
      setAudioError(
        error instanceof Error
          ? error.message
          : "Unable to transcribe the selected audio file.",
      );
    } finally {
      setIsTranscribingAudio(false);
    }
  }, [selectedAudio, isTranscribingAudio]);

  const transcribeVoiceRecording = useCallback(async (wavFile: File) => {
    setVoiceRecordingStatus("processing");
    setVoiceError("");
    setVoiceNotice("");

    try {
      if (!wavFile.size || wavFile.size > MAX_AUDIO_SIZE_BYTES) {
        throw new Error("The WAV recording must be between 1 byte and 10 MiB.");
      }

      const formData = new FormData();
      formData.append("file", wavFile);

      const response = await apiRequest<AudioTranscriptionResponse>(
        "/api/v1/audio/transcribe",
        {
          method: "POST",
          body: formData,
        },
      );

      const transcript = response.transcript.trim();

      if (!transcript) {
        throw new Error("No speech was detected. Please try again.");
      }

      setMessageDraft(transcript);

      setVoiceNotice(
        `Transcript ready · ${response.language} · ` +
        `${(response.language_probability * 100).toFixed(1)}% confidence. ` +
        "Review and edit before sending.",
      );
    } catch (error: unknown) {
      setVoiceError(
        error instanceof Error
          ? error.message
          : "Unable to transcribe the voice recording.",
      );
    } finally {
      setVoiceRecordingStatus("idle");
    }
  }, []);

  const releaseMicrophone = useCallback(() => {
    const processor = audioProcessorRef.current;
    const source = audioSourceRef.current;
    const gain = silentGainRef.current;
    const context = audioContextRef.current;

    if (processor) {
      processor.port.onmessage = null;
      processor.disconnect();
    }

    source?.disconnect();
    gain?.disconnect();

    mediaStreamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });

    mediaStreamRef.current = null;
    audioProcessorRef.current = null;
    audioSourceRef.current = null;
    silentGainRef.current = null;
    audioContextRef.current = null;

    if (context && context.state !== "closed") {
      void context.close().catch(() => {
        // Ignore cleanup failures.
      });
    }
  }, []);

  const handleStartVoiceRecording = useCallback(async () => {
    if (
      voiceRecordingStatus !== "idle" ||
      isSending ||
      isLoadingHistory ||
      isLoadingConversation ||
      isCreatingConversation
    ) {
      return;
    }

    setVoiceError("");
    setVoiceNotice("");

    if (
      !navigator.mediaDevices?.getUserMedia ||
      typeof AudioWorkletNode === "undefined"
    ) {
      setVoiceError("Voice recording is not supported by this browser.");
      return;
    }

    if (!window.isSecureContext) {
      setVoiceError(
        "Microphone access requires HTTPS or secure localhost.",
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 1 },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaStreamRef.current = stream;
      recordedChunksRef.current = [];

      const context = new AudioContext();

      await context.audioWorklet.addModule(
        "/pcm-recorder-processor.js",
      );

      await context.resume();

      audioContextRef.current = context;

      await context.audioWorklet.addModule(
        "/pcm-recorder-processor.js",
      );

      await context.resume();

      const source = context.createMediaStreamSource(stream);

      const processor = new AudioWorkletNode(
        context,
        "pcm-recorder-processor",
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          channelCountMode: "explicit",
          channelInterpretation: "speakers",
          outputChannelCount: [1],
        },
      );

      const gain = context.createGain();
      gain.gain.value = 0;

      processor.port.onmessage = (event) => {
        const data: unknown = event.data;

        if (data instanceof Float32Array) {
          recordedChunksRef.current.push(data);
          return;
        }

        if (
          typeof data === "object" &&
          data !== null &&
          "type" in data &&
          data.type === "flushed"
        ) {
          flushResolveRef.current?.();
          flushResolveRef.current = null;
        }
      };

      source.connect(processor);
      processor.connect(gain);
      gain.connect(context.destination);

      audioSourceRef.current = source;
      audioProcessorRef.current = processor;
      silentGainRef.current = gain;

      setVoiceRecordingStatus("recording");
    } catch (error: unknown) {
      releaseMicrophone();

      setVoiceRecordingStatus("idle");

      setVoiceError(
        error instanceof Error
          ? error.message
          : "Unable to start microphone recording.",
      );
    }
  }, [
    voiceRecordingStatus,
    isSending,
    isLoadingHistory,
    isLoadingConversation,
    isCreatingConversation,
    releaseMicrophone,
  ]);

  const handleStopVoiceRecording = useCallback(async () => {
    const processor = audioProcessorRef.current;
    const context = audioContextRef.current;

    if (!processor || !context) {
      return;
    }

    setVoiceRecordingStatus("processing");

    try {
      await new Promise<void>((resolve) => {
        flushResolveRef.current = resolve;
        processor.port.postMessage({ type: "flush" });
      });

      const chunks = recordedChunksRef.current;
      recordedChunksRef.current = [];

      const inputSampleRate = context.sampleRate;

      releaseMicrophone();

      const wavFile = encodePcmChunksToWav(
        chunks,
        inputSampleRate,
      );

      await transcribeVoiceRecording(wavFile);
    } catch (error: unknown) {
      releaseMicrophone();
      setVoiceRecordingStatus("idle");

      setVoiceError(
        error instanceof Error
          ? error.message
          : "Unable to process the voice recording.",
      );
    }
  }, [releaseMicrophone, transcribeVoiceRecording]);

  const handleCancelVoiceRecording = useCallback(() => {
    recordedChunksRef.current = [];

    flushResolveRef.current?.();
    flushResolveRef.current = null;

    releaseMicrophone();

    setVoiceRecordingStatus("idle");
    setVoiceError("");
    setVoiceNotice("");
  }, [releaseMicrophone]);

  const handleWorkspaceChange = useCallback(
    (workspace: Workspace) => {
      if (
        workspace !== "Chat" &&
        voiceRecordingStatus === "recording"
      ) {
        handleCancelVoiceRecording();
      }

      setActiveWorkspace(workspace);
    },
    [voiceRecordingStatus, handleCancelVoiceRecording],
  );

  const handleAudioMetadataPreview = useCallback(async () => {
    if (isPreviewingAudio || isTranscribingAudio) return;

    if (!selectedAudio) {
      setAudioMetadataError("Please select a WAV audio file first.");
      return;
    }

    if (!selectedAudio.name.toLowerCase().endsWith(".wav")) {
      setAudioMetadataError("Only WAV audio files are supported.");
      return;
    }

    if (selectedAudio.size === 0) {
      setAudioMetadataError("The selected audio file is empty.");
      return;
    }

    if (selectedAudio.size > MAX_AUDIO_SIZE_BYTES) {
      setAudioMetadataError("The audio file must be 10 MiB or smaller.");
      return;
    }

    if (
      selectedAudio.type &&
      !SUPPORTED_AUDIO_TYPES.includes(selectedAudio.type.toLowerCase())
    ) {
      setAudioMetadataError("Please select a supported WAV audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedAudio);

    setAudioMetadataError("");
    setAudioError("");
    setAudioMetadata(null);
    setIsPreviewingAudio(true);

    try {
      const response = await apiRequest<AudioMetadataResponse>(
        "/api/v1/audio",
        {
          method: "POST",
          body: formData,
        },
      );

      setAudioMetadata(response);
    } catch (error: unknown) {
      setAudioMetadataError(
        error instanceof Error
          ? error.message
          : "Unable to preview the selected audio metadata.",
      );
    } finally {
      setIsPreviewingAudio(false);
    }
  }, [selectedAudio, isPreviewingAudio, isTranscribingAudio]);

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

  useEffect(() => {
    let cancelled = false;

    const restoreLatestConversation = async () => {
      setIsLoadingHistory(true);
      setHistoryError("");

      try {
        const savedConversations = await listConversations(chatClientId);

        if (cancelled) return;

        setConversations(savedConversations);

        if (savedConversations.length === 0) {
          setActiveConversationId(null);
          setChatMessages([]);
          return;
        }

        const latestConversation = await getConversation(
          savedConversations[0].id,
          chatClientId,
        );

        if (cancelled) return;

        setActiveConversationId(latestConversation.id);

        setChatMessages(
          latestConversation.messages.map((message) => ({
            id: message.id,
            role: message.role,
            content: message.content,
          })),
        );
      } catch (error: unknown) {
        if (cancelled) return;

        setHistoryError(
          error instanceof Error
            ? error.message
            : "Unable to restore conversation history.",
        );
      } finally {
        if (!cancelled) {
          setIsLoadingHistory(false);
        }
      }
    };

    void restoreLatestConversation();

    return () => {
      cancelled = true;
    };
  }, [chatClientId]);
  useEffect(() => {
    return () => {
      recordedChunksRef.current = [];

      flushResolveRef.current?.();
      flushResolveRef.current = null;

      releaseMicrophone();
    };
  }, [releaseMicrophone]);

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

        <button
          className="new-chat"
          onClick={() => void handleNewConversation()}
          disabled={
            isCreatingConversation ||
            isLoadingHistory ||
            isLoadingConversation ||
            isSending
          }
        >
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
                handleWorkspaceChange(name);
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

        {isLoadingHistory && (
          <div className="empty-history">
            <p>Loading conversations...</p>
          </div>
        )}

        {historyError && (
          <p className="history-error" role="alert">
            {historyError}
          </p>
        )}

        {!isLoadingHistory && conversations.length === 0 && !historyError && (
          <div className="empty-history">
            <MessageSquare size={17} />
            <p>Your conversations will appear here.</p>
          </div>
        )}

        {conversations.length > 0 && (
          <div className="conversation-list">
            {conversations.map((conversation) => (
              <button
                type="button"
                key={conversation.id}
                className={`conversation-item ${activeConversationId === conversation.id
                  ? "conversation-item-active"
                  : ""
                  }`}
                aria-current={
                  activeConversationId === conversation.id ? "true" : undefined
                }
                title={conversation.title}
                disabled={isLoadingConversation || isSending}
                onClick={() => void handleSelectConversation(conversation.id)}
              >
                <MessageSquare size={15} />
                <span>{conversation.title}</span>
              </button>
            ))}
          </div>
        )}

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
                onClick={() => handleWorkspaceChange(name)}
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
                <span className="chat-mode-badge">AI chat</span>
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
                      {message.role === "user" ? "You" : "AI assistant"}
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

              {voiceError && (
                <p className="voice-error" role="alert">
                  {voiceError}
                </p>
              )}

              {voiceRecordingStatus === "recording" && (
                <p className="voice-recording-status" role="status" aria-live="polite">
                  Recording your voice... Press Stop when finished.
                </p>
              )}

              {voiceRecordingStatus === "processing" && (
                <p className="voice-recording-status" role="status" aria-live="polite">
                  Converting audio and transcribing with Whisper...
                </p>
              )}

              {voiceNotice && voiceRecordingStatus === "idle" && (
                <p className="voice-notice" role="status" aria-live="polite">
                  {voiceNotice}
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
                  disabled={
                    isSending ||
                    isLoadingHistory ||
                    isLoadingConversation ||
                    isCreatingConversation ||
                    voiceRecordingStatus !== "idle"
                  }
                />

                {voiceRecordingStatus === "recording" ? (
                  <>
                    <button
                      type="button"
                      className="voice-control-button voice-control-recording"
                      onClick={handleStopVoiceRecording}
                      aria-label="Stop voice recording"
                      title="Stop recording"
                    >
                      <Square size={17} fill="currentColor" />
                    </button>

                    <button
                      type="button"
                      className="voice-control-button"
                      onClick={handleCancelVoiceRecording}
                      aria-label="Cancel voice recording"
                      title="Cancel recording"
                    >
                      <X size={18} />
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="voice-control-button"
                    onClick={() => void handleStartVoiceRecording()}
                    disabled={
                      voiceRecordingStatus === "processing" ||
                      isSending ||
                      isLoadingHistory ||
                      isLoadingConversation ||
                      isCreatingConversation
                    }
                    aria-label="Record voice message"
                    title="Record voice message"
                  >
                    <Mic size={18} />
                  </button>
                )}

                <button
                  type="submit"
                  disabled={
                    isSending ||
                    isLoadingHistory ||
                    isLoadingConversation ||
                    isCreatingConversation ||
                    voiceRecordingStatus !== "idle" ||
                    !messageDraft.trim()
                  }               >
                  {isSending ? "Sending..." : "Send message"}
                </button>
              </form>

              <p className="chat-disclaimer">
                Responses are generated by the connected AI model.
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

          {activeWorkspace === "Audio" && (
            <section className="audio-panel" aria-label="Audio transcription tools">
              <div className="audio-panel-heading">
                <div>
                  <h2>Transcribe your audio</h2>
                  <p>Upload a WAV recording to convert speech into text.</p>
                </div>
                <span className="chat-mode-badge">Speech to Text</span>
              </div>

              <label className="audio-upload">
                <AudioLines size={28} />
                <strong>
                  {selectedAudio ? selectedAudio.name : "Choose a WAV audio file"}
                </strong>
                <span>
                  {selectedAudio
                    ? `${(selectedAudio.size / 1024).toFixed(1)} KB`
                    : "WAV audio · Maximum 10 MiB"}
                </span>
                <input
                  type="file"
                  accept="audio/wav,audio/x-wav,audio/wave,audio/vnd.wave,.wav"
                  disabled={isTranscribingAudio}
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null;

                    setSelectedAudio(file);
                    setAudioResult(null);
                    setAudioError("");
                    setAudioMetadata(null);
                    setAudioMetadataError("");

                    event.target.value = "";
                  }}
                />
              </label>

              <div className="audio-actions">
                <span>Supported format: WAV</span>
                <button
                  type="button"
                  onClick={() => void handleAudioMetadataPreview()}
                  disabled={
                    !selectedAudio ||
                    isPreviewingAudio ||
                    isTranscribingAudio
                  }
                >
                  {isPreviewingAudio ? "Previewing metadata..." : "Preview audio metadata"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleAudioTranscription()}
                  disabled={
                    isTranscribingAudio ||
                    isPreviewingAudio ||
                    !selectedAudio
                  }
                >
                  {isTranscribingAudio ? "Transcribing..." : "Transcribe audio"}
                </button>
              </div>
              {audioMetadataError && (
                <p className="audio-error" role="alert">
                  {audioMetadataError}
                </p>
              )}

              {isPreviewingAudio && (
                <p
                  className="audio-loading"
                  role="status"
                  aria-live="polite"
                >
                  Validating the WAV file and reading its metadata...
                </p>
              )}

              {audioMetadata && (
                <section
                  className="audio-result"
                  aria-label="Audio metadata preview"
                  aria-live="polite"
                >
                  <div className="audio-result-heading">
                    <div>
                      <h3>Audio metadata</h3>
                      <p>{audioMetadata.filename}</p>
                    </div>
                  </div>

                  <div className="audio-metadata">
                    <span>Format: {audioMetadata.audio_format}</span>
                    <span>
                      Size: {(audioMetadata.size_bytes / 1024).toFixed(1)} KB
                    </span>
                    <span>
                      Duration: {audioMetadata.duration_seconds.toFixed(2)} seconds
                    </span>
                    <span>
                      Sample rate: {audioMetadata.sample_rate_hz.toLocaleString()} Hz
                    </span>
                    <span>Channels: {audioMetadata.channels}</span>
                    <span>MIME type: {audioMetadata.content_type}</span>
                  </div>
                </section>
              )}

              {audioError && (
                <p className="audio-error" role="alert">
                  {audioError}
                </p>
              )}

              {isTranscribingAudio && (
                <p className="audio-loading" role="status" aria-live="polite">
                  Transcribing your recording. This may take a little while...
                </p>
              )}

              {audioResult && (
                <section
                  className="audio-result"
                  aria-label="Audio transcription result"
                  aria-live="polite"
                >
                  <div className="audio-result-heading">
                    <div>
                      <h3>Transcription result</h3>
                      <p>{audioResult.filename}</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        void navigator.clipboard
                          .writeText(audioResult.transcript)
                          .catch(() => {
                            setAudioError(
                              "Unable to copy the transcript. Please select and copy the text manually.",
                            );
                          });
                      }}
                    >
                      Copy transcript
                    </button>
                  </div>

                  <div className="audio-metadata">
                    <span>Language: {audioResult.language}</span>
                    <span>
                      Confidence:{" "}
                      {(audioResult.language_probability * 100).toFixed(1)}%
                    </span>
                    <span>
                      Duration: {audioResult.duration_seconds.toFixed(2)} seconds
                    </span>
                  </div>

                  <pre className="audio-result-content">
                    {audioResult.transcript || "No speech was detected."}
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
                Explore the available workspaces to chat with the API, process
                documents, analyze images, and transcribe WAV audio.
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
