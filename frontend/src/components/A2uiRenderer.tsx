
import {
  useEffect,
  useRef,
  useState,
} from "react";
import {
  A2uiSurface,
  type ReactCatalogComponent,
} from "@a2ui/react";
import { injectStyles } from "@a2ui/react/styles";
import { MessageProcessor } from "@a2ui/web_core";
import type { ActionListener } from "@a2ui/web_core/state";
import {
  basicCatalog,
  injectBasicCatalogStyles,
} from "@a2ui/web_core/catalogs/basic/v1";

const ALLOWED_OPERATIONS = new Set([
  "createSurface",
  "updateComponents",
  "updateDataModel",
  "deleteSurface",
]);

const ALLOWED_COMPONENTS = new Set([
  "Text",
  "Row",
  "Column",
  "List",
  "Card",
  "Divider",
  "TextField",
  "CheckBox",
  "ChoicePicker",
  "Slider",
  "DateTimeInput",
  "Button",
]);

const MAX_FORM_MESSAGE_LENGTH = 9000;
const MAX_FORM_DEPTH = 8;
const MAX_FORM_NODES = 200;
const MAX_FORM_STRING_LENGTH = 2000;
const MAX_FORM_COLLECTION_SIZE = 50;

interface RendererSnapshot {
  surfaces: ReturnType<
    MessageProcessor<ReactCatalogComponent>["getSurfaces"]
  > extends ReadonlyMap<string, infer T>
  ? T[]
  : never;
  error: string | null;
}

interface RendererRuntime {
  processor: MessageProcessor<ReactCatalogComponent>;
  publish: (error?: string | null) => void;
  updateCallbacks: (
    onFormSubmit: (message: string) => void,
    isSubmitting: boolean,
  ) => void;
}

interface SanitizationBudget {
  nodes: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function isSafeSubmitButton(
  component: Record<string, unknown>,
): boolean {
  if (component.component !== "Button") {
    return true;
  }

  if (
    typeof component.child !== "string" ||
    component.child.trim().length === 0 ||
    !isRecord(component.action)
  ) {
    return false;
  }

  const action = component.action;

  if (
    Object.keys(action).length !== 1 ||
    !isRecord(action.event)
  ) {
    return false;
  }

  const event = action.event;

  return (
    Object.keys(event).length === 1 &&
    event.name === "submit_form"
  );
}

function getMessageList(payload: unknown): readonly unknown[] | null {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (isRecord(payload) && Array.isArray(payload.messages)) {
    return payload.messages;
  }

  return null;
}

function isAllowedComponentUpdate(
  message: Record<string, unknown>,
): boolean {
  const update = message.updateComponents;

  if (!isRecord(update) || !Array.isArray(update.components)) {
    return false;
  }

  if (update.components.length === 0) {
    return false;
  }

  return update.components.every((entry: unknown) => {
    if (!isRecord(entry)) {
      return false;
    }

    if (
      typeof entry.id !== "string" ||
      entry.id.trim().length === 0 ||
      typeof entry.component !== "string"
    ) {
      return false;
    }

    return (
      ALLOWED_COMPONENTS.has(entry.component) &&
      isSafeSubmitButton(entry)
    );
  });
}

function isSafeMessageBatch(payload: unknown): boolean {
  const messages = getMessageList(payload);

  if (!messages || messages.length === 0) {
    return false;
  }

  return messages.every((entry: unknown) => {
    if (!isRecord(entry) || entry.version !== "1.0") {
      return false;
    }

    const operations = Object.keys(entry).filter(
      (key) => key !== "version",
    );

    if (operations.length !== 1) {
      return false;
    }

    const operation = operations[0];

    if (!ALLOWED_OPERATIONS.has(operation)) {
      return false;
    }

    if (operation === "updateComponents") {
      return isAllowedComponentUpdate(entry);
    }

    return true;
  });
}

function sanitizeFormValue(
  value: unknown,
  depth: number,
  budget: SanitizationBudget,
): unknown {
  budget.nodes += 1;

  if (
    budget.nodes > MAX_FORM_NODES ||
    depth > MAX_FORM_DEPTH
  ) {
    throw new Error("Form data exceeds the allowed complexity.");
  }

  if (
    value === null ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (typeof value === "string") {
    if (value.length > MAX_FORM_STRING_LENGTH) {
      throw new Error("A form value is too long.");
    }

    return value;
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("Form data contains an invalid number.");
    }

    return value;
  }

  if (Array.isArray(value)) {
    if (value.length > MAX_FORM_COLLECTION_SIZE) {
      throw new Error("A form list contains too many values.");
    }

    return value.map((item) =>
      sanitizeFormValue(item, depth + 1, budget),
    );
  }

  if (isRecord(value)) {
    const prototype = Object.getPrototypeOf(value);

    if (
      prototype !== Object.prototype &&
      prototype !== null
    ) {
      throw new Error("Form data contains an unsupported object.");
    }

    const entries = Object.entries(value);

    if (entries.length > MAX_FORM_COLLECTION_SIZE) {
      throw new Error("A form section contains too many fields.");
    }

    const sanitized: Record<string, unknown> = {};

    for (const [key, item] of entries) {
      if (
        !key.trim() ||
        key.length > 100 ||
        ["__proto__", "constructor", "prototype"].includes(key)
      ) {
        throw new Error("Form data contains an invalid field name.");
      }

      sanitized[key] = sanitizeFormValue(
        item,
        depth + 1,
        budget,
      );
    }

    return sanitized;
  }

  throw new Error("Form data contains an unsupported value.");
}

function formatFieldLabel(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (character) => character.toUpperCase());
}

function createFormSubmissionMessage(
  rawData: unknown,
): string | null {
  try {
    if (!isRecord(rawData)) {
      return null;
    }

    const sanitized = sanitizeFormValue(
      rawData,
      0,
      { nodes: 0 },
    );

    if (!isRecord(sanitized)) {
      return null;
    }

    const fields = Object.entries(sanitized);

    if (fields.length === 0) {
      return null;
    }

    const lines = fields.map(([key, value]) => {
      const formattedValue =
        typeof value === "string"
          ? value
          : JSON.stringify(value);

      return `- ${formatFieldLabel(key)}: ${formattedValue}`;
    });

    const message = `Form submission:\n${lines.join("\n")}`;

    if (message.length > MAX_FORM_MESSAGE_LENGTH) {
      return null;
    }

    return message;
  } catch {
    return null;
  }
}

export interface A2uiRendererProps {
  messages?: unknown;
  onFormSubmit: (message: string) => void;
  isSubmitting: boolean;
}

export default function A2uiRenderer({
  messages,
  onFormSubmit,
  isSubmitting,
}: A2uiRendererProps) {
  const [snapshot, setSnapshot] = useState<RendererSnapshot>({
    surfaces: [],
    error: null,
  });

  const lastProcessedMessagesRef = useRef<unknown>(undefined);

  const [runtime] = useState<RendererRuntime>(() => {
    const callbacks: {
      onFormSubmit: (message: string) => void;
      isSubmitting: boolean;
      submissionInProgress: boolean;
    } = {
      onFormSubmit: () => { },
      isSubmitting: false,
      submissionInProgress: false,
    };

    const processorHolder: {
      current: MessageProcessor<ReactCatalogComponent> | null;
    } = {
      current: null,
    };

    const actionHandler: ActionListener = (action) => {
      if (
        action.name !== "submit_form" ||
        typeof action.surfaceId !== "string" ||
        typeof action.sourceComponentId !== "string" ||
        callbacks.isSubmitting ||
        callbacks.submissionInProgress
      ) {
        return;
      }

      const surface = processorHolder.current?.getSurface(
        action.surfaceId,
      );

      if (!surface) return;

      const component = surface.componentsModel.get(
        action.sourceComponentId,
      );

      if (!component || component.type !== "Button") return;

      const configuredAction = component.properties.action;

      if (
        !isRecord(configuredAction) ||
        Object.keys(configuredAction).length !== 1 ||
        !isRecord(configuredAction.event) ||
        Object.keys(configuredAction.event).length !== 1 ||
        configuredAction.event.name !== "submit_form"
      ) {
        return;
      }

      const message = createFormSubmissionMessage(
        surface.dataModel.get("/"),
      );

      if (!message) return;

      callbacks.submissionInProgress = true;

      try {
        callbacks.onFormSubmit(message);
      } catch {
        callbacks.submissionInProgress = false;
      }
    };

    const processor = new MessageProcessor<ReactCatalogComponent>(
      [basicCatalog],
      actionHandler,
      { version: "1.0" },
    );

    processorHolder.current = processor;

    let currentError: string | null = null;

    const publish = (error: string | null = currentError) => {
      currentError = error;

      setSnapshot({
        surfaces: Array.from(processor.getSurfaces().values()),
        error,
      });
    };

    return {
      processor,
      publish,
      updateCallbacks(nextOnFormSubmit, nextIsSubmitting) {
        const wasSubmitting = callbacks.isSubmitting;

        callbacks.onFormSubmit = nextOnFormSubmit;
        callbacks.isSubmitting = nextIsSubmitting;

        if (wasSubmitting && !nextIsSubmitting) {
          callbacks.submissionInProgress = false;
        }
      },
    };
  });

  useEffect(() => {
    runtime.updateCallbacks(onFormSubmit, isSubmitting);
  }, [runtime, onFormSubmit, isSubmitting]);

  useEffect(() => {
    injectStyles();
    injectBasicCatalogStyles();
  }, []);

  useEffect(() => {
    const syncSurfaces = () => {
      runtime.publish();
    };

    const createdSubscription =
      runtime.processor.onSurfaceCreated(syncSurfaces);

    const deletedSubscription =
      runtime.processor.onSurfaceDeleted(syncSurfaces);

    return () => {
      createdSubscription.unsubscribe();
      deletedSubscription.unsubscribe();
    };
  }, [runtime]);

  useEffect(() => {
    if (messages === undefined || messages === null) {
      return;
    }

    if (lastProcessedMessagesRef.current === messages) {
      return;
    }

    lastProcessedMessagesRef.current = messages;

    if (!isSafeMessageBatch(messages)) {
      runtime.publish(
        "The A2UI message batch contains unsupported content.",
      );
      return;
    }

    const errors =
      runtime.processor.processMessagesWithErrors(messages);

    const firstError = errors[0]?.error;

    runtime.publish(
      errors.length > 0
        ? `A2UI ${firstError?.code ?? "PROCESSING_ERROR"}: ${firstError?.message ??
        "An unknown processing error occurred."
        }`
        : null,
    );
  }, [messages, runtime]);

  const validationError =
    messages !== undefined &&
      messages !== null &&
      !isSafeMessageBatch(messages)
      ? "The A2UI message batch contains unsupported content."
      : null;

  const error = validationError ?? snapshot.error;

  if (error) {
    return (
      <div className="a2ui-container" role="alert">
        {error}
      </div>
    );
  }

  if (snapshot.surfaces.length === 0) {
    return null;
  }

  return (
    <div className="a2ui-container">
      {snapshot.surfaces.map((surface) => (
        <A2uiSurface key={surface.id} surface={surface} />
      ))}
    </div>
  );
}
