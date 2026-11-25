
import {
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  A2uiSurface,
  type ReactCatalogComponent,
} from "@a2ui/react";
import { injectStyles } from "@a2ui/react/styles";
import { MessageProcessor } from "@a2ui/web_core";
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
]);

interface RendererSnapshot {
  surfaces: ReturnType<
    MessageProcessor<ReactCatalogComponent>["getSurfaces"]
  > extends ReadonlyMap<string, infer T>
  ? T[]
  : never;
  error: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
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

    return ALLOWED_COMPONENTS.has(entry.component);
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

export interface A2uiRendererProps {
  messages?: unknown;
}

export default function A2uiRenderer({
  messages,
}: A2uiRendererProps) {
  const [runtime] = useState(() => {
    const processor = new MessageProcessor<ReactCatalogComponent>(
      [basicCatalog],
      undefined,
      { version: "1.0" },
    );

    let snapshot: RendererSnapshot = {
      surfaces: Array.from(processor.getSurfaces().values()),
      error: null,
    };

    const listeners = new Set<() => void>();

    const publish = (error: string | null = snapshot.error) => {
      snapshot = {
        surfaces: Array.from(processor.getSurfaces().values()),
        error,
      };

      listeners.forEach((listener) => listener());
    };

    return {
      processor,

      subscribe(listener: () => void) {
        listeners.add(listener);

        return () => {
          listeners.delete(listener);
        };
      },

      getSnapshot() {
        return snapshot;
      },

      publish,
    };
  });

  const snapshot = useSyncExternalStore(
    runtime.subscribe,
    runtime.getSnapshot,
    runtime.getSnapshot,
  );

  const lastProcessedMessagesRef = useRef<unknown>(undefined);

  const validationError =
    messages !== undefined &&
      messages !== null &&
      !isSafeMessageBatch(messages)
      ? "The A2UI message batch contains unsupported content."
      : null;

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
        ? `A2UI ${firstError?.code ?? "PROCESSING_ERROR"}: ${firstError?.message ?? "An unknown processing error occurred."
        }`
        : null,
    );
  }, [messages, runtime]);

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
