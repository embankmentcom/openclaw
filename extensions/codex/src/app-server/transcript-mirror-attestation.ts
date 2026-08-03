import { createHash } from "node:crypto";
import type { AgentMessage } from "openclaw/plugin-sdk/agent-harness-runtime";
import { readMirrorIdentity, readUpstreamUserText } from "./upstream-prompt-provenance.js";

type MirroredAgentMessage = Extract<AgentMessage, { role: "user" | "assistant" | "toolResult" }>;

const MIRROR_ORIGIN_META_KEY = "mirrorOrigin" as const;
const MIRROR_SOURCE_FINGERPRINT_META_KEY = "mirrorSourceFingerprint" as const;
const CODEX_APP_SERVER_MIRROR_ORIGIN = "codex-app-server" as const;

export type CodexCanonicalPromptEvidence = {
  idempotencyKey: string;
  mirrorIdentity: string;
  sourceFingerprint: string;
};

function stableStringify(value: unknown): string {
  return (
    JSON.stringify(value, (_key, nested) => {
      if (!nested || typeof nested !== "object" || Array.isArray(nested)) {
        return nested;
      }
      return Object.fromEntries(
        Object.entries(nested as Record<string, unknown>).toSorted(([left], [right]) =>
          left.localeCompare(right),
        ),
      );
    }) ?? "undefined"
  );
}

/** Fingerprints a canonical gateway user row without provider-owned mirror metadata. */
export function fingerprintCodexCanonicalPrompt(message: AgentMessage): string {
  const record = { ...(message as unknown as Record<string, unknown>) };
  const meta = record["__openclaw"];
  if (meta && typeof meta === "object" && !Array.isArray(meta)) {
    const canonicalMeta = { ...(meta as Record<string, unknown>) };
    delete canonicalMeta.mirrorIdentity;
    delete canonicalMeta[MIRROR_ORIGIN_META_KEY];
    delete canonicalMeta[MIRROR_SOURCE_FINGERPRINT_META_KEY];
    delete canonicalMeta.upstreamUserText;
    if (Object.keys(canonicalMeta).length > 0) {
      record["__openclaw"] = canonicalMeta;
    } else {
      delete record["__openclaw"];
    }
  }
  return createHash("sha256").update(stableStringify(record)).digest("hex").slice(0, 32);
}

export function buildCodexCanonicalPromptEvidence(
  source: AgentMessage,
  persisted: AgentMessage,
): CodexCanonicalPromptEvidence | undefined {
  const sourceKey = (source as { idempotencyKey?: unknown }).idempotencyKey;
  const persistedKey = (persisted as { idempotencyKey?: unknown }).idempotencyKey;
  const mirrorIdentity = readMirrorIdentity(source);
  if (
    source.role !== "user" ||
    persisted.role !== "user" ||
    typeof sourceKey !== "string" ||
    !sourceKey.trim() ||
    sourceKey !== persistedKey ||
    !mirrorIdentity?.endsWith(":prompt") ||
    readMirrorIdentity(persisted) ||
    fingerprintCodexCanonicalPrompt(source) !== fingerprintCodexCanonicalPrompt(persisted)
  ) {
    return undefined;
  }
  return {
    idempotencyKey: sourceKey,
    mirrorIdentity,
    sourceFingerprint: fingerprintCodexCanonicalPrompt(source),
  };
}

export function resolvePromptEvidence(
  messages: readonly AgentMessage[],
  persistedUserMessages: readonly AgentMessage[],
): CodexCanonicalPromptEvidence | undefined {
  const sources = messages.filter(
    (message) => message.role === "user" && readMirrorIdentity(message)?.endsWith(":prompt"),
  );
  const source = sources.length === 1 ? sources[0] : undefined;
  const sourceKey = (source as { idempotencyKey?: unknown } | undefined)?.idempotencyKey;
  const persisted = persistedUserMessages.find(
    (message) => (message as { idempotencyKey?: unknown }).idempotencyKey === sourceKey,
  );
  return source && persisted ? buildCodexCanonicalPromptEvidence(source, persisted) : undefined;
}

export function attachCodexMirrorAttestation(
  message: AgentMessage,
  sourceFingerprint?: string,
): AgentMessage {
  const record = message as unknown as Record<string, unknown>;
  const existing = record["__openclaw"];
  const baseMeta =
    existing && typeof existing === "object" && !Array.isArray(existing)
      ? (existing as Record<string, unknown>)
      : {};
  return {
    ...record,
    __openclaw: {
      ...baseMeta,
      [MIRROR_ORIGIN_META_KEY]: CODEX_APP_SERVER_MIRROR_ORIGIN,
      ...(sourceFingerprint ? { [MIRROR_SOURCE_FINGERPRINT_META_KEY]: sourceFingerprint } : {}),
    },
  } as unknown as AgentMessage;
}

export function readCodexMirrorSourceFingerprint(message: AgentMessage): string | undefined {
  const meta = (message as unknown as Record<string, unknown>)["__openclaw"];
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) {
    return undefined;
  }
  const value = (meta as Record<string, unknown>)[MIRROR_SOURCE_FINGERPRINT_META_KEY];
  return typeof value === "string" && value ? value : undefined;
}

export function serializeCodexMirrorSourceEvidence(message: AgentMessage): string {
  const record = message as unknown as Record<string, unknown>;
  return JSON.stringify({
    role: message.role,
    content: record.content,
    ...(message.role === "user" ? { upstreamUserText: readUpstreamUserText(message) } : {}),
    ...(message.role === "toolResult"
      ? {
          toolCallId: record.toolCallId,
          toolName: record.toolName,
          isError: record.isError === true,
        }
      : {}),
  });
}

export function fingerprintCodexMirrorSourceMessage(message: MirroredAgentMessage): string {
  return createHash("sha256")
    .update(serializeCodexMirrorSourceEvidence(message))
    .digest("hex")
    .slice(0, 32);
}
