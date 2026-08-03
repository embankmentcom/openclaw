import {
  embeddedAgentLog,
  formatErrorMessage,
  type AgentMessage,
} from "openclaw/plugin-sdk/agent-harness-runtime";
import type { EmbeddedRunAttemptResult } from "./attempt-terminal.js";
import {
  readCodexMirroredSessionHistoryMessages,
  type CodexMirroredSessionHistoryTarget,
} from "./session-history.js";
import {
  fingerprintCodexCanonicalPrompt,
  serializeCodexMirrorSourceEvidence,
  type CodexCanonicalPromptEvidence,
} from "./transcript-mirror-attestation.js";
import { readMirrorIdentity } from "./upstream-prompt-provenance.js";

type SettledTurnFinalizationContext = EmbeddedRunAttemptResult["settledTurnFinalizationContext"];

function collectUniqueMessageIdentities(
  messages: readonly AgentMessage[],
): Map<string, number> | undefined {
  const identities = new Map<string, number>();
  for (const [index, message] of messages.entries()) {
    const identity = readMirrorIdentity(message);
    if (!identity) {
      continue;
    }
    if (identities.has(identity)) {
      return undefined;
    }
    identities.set(identity, index);
  }
  return identities;
}

/** Freezes one complete active transcript branch through the settled tool-result boundary. */
function buildCodexSettledTurnFinalizationContext(params: {
  canonicalPromptEvidence?: CodexCanonicalPromptEvidence;
  historyMessages: readonly AgentMessage[];
  mirroredMessages: readonly AgentMessage[];
  settledMessages: readonly AgentMessage[];
  turnId: string;
}): SettledTurnFinalizationContext | undefined {
  const boundaryMessage = params.settledMessages.findLast(
    (message) => message.role === "toolResult",
  );
  const boundaryIdentity = boundaryMessage ? readMirrorIdentity(boundaryMessage) : undefined;
  if (
    !boundaryMessage ||
    !boundaryIdentity ||
    !boundaryIdentity.startsWith(`${params.turnId}:tool:`)
  ) {
    return undefined;
  }

  const settledBoundaryIndex = params.settledMessages.indexOf(boundaryMessage);
  const requiredIdentities: string[] = [];
  for (const message of params.settledMessages.slice(0, settledBoundaryIndex + 1)) {
    const identity = readMirrorIdentity(message);
    if (!identity || requiredIdentities.includes(identity)) {
      return undefined;
    }
    requiredIdentities.push(identity);
  }
  if (requiredIdentities.length === 0 || !requiredIdentities.includes(`${params.turnId}:prompt`)) {
    return undefined;
  }

  const historyIdentities = collectUniqueMessageIdentities(params.historyMessages);
  const mirroredIdentities = collectUniqueMessageIdentities(params.mirroredMessages);
  if (!historyIdentities || !mirroredIdentities) {
    return undefined;
  }
  const promptIdentity = `${params.turnId}:prompt`;
  const canonicalPromptEvidence =
    params.canonicalPromptEvidence?.mirrorIdentity === promptIdentity
      ? params.canonicalPromptEvidence
      : undefined;
  const mirroredBoundaryIndex = mirroredIdentities.get(boundaryIdentity);
  if (mirroredBoundaryIndex === undefined) {
    return undefined;
  }
  const mirroredThroughBoundary = params.mirroredMessages.slice(0, mirroredBoundaryIndex + 1);
  const expectedMirroredIdentities = canonicalPromptEvidence
    ? requiredIdentities.filter((identity) => identity !== promptIdentity)
    : requiredIdentities;
  if (
    mirroredThroughBoundary.length !== expectedMirroredIdentities.length ||
    mirroredThroughBoundary.some(
      (message, index) => readMirrorIdentity(message) !== expectedMirroredIdentities[index],
    )
  ) {
    return undefined;
  }
  const historyBoundaryIndex = historyIdentities.get(boundaryIdentity);
  if (historyBoundaryIndex === undefined) {
    return undefined;
  }
  const mirroredByIdentity = new Map(
    mirroredThroughBoundary.map((message) => [readMirrorIdentity(message), message] as const),
  );
  const canonicalHistoryMatches = canonicalPromptEvidence
    ? params.historyMessages.flatMap((message, index) => {
        const idempotencyKey = (message as { idempotencyKey?: unknown }).idempotencyKey;
        return message.role === "user" &&
          idempotencyKey === canonicalPromptEvidence.idempotencyKey &&
          !readMirrorIdentity(message) &&
          fingerprintCodexCanonicalPrompt(message) === canonicalPromptEvidence.sourceFingerprint
          ? [index]
          : [];
      })
    : [];
  if (canonicalPromptEvidence && canonicalHistoryMatches.length !== 1) {
    return undefined;
  }

  let previousHistoryIndex = -1;
  for (const identity of requiredIdentities) {
    const mirroredMessage = mirroredByIdentity.get(identity);
    const historyIndex =
      identity === promptIdentity && canonicalPromptEvidence
        ? canonicalHistoryMatches[0]
        : historyIdentities.get(identity);
    const historyMessage =
      historyIndex === undefined ? undefined : params.historyMessages[historyIndex];
    if (
      historyIndex === undefined ||
      historyIndex <= previousHistoryIndex ||
      historyIndex > historyBoundaryIndex ||
      !historyMessage ||
      ((!canonicalPromptEvidence || identity !== promptIdentity) &&
        (!mirroredMessage ||
          serializeCodexMirrorSourceEvidence(historyMessage) !==
            serializeCodexMirrorSourceEvidence(mirroredMessage)))
    ) {
      return undefined;
    }
    previousHistoryIndex = historyIndex;
  }

  // Clone before returning so later transcript/cache mutation cannot change the
  // exact application evidence authorized for the isolated finalization turn.
  const messages = Object.freeze(
    structuredClone(params.historyMessages.slice(0, historyBoundaryIndex + 1)),
  );
  return { source: "openclaw-transcript", messages };
}

/** Reads and freezes the current active transcript branch after mirroring has settled. */
export async function captureCodexSettledTurnFinalizationContext(
  params: CodexMirroredSessionHistoryTarget & {
    canonicalPromptEvidence?: CodexCanonicalPromptEvidence;
    mirroredMessages: readonly AgentMessage[];
    settledMessages: readonly AgentMessage[];
    turnId: string;
  },
): Promise<SettledTurnFinalizationContext | undefined> {
  try {
    const historyMessages = await readCodexMirroredSessionHistoryMessages(params);
    if (!historyMessages) {
      return undefined;
    }
    return buildCodexSettledTurnFinalizationContext({
      ...(params.canonicalPromptEvidence
        ? { canonicalPromptEvidence: params.canonicalPromptEvidence }
        : {}),
      historyMessages,
      mirroredMessages: params.mirroredMessages,
      settledMessages: params.settledMessages,
      turnId: params.turnId,
    });
  } catch (error) {
    // Capture runs after tools have settled. Never let transcript I/O or cloning
    // bypass the caller's side-effect-aware incomplete-turn result.
    embeddedAgentLog.warn("codex settled-turn finalization context capture failed", {
      error: formatErrorMessage(error),
      turnId: params.turnId,
    });
    return undefined;
  }
}
