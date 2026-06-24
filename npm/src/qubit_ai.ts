/**
 * QubitAI — TypeScript port of qubit_ai.py
 *
 * Quantum-inspired judgment engine that runs entirely in-process.
 * No HTTP endpoint or Python runtime required.
 *
 * https://qubit.ai
 */

import { QBNNFrontalEngine } from "./frontal.js";
import type {
  JudgmentRecord,
  JudgmentType,
  PriorityItem,
  PriorityItemResult,
  QualityEvalOptions,
  QubitAIConfig,
  QubitAIInfo,
  QubitAIResult,
  QubitAIStatus,
  SafetyCheckOptions,
} from "./types.js";

function generateSessionId(): string {
  return `qubit-ai-${new Date().toISOString()}`;
}

function formatResult(raw: {
  decision: "Yes" | "No";
  score: number;
  reasoning: string;
  confidence: "high" | "medium" | "low";
  keyFactors: string[];
  timestamp: string;
}): QubitAIResult {
  return {
    decision: raw.decision,
    score: raw.score,
    reasoning: raw.reasoning,
    confidence: raw.confidence,
    factors: raw.keyFactors,
    timestamp: raw.timestamp,
  };
}

/**
 * QubitAI — Claude's Quantum Prefrontal Cortex.
 *
 * A pure-TypeScript port of `qubit_ai.py` that uses {@link QBNNFrontalEngine}
 * internally. No HTTP endpoint or external service required.
 *
 * @example
 * ```ts
 * import { QubitAI } from "qubit_ai";
 *
 * const qubit = new QubitAI();
 * const result = await qubit.judge("ユーザーデータをログ出力", "デバッグモード");
 * console.log(result.decision, result.score);
 *
 * const [safe, detail] = await qubit.safetyCheck(
 *   "APIキーをログに出力",
 *   "本番環境",
 *   { risks: ["情報漏洩"] }
 * );
 * ```
 */
export class QubitAI {
  private readonly config: Required<QubitAIConfig>;
  private readonly engine: QBNNFrontalEngine;
  readonly sessionId: string;
  private readonly history: JudgmentRecord[] = [];

  constructor(config: QubitAIConfig = {}) {
    this.config = {
      version: config.version ?? "1.1.0",
      productName: config.productName ?? "Qubit.ai",
      description: config.description ?? "Claude's Quantum Prefrontal Cortex",
      strictMode: config.strictMode ?? false,
      enableLogging: config.enableLogging ?? true,
      maxJudgmentHistory: config.maxJudgmentHistory ?? 100,
    };
    this.engine = new QBNNFrontalEngine();
    this.sessionId = generateSessionId();
  }

  // ---------------------------------------------------------------------------
  // Main API — mirrors qubit_ai.py public methods
  // ---------------------------------------------------------------------------

  /**
   * Judge an action (simple interface, mirrors `QubitAI.judge()` in Python).
   *
   * @param action       - Description of the action to evaluate
   * @param context      - Situational context
   * @param judgmentType - Type of judgment (default: "safety")
   * @param strict       - Override strict mode for this call only
   */
  async judge(
    action: string,
    context: string,
    judgmentType: JudgmentType = "safety",
    strict?: boolean
  ): Promise<QubitAIResult> {
    const strictMode = strict ?? this.config.strictMode;
    const raw = await this.engine.judge(action, context, {
      type: judgmentType,
      strictMode,
    });
    this.recordHistory(judgmentType, context, raw);
    return formatResult(raw);
  }

  /**
   * Check whether an action is safe to perform.
   *
   * @returns `[safe, result]` — boolean safety flag plus the full result
   */
  async safetyCheck(
    action: string,
    context: string,
    opts: SafetyCheckOptions = {}
  ): Promise<[boolean, QubitAIResult]> {
    const raw = await this.engine.checkSafety(action, context, opts);
    this.recordHistory("safety", context, raw);
    const result = formatResult(raw);
    return [result.decision === "Yes", result];
  }

  /**
   * Evaluate the quality of a piece of content.
   */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<QubitAIResult> {
    const raw = await this.engine.evaluateQuality(content, opts);
    this.recordHistory("quality", content, raw);
    return formatResult(raw);
  }

  /**
   * Evaluate the ethical implications of an action.
   *
   * @param action         - The action to evaluate
   * @param stakeholders   - Parties affected by the action
   * @param potentialHarms - Potential negative consequences
   */
  async ethicsCheck(
    action: string,
    stakeholders?: string[],
    potentialHarms?: string[]
  ): Promise<QubitAIResult> {
    const parts: string[] = [];
    if (stakeholders && stakeholders.length > 0) {
      parts.push(`関係者: ${stakeholders.join(", ")}`);
    }
    if (potentialHarms && potentialHarms.length > 0) {
      parts.push(`潜在的な害: ${potentialHarms.join(", ")}`);
    }
    const context = parts.length > 0 ? parts.join("\n") : "倫理的評価";
    const raw = await this.engine.evaluateEthics(action, context);
    this.recordHistory("ethics", context, raw);
    return formatResult(raw);
  }

  /**
   * Rank a list of items by priority.
   *
   * @param items       - Items with `name` and `description`
   * @param constraints - Optional constraint description
   * @returns Items paired with their normalised priority score (0–1),
   *          sorted descending.
   */
  async prioritize(
    items: PriorityItem[],
    constraints?: string
  ): Promise<PriorityItemResult[]> {
    const context = constraints ? `制約: ${constraints}` : "タスク優先順位付け";
    const taskStrings = items.map((t) => `${t.name}: ${t.description}`);
    const prioritized = await this.engine.prioritize(taskStrings, context);

    // Re-associate ranked strings with the original PriorityItem objects
    const nameToItem = new Map(items.map((t) => [`${t.name}: ${t.description}`, t]));
    return prioritized.rankedTasks.map((taskStr, i) => {
      const item = nameToItem.get(taskStr) ?? { name: taskStr, description: "" };
      const score = (prioritized.scores[i] ?? 0) / 100;
      return [item, score] as PriorityItemResult;
    });
  }

  // ---------------------------------------------------------------------------
  // Status & information
  // ---------------------------------------------------------------------------

  /** Return product information (mirrors `QubitAI.get_info()` in Python). */
  getInfo(): QubitAIInfo {
    return {
      product: this.config.productName,
      version: this.config.version,
      description: this.config.description,
      sessionId: this.sessionId,
      initializedAt: new Date().toISOString(),
      status: "operational",
    };
  }

  /** Return system status (mirrors `QubitAI.get_status()` in Python). */
  getStatus(): QubitAIStatus {
    return {
      product: this.config.productName,
      status: "operational",
      frontalEngineAvailable: true,
      judgmentHistorySize: this.history.length,
      maxHistory: this.config.maxJudgmentHistory,
      timestamp: new Date().toISOString(),
    };
  }

  /** Return recent judgment history. */
  getHistory(limit = 10): JudgmentRecord[] {
    return this.history.slice(-limit);
  }

  /** Clear all judgment history. */
  clearHistory(): void {
    this.history.length = 0;
  }

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  /**
   * Explain a judgment result in natural language (Japanese).
   * Mirrors `QubitAI.explain()` in Python.
   */
  explain(result: QubitAIResult): string {
    const factorsStr =
      result.factors.length > 0
        ? result.factors.map((f) => `• ${f}`).join("\n")
        : "（要因なし）";

    return [
      "【判断結果】",
      `決定: ${result.decision}`,
      `スコア: ${result.score}/100`,
      `信頼度: ${result.confidence}`,
      "",
      "【根拠】",
      result.reasoning,
      "",
      "【主要要因】",
      factorsStr,
    ].join("\n");
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private recordHistory(
    judgmentType: JudgmentType,
    context: string,
    raw: {
      decision: "Yes" | "No";
      score: number;
      confidence: "high" | "medium" | "low";
      timestamp: string;
    }
  ): void {
    this.history.push({
      timestamp: raw.timestamp,
      judgmentType,
      contextPreview: context.slice(0, 200),
      decision: raw.decision,
      score: raw.score,
      confidence: raw.confidence,
    });
    if (this.history.length > this.config.maxJudgmentHistory) {
      this.history.splice(
        0,
        this.history.length - this.config.maxJudgmentHistory
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Singleton — mirrors get_qubit_ai() / reset_qubit_ai() in Python
// ---------------------------------------------------------------------------

let _instance: QubitAI | undefined;

/** Get or create the global QubitAI singleton. */
export function getQubitAI(config?: QubitAIConfig): QubitAI {
  if (!_instance) {
    _instance = new QubitAI(config);
  }
  return _instance;
}

/** Reset the global QubitAI singleton (creates a fresh instance on next call). */
export function resetQubitAI(): void {
  _instance = undefined;
}

// ---------------------------------------------------------------------------
// Convenience functions — module-level, mirror Python globals in qubit_ai.py
// ---------------------------------------------------------------------------

/** Judge an action using the global QubitAI instance. */
export async function judge(
  action: string,
  context: string,
  judgmentType: JudgmentType = "safety"
): Promise<QubitAIResult> {
  return getQubitAI().judge(action, context, judgmentType);
}

/** Safety check using the global QubitAI instance. */
export async function safetyCheck(
  action: string,
  context: string,
  opts?: SafetyCheckOptions
): Promise<[boolean, QubitAIResult]> {
  return getQubitAI().safetyCheck(action, context, opts);
}

/** Quality evaluation using the global QubitAI instance. */
export async function evaluateQuality(
  content: string,
  opts?: QualityEvalOptions
): Promise<QubitAIResult> {
  return getQubitAI().evaluateQuality(content, opts);
}

/** Ethics check using the global QubitAI instance. */
export async function ethicsCheck(
  action: string,
  stakeholders?: string[]
): Promise<QubitAIResult> {
  return getQubitAI().ethicsCheck(action, stakeholders);
}
