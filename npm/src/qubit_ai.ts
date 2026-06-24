/**
 * QubitAI — TypeScript port of qubit_ai.py
 *
 * Uses NeuroQuantumEngine (pure-TS port of neuroquantum_layered.py) for
 * judgment inference.  No HTTP endpoints, no Python runtime required.
 *
 * https://qubit.ai
 */

import { NeuroQuantumEngine } from "./neuroquantum.js";
import type {
  JudgmentRecord,
  JudgmentResult,
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

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

function fromJudgmentResult(raw: JudgmentResult): QubitAIResult {
  return {
    decision: raw.decision,
    score: raw.score,
    reasoning: raw.reasoning,
    confidence: raw.confidence,
    factors: raw.keyFactors,
    timestamp: raw.timestamp,
  };
}

function generateSessionId(): string {
  return `qubit-ai-${new Date().toISOString()}`;
}

// ---------------------------------------------------------------------------
// QubitAI
// ---------------------------------------------------------------------------

/**
 * QubitAI — Claude's Quantum Prefrontal Cortex.
 *
 * All judgment inference is performed by {@link NeuroQuantumEngine}, a
 * pure-TypeScript port of `neuroquantum_layered.py` that implements:
 *   - QBNN entanglement correction with dynamic sinusoidal λ
 *   - Multi-head QBNN attention (action × context)
 *   - APQB scoring: r = cos(2θ), score = ((r+1)/2) × 100
 *
 * No HTTP endpoint is used.
 *
 * @example
 * ```ts
 * import { QubitAI } from "qubit_ai";
 *
 * const qubit = new QubitAI();
 * const result = await qubit.judge("APIキーをログに記録", "本番環境", "safety");
 * console.log(result.decision, result.score, result.reasoning);
 *
 * const [safe] = await qubit.safetyCheck("安全な操作", "テスト環境");
 * ```
 */
export class QubitAI {
  private readonly config: Required<Omit<QubitAIConfig, "neuroQuantumConfig">>;
  private readonly engine: NeuroQuantumEngine;
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

    this.engine = new NeuroQuantumEngine({ numLayers: 3, numHeads: 4, lambdaEntangle: 0.5 });
    this.sessionId = generateSessionId();
  }

  // ---------------------------------------------------------------------------
  // Main API
  // ---------------------------------------------------------------------------

  /**
   * Judge an action using the NeuroQuantum engine.
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
    const raw = await this.engine.judge(action, context, { type: judgmentType, strictMode });
    const result = fromJudgmentResult(raw);
    this.recordHistory(judgmentType, context, result);
    return result;
  }

  /**
   * Check whether an action is safe to perform.
   *
   * @returns `[safe, result]` — boolean flag plus the full result
   */
  async safetyCheck(
    action: string,
    context: string,
    opts: SafetyCheckOptions = {}
  ): Promise<[boolean, QubitAIResult]> {
    const riskText =
      opts.risks && opts.risks.length > 0
        ? `\n考慮するリスク: ${opts.risks.join(", ")}`
        : "";
    const constraintText = opts.constraints
      ? `\n制約: ${JSON.stringify(opts.constraints)}`
      : "";
    const fullContext = `${context}${riskText}${constraintText}`;

    const result = await this.judge(action, fullContext, "safety");
    return [result.decision === "Yes", result];
  }

  /**
   * Evaluate the quality of a piece of content.
   */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<QubitAIResult> {
    const reqText =
      opts.requirements && opts.requirements.length > 0
        ? `\n要件: ${opts.requirements.join(", ")}`
        : "";
    const intentText = opts.userIntent ? `\nユーザーの意図: ${opts.userIntent}` : "";
    const context = `品質評価${reqText}${intentText}`;
    return this.judge(content, context, "quality");
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
    return this.judge(action, context, "ethics");
  }

  /**
   * Rank a list of items by priority using the NeuroQuantum engine.
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

    const scored = await Promise.all(
      items.map(async (item) => {
        const action = `${item.name}: ${item.description}`;
        const raw = await this.engine.judge(action, context, {
          type: "priority",
          strictMode: this.config.strictMode,
        });
        return { item, score: raw.score / 100 };
      })
    );

    scored.sort((a, b) => b.score - a.score);
    return scored.map(({ item, score }) => [item, score] as PriorityItemResult);
  }

  // ---------------------------------------------------------------------------
  // Status & information
  // ---------------------------------------------------------------------------

  /** Return product information. */
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

  /** Return system status. */
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
  // Private
  // ---------------------------------------------------------------------------

  private recordHistory(
    judgmentType: JudgmentType,
    context: string,
    result: QubitAIResult
  ): void {
    this.history.push({
      timestamp: result.timestamp,
      judgmentType,
      contextPreview: context.slice(0, 200),
      decision: result.decision,
      score: result.score,
      confidence: result.confidence,
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
// Singleton
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
// Module-level convenience functions (mirror qubit_ai.py globals)
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
