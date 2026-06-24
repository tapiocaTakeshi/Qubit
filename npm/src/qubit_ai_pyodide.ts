/**
 * QubitAI with Pyodide backend — Simplified version for HF dataset training
 *
 * Focus: Quantum-inspired inference + HuggingFace dataset fine-tuning
 * No external APIs required — all Python runs in WebAssembly via Pyodide
 */

import { NeuroQuantumPyodide, getNeuroQuantumPyodide } from "./pyodide-wrapper.js";
import { HFDatasetLoader } from "./dataset.js";
import type {
  JudgmentType,
  QubitAIConfig,
  QubitAIResult,
  QubitAIInfo,
  QubitAIStatus,
  JudgmentRecord,
  TrainingProgress,
  TrainingResult,
} from "./types.js";

/**
 * QubitAI with Pyodide backend — Simplified for quantum inference + HF training
 */
export class QubitAIPyodide {
  private pyodideBackend: NeuroQuantumPyodide;
  private datasetLoader: HFDatasetLoader;
  readonly sessionId: string;
  private readonly history: JudgmentRecord[] = [];
  private readonly config: Required<QubitAIConfig>;

  constructor(config: QubitAIConfig = {}) {
    this.config = {
      version: config.version ?? "2.1.0",
      productName: config.productName ?? "Qubit.ai",
      description: config.description ?? "Quantum-Inspired AI with Pyodide",
      strictMode: config.strictMode ?? false,
      enableLogging: config.enableLogging ?? true,
      maxJudgmentHistory: config.maxJudgmentHistory ?? 100,
      llmEnabled: false, // Disabled for Pyodide-only version
      llmProvider: "claude",
      fallbackToHeuristics: false,
      llmBlendStrategy: "weighted",
    } as Required<QubitAIConfig>;

    this.pyodideBackend = getNeuroQuantumPyodide();
    this.datasetLoader = new HFDatasetLoader(config.llmConfig);
    this.sessionId = `qubit-ai-${new Date().toISOString()}`;
  }

  /**
   * Initialize Pyodide backend
   */
  async initialize(): Promise<void> {
    await this.pyodideBackend.initialize();
  }

  /**
   * Make a judgment using Pyodide quantum-inspired backend
   */
  async judge(
    action: string,
    context: string,
    judgmentType: JudgmentType = "safety",
    strict?: boolean
  ): Promise<QubitAIResult> {
    await this.initialize();

    const raw = await this.pyodideBackend.judge(action, context, judgmentType);

    // Apply strict mode if needed
    const strictMode = strict ?? this.config.strictMode;
    const decision =
      strictMode && raw.score < 70 ? "No" : raw.decision;

    const result: QubitAIResult = {
      decision: decision as "Yes" | "No",
      score: raw.score,
      reasoning: raw.reasoning,
      confidence: raw.confidence,
      factors: raw.keyFactors,
      timestamp: raw.timestamp,
    };

    this.recordHistory(judgmentType, context, raw);
    return result;
  }

  /**
   * Safety check
   */
  async safetyCheck(
    action: string,
    context: string
  ): Promise<[boolean, QubitAIResult]> {
    const result = await this.judge(action, context, "safety");
    return [result.decision === "Yes", result];
  }

  /**
   * Ethics evaluation
   */
  async ethicsCheck(
    action: string,
    stakeholders?: string[],
    potentialHarms?: string[]
  ): Promise<QubitAIResult> {
    const parts: string[] = [action];
    if (stakeholders && stakeholders.length > 0) {
      parts.push(`Stakeholders: ${stakeholders.join(", ")}`);
    }
    if (potentialHarms && potentialHarms.length > 0) {
      parts.push(`Potential harms: ${potentialHarms.join(", ")}`);
    }
    const context = parts.join("\n");

    return this.judge(action, context, "ethics");
  }

  /**
   * Quality evaluation
   */
  async evaluateQuality(content: string): Promise<QubitAIResult> {
    return this.judge(content, "Quality evaluation", "quality");
  }

  /**
   * Prioritize items
   */
  async prioritize(
    items: Array<{ name: string; description: string }>,
    constraints?: string
  ): Promise<Array<[{ name: string; description: string }, number]>> {
    const taskStrings = items.map((item) => `${item.name}: ${item.description}`);
    const context = constraints || "Task prioritization";

    // Score each task
    const scores: number[] = [];
    for (const task of taskStrings) {
      const result = await this.judge(task, context, "priority");
      scores.push(result.score / 100);
    }

    // Rank by score
    const ranked = items
      .map((item, i) => [item, scores[i]!] as const)
      .sort((a, b) => b[1] - a[1]);

    return ranked;
  }

  /**
   * Train on HuggingFace dataset
   */
  async trainOnHFDataset(opts: {
    dataset: string;
    judgmentType: JudgmentType;
    maxExamples?: number;
    onProgress?: (progress: TrainingProgress) => void;
  }): Promise<TrainingResult> {
    const startTime = Date.now();

    try {
      await this.initialize();

      // Load dataset using HFDatasetLoader
      const examples = [];
      let count = 0;
      const maxExamples = opts.maxExamples ?? 1000;

      for await (const row of this.datasetLoader.streamRows({
        dataset: opts.dataset,
        split: "train",
        maxRows: maxExamples,
      })) {
        examples.push(row);
        count++;

        if (opts.onProgress) {
          opts.onProgress({
            processedExamples: count,
            totalExamples: maxExamples,
            currentBatch: Math.floor(count / 32) + 1,
            totalBatches: Math.ceil(maxExamples / 32),
          });
        }

        if (count >= maxExamples) {
          break;
        }
      }

      // Train using Pyodide backend
      const trainResult = await this.pyodideBackend.trainOnHFDataset({
        dataset: opts.dataset,
        judgmentType: opts.judgmentType,
        maxExamples,
        onProgress: opts.onProgress,
      });

      return {
        totalExamples: examples.length,
        batches: Math.ceil(examples.length / 32),
        durationMs: Date.now() - startTime,
        status: trainResult.status as "success" | "failed",
        errors: [],
      };
    } catch (error) {
      return {
        totalExamples: 0,
        batches: 0,
        durationMs: Date.now() - startTime,
        status: "failed",
        errors: [error instanceof Error ? error.message : String(error)],
      };
    }
  }

  /**
   * Get product info
   */
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

  /**
   * Get system status
   */
  async getStatus(): Promise<QubitAIStatus> {
    const backendStatus = await this.pyodideBackend.getStatus();

    return {
      product: this.config.productName,
      status: backendStatus.available ? "operational" : "unavailable",
      frontalEngineAvailable: backendStatus.available,
      judgmentHistorySize: this.history.length,
      maxHistory: this.config.maxJudgmentHistory,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get judgment history
   */
  getHistory(limit = 10): JudgmentRecord[] {
    return this.history.slice(-limit);
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.history.length = 0;
  }

  /**
   * Explain a result in natural language
   */
  explain(result: QubitAIResult): string {
    const factorsStr =
      result.factors.length > 0
        ? result.factors.map((f) => `• ${f}`).join("\n")
        : "（No factors）";

    return [
      "【Judgment Result】",
      `Decision: ${result.decision}`,
      `Score: ${result.score}/100`,
      `Confidence: ${result.confidence}`,
      "",
      "【Reasoning】",
      result.reasoning,
      "",
      "【Key Factors】",
      factorsStr,
    ].join("\n");
  }

  /**
   * Private: Record judgment in history
   */
  private recordHistory(
    judgmentType: JudgmentType,
    context: string,
    raw: any
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
      this.history.splice(0, this.history.length - this.config.maxJudgmentHistory);
    }
  }
}

// Singleton
let _instance: QubitAIPyodide | undefined;

/**
 * Get or create global QubitAIPyodide instance
 */
export function getQubitAIPyodide(config?: QubitAIConfig): QubitAIPyodide {
  if (!_instance) {
    _instance = new QubitAIPyodide(config);
  }
  return _instance;
}

/**
 * Reset global instance
 */
export function resetQubitAIPyodide(): void {
  _instance = undefined;
}

// Convenience functions
export async function judge(
  action: string,
  context: string,
  judgmentType: JudgmentType = "safety"
): Promise<QubitAIResult> {
  return getQubitAIPyodide().judge(action, context, judgmentType);
}

export async function safetyCheck(
  action: string,
  context: string
): Promise<[boolean, QubitAIResult]> {
  return getQubitAIPyodide().safetyCheck(action, context);
}

export async function ethicsCheck(
  action: string,
  stakeholders?: string[]
): Promise<QubitAIResult> {
  return getQubitAIPyodide().ethicsCheck(action, stakeholders);
}

export async function evaluateQuality(
  content: string
): Promise<QubitAIResult> {
  return getQubitAIPyodide().evaluateQuality(content);
}
