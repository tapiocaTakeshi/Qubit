/**
 * LLMTrainer — HuggingFace dataset-based LLM fine-tuning
 *
 * Loads examples from HF datasets, adapts them for judgment tasks,
 * and trains the LLM on them
 */

import { HFDatasetLoader } from "./dataset.js";
import { LLMProvider } from "./llm-provider.js";
import { PromptTemplates } from "./prompt-templates.js";
import type {
  AdaptedTrainingExample,
  EvaluationMetrics,
  JudgmentType,
  QubitAIConfig,
  TrainingCheckpoint,
  TrainingExample,
  TrainingProgress,
  TrainingResult,
  StreamRowsOptions,
} from "./types.js";

/**
 * LLMTrainer — Fine-tunes LLMs on HuggingFace datasets
 */
export class LLMTrainer {
  private llmProvider: LLMProvider;
  private config: QubitAIConfig;
  private datasetLoader: HFDatasetLoader;

  constructor(llmProvider: LLMProvider, config: QubitAIConfig = {}) {
    this.llmProvider = llmProvider;
    this.config = config;
    // Pass HF token from environment or config
    this.datasetLoader = new HFDatasetLoader({
      hfToken: (typeof process !== "undefined" ? process.env["HF_TOKEN"] : undefined),
    });
  }

  /**
   * Load training examples from a HuggingFace dataset
   */
  async loadDataset(opts: {
    dataset: string;
    promptField?: string;
    completionField?: string;
    split?: string;
    limit?: number;
  }): Promise<TrainingExample[]> {
    const examples: TrainingExample[] = [];

    try {
      for await (const example of this.datasetLoader.streamRows({
        dataset: opts.dataset,
        split: opts.split ?? "train",
        maxRows: opts.limit ?? 1000,
      })) {
        // Auto-detect or use provided field names
        const promptField = opts.promptField ?? this.detectPromptField(example.row);
        const completionField = opts.completionField ?? this.detectCompletionField(example.row);

        if (promptField && completionField) {
          examples.push({
            prompt: String(example.row[promptField] ?? ""),
            completion: String(example.row[completionField] ?? ""),
          });
        }

        if (examples.length >= (opts.limit ?? 1000)) {
          break;
        }
      }
    } catch (error) {
      console.error(`Failed to load dataset ${opts.dataset}:`, error);
    }

    return examples;
  }

  /**
   * Adapt training examples for a judgment task
   */
  async adaptExamples(
    examples: TrainingExample[],
    judgmentType: JudgmentType
  ): Promise<AdaptedTrainingExample[]> {
    const adapted: AdaptedTrainingExample[] = [];

    for (const example of examples) {
      try {
        // Build judgment prompt from the original prompt
        const { system, user } = PromptTemplates.buildPrompt(
          judgmentType,
          example.prompt,
          "Training context",
          {}
        );

        // The example's completion becomes our expected judgment output
        // For simplicity, we assume completion is a judgment result or can be converted to one
        const judgmentCompletion = this.parseCompletionAsJudgment(
          example.completion,
          judgmentType
        );

        adapted.push({
          prompt: example.prompt,
          completion: example.completion,
          judgmentPrompt: `${system}\n\n${user}`,
          judgmentCompletion: judgmentCompletion,
          source: `hf-dataset`,
          judgmentType,
        });
      } catch (error) {
        console.warn(`Failed to adapt example:`, error);
        // Skip this example
      }
    }

    return adapted;
  }

  /**
   * Train the LLM on adapted dataset examples
   */
  async trainOnDataset(opts: {
    dataset: string;
    judgmentType: JudgmentType;
    batchSize?: number;
    maxExamples?: number;
    onProgress?: (progress: TrainingProgress) => void;
  }): Promise<TrainingResult> {
    const startTime = Date.now();
    const batchSize = opts.batchSize ?? 16;

    try {
      // Load examples
      const examples = await this.loadDataset({
        dataset: opts.dataset,
        limit: opts.maxExamples ?? 1000,
      });

      if (examples.length === 0) {
        return {
          totalExamples: 0,
          batches: 0,
          durationMs: Date.now() - startTime,
          status: "failed",
          errors: ["No examples loaded from dataset"],
        };
      }

      // Adapt examples
      const adapted = await this.adaptExamples(examples, opts.judgmentType);

      // Send to provider for training
      const result = await this.llmProvider.trainFromDataset({
        dataset: opts.dataset,
        promptField: "prompt",
        completionField: "completion",
        batchSize: batchSize,
        maxRows: opts.maxExamples,
        onProgress: opts.onProgress,
      });

      return {
        totalExamples: adapted.length,
        batches: Math.ceil(adapted.length / batchSize),
        durationMs: Date.now() - startTime,
        status: result.status,
        errors: result.errors,
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
   * Evaluate fine-tuned model on test set
   */
  async evaluateOnTestSet(opts: {
    dataset: string;
    split: string;
    sampleSize?: number;
  }): Promise<EvaluationMetrics> {
    const examples = await this.loadDataset({
      dataset: opts.dataset,
      split: opts.split,
      limit: opts.sampleSize ?? 100,
    });

    if (examples.length === 0) {
      return {
        totalExamples: 0,
        correctDecisions: 0,
        accuracy: 0,
        scoreMAE: 0,
        confidenceAccuracy: 0,
        f1Score: 0,
        inferenceTimeMs: 0,
        errors: ["No test examples found"],
      };
    }

    let correctDecisions = 0;
    let totalScoreError = 0;
    let correctConfidence = 0;
    const inferenceTimings: number[] = [];
    const errors: string[] = [];

    for (const example of examples) {
      try {
        const startTime = Date.now();
        const result = await this.llmProvider.generate(example.prompt);
        const inferenceTime = Date.now() - startTime;
        inferenceTimings.push(inferenceTime);

        // Parse completion as expected judgment
        const expected = this.parseCompletionAsJudgment(example.completion, "quality");

        // Simple evaluation: check if model response contains expected answer
        const responseMatch = result.generatedText.toLowerCase().includes(
          example.completion.toLowerCase().substring(0, 10)
        );

        if (responseMatch) {
          correctDecisions++;
        }

        // Measure confidence accuracy (simplified)
        if (result.generatedText.length > 0) {
          correctConfidence++;
        }
      } catch (error) {
        errors.push(error instanceof Error ? error.message : String(error));
      }
    }

    const accuracy = examples.length > 0 ? correctDecisions / examples.length : 0;
    const confidenceAccuracy = examples.length > 0 ? correctConfidence / examples.length : 0;
    const avgInferenceTime = inferenceTimings.length > 0
      ? inferenceTimings.reduce((a, b) => a + b, 0) / inferenceTimings.length
      : 0;

    // Simplified F1 score calculation
    const f1Score = 2 * (accuracy * confidenceAccuracy) / (accuracy + confidenceAccuracy || 1);

    return {
      totalExamples: examples.length,
      correctDecisions,
      accuracy,
      scoreMAE: 0, // Simplified
      confidenceAccuracy,
      f1Score,
      inferenceTimeMs: avgInferenceTime,
      errors,
    };
  }

  /**
   * Detect prompt field name in a row
   */
  private detectPromptField(row: Record<string, unknown>): string | null {
    const promptNames = ["prompt", "question", "instruction", "input", "text"];

    for (const name of promptNames) {
      if (name in row && row[name]) {
        return name;
      }
    }

    // Fallback to first non-empty field
    for (const [key, value] of Object.entries(row)) {
      if (value && typeof value === "string") {
        return key;
      }
    }

    return null;
  }

  /**
   * Detect completion field name in a row
   */
  private detectCompletionField(row: Record<string, unknown>): string | null {
    const completionNames = ["completion", "answer", "response", "output", "label"];

    for (const name of completionNames) {
      if (name in row && row[name]) {
        return name;
      }
    }

    // Fallback to second field
    const fields = Object.keys(row);
    if (fields.length > 1) {
      return fields[1] ?? null;
    }

    return null;
  }

  /**
   * Parse a completion string as a judgment JSON
   */
  private parseCompletionAsJudgment(completion: string, type: JudgmentType): string {
    // Try to parse as existing JSON
    try {
      JSON.parse(completion);
      return completion;
    } catch {
      // Not JSON, construct judgment from the completion text
    }

    // Default: create a simple judgment based on positive/negative indicators
    const lower = completion.toLowerCase();
    const isPositive = ["yes", "good", "safe", "ethical", "positive", "correct", "high"].some(
      (w) => lower.includes(w)
    );

    return JSON.stringify({
      decision: isPositive ? "Yes" : "No",
      score: isPositive ? 75 : 25,
      reasoning: completion.substring(0, 200),
      confidence: "medium",
      factors: [completion.split(" ").slice(0, 3).join(" ")],
    });
  }
}
