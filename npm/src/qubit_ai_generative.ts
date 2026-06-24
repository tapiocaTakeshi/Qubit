/**
 * QubitAI Generative — LLM-based text generation + HuggingFace training
 *
 * Focus: Content generation and fine-tuning
 * No judgment/reasoning - pure generative AI
 */

import { HFDatasetLoader } from "./dataset.js";
import { LLMProvider } from "./llm-provider.js";
import { HuggingFaceProvider } from "./llm-provider-hf.js";
import { ClaudeProvider } from "./llm-provider-claude.js";
import { OpenAIProvider } from "./llm-provider-openai.js";
import { PromptTemplates } from "./prompt-templates.js";
import type {
  QubitAIConfig,
  TrainingProgress,
  TrainingResult,
  JudgmentType,
} from "./types.js";

/**
 * Generation options
 */
export interface GenerationOptions {
  temperature?: number;
  maxTokens?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
}

/**
 * Generation result
 */
export interface GenerationResult {
  text: string;
  finishReason: string;
  tokensUsed: number;
  generatedAt: string;
}

/**
 * QubitAI Generative — LLM-based text generation and training
 */
export class QubitAIGenerative {
  private llmProvider: LLMProvider;
  private datasetLoader: HFDatasetLoader;
  readonly sessionId: string;
  private readonly config: Required<QubitAIConfig>;

  constructor(
    llmProvider: "claude" | "openai" | "hf",
    llmConfig?: Record<string, unknown>
  ) {
    this.sessionId = `qubit-generative-${new Date().toISOString()}`;

    this.config = {
      version: "3.1.0",
      productName: "Qubit.ai Generative",
      description: "Generative AI + HuggingFace training",
      strictMode: false,
      enableLogging: true,
      maxJudgmentHistory: 0,
      llmEnabled: true,
      llmProvider: llmProvider as any,
      llmConfig: llmConfig,
      fallbackToHeuristics: false,
      llmBlendStrategy: "weighted",
    } as Required<QubitAIConfig>;

    // Initialize LLM provider
    this.llmProvider = this.createLLMProvider();
    this.datasetLoader = new HFDatasetLoader(llmConfig);
  }

  /**
   * Create LLM provider
   */
  private createLLMProvider(): LLMProvider {
    const provider = this.config.llmProvider;

    switch (provider) {
      case "claude":
        return new ClaudeProvider(this.config.llmConfig);
      case "openai":
        return new OpenAIProvider(this.config.llmConfig);
      case "hf":
        return new HuggingFaceProvider(this.config.llmConfig);
      default:
        throw new Error(`Unknown LLM provider: ${provider}`);
    }
  }

  /**
   * Generate text
   */
  async generate(
    prompt: string,
    options: GenerationOptions = {}
  ): Promise<GenerationResult> {
    const result = await this.llmProvider.generate(prompt, {
      temperature: options.temperature ?? 0.7,
      maxTokens: options.maxTokens ?? 500,
      topK: options.topK,
      topP: options.topP,
      repetitionPenalty: options.repetitionPenalty,
    });

    return {
      text: result.generatedText,
      finishReason: result.finishReason || "max_tokens",
      tokensUsed: result.tokensUsed || 0,
      generatedAt: new Date().toISOString(),
    };
  }

  /**
   * Generate with examples (few-shot)
   */
  async generateWithExamples(
    prompt: string,
    examplePrompts: string[],
    options: GenerationOptions = {}
  ): Promise<GenerationResult> {
    // Build few-shot prompt
    const examplesText = examplePrompts
      .map((ex, i) => `Example ${i + 1}: ${ex}`)
      .join("\n\n");

    const fullPrompt = `${examplesText}\n\nNow: ${prompt}`;

    return this.generate(fullPrompt, options);
  }

  /**
   * Generate based on judgment type (for training data generation)
   */
  async generateForType(
    prompt: string,
    judgmentType: JudgmentType,
    options: GenerationOptions = {}
  ): Promise<GenerationResult> {
    const templates = new PromptTemplates();
    const { system, user } = templates.buildPrompt(
      judgmentType,
      prompt,
      "Generation",
      {}
    );

    const fullPrompt = `${system}\n\n${user}`;
    return this.generate(fullPrompt, options);
  }

  /**
   * Batch generation
   */
  async generateBatch(
    prompts: string[],
    options: GenerationOptions = {}
  ): Promise<GenerationResult[]> {
    const results: GenerationResult[] = [];

    for (const prompt of prompts) {
      try {
        const result = await this.generate(prompt, options);
        results.push(result);
      } catch (error) {
        console.error(`Failed to generate for prompt: ${prompt}`, error);
        results.push({
          text: "",
          finishReason: "error",
          tokensUsed: 0,
          generatedAt: new Date().toISOString(),
        });
      }
    }

    return results;
  }

  /**
   * Train on HuggingFace dataset
   */
  async trainOnHFDataset(opts: {
    dataset: string;
    judgmentType?: JudgmentType;
    split?: string;
    maxExamples?: number;
    batchSize?: number;
    onProgress?: (progress: TrainingProgress) => void;
  }): Promise<TrainingResult> {
    const startTime = Date.now();

    try {
      // Load examples from HF dataset
      const examples = [];
      let count = 0;
      const maxExamples = opts.maxExamples ?? 1000;

      for await (const row of this.datasetLoader.streamRows({
        dataset: opts.dataset,
        split: opts.split ?? "train",
        maxRows: maxExamples,
      })) {
        examples.push(row);
        count++;

        if (opts.onProgress) {
          opts.onProgress({
            processedExamples: count,
            totalExamples: maxExamples,
            currentBatch: Math.floor(count / (opts.batchSize ?? 32)) + 1,
            totalBatches: Math.ceil(maxExamples / (opts.batchSize ?? 32)),
          });
        }

        if (count >= maxExamples) {
          break;
        }
      }

      // Send to provider for training
      const result = await this.llmProvider.trainFromDataset({
        dataset: opts.dataset,
        promptField: "prompt",
        completionField: "completion",
        batchSize: opts.batchSize ?? 32,
        maxRows: maxExamples,
        onProgress: opts.onProgress,
      });

      return {
        totalExamples: examples.length,
        batches: Math.ceil(examples.length / (opts.batchSize ?? 32)),
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
   * Get LLM provider status
   */
  async getStatus(): Promise<{
    available: boolean;
    provider: string;
    version: string;
  }> {
    try {
      const status = await this.llmProvider.getStatus?.();
      return {
        available: true,
        provider: this.config.llmProvider,
        version: this.config.version,
        ...status,
      } as any;
    } catch (error) {
      return {
        available: false,
        provider: this.config.llmProvider,
        version: this.config.version,
      };
    }
  }

  /**
   * Get configuration
   */
  getConfig(): Record<string, unknown> {
    return {
      provider: this.config.llmProvider,
      version: this.config.version,
      productName: this.config.productName,
      sessionId: this.sessionId,
    };
  }
}

// Singleton instances by provider
const instances: Map<string, QubitAIGenerative> = new Map();

/**
 * Get or create QubitAI Generative instance
 */
export function getQubitAIGenerative(
  provider: "claude" | "openai" | "hf" = "claude",
  llmConfig?: Record<string, unknown>
): QubitAIGenerative {
  const key = `${provider}`;

  if (!instances.has(key)) {
    instances.set(key, new QubitAIGenerative(provider, llmConfig));
  }

  return instances.get(key)!;
}

/**
 * Reset all instances
 */
export function resetQubitAIGenerative(): void {
  instances.clear();
}

/**
 * Reset specific provider instance
 */
export function resetQubitAIGenerativeProvider(
  provider: "claude" | "openai" | "hf"
): void {
  instances.delete(provider);
}

// Convenience functions
export async function generate(
  prompt: string,
  provider: "claude" | "openai" | "hf" = "claude",
  options?: GenerationOptions
): Promise<GenerationResult> {
  return getQubitAIGenerative(provider).generate(prompt, options);
}

export async function generateWithExamples(
  prompt: string,
  examples: string[],
  provider: "claude" | "openai" | "hf" = "claude",
  options?: GenerationOptions
): Promise<GenerationResult> {
  return getQubitAIGenerative(provider).generateWithExamples(
    prompt,
    examples,
    options
  );
}

export async function generateBatch(
  prompts: string[],
  provider: "claude" | "openai" | "hf" = "claude",
  options?: GenerationOptions
): Promise<GenerationResult[]> {
  return getQubitAIGenerative(provider).generateBatch(prompts, options);
}

export async function trainOnHFDataset(
  opts: Parameters<QubitAIGenerative["trainOnHFDataset"]>[0] & {
    provider?: "claude" | "openai" | "hf";
  }
): Promise<TrainingResult> {
  const { provider = "claude", ...trainOpts } = opts;
  return getQubitAIGenerative(provider).trainOnHFDataset(trainOpts);
}
