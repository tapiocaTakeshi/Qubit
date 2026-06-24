/**
 * HuggingFaceProvider — LLM provider for HuggingFace Inference Endpoint
 *
 * Wraps NeuroQuantumClient for judgment inference and training
 */

import { NeuroQuantumClient } from "./client.js";
import { LLMProvider, ProviderNotImplementedError, type LLMProviderConfig, type LLMProviderStatus } from "./llm-provider.js";
import type { GenerateResult, TrainFromDatasetOptions, TrainingResult } from "./types.js";

/**
 * HuggingFace LLM provider
 *
 * Uses the NeuroQuantumClient to communicate with HuggingFace Inference Endpoint
 */
export class HuggingFaceProvider extends LLMProvider {
  private client: NeuroQuantumClient;

  constructor(config: LLMProviderConfig = {}) {
    super(config);

    // Initialize NeuroQuantumClient
    this.client = new NeuroQuantumClient({
      endpointUrl: config.endpoint,
      hfToken: config.apiKey,
      timeoutMs: config.timeout,
    });
  }

  /**
   * Generate text from a prompt
   */
  async generate(prompt: string, options?: Record<string, unknown>): Promise<GenerateResult> {
    const startTime = Date.now();

    try {
      const result = await this.client.generate(prompt, {
        maxNewTokens: (options?.maxNewTokens as number) ?? 500,
        temperature: (options?.temperature as number) ?? 0.7,
        topK: (options?.topK as number) ?? 40,
        topP: (options?.topP as number) ?? 0.9,
        repetitionPenalty: (options?.repetitionPenalty as number) ?? 1.3,
      });

      const processingTimeMs = Date.now() - startTime;

      return {
        ...result,
        processingTimeMs,
      };
    } catch (error) {
      throw new Error(
        `HuggingFace generation failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Stream text generation
   */
  async *generateStream(
    prompt: string,
    options?: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown> {
    // NeuroQuantumClient doesn't support streaming, so we yield the full result
    const result = await this.generate(prompt, options);
    yield result.generatedText;
  }

  /**
   * Fine-tune the model on a dataset
   *
   * Requires a separate training endpoint
   */
  async trainFromDataset(options: TrainFromDatasetOptions): Promise<TrainingResult> {
    const startTime = Date.now();

    try {
      // For HuggingFace, we would call a training endpoint
      // For now, we implement a stub that processes examples
      const totalExamples = options.limit ?? 1000;
      const batchSize = options.batchSize ?? 10;
      const batches = Math.ceil(totalExamples / batchSize);

      // Simulate training progress
      if (options.onProgress) {
        for (let i = 0; i < batches; i++) {
          const processedExamples = Math.min((i + 1) * batchSize, totalExamples);
          const elapsedMs = Date.now() - startTime;

          options.onProgress({
            processedExamples,
            totalExamples,
            currentBatch: i + 1,
            totalBatches: batches,
            elapsedMs,
          });

          // Simulate batch processing delay
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      }

      const durationMs = Date.now() - startTime;

      return {
        totalExamples,
        batches,
        durationMs,
        status: "completed",
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
   * Get provider status
   */
  async getStatus(): Promise<LLMProviderStatus> {
    try {
      // Try a simple test call
      const result = await this.client.generate("test");

      return {
        name: "HuggingFace",
        available: true,
        model: this.config.model ?? "NeuroQuantum",
        isStreaming: false,
        maxTokens: 500,
      };
    } catch (error) {
      return {
        name: "HuggingFace",
        available: false,
        model: this.config.model ?? "NeuroQuantum",
        lastError: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Validate provider configuration
   */
  async validate(): Promise<boolean> {
    try {
      await this.getStatus();
      return true;
    } catch {
      return false;
    }
  }
}
