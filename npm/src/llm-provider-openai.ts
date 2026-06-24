/**
 * OpenAIProvider — LLM provider for OpenAI API
 *
 * Uses the OpenAI SDK for judgment inference
 */

import { LLMProvider, ProviderNotImplementedError, ProviderConfigError, type LLMProviderConfig, type LLMProviderStatus } from "./llm-provider.js";
import type { GenerateResult, TrainFromDatasetOptions, TrainingResult } from "./types.js";

/**
 * OpenAI LLM provider
 */
export class OpenAIProvider extends LLMProvider {
  private client: any; // Would be OpenAI client
  private apiKey: string;
  private model: string;

  constructor(config: LLMProviderConfig = {}) {
    super(config);

    this.apiKey = config.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    if (!this.apiKey) {
      throw new ProviderConfigError(
        "OpenAI provider requires OPENAI_API_KEY environment variable or apiKey config"
      );
    }

    this.model = config.model ?? "gpt-4o";

    // Lazy load OpenAI SDK to avoid dependency if not using OpenAI
    try {
      const { OpenAI } = require("openai") as any;
      this.client = new OpenAI({ apiKey: this.apiKey });
    } catch (error) {
      throw new ProviderConfigError("Failed to load OpenAI SDK. Install with: npm install openai");
    }
  }

  /**
   * Generate text from a prompt
   */
  async generate(prompt: string, options?: Record<string, unknown>): Promise<GenerateResult> {
    const startTime = Date.now();

    try {
      const response = await this.client.chat.completions.create({
        model: this.model,
        max_tokens: (options?.maxTokens as number) ?? 1024,
        temperature: (options?.temperature as number) ?? 0.7,
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
      });

      const generatedText = response.choices[0]?.message?.content ?? "";
      const processingTimeMs = Date.now() - startTime;

      return {
        generatedText,
        processingTimeMs,
        raw: response,
      };
    } catch (error) {
      throw new Error(
        `OpenAI generation failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Stream text generation using OpenAI streaming API
   */
  async *generateStream(
    prompt: string,
    options?: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown> {
    try {
      const stream = await this.client.chat.completions.create({
        model: this.model,
        max_tokens: (options?.maxTokens as number) ?? 1024,
        temperature: (options?.temperature as number) ?? 0.7,
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
        stream: true,
      });

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? "";
        if (delta) {
          yield delta;
        }
      }
    } catch (error) {
      throw new Error(
        `OpenAI streaming failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Fine-tuning not supported via this provider interface
   *
   * OpenAI fine-tuning requires saving examples to JSONL format
   * and calling the fine-tuning API separately
   */
  async trainFromDataset(options: TrainFromDatasetOptions): Promise<TrainingResult> {
    throw new ProviderNotImplementedError("trainFromDataset", "OpenAI");
  }

  /**
   * Get provider status
   */
  async getStatus(): Promise<LLMProviderStatus> {
    try {
      // Try a simple test call
      const response = await this.client.chat.completions.create({
        model: this.model,
        max_tokens: 10,
        messages: [
          {
            role: "user",
            content: "test",
          },
        ],
      });

      return {
        name: "OpenAI",
        available: true,
        model: this.model,
        isStreaming: true,
        maxTokens: 4096, // Context window varies by model
      };
    } catch (error) {
      return {
        name: "OpenAI",
        available: false,
        model: this.model,
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
