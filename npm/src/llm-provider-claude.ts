/**
 * ClaudeProvider — LLM provider for Claude API (Anthropic)
 *
 * Uses the Anthropic SDK for judgment inference
 */

import { LLMProvider, ProviderNotImplementedError, ProviderConfigError, type LLMProviderConfig, type LLMProviderStatus } from "./llm-provider.js";
import type { GenerateResult, TrainFromDatasetOptions, TrainingResult } from "./types.js";

/**
 * Claude LLM provider using Anthropic API
 */
export class ClaudeProvider extends LLMProvider {
  private client: any; // Would be Anthropic client
  private apiKey: string;
  private model: string;

  constructor(config: LLMProviderConfig = {}) {
    super(config);

    this.apiKey = config.apiKey ?? process.env.ANTHROPIC_API_KEY ?? "";
    if (!this.apiKey) {
      throw new ProviderConfigError(
        "Claude provider requires ANTHROPIC_API_KEY environment variable or apiKey config"
      );
    }

    this.model = config.model ?? "claude-3-5-sonnet-20241022";

    // Lazy load Anthropic SDK to avoid dependency if not using Claude
    try {
      const { Anthropic } = require("@anthropic-ai/sdk") as any;
      this.client = new Anthropic({ apiKey: this.apiKey });
    } catch (error) {
      throw new ProviderConfigError(
        "Failed to load Anthropic SDK. Install with: npm install @anthropic-ai/sdk"
      );
    }
  }

  /**
   * Generate text from a prompt
   */
  async generate(prompt: string, options?: Record<string, unknown>): Promise<GenerateResult> {
    const startTime = Date.now();

    try {
      const message = await this.client.messages.create({
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

      const generatedText = message.content
        .filter((block: any) => block.type === "text")
        .map((block: any) => block.text)
        .join("");

      const processingTimeMs = Date.now() - startTime;

      return {
        generatedText,
        processingTimeMs,
        raw: message,
      };
    } catch (error) {
      throw new Error(
        `Claude generation failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Stream text generation using Claude streaming API
   */
  async *generateStream(
    prompt: string,
    options?: Record<string, unknown>
  ): AsyncGenerator<string, void, unknown> {
    try {
      const stream = await this.client.messages.create({
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

      for await (const event of stream) {
        if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
          yield event.delta.text;
        }
      }
    } catch (error) {
      throw new Error(
        `Claude streaming failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Fine-tuning not supported for Claude via this API
   */
  async trainFromDataset(options: TrainFromDatasetOptions): Promise<TrainingResult> {
    throw new ProviderNotImplementedError("trainFromDataset", "Claude");
  }

  /**
   * Get provider status
   */
  async getStatus(): Promise<LLMProviderStatus> {
    try {
      // Try a simple test call
      const message = await this.client.messages.create({
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
        name: "Claude",
        available: true,
        model: this.model,
        isStreaming: true,
        maxTokens: 8192, // Claude 3.5 Sonnet context
      };
    } catch (error) {
      return {
        name: "Claude",
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
