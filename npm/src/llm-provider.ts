/**
 * LLMProvider — Abstract interface for pluggable LLM backends
 *
 * Supports Claude, OpenAI, HuggingFace, or custom implementations
 */

import type { GenerateOptions, GenerateResult, TrainFromDatasetOptions, TrainingResult } from "./types.js";

/**
 * Configuration for any LLM provider
 */
export interface LLMProviderConfig {
  /** API key or authentication token */
  apiKey?: string;
  /** Custom endpoint URL (if supported) */
  endpoint?: string;
  /** Model identifier (e.g., "claude-3-5-sonnet", "gpt-4", "mistral-7b") */
  model?: string;
  /** Request timeout in milliseconds */
  timeoutMs?: number;
  /** Additional provider-specific config */
  [key: string]: unknown;
}

/**
 * Options for generate() calls
 */
export interface LLMGenerateOptions extends GenerateOptions {
  /** Maximum retries on transient failures */
  maxRetries?: number;
  /** Timeout override for this specific call */
  timeoutMs?: number;
}

/**
 * Status information from an LLM provider
 */
export interface LLMProviderStatus {
  name: string;
  available: boolean;
  model?: string;
  lastError?: string;
  isStreaming?: boolean;
  maxTokens?: number;
}

/**
 * Abstract LLMProvider base class
 *
 * All concrete LLM implementations (Claude, OpenAI, HuggingFace) extend this.
 */
export abstract class LLMProvider {
  protected config: LLMProviderConfig;

  constructor(config: LLMProviderConfig = {}) {
    this.config = config;
  }

  /**
   * Generate text from a prompt
   *
   * @param prompt - The input prompt
   * @param options - Generation options (temperature, maxTokens, etc.)
   * @returns Generated text result
   */
  abstract generate(
    prompt: string,
    options?: LLMGenerateOptions
  ): Promise<GenerateResult>;

  /**
   * Generate text with streaming support
   *
   * @param prompt - The input prompt
   * @param options - Generation options
   * @returns Async generator yielding text chunks
   */
  abstract generateStream(
    prompt: string,
    options?: LLMGenerateOptions
  ): AsyncGenerator<string, void, unknown>;

  /**
   * Fine-tune the model on a dataset
   *
   * Supported by HuggingFace. Other providers may throw NotImplementedError.
   *
   * @param options - Training options (dataset, model, batches, etc.)
   * @returns Training result with stats
   */
  abstract trainFromDataset(options: TrainFromDatasetOptions): Promise<TrainingResult>;

  /**
   * Get provider status and capabilities
   */
  abstract getStatus(): Promise<LLMProviderStatus>;

  /**
   * Get the provider's configuration
   */
  getConfig(): LLMProviderConfig {
    return { ...this.config };
  }

  /**
   * Update configuration at runtime
   */
  setConfig(newConfig: Partial<LLMProviderConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Validate that the provider is properly configured
   */
  abstract validate(): Promise<boolean>;
}

/**
 * Provider not implemented error
 */
export class ProviderNotImplementedError extends Error {
  constructor(method: string, providerName: string) {
    super(`${method} is not implemented for provider: ${providerName}`);
    this.name = "ProviderNotImplementedError";
  }
}

/**
 * Provider configuration error
 */
export class ProviderConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProviderConfigError";
  }
}
