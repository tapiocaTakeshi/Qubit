/**
 * QubitAIConfigManager — Configuration management for Qubit AI
 *
 * Loads and validates configuration from environment variables and runtime config
 */

import type { QubitAIConfig } from "./types.js";

/**
 * Default configuration values
 */
const DEFAULTS: Required<QubitAIConfig> = {
  version: "1.2.2",
  productName: "Qubit.ai",
  description: "Claude's Quantum Prefrontal Cortex",
  strictMode: false,
  enableLogging: true,
  maxJudgmentHistory: 100,

  llmEnabled: false,
  llmProvider: "claude",
  llmConfig: {
    apiKey: "",
    endpoint: "",
    model: "",
    temperature: 0.7,
    maxTokens: 500,
    timeout: 30000,
  },
  fallbackToHeuristics: false,
  llmPromptTemplate: "",
  llmBlendStrategy: "weighted",

  neuroquantumEnabled: false,
  neuroquantumConfig: {
    baseUrl: "http://localhost:5000",
    timeout: 30000,
    maxRetries: 3,
    retryDelayMs: 1000,
  },
};

/**
 * QubitAIConfigManager — Manages configuration loading and validation
 */
export class QubitAIConfigManager {
  /**
   * Load configuration from environment variables
   *
   * Supported env vars:
   * - QUBIT_LLM_ENABLED: "true" or "false"
   * - QUBIT_LLM_PROVIDER: "hf", "claude", or "openai"
   * - QUBIT_LLM_API_KEY: API key/token
   * - QUBIT_LLM_ENDPOINT: Custom endpoint for HuggingFace
   * - QUBIT_LLM_MODEL: Model name
   * - QUBIT_LLM_TEMPERATURE: 0.0-2.0
   * - QUBIT_LLM_MAX_TOKENS: Max tokens to generate
   * - QUBIT_LLM_TIMEOUT: Timeout in ms
   * - QUBIT_FALLBACK_TO_HEURISTICS: "true" or "false"
   * - ANTHROPIC_API_KEY: Claude API key (alternate)
   * - OPENAI_API_KEY: OpenAI API key (alternate)
   * - HF_TOKEN: HuggingFace token (alternate)
   */
  static loadFromEnv(): QubitAIConfig {
    const env = typeof process !== "undefined" ? process.env : ({} as Record<string, string>);

    return {
      llmEnabled: env.QUBIT_LLM_ENABLED === "true",
      llmProvider: (env.QUBIT_LLM_PROVIDER as "hf" | "claude" | "openai") || "claude",
      llmConfig: {
        // Prefer QUBIT_LLM_API_KEY, fallback to provider-specific keys
        apiKey:
          env.QUBIT_LLM_API_KEY ||
          env.ANTHROPIC_API_KEY ||
          env.OPENAI_API_KEY ||
          env.HF_TOKEN ||
          "",
        endpoint: env.QUBIT_LLM_ENDPOINT || "",
        model: env.QUBIT_LLM_MODEL || "",
        temperature: parseFloat(env.QUBIT_LLM_TEMPERATURE || "0.7"),
        maxTokens: parseInt(env.QUBIT_LLM_MAX_TOKENS || "500", 10),
        timeout: parseInt(env.QUBIT_LLM_TIMEOUT || "30000", 10),
      },
      fallbackToHeuristics: env.QUBIT_FALLBACK_TO_HEURISTICS === "true",
    };
  }

  /**
   * Get default configuration
   */
  static getDefaults(): QubitAIConfig {
    return { ...DEFAULTS };
  }

  /**
   * Merge multiple config objects (later takes precedence)
   */
  static mergeConfigs(...configs: (QubitAIConfig | undefined)[]): QubitAIConfig {
    const merged = { ...DEFAULTS };

    for (const config of configs) {
      if (!config) continue;

      // Merge top-level properties
      Object.assign(merged, {
        version: config.version ?? merged.version,
        productName: config.productName ?? merged.productName,
        description: config.description ?? merged.description,
        strictMode: config.strictMode ?? merged.strictMode,
        enableLogging: config.enableLogging ?? merged.enableLogging,
        maxJudgmentHistory: config.maxJudgmentHistory ?? merged.maxJudgmentHistory,
        llmEnabled: config.llmEnabled ?? merged.llmEnabled,
        llmProvider: config.llmProvider ?? merged.llmProvider,
        fallbackToHeuristics: config.fallbackToHeuristics ?? merged.fallbackToHeuristics,
        llmPromptTemplate: config.llmPromptTemplate ?? merged.llmPromptTemplate,
        llmBlendStrategy: config.llmBlendStrategy ?? merged.llmBlendStrategy,
        neuroquantumEnabled: config.neuroquantumEnabled ?? merged.neuroquantumEnabled,
      });

      // Merge llmConfig
      if (config.llmConfig) {
        merged.llmConfig = {
          ...merged.llmConfig,
          ...config.llmConfig,
        };
      }

      // Merge neuroquantumConfig
      if (config.neuroquantumConfig) {
        merged.neuroquantumConfig = {
          ...merged.neuroquantumConfig,
          ...config.neuroquantumConfig,
        };
      }
    }

    return merged;
  }

  /**
   * Validate configuration
   *
   * @returns { valid: boolean, errors: string[] }
   */
  static validate(config: QubitAIConfig): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate LLM config if enabled
    if (config.llmEnabled) {
      if (!config.llmProvider) {
        errors.push("llmProvider required when llmEnabled is true");
      }

      if (config.llmProvider === "claude" && !config.llmConfig?.apiKey) {
        errors.push("ANTHROPIC_API_KEY required for Claude provider");
      }

      if (config.llmProvider === "openai" && !config.llmConfig?.apiKey) {
        errors.push("OPENAI_API_KEY required for OpenAI provider");
      }

      // Validate temperature range
      if (config.llmConfig?.temperature !== undefined) {
        if (config.llmConfig.temperature < 0 || config.llmConfig.temperature > 2) {
          errors.push("llmConfig.temperature must be between 0 and 2");
        }
      }

      // Validate maxTokens
      if (config.llmConfig?.maxTokens !== undefined) {
        if (config.llmConfig.maxTokens < 10 || config.llmConfig.maxTokens > 8192) {
          errors.push("llmConfig.maxTokens must be between 10 and 8192");
        }
      }
    }

    // Validate strictMode
    if (config.strictMode !== undefined && typeof config.strictMode !== "boolean") {
      errors.push("strictMode must be a boolean");
    }

    // Validate maxJudgmentHistory
    if (config.maxJudgmentHistory !== undefined) {
      if (!Number.isInteger(config.maxJudgmentHistory) || config.maxJudgmentHistory < 0) {
        errors.push("maxJudgmentHistory must be a non-negative integer");
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Create a config builder for fluent configuration
   */
  static builder(baseConfig: QubitAIConfig = {}): ConfigBuilder {
    return new ConfigBuilder(baseConfig);
  }
}

/**
 * Fluent configuration builder
 */
class ConfigBuilder {
  private config: QubitAIConfig;

  constructor(baseConfig: QubitAIConfig) {
    this.config = { ...baseConfig };
  }

  withLLM(provider: "claude" | "openai" | "hf", apiKey: string): ConfigBuilder {
    this.config.llmEnabled = true;
    this.config.llmProvider = provider;
    if (!this.config.llmConfig) {
      this.config.llmConfig = {};
    }
    this.config.llmConfig.apiKey = apiKey;
    return this;
  }

  withModel(model: string): ConfigBuilder {
    if (!this.config.llmConfig) {
      this.config.llmConfig = {};
    }
    this.config.llmConfig.model = model;
    return this;
  }

  withTemperature(temp: number): ConfigBuilder {
    if (!this.config.llmConfig) {
      this.config.llmConfig = {};
    }
    this.config.llmConfig.temperature = temp;
    return this;
  }

  withStrictMode(enabled: boolean): ConfigBuilder {
    this.config.strictMode = enabled;
    return this;
  }

  withHybridMode(fallback: boolean = true): ConfigBuilder {
    this.config.llmEnabled = true;
    this.config.fallbackToHeuristics = fallback;
    return this;
  }

  build(): QubitAIConfig {
    return { ...this.config };
  }
}
