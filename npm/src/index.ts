/**
 * qubit_ai — Generative AI with LLM + HuggingFace training
 *
 * Generate content and fine-tune LLMs on HuggingFace datasets
 * Support for Claude, OpenAI, and HuggingFace providers
 *
 * @packageDocumentation
 */

// Primary export: QubitAI Generative (LLM + HF training)
export {
  QubitAIGenerative,
  getQubitAIGenerative,
  resetQubitAIGenerative,
  resetQubitAIGenerativeProvider,
  generate,
  generateWithExamples,
  generateBatch,
  trainOnHFDataset,
  type GenerationOptions,
  type GenerationResult,
} from "./qubit_ai_generative.js";

// LLM providers
export { LLMProvider } from "./llm-provider.js";
export { ClaudeProvider } from "./llm-provider-claude.js";
export { OpenAIProvider } from "./llm-provider-openai.js";
export { HuggingFaceProvider } from "./llm-provider-hf.js";

// HuggingFace dataset loader (for Pyodide training)
export { HFDatasetLoader } from "./dataset.js";

export type {
  // Config
  QubitAIConfig,
  // Training types
  TrainingProgress,
  TrainingResult,
  // Dataset types
  HFDatasetLoaderConfig,
  HFDatasetRow,
  StreamRowsOptions,
  // LLM types
  LLMProviderConfig,
  LLMProviderStatus,
} from "./types.js";

export { ProviderNotImplementedError, ProviderConfigError } from "./llm-provider.js";
