/**
 * qubit_ai — quantum-inspired QBNN inference & judgment engine
 *
 * Supports both keyword-based heuristics and LLM-based generative inference
 *
 * @packageDocumentation
 */

// Primary export: QubitAI class (TypeScript port of qubit_ai.py)
export {
  QubitAI,
  getQubitAI,
  resetQubitAI,
  judge,
  safetyCheck,
  evaluateQuality,
  ethicsCheck,
} from "./qubit_ai.js";

// Lower-level engines (pure QBNN scoring, no external deps)
export { QBNNFrontalEngine } from "./frontal.js";
export { LLMFrontalEngine } from "./llm-frontal.js";
export { HybridFrontalEngine } from "./hybrid-frontal.js";
export { NeuroQuantumFrontalEngine } from "./neuroquantum-frontal.js";

// LLM providers (pluggable backends)
export { LLMProvider } from "./llm-provider.js";
export { HuggingFaceProvider } from "./llm-provider-hf.js";
export { ClaudeProvider } from "./llm-provider-claude.js";
export { OpenAIProvider } from "./llm-provider-openai.js";

// LLM support classes
export { PromptTemplates } from "./prompt-templates.js";
export { ResponseParser } from "./response-parser.js";
export { QubitAIConfigManager } from "./config.js";
export { LLMTrainer } from "./llm-trainer.js";

// HuggingFace endpoint client (optional; requires HF_TOKEN)
export { NeuroQuantumClient } from "./client.js";

// NeuroQuantum REST API client (for Python backend integration)
export { NeuroQuantumAPIClient } from "./neuroquantum-api-client.js";

// HuggingFace dataset loader
export { HFDatasetLoader } from "./dataset.js";

export type {
  // QubitAI types
  QubitAIConfig,
  QubitAIResult,
  QubitAIInfo,
  QubitAIStatus,
  JudgmentRecord,
  PriorityItem,
  PriorityItemResult,
  // Judgment engine types
  GenerateOptions,
  GenerateResult,
  JudgmentCriteria,
  JudgeOptions,
  JudgmentResult,
  JudgmentType,
  NeuroQuantumClientConfig,
  PrioritizationResult,
  QualityEvalOptions,
  RiskAssessmentOptions,
  SafetyCheckOptions,
  // Dataset types
  DatasetToExamplesOptions,
  FetchRowsOptions,
  GenerateWithExamplesOptions,
  HFDatasetLoaderConfig,
  HFDatasetPage,
  HFDatasetRow,
  StreamRowsOptions,
  TrainFromDatasetOptions,
  TrainingExample,
  TrainingProgress,
  TrainingResult,
  // LLM provider types
  AdaptedTrainingExample,
  EvaluationMetrics,
  TrainingCheckpoint,
} from "./types.js";

// NeuroQuantum API client types
export type { NeuroQuantumResponse, NeuroQuantumAPIClientConfig } from "./neuroquantum-api-client.js";

// LLM provider types and errors
export type { LLMProviderConfig, LLMProviderStatus } from "./llm-provider.js";
export { ProviderNotImplementedError, ProviderConfigError } from "./llm-provider.js";
