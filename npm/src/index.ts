/**
 * qubit_ai — Generative AI with Pyodide + NeuroQuantum
 *
 * Quantum-inspired text generation without external APIs
 * Pure in-browser or Node.js generative AI using Pyodide
 * Also includes judgment engine for safety, ethics, and quality evaluation
 *
 * @packageDocumentation
 */

// Primary export: QubitAI Generative (Pyodide-based)
export {
  QubitAIGenerative,
  getQubitAIGenerative,
  resetQubitAIGenerative,
  resetQubitAIGenerativeSession,
  generate,
  generateWithExamples,
  generateBatch,
  trainOnData,
  trainOnHFDataset,
  type GenerationOptions,
  type GenerationResult,
  type TrainingProgress,
  type TrainingResult,
} from "./qubit_ai_generative_pyodide.js";

// Core generation engine
export {
  NeuroQuantumGenerator,
  SimpleTokenizer,
  LightweightLanguageModel,
  type GenerationConfig,
  type GenerationResult as GeneratorResult,
  type ModelStatus,
} from "./pyodide-generator.js";

// Judgment engine exports
export {
  QubitAI,
  getQubitAI,
  resetQubitAI,
  judge,
  safetyCheck,
  evaluateQuality,
  ethicsCheck,
} from "./qubit_ai.js";

export { NeuroQuantumAPIClient } from "./neuroquantum-api-client.js";
export { NeuroQuantumFrontalEngine } from "./neuroquantum-frontal.js";
export { QBNNFrontalEngine } from "./frontal.js";
export { LLMFrontalEngine } from "./llm-frontal.js";
export { HybridFrontalEngine } from "./hybrid-frontal.js";
export { HFDatasetLoader } from "./dataset.js";
export { NeuroQuantumClient } from "./client.js";
export { LLMProvider } from "./llm-provider.js";
export { ClaudeProvider } from "./llm-provider-claude.js";
export { OpenAIProvider } from "./llm-provider-openai.js";
export { HuggingFaceProvider } from "./llm-provider-hf.js";

export type {
  QubitAIConfig,
  QubitAIInfo,
  QubitAIStatus,
  QubitAIResult,
  JudgmentRecord,
  JudgmentType,
  SafetyCheckOptions,
  QualityEvalOptions,
  PriorityItem,
  PriorityItemResult,
  EvaluationMetrics,
  NeuroQuantumResponse,
  NeuroQuantumConfig,
} from "./types.js";
