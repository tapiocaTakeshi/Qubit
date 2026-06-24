/**
 * qubit_ai — Generative AI with Pyodide + NeuroQuantum
 *
 * Quantum-inspired text generation without external APIs
 * Pure in-browser or Node.js generative AI using Pyodide
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
