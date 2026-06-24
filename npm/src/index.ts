/**
 * qubit_ai — Quantum-inspired AI with Pyodide backend
 *
 * Execute neuroquantum_layered.py in WebAssembly + HuggingFace dataset training
 *
 * @packageDocumentation
 */

// Primary export: QubitAI with Pyodide backend (simplified for quantum + HF training)
export {
  QubitAIPyodide,
  getQubitAIPyodide,
  resetQubitAIPyodide,
  judge,
  safetyCheck,
  ethicsCheck,
  evaluateQuality,
} from "./qubit_ai_pyodide.js";

// Legacy exports (deprecated, use QubitAIPyodide instead)
export {
  QubitAI,
  getQubitAI,
  resetQubitAI,
} from "./qubit_ai.js";

// Pyodide backend
export {
  NeuroQuantumPyodide,
  getNeuroQuantumPyodide,
  resetNeuroQuantumPyodide,
  initPyodide,
  executePython,
  loadNeuroQuantumModule,
} from "./pyodide-wrapper.js";

// Lower-level engines (legacy, not used with Pyodide backend)
export { QBNNFrontalEngine } from "./frontal.js";

// HuggingFace dataset loader (for Pyodide training)
export { HFDatasetLoader } from "./dataset.js";

export type {
  // Core Qubit AI types
  QubitAIConfig,
  QubitAIResult,
  QubitAIInfo,
  QubitAIStatus,
  JudgmentRecord,
  JudgmentType,
  // Training types
  TrainingProgress,
  TrainingResult,
  // Dataset types
  HFDatasetLoaderConfig,
  HFDatasetRow,
  StreamRowsOptions,
} from "./types.js";
