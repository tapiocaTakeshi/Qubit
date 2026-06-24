/**
 * qubit_ai — quantum-inspired QBNN inference & judgment engine
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

// Lower-level engine (pure QBNN scoring, no external deps)
export { QBNNFrontalEngine } from "./frontal.js";

// HuggingFace endpoint client (optional; requires HF_TOKEN)
export { NeuroQuantumClient } from "./client.js";

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
} from "./types.js";
