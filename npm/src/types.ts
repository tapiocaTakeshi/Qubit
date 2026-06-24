/**
 * Qubit AI v4.0.0 - Pyodide + NeuroQuantum Type Definitions
 *
 * Types for generative AI with quantum-inspired sampling
 */

// ============================================
// Generation Types
// ============================================

/** Generation parameters for text inference */
export interface GenerateOptions {
  /** Maximum number of tokens to generate (default: 100) */
  maxNewTokens?: number;
  /** Sampling temperature — higher = more creative (default: 0.7) */
  temperature?: number;
  /** Top-K sampling cutoff (default: 40) */
  topK?: number;
  /** Nucleus sampling threshold (default: 0.9) */
  topP?: number;
  /** Penalty to reduce token repetition (default: 1.3) */
  repetitionPenalty?: number;
}

/** Result returned by the generation engine */
export interface GenerateResult {
  /** Generated text */
  generatedText: string;
  /** Internal debug info, only present when available */
  debug?: Record<string, unknown>;
  /** Raw response from the backend (preserved for advanced usage) */
  raw?: unknown;
}

// ============================================
// HuggingFace Dataset Types
// ============================================

/** Configuration for HFDatasetLoader */
export interface HFDatasetLoaderConfig {
  /**
   * HuggingFace API token.
   * Falls back to HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable.
   */
  hfToken?: string;
  /** Request timeout in milliseconds (default: 30_000) */
  timeoutMs?: number;
  /** Number of retry attempts (default: 3) */
  maxRetries?: number;
}

/** A single row from a HuggingFace dataset */
export interface HFDatasetRow {
  [key: string]: unknown;
}

/** Options for streaming rows from a dataset */
export interface StreamRowsOptions {
  /** HuggingFace dataset name (e.g., "wikitext", "openwebtext") */
  dataset: string;
  /** Dataset split (default: "train") */
  split?: string;
  /** Maximum number of rows to stream */
  maxRows?: number;
  /** Streaming batch size (default: 100) */
  batchSize?: number;
}

// ============================================
// Training Types
// ============================================

/** Progress update during training */
export interface TrainingProgress {
  /** Number of examples processed so far */
  processedExamples: number;
  /** Total number of examples */
  totalExamples: number;
  /** Current batch number */
  currentBatch: number;
  /** Total number of batches */
  totalBatches: number;
}

/** Result of training operation */
export interface TrainingResult {
  /** Total examples processed */
  totalExamples: number;
  /** Number of batches */
  batches: number;
  /** Training duration in milliseconds */
  durationMs: number;
  /** Training status (e.g., "completed", "failed") */
  status: string;
  /** Errors encountered during training */
  errors: string[];
}

// ============================================
// Model Configuration Types
// ============================================

/** Configuration for NeuroQuantumClient (legacy compatibility) */
export interface NeuroQuantumClientConfig {
  /**
   * HuggingFace inference endpoint URL.
   * Defaults to the public neuroQ endpoint.
   */
  endpointUrl?: string;
  /**
   * HuggingFace API token.
   * Falls back to HF_TOKEN environment variable.
   */
  hfToken?: string;
  /** Request timeout in milliseconds (default: 600_000) */
  timeoutMs?: number;
  /** Number of retry attempts on 503 / network error (default: 12) */
  maxRetries?: number;
}

/** Generic QubitAI configuration */
export interface QubitAIConfig {
  /** Version string */
  version?: string;
  /** Product name */
  productName?: string;
  /** Product description */
  description?: string;
  /** Strict mode: require higher confidence threshold */
  strictMode?: boolean;
  /** Enable detailed logging */
  enableLogging?: boolean;
  /** Maximum judgment history size */
  maxJudgmentHistory?: number;
  /** Vocabulary size for tokenizer */
  vocabSize?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

// ============================================
// Backward Compatibility (Legacy)
// ============================================

/** The type of judgment to perform (legacy - kept for compatibility) */
export type JudgmentType =
  | "safety"
  | "ethics"
  | "quality"
  | "risk"
  | "decision"
  | "priority";

/** Criteria dictionary (legacy) */
export type JudgmentCriteria = Record<string, string | number | boolean>;

/** Options for judgment calls (legacy) */
export interface JudgeOptions {
  type?: JudgmentType;
  criteria?: JudgmentCriteria;
  options?: string[];
  strictMode?: boolean;
}

/** Result of a judgment call (legacy) */
export interface JudgmentResult {
  decision: "Yes" | "No";
  score: number;
  reasoning: string;
  confidence: "high" | "medium" | "low";
  keyFactors: string[];
  timestamp: string;
  system: "qbnn" | "heuristic" | "llm" | "hybrid";
  llmModelUsed?: string;
  processingTimeMs?: number;
}

/** Safety check options (legacy) */
export interface SafetyCheckOptions {
  risks?: string[];
  constraints?: Record<string, unknown>;
}
