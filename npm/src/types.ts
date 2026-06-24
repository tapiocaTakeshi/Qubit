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

/** Result returned by the inference endpoint */
export interface GenerateResult {
  /** Generated text */
  generatedText: string;
  /** Internal debug info, only present when the endpoint returns it */
  debug?: Record<string, unknown>;
  /** Raw response from the endpoint (preserved for advanced usage) */
  raw?: unknown;
}

/** Configuration for NeuroQuantumClient */
export interface NeuroQuantumClientConfig {
  /**
   * HuggingFace inference endpoint URL.
   * Defaults to the public neuroQ endpoint.
   */
  endpointUrl?: string;
  /**
   * HuggingFace API token.
   * Falls back to the HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable.
   */
  hfToken?: string;
  /** Request timeout in milliseconds (default: 600_000) */
  timeoutMs?: number;
  /** Number of retry attempts on 503 / network error (default: 12) */
  maxRetries?: number;
}

// ---------------------------------------------------------------------------
// Judgment / Frontal Engine types
// ---------------------------------------------------------------------------

/** The type of judgment to perform */
export type JudgmentType =
  | "safety"
  | "ethics"
  | "quality"
  | "risk"
  | "decision"
  | "priority";

/** Criteria dictionary used to guide judgment scoring */
export type JudgmentCriteria = Record<string, string | number | boolean>;

/** Options passed to a judgment call */
export interface JudgeOptions {
  /** Type of judgment (default: "decision") */
  type?: JudgmentType;
  /** Structured criteria to evaluate against */
  criteria?: JudgmentCriteria;
  /** Candidate options / actions to rank */
  options?: string[];
  /** Strict mode: require score ≥ 70 for a "Yes" decision (default: false) */
  strictMode?: boolean;
}

/** Result of a judgment call */
export interface JudgmentResult {
  /** Binary decision */
  decision: "Yes" | "No";
  /** Confidence score 0–100 */
  score: number;
  /** Human-readable reasoning */
  reasoning: string;
  /** Confidence level */
  confidence: "high" | "medium" | "low";
  /** Key factors that influenced the score */
  keyFactors: string[];
  /** ISO timestamp */
  timestamp: string;
  /** System that produced the judgment */
  system: "qbnn" | "heuristic" | "llm" | "hybrid" | "neuroquantum";
  /** Optional: LLM model used (if system is "llm" or "hybrid") */
  llmModelUsed?: string;
  /** Optional: Processing time in milliseconds */
  processingTimeMs?: number;
}

/** Options for safety / ethics / risk / quality convenience wrappers */
export interface SafetyCheckOptions {
  /** Additional known risks to consider */
  risks?: string[];
  /** Constraint map (e.g. { "pii": true, "scope": "internal" }) */
  constraints?: Record<string, unknown>;
}

export interface QualityEvalOptions {
  /** Requirements the content must satisfy */
  requirements?: string[];
  /** Original user intent (for context) */
  userIntent?: string;
}

export interface RiskAssessmentOptions {
  /** Risk tolerance 0–100 (default: 50) */
  riskTolerance?: number;
}

export interface PrioritizationResult {
  /** Original task descriptions in ranked order */
  rankedTasks: string[];
  /** Score for each task */
  scores: number[];
  /** Reasoning per task */
  reasonings: string[];
}

// ---------------------------------------------------------------------------
// HuggingFace Dataset types
// ---------------------------------------------------------------------------

/** A single row from a HuggingFace dataset */
export interface HFDatasetRow {
  /** Zero-based row index within the split */
  rowIdx: number;
  /** Column values for this row */
  row: Record<string, unknown>;
}

/** A page of rows returned from the HF Datasets Server API */
export interface HFDatasetPage {
  rows: HFDatasetRow[];
  /** Total number of rows in the split */
  numRowsTotal: number;
}

/** Options for loading data from a HuggingFace dataset */
export interface HFDatasetLoaderConfig {
  /**
   * HuggingFace API token.
   * Required for private datasets; falls back to HF_TOKEN env var.
   */
  hfToken?: string;
  /**
   * Base URL for the HF Datasets Server.
   * Defaults to https://datasets-server.huggingface.co
   */
  datasetsServerUrl?: string;
  /** Request timeout in milliseconds (default: 30_000) */
  timeoutMs?: number;
}

/** Options for fetching rows from a dataset */
export interface FetchRowsOptions {
  /** Dataset name on HuggingFace Hub (e.g. "llm-jp/oasst2-33k-ja") */
  dataset: string;
  /** Configuration name (default: "default") */
  config?: string;
  /** Split name (default: "train") */
  split?: string;
  /** Row offset for pagination (default: 0) */
  offset?: number;
  /** Number of rows to fetch per page, max 100 (default: 100) */
  limit?: number;
}

/** Options for streaming all rows from a dataset */
export interface StreamRowsOptions extends Omit<FetchRowsOptions, "offset" | "limit"> {
  /** Maximum number of rows to stream in total (default: unlimited) */
  maxRows?: number;
  /** Rows per page internally (default: 100) */
  pageSize?: number;
}

/** A prompt–completion pair used for fine-tuning or few-shot learning */
export interface TrainingExample {
  prompt: string;
  completion: string;
}

/** Options for converting a HF dataset into training examples */
export interface DatasetToExamplesOptions extends StreamRowsOptions {
  /** Column to use as the prompt (default: "input" or first column) */
  promptField?: string;
  /** Column to use as the completion (default: "output" or second column) */
  completionField?: string;
  /**
   * Custom row-to-example transformer. When provided, promptField and
   * completionField are ignored.
   */
  transform?: (row: Record<string, unknown>) => TrainingExample | null;
}

/** Progress event emitted during dataset training */
export interface TrainingProgress {
  processedExamples: number;
  totalExamples: number;
  currentBatch: number;
  totalBatches: number;
  /** Elapsed time in milliseconds */
  elapsedMs: number;
}

/** Result of a trainFromDataset call */
export interface TrainingResult {
  totalExamples: number;
  batches: number;
  durationMs: number;
  status: "completed" | "partial" | "failed";
  errors?: string[];
}

/** Options for training from a HF dataset via a remote endpoint */
export interface TrainFromDatasetOptions extends DatasetToExamplesOptions {
  /**
   * URL of the fine-tuning endpoint.
   * Falls back to NeuroQuantumClientConfig.endpointUrl with path "/train".
   */
  trainingEndpointUrl?: string;
  /** Training examples per HTTP batch (default: 10) */
  batchSize?: number;
  /** Callback invoked after each batch */
  onProgress?: (progress: TrainingProgress) => void;
}

// ---------------------------------------------------------------------------
// QubitAI types (TypeScript port of qubit_ai.py)
// ---------------------------------------------------------------------------

/** Configuration for QubitAI */
export interface QubitAIConfig {
  /** Library version (default: "1.1.0") */
  version?: string;
  /** Product name (default: "Qubit.ai") */
  productName?: string;
  /** Product description */
  description?: string;
  /** Strict mode: require score ≥ 70 for "Yes" decisions (default: false) */
  strictMode?: boolean;
  /** Enable logging (default: true) */
  enableLogging?: boolean;
  /** Maximum number of judgment records to keep in history (default: 100) */
  maxJudgmentHistory?: number;

  // LLM-based judgment configuration
  /** Enable LLM-based inference (default: false, uses heuristics) */
  llmEnabled?: boolean;
  /** Which LLM provider to use: 'hf' | 'claude' | 'openai' */
  llmProvider?: "hf" | "claude" | "openai" | "custom";
  /** LLM provider configuration */
  llmConfig?: {
    /** API key or authentication token */
    apiKey?: string;
    /** Custom endpoint URL (for HuggingFace) */
    endpoint?: string;
    /** Model identifier */
    model?: string;
    /** Sampling temperature (0.0-2.0, default: 0.7) */
    temperature?: number;
    /** Maximum tokens to generate (default: 500) */
    maxTokens?: number;
    /** Request timeout in milliseconds (default: 30000) */
    timeout?: number;
    /** Number of retries on failure (default: 3) */
    maxRetries?: number;
  };
  /** Fall back to heuristic engine if LLM fails (default: false) */
  fallbackToHeuristics?: boolean;
  /** Custom LLM prompt template override */
  llmPromptTemplate?: string;
  /** Blending strategy for hybrid mode: 'weighted' | 'confidence-based' (default: 'weighted') */
  llmBlendStrategy?: "weighted" | "confidence-based";

  // NeuroQuantum backend configuration (Python REST API)
  /** Enable NeuroQuantum backend (Python quantum-inspired neural reasoning) */
  neuroquantumEnabled?: boolean;
  /** NeuroQuantum API client configuration */
  neuroquantumConfig?: {
    /** Base URL of NeuroQuantum REST API (default: http://localhost:5000) */
    baseUrl?: string;
    /** Request timeout in milliseconds (default: 30000) */
    timeout?: number;
    /** Number of retries on failure (default: 3) */
    maxRetries?: number;
    /** Retry delay in milliseconds (default: 1000) */
    retryDelayMs?: number;
  };
}

/** Result returned by QubitAI judgment methods */
export interface QubitAIResult {
  /** Binary decision */
  decision: "Yes" | "No";
  /** Confidence score 0–100 */
  score: number;
  /** Human-readable reasoning */
  reasoning: string;
  /** Confidence level */
  confidence: "high" | "medium" | "low";
  /** Key factors that influenced the decision */
  factors: string[];
  /** ISO timestamp */
  timestamp: string;
}

/** System information returned by getInfo() */
export interface QubitAIInfo {
  product: string;
  version: string;
  description: string;
  sessionId: string;
  initializedAt: string;
  status: "operational";
}

/** System status returned by getStatus() */
export interface QubitAIStatus {
  product: string;
  status: "operational";
  frontalEngineAvailable: boolean;
  judgmentHistorySize: number;
  maxHistory: number;
  timestamp: string;
}

/** A single judgment history record */
export interface JudgmentRecord {
  timestamp: string;
  judgmentType: JudgmentType;
  contextPreview: string;
  decision: "Yes" | "No";
  score: number;
  confidence: "high" | "medium" | "low";
}

/** An item to prioritize */
export interface PriorityItem {
  name: string;
  description: string;
}

/** Result of a prioritize() call — item paired with its normalised score (0–1) */
export type PriorityItemResult = [PriorityItem, number];

/** Options for few-shot generation using dataset examples */
export interface GenerateWithExamplesOptions extends GenerateOptions {
  /** Number of few-shot examples to include in context (default: 3) */
  numExamples?: number;
  /** Separator between examples in the prompt (default: "\n\n") */
  exampleSeparator?: string;
  /** Format string for each example; use {prompt} and {completion} (default: "Q: {prompt}\nA: {completion}") */
  exampleTemplate?: string;
  /** Suffix appended before the model generates; use {prompt} (default: "Q: {prompt}\nA:") */
  queryTemplate?: string;
}

// ---------------------------------------------------------------------------
// LLM Training types
// ---------------------------------------------------------------------------

/** Training example adapted for judgment tasks */
export interface AdaptedTrainingExample extends TrainingExample {
  /** Original prompt from dataset */
  prompt: string;
  /** Original completion from dataset */
  completion: string;
  /** Adapted prompt for judgment */
  judgmentPrompt: string;
  /** Adapted completion in JSON format */
  judgmentCompletion: string;
  /** Source dataset and row ID */
  source: string;
  /** Which judgment type this example is for */
  judgmentType: JudgmentType;
}

/** Evaluation metrics from model testing */
export interface EvaluationMetrics {
  /** Total examples evaluated */
  totalExamples: number;
  /** Correct decision predictions (Yes/No match) */
  correctDecisions: number;
  /** Accuracy rate (0-1) */
  accuracy: number;
  /** Mean Absolute Error on scores */
  scoreMAE: number;
  /** Accuracy on confidence predictions */
  confidenceAccuracy: number;
  /** Weighted F1 score */
  f1Score: number;
  /** Average inference time per example (ms) */
  inferenceTimeMs: number;
  /** Failed predictions and errors */
  errors: string[];
}

/** Checkpoint for resuming training */
export interface TrainingCheckpoint {
  /** Dataset name being trained on */
  datasetName: string;
  /** Judgment type being trained for */
  judgmentType: JudgmentType;
  /** Examples processed so far */
  examplesProcessed: number;
  /** Batches processed so far */
  batchesProcessed: number;
  /** Last example processed */
  lastExample?: TrainingExample;
  /** Checkpoint timestamp */
  timestamp: string;
  /** Model checkpoint ID or path */
  modelCheckpoint: string;
}
