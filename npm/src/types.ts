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
  system: "qbnn" | "heuristic";
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
