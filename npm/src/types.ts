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
  /**
   * Base URL of the NeuroQuantum search API (the FastAPI server from
   * `api.py`). `/search`, `/search/documents` and `/search/status` are
   * appended to it. Defaults to `endpointUrl`.
   */
  searchEndpointUrl?: string;
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
// NeuroQuantum search (retrieval / RAG) types
// ---------------------------------------------------------------------------

/** A document to register in the NeuroQuantum search index */
export interface SearchDocumentInput {
  /** Document text (required) */
  text: string;
  /** Optional stable id; re-registering the same id replaces the document */
  id?: string;
  /** Arbitrary metadata, usable with `metadataFilter` when searching */
  metadata?: Record<string, unknown>;
}

/** Result of registering documents */
export interface AddDocumentsResult {
  /** Number of documents added in this call */
  added: number;
  /** Total number of documents now in the index */
  total: number;
  /** Ids assigned to the added documents (in input order) */
  docIds: string[];
}

/** Search scoring mode. `hybrid` = BM25 + NeuroQuantum embeddings. */
export type SearchMode = "hybrid" | "bm25" | "dense";

/** Options for searching the NeuroQuantum document index */
export interface SearchOptions {
  /** Number of results to return (default: 5) */
  topK?: number;
  /** Scoring mode; defaults to the server's index mode */
  mode?: SearchMode;
  /** Drop results scoring below this value (default: 0) */
  minScore?: number;
  /** Only return documents whose metadata matches every key/value */
  metadataFilter?: Record<string, unknown>;
}

/** One search hit */
export interface SearchHit {
  docId: string;
  text: string;
  /** Combined score in [0, 1] (higher is better) */
  score: number;
  /** Raw BM25 score */
  bm25Score: number;
  /** Cosine similarity from NeuroQuantum embeddings (0 when unavailable) */
  denseScore: number;
  metadata: Record<string, unknown>;
  /** 1-based rank */
  rank: number;
}

/** Result of a search call */
export interface SearchResult {
  query: string;
  mode: string;
  /** Number of documents in the index */
  totalDocuments: number;
  results: SearchHit[];
  /** Raw response from the endpoint */
  raw?: unknown;
}

/** Status of the server-side search index */
export interface SearchStatus {
  documents: number;
  mode: string;
  alpha: number;
  denseAvailable: boolean;
  ngram: number;
}

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
