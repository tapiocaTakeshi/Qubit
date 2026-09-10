import type {
  AddDocumentsResult,
  GenerateResult,
  GenerateWithExamplesOptions,
  NeuroQuantumClientConfig,
  SearchDocumentInput,
  SearchHit,
  SearchOptions,
  SearchResult,
  SearchStatus,
  TrainFromDatasetOptions,
  TrainingProgress,
  TrainingResult,
} from "./types.js";
import { HFDatasetLoader } from "./dataset.js";

const DEFAULT_ENDPOINT_URL = "https://api-inference.huggingface.co/models/neuroq-ai/quantum-llm";
const DEFAULT_TIMEOUT_MS = 600_000;
const DEFAULT_MAX_RETRIES = 12;
const DEFAULT_BATCH_SIZE = 10;
const DEFAULT_NUM_EXAMPLES = 3;
const DEFAULT_EXAMPLE_SEPARATOR = "\n\n";
const DEFAULT_EXAMPLE_TEMPLATE = "Q: {prompt}\nA: {completion}";
const DEFAULT_QUERY_TEMPLATE = "Q: {prompt}\nA:";

/**
 * Client for the NeuroQuantum inference service.
 *
 * Supports text generation with few-shot examples and fine-tuning from datasets.
 *
 * @example
 * ```ts
 * import { NeuroQuantumClient } from "qubit_ai";
 *
 * const client = new NeuroQuantumClient({ hfToken: process.env.HF_TOKEN });
 *
 * // Generate with few-shot examples
 * const result = await client.generateWithExamples(
 *   "何ですか？",
 *   [
 *     { prompt: "これは何？", completion: "これはペンです" },
 *     { prompt: "あれは何？", completion: "あれは猫です" },
 *   ]
 * );
 * console.log(result.generatedText);
 * ```
 */
export class NeuroQuantumClient {
  private readonly endpointUrl: string;
  private readonly hfToken: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly searchEndpointUrl: string;

  constructor(config: NeuroQuantumClientConfig = {}) {
    this.endpointUrl = config.endpointUrl ?? DEFAULT_ENDPOINT_URL;
    this.searchEndpointUrl = (config.searchEndpointUrl ?? this.endpointUrl).replace(/\/+$/, "");
    this.hfToken =
      config.hfToken ??
      (typeof globalThis !== "undefined" &&
      (globalThis as any).process?.env
        ? ((globalThis as any).process.env["HF_TOKEN"] ??
            (globalThis as any).process.env["HUGGING_FACE_HUB_TOKEN"] ??
            "")
        : "");
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.hfToken) {
      headers["Authorization"] = `Bearer ${this.hfToken}`;
    }
    return headers;
  }

  private async post(
    url: string,
    body: Record<string, unknown>
  ): Promise<unknown> {
    return this.request("POST", url, body);
  }

  private async request(
    method: "GET" | "POST" | "DELETE",
    url: string,
    body?: Record<string, unknown>
  ): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let lastError: Error | null = null;

    try {
      for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
        try {
          const res = await fetch(url, {
            method,
            headers: this.buildHeaders(),
            body: body === undefined ? undefined : JSON.stringify(body),
            signal: controller.signal,
          });

          if (!res.ok) {
            const text = await res.text();
            if (res.status === 503 && attempt < this.maxRetries) {
              // Service unavailable, retry
              lastError = new Error(`Service unavailable (503): ${text}`);
              continue;
            }
            throw new Error(
              `NeuroQuantum API error ${res.status}: ${text}`
            );
          }

          return await res.json();
        } catch (e) {
          lastError = e instanceof Error ? e : new Error(String(e));
          if (
            (lastError.message.includes("503") ||
              lastError.message.includes("Network")) &&
            attempt < this.maxRetries
          ) {
            // Retry on service unavailable or network error
            continue;
          }
          throw lastError;
        }
      }
    } finally {
      // Always release the abort timer so a resolved request cannot keep the
      // event loop (or a test runner) alive for the full timeout.
      clearTimeout(timer);
    }

    throw lastError || new Error("Max retries exceeded");
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Generate text conditioned on few-shot examples.
   *
   * @param prompt - The input prompt to generate from
   * @param examples - Array of prompt-completion pairs for few-shot learning
   * @param opts - Generation options
   * @returns The generation result
   */
  async generateWithExamples(
    prompt: string,
    examples: Array<{ prompt: string; completion: string }> = [],
    opts: GenerateWithExamplesOptions = {}
  ): Promise<GenerateResult> {
    const numExamples = opts.numExamples ?? DEFAULT_NUM_EXAMPLES;
    const exampleSeparator = opts.exampleSeparator ?? DEFAULT_EXAMPLE_SEPARATOR;
    const exampleTemplate = opts.exampleTemplate ?? DEFAULT_EXAMPLE_TEMPLATE;
    const queryTemplate = opts.queryTemplate ?? DEFAULT_QUERY_TEMPLATE;

    // Build the few-shot prompt
    const selectedExamples = examples.slice(0, numExamples);
    const exampleStrings = selectedExamples.map((ex) =>
      exampleTemplate
        .replace("{prompt}", ex.prompt)
        .replace("{completion}", ex.completion)
    );
    const querySuffix = queryTemplate.replace("{prompt}", prompt);

    const fullPrompt =
      exampleStrings.length > 0
        ? exampleStrings.join(exampleSeparator) + exampleSeparator + querySuffix
        : querySuffix;

    const body: Record<string, unknown> = {
      inputs: fullPrompt,
    };

    // Add generation options
    if (opts.maxNewTokens !== undefined) {
      (body as any).parameters = {
        ...((body as any).parameters ?? {}),
        max_new_tokens: opts.maxNewTokens,
      };
    }
    if (opts.temperature !== undefined) {
      (body as any).parameters = {
        ...((body as any).parameters ?? {}),
        temperature: opts.temperature,
      };
    }
    if (opts.topK !== undefined) {
      (body as any).parameters = {
        ...((body as any).parameters ?? {}),
        top_k: opts.topK,
      };
    }
    if (opts.topP !== undefined) {
      (body as any).parameters = {
        ...((body as any).parameters ?? {}),
        top_p: opts.topP,
      };
    }
    if (opts.repetitionPenalty !== undefined) {
      (body as any).parameters = {
        ...((body as any).parameters ?? {}),
        repetition_penalty: opts.repetitionPenalty,
      };
    }

    const result = (await this.post(this.endpointUrl, body)) as Array<
      Record<string, unknown>
    >;
    const firstResult = result[0];

    return {
      generatedText: String(firstResult?.["generated_text"] ?? ""),
      raw: result,
    };
  }

  // -------------------------------------------------------------------------
  // Search (retrieval / RAG)
  // -------------------------------------------------------------------------

  private static toSearchHit(raw: Record<string, unknown>): SearchHit {
    return {
      docId: String(raw["doc_id"] ?? ""),
      text: String(raw["text"] ?? ""),
      score: Number(raw["score"] ?? 0),
      bm25Score: Number(raw["bm25_score"] ?? 0),
      denseScore: Number(raw["dense_score"] ?? 0),
      metadata: (raw["metadata"] as Record<string, unknown>) ?? {},
      rank: Number(raw["rank"] ?? 0),
    };
  }

  /**
   * Register documents in the NeuroQuantum search index.
   *
   * Plain strings are accepted as well as `{ text, id?, metadata? }` objects.
   * Re-registering an existing `id` replaces that document.
   */
  async addDocuments(
    documents: Array<string | SearchDocumentInput>
  ): Promise<AddDocumentsResult> {
    if (documents.length === 0) {
      throw new Error("addDocuments: documents must not be empty");
    }
    const payload = documents.map((d) =>
      typeof d === "string" ? { text: d } : { text: d.text, id: d.id, metadata: d.metadata ?? {} }
    );
    const raw = (await this.post(`${this.searchEndpointUrl}/search/documents`, {
      documents: payload,
    })) as Record<string, unknown>;
    return {
      added: Number(raw["added"] ?? 0),
      total: Number(raw["total"] ?? 0),
      docIds: ((raw["doc_ids"] as unknown[]) ?? []).map(String),
    };
  }

  /**
   * Search the registered documents.
   *
   * Results are ordered by descending score. With a loaded NeuroQuantum model
   * the server combines BM25 keyword matching with embedding similarity
   * (`hybrid`); without one it falls back to BM25.
   */
  async search(query: string, opts: SearchOptions = {}): Promise<SearchResult> {
    if (!query.trim()) {
      throw new Error("search: query must not be empty");
    }
    const body: Record<string, unknown> = { query };
    if (opts.topK !== undefined) body["top_k"] = opts.topK;
    if (opts.mode !== undefined) body["mode"] = opts.mode;
    if (opts.minScore !== undefined) body["min_score"] = opts.minScore;
    if (opts.metadataFilter !== undefined) body["metadata_filter"] = opts.metadataFilter;

    const raw = (await this.post(`${this.searchEndpointUrl}/search`, body)) as Record<
      string,
      unknown
    >;
    const results = ((raw["results"] as Array<Record<string, unknown>>) ?? []).map(
      NeuroQuantumClient.toSearchHit
    );
    return {
      query: String(raw["query"] ?? query),
      mode: String(raw["mode"] ?? ""),
      totalDocuments: Number(raw["total_documents"] ?? 0),
      results,
      raw,
    };
  }

  /** Remove one document (by id) or every document from the search index. */
  async clearDocuments(docId?: string): Promise<{ total: number }> {
    const url =
      docId === undefined
        ? `${this.searchEndpointUrl}/search/documents`
        : `${this.searchEndpointUrl}/search/documents?doc_id=${encodeURIComponent(docId)}`;
    const raw = (await this.request("DELETE", url)) as Record<string, unknown>;
    return { total: Number(raw["total"] ?? 0) };
  }

  /** Current state of the server-side search index. */
  async searchStatus(): Promise<SearchStatus> {
    const raw = (await this.request("GET", `${this.searchEndpointUrl}/search/status`)) as Record<
      string,
      unknown
    >;
    return {
      documents: Number(raw["documents"] ?? 0),
      mode: String(raw["mode"] ?? ""),
      alpha: Number(raw["alpha"] ?? 0),
      denseAvailable: Boolean(raw["dense_available"]),
      ngram: Number(raw["ngram"] ?? 0),
    };
  }

  /**
   * Train (fine-tune) the model on examples from a HuggingFace dataset.
   *
   * @param opts - Training configuration
   * @returns Training result with status and metrics
   */
  async trainFromDataset(
    opts: TrainFromDatasetOptions
  ): Promise<TrainingResult> {
    const batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;
    const trainingEndpointUrl =
      opts.trainingEndpointUrl ?? `${this.endpointUrl}/train`;

    const loader = new HFDatasetLoader();

    const examples = await loader.loadExamples(opts);
    const startTime = Date.now();
    const errors: string[] = [];
    let processedExamples = 0;
    let totalBatches = Math.ceil(examples.length / batchSize);

    for (let i = 0; i < examples.length; i += batchSize) {
      const batch = examples.slice(i, i + batchSize);
      const currentBatch = Math.floor(i / batchSize) + 1;

      try {
        await this.post(trainingEndpointUrl, {
          examples: batch,
        });
      } catch (e) {
        errors.push(
          e instanceof Error ? e.message : String(e)
        );
        // Continue processing other batches
      }

      processedExamples += batch.length;

      if (opts.onProgress) {
        opts.onProgress({
          processedExamples,
          totalExamples: examples.length,
          currentBatch,
          totalBatches,
          elapsedMs: Date.now() - startTime,
        });
      }
    }

    const durationMs = Date.now() - startTime;
    let status: "completed" | "partial" | "failed" = "completed";
    if (errors.length > 0) {
      status = processedExamples === 0 ? "failed" : "partial";
    }

    return {
      totalExamples: examples.length,
      batches: totalBatches,
      durationMs,
      status,
      errors: errors.length > 0 ? errors : undefined,
    };
  }
}
