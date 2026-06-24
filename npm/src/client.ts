import type {
  GenerateOptions,
  GenerateResult,
  GenerateWithExamplesOptions,
  NeuroQuantumClientConfig,
  TrainFromDatasetOptions,
  TrainingExample,
  TrainingProgress,
  TrainingResult,
} from "./types.js";
import { HFDatasetLoader } from "./dataset.js";

const DEFAULT_ENDPOINT =
  "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud";

const DEFAULT_TIMEOUT_MS = 600_000;
const DEFAULT_MAX_RETRIES = 12;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * HTTP client for the neuroQ HuggingFace Inference Endpoint.
 *
 * Mirrors the retry and payload logic of `hf_inference.py` in TypeScript.
 *
 * @example
 * ```ts
 * import { NeuroQuantumClient } from "neuroquantum";
 *
 * const client = new NeuroQuantumClient({ hfToken: process.env.HF_TOKEN });
 * const result = await client.generate("量子コンピュータとは何ですか？");
 * console.log(result.generatedText);
 * ```
 */
export class NeuroQuantumClient {
  private readonly endpointUrl: string;
  private readonly hfToken: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(config: NeuroQuantumClientConfig = {}) {
    this.endpointUrl = config.endpointUrl ?? DEFAULT_ENDPOINT;
    this.hfToken =
      config.hfToken ??
      (typeof process !== "undefined"
        ? (process.env["HF_TOKEN"] ?? process.env["HUGGING_FACE_HUB_TOKEN"] ?? "")
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

  private async sendRequest(payload: unknown): Promise<unknown> {
    const body = JSON.stringify(payload);
    const headers = this.buildHeaders();
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const res = await fetch(this.endpointUrl, {
          method: "POST",
          headers,
          body,
          signal: controller.signal,
        });
        clearTimeout(timer);

        if (res.status === 503 && attempt < this.maxRetries - 1) {
          const waitMs = Math.min((10 + attempt * 5) * 1000, 30_000);
          console.warn(
            `[neuroQ] Endpoint unavailable (503), retrying in ${waitMs / 1000}s… ` +
              `(${attempt + 1}/${this.maxRetries})`
          );
          await sleep(waitMs);
          continue;
        }

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
        }

        return await res.json();
      } catch (err) {
        clearTimeout(timer);
        lastError = err instanceof Error ? err : new Error(String(err));

        // Retry on network / abort errors
        const isRetryable =
          lastError.name === "AbortError" ||
          lastError.name === "TypeError" ||
          lastError.message.includes("fetch");

        if (isRetryable && attempt < this.maxRetries - 1) {
          const waitMs = Math.min((10 + attempt * 5) * 1000, 30_000);
          console.warn(
            `[neuroQ] Connection error, retrying in ${waitMs / 1000}s… ` +
              `(${attempt + 1}/${this.maxRetries}): ${lastError.message}`
          );
          await sleep(waitMs);
          continue;
        }
        throw lastError;
      }
    }
    throw lastError ?? new Error("Max retries exceeded");
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Generate text from a prompt.
   *
   * @param prompt - Input text (Japanese or English)
   * @param options - Sampling / generation parameters
   */
  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult> {
    const payload = {
      inputs: prompt,
      parameters: {
        max_new_tokens: options.maxNewTokens ?? 100,
        temperature: options.temperature ?? 0.7,
        top_k: options.topK ?? 40,
        top_p: options.topP ?? 0.9,
        repetition_penalty: options.repetitionPenalty ?? 1.3,
      },
    };

    const raw = await this.sendRequest(payload);

    // The endpoint returns either [{generated_text: "..."}] or {generated_text: "..."}
    if (Array.isArray(raw) && raw.length > 0) {
      const first = raw[0] as Record<string, unknown>;
      return {
        generatedText: String(first["generated_text"] ?? ""),
        debug: first["debug"] as Record<string, unknown> | undefined,
        raw: first,
      };
    }
    if (raw && typeof raw === "object") {
      const obj = raw as Record<string, unknown>;
      return {
        generatedText: String(obj["generated_text"] ?? ""),
        debug: obj["debug"] as Record<string, unknown> | undefined,
        raw: obj,
      };
    }
    return { generatedText: String(raw ?? ""), raw };
  }

  /**
   * Check the health / status of the model endpoint.
   */
  async status(): Promise<Record<string, unknown>> {
    const raw = await this.sendRequest({
      inputs: "__status__",
      parameters: { action: "status" },
    });
    return raw as Record<string, unknown>;
  }

  // -------------------------------------------------------------------------
  // HuggingFace Dataset learning
  // -------------------------------------------------------------------------

  /**
   * Generate text using few-shot examples as in-context learning context.
   *
   * Examples are prepended to the prompt using the specified templates, so
   * the model can learn the expected input→output pattern at inference time.
   *
   * @param prompt   - The query to answer
   * @param examples - Prompt–completion pairs to use as few-shot context
   * @param options  - Generation and template options
   *
   * @example
   * ```ts
   * const loader = new HFDatasetLoader({ hfToken: process.env.HF_TOKEN });
   * const examples = await loader.preview("llm-jp/oasst2-33k-ja", 3);
   *
   * const result = await client.generateWithExamples(
   *   "量子コンピュータの利点を教えてください",
   *   examples,
   *   { numExamples: 3 }
   * );
   * ```
   */
  async generateWithExamples(
    prompt: string,
    examples: TrainingExample[],
    options: GenerateWithExamplesOptions = {}
  ): Promise<GenerateResult> {
    const numExamples = options.numExamples ?? 3;
    const separator = options.exampleSeparator ?? "\n\n";
    const exampleTemplate =
      options.exampleTemplate ?? "Q: {prompt}\nA: {completion}";
    const queryTemplate = options.queryTemplate ?? "Q: {prompt}\nA:";

    const selected = examples.slice(0, numExamples);
    const fewShotBlock = selected
      .map((ex) =>
        exampleTemplate
          .replace("{prompt}", ex.prompt)
          .replace("{completion}", ex.completion)
      )
      .join(separator);

    const queryBlock = queryTemplate.replace("{prompt}", prompt);
    const fullPrompt = fewShotBlock
      ? `${fewShotBlock}${separator}${queryBlock}`
      : queryBlock;

    const { numExamples: _n, exampleSeparator: _s, exampleTemplate: _et, queryTemplate: _qt, ...genOptions } =
      options;
    return this.generate(fullPrompt, genOptions);
  }

  /**
   * Train the remote endpoint using examples streamed from a HuggingFace dataset.
   *
   * Rows are loaded via {@link HFDatasetLoader}, converted to prompt–completion
   * pairs, and sent to the training endpoint in batches.  A progress callback
   * can be provided to track progress.
   *
   * @param opts - Dataset, field-mapping, batching, and progress options
   *
   * @example
   * ```ts
   * const result = await client.trainFromDataset({
   *   dataset: "llm-jp/oasst2-33k-ja",
   *   promptField: "input",
   *   completionField: "output",
   *   maxRows: 500,
   *   batchSize: 10,
   *   trainingEndpointUrl: "https://my-training-endpoint/train",
   *   onProgress: (p) =>
   *     console.log(`${p.processedExamples}/${p.totalExamples} examples`),
   * });
   * console.log(result.status, result.totalExamples);
   * ```
   */
  async trainFromDataset(opts: TrainFromDatasetOptions): Promise<TrainingResult> {
    const trainingUrl =
      opts.trainingEndpointUrl ??
      this.endpointUrl.replace(/\/?$/, "/train");
    const batchSize = opts.batchSize ?? 10;

    const loader = new HFDatasetLoader({ hfToken: this.hfToken });

    const startTime = Date.now();
    const errors: string[] = [];
    let processedExamples = 0;
    let currentBatch = 0;

    const queue: TrainingExample[] = [];

    const flushBatch = async (
      batch: TrainingExample[],
      totalExamples: number,
      totalBatches: number
    ): Promise<void> => {
      currentBatch++;
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
          const res = await fetch(trainingUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(this.hfToken ? { Authorization: `Bearer ${this.hfToken}` } : {}),
            },
            body: JSON.stringify({ examples: batch }),
            signal: controller.signal,
          });
          if (!res.ok) {
            const text = await res.text();
            errors.push(`Batch ${currentBatch}: HTTP ${res.status} – ${text}`);
          }
        } finally {
          clearTimeout(timer);
        }
      } catch (err) {
        errors.push(
          `Batch ${currentBatch}: ${err instanceof Error ? err.message : String(err)}`
        );
      }

      processedExamples += batch.length;

      if (opts.onProgress) {
        const progress: TrainingProgress = {
          processedExamples,
          totalExamples,
          currentBatch,
          totalBatches,
          elapsedMs: Date.now() - startTime,
        };
        opts.onProgress(progress);
      }
    };

    // Two-pass approach: first collect all examples to know total count,
    // then send in batches. For large datasets we stream and batch on-the-fly.
    const allExamples: TrainingExample[] = [];
    for await (const example of loader.streamExamples(opts)) {
      allExamples.push(example);
    }

    const totalExamples = allExamples.length;
    const totalBatches = Math.ceil(totalExamples / batchSize);

    for (let i = 0; i < allExamples.length; i += batchSize) {
      const batch = allExamples.slice(i, i + batchSize);
      queue.push(...batch);
      await flushBatch(batch, totalExamples, totalBatches);
      queue.length = 0;
    }

    const durationMs = Date.now() - startTime;
    const status: TrainingResult["status"] =
      errors.length === 0
        ? "completed"
        : processedExamples > 0
        ? "partial"
        : "failed";

    return {
      totalExamples: processedExamples,
      batches: currentBatch,
      durationMs,
      status,
      ...(errors.length > 0 ? { errors } : {}),
    };
  }
}
