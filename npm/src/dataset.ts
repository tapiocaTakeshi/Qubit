import type {
  DatasetToExamplesOptions,
  FetchRowsOptions,
  HFDatasetLoaderConfig,
  HFDatasetPage,
  HFDatasetRow,
  StreamRowsOptions,
  TrainingExample,
} from "./types.js";

const DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co";
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Client for the HuggingFace Datasets Server API.
 *
 * Supports paginated fetching, async streaming, and conversion to
 * prompt–completion training examples.
 *
 * @example
 * ```ts
 * import { HFDatasetLoader } from "qubit_ai";
 *
 * const loader = new HFDatasetLoader({ hfToken: process.env.HF_TOKEN });
 *
 * // Stream all rows and convert to training examples
 * for await (const example of loader.streamExamples({
 *   dataset: "llm-jp/oasst2-33k-ja",
 *   promptField: "input",
 *   completionField: "output",
 *   maxRows: 200,
 * })) {
 *   console.log(example.prompt, "->", example.completion);
 * }
 * ```
 */
export class HFDatasetLoader {
  private readonly datasetsServerUrl: string;
  private readonly hfToken: string;
  private readonly timeoutMs: number;

  constructor(config: HFDatasetLoaderConfig = {}) {
    this.datasetsServerUrl =
      config.datasetsServerUrl ?? DEFAULT_DATASETS_SERVER;
    this.hfToken =
      config.hfToken ??
      (typeof process !== "undefined"
        ? (process.env["HF_TOKEN"] ?? process.env["HUGGING_FACE_HUB_TOKEN"] ?? "")
        : "");
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};
    if (this.hfToken) {
      headers["Authorization"] = `Bearer ${this.hfToken}`;
    }
    return headers;
  }

  private async get(path: string): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(`${this.datasetsServerUrl}${path}`, {
        headers: this.buildHeaders(),
        signal: controller.signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HF Datasets API error ${res.status}: ${text}`);
      }
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Fetch a single page of rows from the dataset.
   *
   * @param opts - Dataset identifier and pagination options
   */
  async fetchRows(opts: FetchRowsOptions): Promise<HFDatasetPage> {
    const config = opts.config ?? "default";
    const split = opts.split ?? "train";
    const offset = opts.offset ?? 0;
    const limit = Math.min(opts.limit ?? 100, 100);

    const qs = new URLSearchParams({
      dataset: opts.dataset,
      config,
      split,
      offset: String(offset),
      limit: String(limit),
    });

    const raw = await this.get(`/rows?${qs.toString()}`);
    const data = raw as {
      rows: Array<{ row_idx: number; row: Record<string, unknown> }>;
      num_rows_total: number;
    };

    return {
      rows: data.rows.map((r) => ({
        rowIdx: r.row_idx,
        row: r.row,
      })),
      numRowsTotal: data.num_rows_total,
    };
  }

  /**
   * Fetch available splits for a dataset.
   *
   * @param dataset - Dataset name on HuggingFace Hub
   * @param config  - Configuration name (default: "default")
   */
  async fetchSplits(
    dataset: string,
    config = "default"
  ): Promise<string[]> {
    const qs = new URLSearchParams({ dataset, config });
    const raw = await this.get(`/splits?${qs.toString()}`);
    const data = raw as {
      splits: Array<{ split: string }>;
    };
    return data.splits.map((s) => s.split);
  }

  /**
   * Async generator that yields rows page-by-page.
   *
   * @param opts - Stream options including optional maxRows cap
   */
  async *streamRows(opts: StreamRowsOptions): AsyncGenerator<HFDatasetRow> {
    const pageSize = opts.pageSize ?? 100;
    const maxRows = opts.maxRows ?? Infinity;
    let offset = 0;
    let yielded = 0;

    while (yielded < maxRows) {
      const remaining = maxRows - yielded;
      const limit = Math.min(pageSize, remaining, 100);
      const page = await this.fetchRows({ ...opts, offset, limit });

      for (const row of page.rows) {
        yield row;
        yielded++;
        if (yielded >= maxRows) return;
      }

      offset += page.rows.length;
      if (offset >= page.numRowsTotal) return;
    }
  }

  /**
   * Infer prompt and completion field names from the first row.
   * Prefers common field names used in Japanese instruction datasets.
   */
  private inferFields(
    row: Record<string, unknown>
  ): { promptField: string; completionField: string } {
    const keys = Object.keys(row);
    const PROMPT_CANDIDATES = ["input", "instruction", "question", "prompt", "text"];
    const COMPLETION_CANDIDATES = ["output", "response", "answer", "completion", "label"];

    const promptField =
      PROMPT_CANDIDATES.find((k) => keys.includes(k)) ?? keys[0] ?? "input";
    const completionField =
      COMPLETION_CANDIDATES.find((k) => keys.includes(k)) ??
      keys.find((k) => k !== promptField) ??
      "output";

    return { promptField, completionField };
  }

  /**
   * Convert a raw dataset row into a {@link TrainingExample}.
   *
   * Returns `null` if the row cannot be converted (e.g. empty fields).
   */
  private rowToExample(
    row: Record<string, unknown>,
    promptField: string,
    completionField: string,
    transform?: (row: Record<string, unknown>) => TrainingExample | null
  ): TrainingExample | null {
    if (transform) return transform(row);

    const prompt = String(row[promptField] ?? "").trim();
    const completion = String(row[completionField] ?? "").trim();
    if (!prompt || !completion) return null;
    return { prompt, completion };
  }

  /**
   * Async generator that yields {@link TrainingExample} objects.
   *
   * Field names are inferred automatically from the dataset schema unless
   * `promptField` / `completionField` are provided.
   *
   * @param opts - Stream + field-mapping options
   */
  async *streamExamples(
    opts: DatasetToExamplesOptions
  ): AsyncGenerator<TrainingExample> {
    let resolvedPromptField: string | undefined = opts.promptField;
    let resolvedCompletionField: string | undefined = opts.completionField;

    for await (const { row } of this.streamRows(opts)) {
      if (!resolvedPromptField || !resolvedCompletionField) {
        const inferred = this.inferFields(row);
        resolvedPromptField ??= inferred.promptField;
        resolvedCompletionField ??= inferred.completionField;
      }

      const example = this.rowToExample(
        row,
        resolvedPromptField,
        resolvedCompletionField,
        opts.transform
      );
      if (example) yield example;
    }
  }

  /**
   * Load up to `maxRows` examples eagerly into memory.
   *
   * Convenience wrapper around {@link streamExamples} for small datasets.
   *
   * @param opts - Same options as streamExamples
   */
  async loadExamples(
    opts: DatasetToExamplesOptions
  ): Promise<TrainingExample[]> {
    const examples: TrainingExample[] = [];
    for await (const ex of this.streamExamples(opts)) {
      examples.push(ex);
    }
    return examples;
  }

  /**
   * Return a small preview of examples from the dataset.
   *
   * @param dataset - Dataset name on HuggingFace Hub
   * @param n       - Number of examples to fetch (default: 5, max: 100)
   */
  async preview(
    dataset: string,
    n = 5
  ): Promise<TrainingExample[]> {
    return this.loadExamples({ dataset, maxRows: Math.min(n, 100) });
  }
}
