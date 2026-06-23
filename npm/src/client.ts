import type {
  GenerateOptions,
  GenerateResult,
  NeuroQuantumClientConfig,
} from "./types.js";

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
}
