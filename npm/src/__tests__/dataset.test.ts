import { jest, describe, it, expect, afterEach } from "@jest/globals";
import { HFDatasetLoader } from "../dataset.js";
import { NeuroQuantumClient } from "../client.js";

// ---------------------------------------------------------------------------
// HFDatasetLoader unit tests (no network — fetch is mocked)
// ---------------------------------------------------------------------------

type FakeFetch = (url: unknown, init?: unknown) => Promise<{
  ok: boolean;
  status: number;
  statusText: string;
  json: () => Promise<unknown>;
  text: () => Promise<string>;
}>;

function makeFetch(response: unknown, status = 200): FakeFetch {
  return async () => ({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => response,
    text: async () => JSON.stringify(response),
  });
}

function setFetch(impl: FakeFetch): void {
  global.fetch = impl as unknown as typeof fetch;
}

describe("HFDatasetLoader", () => {
  afterEach(() => jest.restoreAllMocks());

  describe("fetchRows()", () => {
    it("fetches rows and normalises the shape", async () => {
      const payload = {
        rows: [
          { row_idx: 0, row: { input: "こんにちは", output: "Hello" }, truncated_cells: [] },
          { row_idx: 1, row: { input: "ありがとう", output: "Thank you" }, truncated_cells: [] },
        ],
        num_rows_total: 2,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const page = await loader.fetchRows({ dataset: "test/ds" });

      expect(page.rows).toHaveLength(2);
      expect(page.rows[0]).toEqual({ rowIdx: 0, row: { input: "こんにちは", output: "Hello" } });
      expect(page.numRowsTotal).toBe(2);
    });

    it("passes pagination params to the API", async () => {
      let capturedUrl = "";
      global.fetch = (async (url: unknown) => {
        capturedUrl = String(url);
        return { ok: true, json: async () => ({ rows: [], num_rows_total: 1000 }), text: async () => "" };
      }) as unknown as typeof fetch;

      const loader = new HFDatasetLoader();
      await loader.fetchRows({ dataset: "test/ds", offset: 50, limit: 25 });

      expect(capturedUrl).toContain("offset=50");
      expect(capturedUrl).toContain("limit=25");
    });

    it("caps limit at 100", async () => {
      let capturedUrl = "";
      global.fetch = (async (url: unknown) => {
        capturedUrl = String(url);
        return { ok: true, json: async () => ({ rows: [], num_rows_total: 0 }), text: async () => "" };
      }) as unknown as typeof fetch;

      const loader = new HFDatasetLoader();
      await loader.fetchRows({ dataset: "test/ds", limit: 500 });

      expect(capturedUrl).toContain("limit=100");
    });

    it("throws on API error", async () => {
      setFetch(makeFetch({ error: "not found" }, 404));

      const loader = new HFDatasetLoader();
      await expect(loader.fetchRows({ dataset: "missing/ds" })).rejects.toThrow(
        "HF Datasets API error 404"
      );
    });
  });

  describe("streamRows()", () => {
    it("yields rows across multiple pages", async () => {
      const page1 = {
        rows: [
          { row_idx: 0, row: { text: "a" }, truncated_cells: [] },
          { row_idx: 1, row: { text: "b" }, truncated_cells: [] },
        ],
        num_rows_total: 3,
      };
      const page2 = {
        rows: [{ row_idx: 2, row: { text: "c" }, truncated_cells: [] }],
        num_rows_total: 3,
      };
      let call = 0;
      global.fetch = (async () => {
        const data = call++ === 0 ? page1 : page2;
        return { ok: true, json: async () => data, text: async () => "" };
      }) as unknown as typeof fetch;

      const loader = new HFDatasetLoader();
      const rows = [];
      for await (const row of loader.streamRows({ dataset: "test/ds", pageSize: 2 })) {
        rows.push(row);
      }

      expect(rows).toHaveLength(3);
      expect(rows[2]?.row["text"]).toBe("c");
    });

    it("respects maxRows", async () => {
      const page = {
        rows: Array.from({ length: 5 }, (_, i) => ({
          row_idx: i,
          row: { text: String(i) },
          truncated_cells: [],
        })),
        num_rows_total: 100,
      };
      setFetch(makeFetch(page));

      const loader = new HFDatasetLoader();
      const rows = [];
      for await (const row of loader.streamRows({ dataset: "test/ds", maxRows: 3 })) {
        rows.push(row);
      }
      expect(rows).toHaveLength(3);
    });
  });

  describe("streamExamples()", () => {
    it("converts rows to TrainingExamples", async () => {
      const payload = {
        rows: [
          { row_idx: 0, row: { input: "質問A", output: "回答A" }, truncated_cells: [] },
          { row_idx: 1, row: { input: "質問B", output: "回答B" }, truncated_cells: [] },
        ],
        num_rows_total: 2,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const examples = [];
      for await (const ex of loader.streamExamples({
        dataset: "test/ds",
        promptField: "input",
        completionField: "output",
      })) {
        examples.push(ex);
      }

      expect(examples).toHaveLength(2);
      expect(examples[0]).toEqual({ prompt: "質問A", completion: "回答A" });
    });

    it("skips rows with empty fields", async () => {
      const payload = {
        rows: [
          { row_idx: 0, row: { input: "", output: "回答A" }, truncated_cells: [] },
          { row_idx: 1, row: { input: "質問B", output: "回答B" }, truncated_cells: [] },
        ],
        num_rows_total: 2,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const examples = await loader.loadExamples({
        dataset: "test/ds",
        promptField: "input",
        completionField: "output",
      });

      expect(examples).toHaveLength(1);
      expect(examples[0]!.prompt).toBe("質問B");
    });

    it("uses a custom transform function", async () => {
      const payload = {
        rows: [
          { row_idx: 0, row: { q: "Q1", a: "A1", score: 5 }, truncated_cells: [] },
        ],
        num_rows_total: 1,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const examples = await loader.loadExamples({
        dataset: "test/ds",
        transform: (row) => ({
          prompt: String(row["q"]),
          completion: `${String(row["a"])} (score: ${String(row["score"])})`,
        }),
      });

      expect(examples[0]).toEqual({ prompt: "Q1", completion: "A1 (score: 5)" });
    });

    it("infers field names automatically from known candidates", async () => {
      const payload = {
        rows: [
          { row_idx: 0, row: { question: "自動?", answer: "はい" }, truncated_cells: [] },
        ],
        num_rows_total: 1,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const examples = await loader.loadExamples({ dataset: "test/ds" });

      expect(examples[0]?.prompt).toBe("自動?");
      expect(examples[0]?.completion).toBe("はい");
    });
  });

  describe("preview()", () => {
    it("loads n examples", async () => {
      const payload = {
        rows: Array.from({ length: 5 }, (_, i) => ({
          row_idx: i,
          row: { input: `Q${i}`, output: `A${i}` },
          truncated_cells: [],
        })),
        num_rows_total: 100,
      };
      setFetch(makeFetch(payload));

      const loader = new HFDatasetLoader();
      const examples = await loader.preview("test/ds", 5);
      expect(examples.length).toBeLessThanOrEqual(5);
    });
  });
});

// ---------------------------------------------------------------------------
// NeuroQuantumClient dataset methods
// ---------------------------------------------------------------------------

describe("NeuroQuantumClient dataset methods", () => {
  afterEach(() => jest.restoreAllMocks());

  describe("generateWithExamples()", () => {
    it("prepends few-shot examples to the prompt", async () => {
      let capturedBody: Record<string, unknown> = {};
      global.fetch = (async (_url: unknown, init: unknown) => {
        capturedBody = JSON.parse((init as RequestInit).body as string) as Record<string, unknown>;
        return { ok: true, json: async () => [{ generated_text: "答え" }], text: async () => "" };
      }) as unknown as typeof fetch;

      const client = new NeuroQuantumClient({ hfToken: "tok", maxRetries: 1 });
      const examples = [
        { prompt: "例の質問1", completion: "例の答え1" },
        { prompt: "例の質問2", completion: "例の答え2" },
      ];
      await client.generateWithExamples("本番の質問", examples, { numExamples: 2 });

      expect(String(capturedBody["inputs"])).toContain("例の質問1");
      expect(String(capturedBody["inputs"])).toContain("例の答え1");
      expect(String(capturedBody["inputs"])).toContain("本番の質問");
    });

    it("uses custom templates", async () => {
      let capturedBody: Record<string, unknown> = {};
      global.fetch = (async (_url: unknown, init: unknown) => {
        capturedBody = JSON.parse((init as RequestInit).body as string) as Record<string, unknown>;
        return { ok: true, json: async () => [{ generated_text: "ok" }], text: async () => "" };
      }) as unknown as typeof fetch;

      const client = new NeuroQuantumClient({ hfToken: "tok", maxRetries: 1 });
      await client.generateWithExamples(
        "テスト",
        [{ prompt: "P", completion: "C" }],
        {
          exampleTemplate: "入力: {prompt}\n出力: {completion}",
          queryTemplate: "入力: {prompt}\n出力:",
        }
      );

      expect(String(capturedBody["inputs"])).toContain("入力: P\n出力: C");
      expect(String(capturedBody["inputs"])).toContain("入力: テスト\n出力:");
    });

    it("works with empty examples array", async () => {
      global.fetch = (async () => ({
        ok: true,
        json: async () => [{ generated_text: "ok" }],
        text: async () => "",
      })) as unknown as typeof fetch;

      const client = new NeuroQuantumClient({ hfToken: "tok", maxRetries: 1 });
      const result = await client.generateWithExamples("質問", []);
      expect(result.generatedText).toBe("ok");
    });
  });

  describe("trainFromDataset()", () => {
    it("sends batches to the training endpoint", async () => {
      const dsPayload = {
        rows: Array.from({ length: 4 }, (_, i) => ({
          row_idx: i,
          row: { input: `Q${i}`, output: `A${i}` },
          truncated_cells: [],
        })),
        num_rows_total: 4,
      };

      const trainCalls: unknown[] = [];
      let call = 0;
      global.fetch = (async (_url: unknown, init: unknown) => {
        if (call++ === 0) {
          return { ok: true, json: async () => dsPayload, text: async () => "" };
        }
        trainCalls.push(JSON.parse((init as RequestInit).body as string));
        return { ok: true, json: async () => ({ ok: true }), text: async () => "" };
      }) as unknown as typeof fetch;

      const client = new NeuroQuantumClient({
        hfToken: "tok",
        maxRetries: 1,
        endpointUrl: "https://example.com/infer",
      });

      const result = await client.trainFromDataset({
        dataset: "test/ds",
        promptField: "input",
        completionField: "output",
        batchSize: 2,
        trainingEndpointUrl: "https://example.com/train",
      });

      expect(result.status).toBe("completed");
      expect(result.totalExamples).toBe(4);
      expect(result.batches).toBe(2);
      expect(trainCalls).toHaveLength(2);
    });

    it("reports progress via onProgress callback", async () => {
      const dsPayload = {
        rows: Array.from({ length: 3 }, (_, i) => ({
          row_idx: i,
          row: { input: `Q${i}`, output: `A${i}` },
          truncated_cells: [],
        })),
        num_rows_total: 3,
      };

      let call = 0;
      global.fetch = (async () => {
        const data = call++ === 0 ? dsPayload : {};
        return { ok: true, json: async () => data, text: async () => "" };
      }) as unknown as typeof fetch;

      const progressEvents: number[] = [];
      const client = new NeuroQuantumClient({ hfToken: "tok", maxRetries: 1 });

      await client.trainFromDataset({
        dataset: "test/ds",
        promptField: "input",
        completionField: "output",
        batchSize: 2,
        trainingEndpointUrl: "https://example.com/train",
        onProgress: (p) => progressEvents.push(p.processedExamples),
      });

      expect(progressEvents.length).toBeGreaterThan(0);
      expect(progressEvents[progressEvents.length - 1]).toBe(3);
    });

    it("returns partial status on batch errors", async () => {
      const dsPayload = {
        rows: Array.from({ length: 2 }, (_, i) => ({
          row_idx: i,
          row: { input: `Q${i}`, output: `A${i}` },
          truncated_cells: [],
        })),
        num_rows_total: 2,
      };

      let call = 0;
      global.fetch = (async () => {
        if (call++ === 0) {
          return { ok: true, json: async () => dsPayload, text: async () => "" };
        }
        if (call === 2) {
          return {
            ok: false,
            status: 500,
            statusText: "Server Error",
            json: async () => ({}),
            text: async () => "Internal Server Error",
          };
        }
        return { ok: true, json: async () => ({}), text: async () => "" };
      }) as unknown as typeof fetch;

      const client = new NeuroQuantumClient({ hfToken: "tok", maxRetries: 1 });
      const result = await client.trainFromDataset({
        dataset: "test/ds",
        promptField: "input",
        completionField: "output",
        batchSize: 1,
        trainingEndpointUrl: "https://example.com/train",
      });

      expect(result.status).toBe("partial");
      expect(result.errors).toBeDefined();
      expect(result.errors!.length).toBeGreaterThan(0);
    });
  });
});
