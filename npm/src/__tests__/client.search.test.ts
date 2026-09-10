import { jest, describe, it, expect, afterEach } from "@jest/globals";
import { NeuroQuantumClient } from "../client.js";

// ---------------------------------------------------------------------------
// NeuroQuantumClient search API (no network — fetch is mocked)
// ---------------------------------------------------------------------------

type Call = { url: string; method?: string; body?: unknown };

function mockFetch(response: unknown, status = 200): Call[] {
  const calls: Call[] = [];
  global.fetch = (async (url: unknown, init?: RequestInit) => {
    calls.push({
      url: String(url),
      method: init?.method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    });
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? "OK" : "Error",
      json: async () => response,
      text: async () => JSON.stringify(response),
    };
  }) as unknown as typeof fetch;
  return calls;
}

const client = () =>
  new NeuroQuantumClient({
    endpointUrl: "http://generate.local",
    searchEndpointUrl: "http://search.local/",
    maxRetries: 0,
  });

describe("NeuroQuantumClient search", () => {
  afterEach(() => jest.restoreAllMocks());

  it("addDocuments() posts strings and objects in the server shape", async () => {
    const calls = mockFetch({ added: 2, total: 2, doc_ids: ["doc-1", "nn"] });
    const result = await client().addDocuments([
      "量子コンピュータの説明",
      { text: "ニューラルネットワーク", id: "nn", metadata: { topic: "ai" } },
    ]);

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://search.local/search/documents");
    expect(calls[0].method).toBe("POST");
    expect(calls[0].body).toEqual({
      documents: [
        { text: "量子コンピュータの説明" },
        { text: "ニューラルネットワーク", id: "nn", metadata: { topic: "ai" } },
      ],
    });
    expect(result).toEqual({ added: 2, total: 2, docIds: ["doc-1", "nn"] });
  });

  it("addDocuments() rejects an empty list without calling the server", async () => {
    const calls = mockFetch({});
    await expect(client().addDocuments([])).rejects.toThrow(/must not be empty/);
    expect(calls).toHaveLength(0);
  });

  it("search() sends options and normalises hits to camelCase", async () => {
    const calls = mockFetch({
      query: "量子",
      mode: "hybrid",
      total_documents: 3,
      results: [
        {
          doc_id: "doc-1",
          text: "量子コンピュータ",
          score: 0.9,
          bm25_score: 4.2,
          dense_score: 0.8,
          metadata: { topic: "quantum" },
          rank: 1,
        },
      ],
    });
    const result = await client().search("量子", {
      topK: 2,
      mode: "hybrid",
      minScore: 0.1,
      metadataFilter: { topic: "quantum" },
    });

    expect(calls[0].url).toBe("http://search.local/search");
    expect(calls[0].body).toEqual({
      query: "量子",
      top_k: 2,
      mode: "hybrid",
      min_score: 0.1,
      metadata_filter: { topic: "quantum" },
    });
    expect(result.mode).toBe("hybrid");
    expect(result.totalDocuments).toBe(3);
    expect(result.results).toEqual([
      {
        docId: "doc-1",
        text: "量子コンピュータ",
        score: 0.9,
        bm25Score: 4.2,
        denseScore: 0.8,
        metadata: { topic: "quantum" },
        rank: 1,
      },
    ]);
  });

  it("search() rejects a blank query", async () => {
    mockFetch({});
    await expect(client().search("   ")).rejects.toThrow(/must not be empty/);
  });

  it("search() surfaces server errors", async () => {
    mockFetch({ detail: "dense 検索にはモデルとトークナイザーが必要です" }, 400);
    await expect(client().search("x", { mode: "dense" })).rejects.toThrow(/400/);
  });

  it("clearDocuments() issues DELETE with and without doc_id", async () => {
    const calls = mockFetch({ status: "deleted", total: 1 });
    await client().clearDocuments("a b");
    await client().clearDocuments();

    expect(calls[0].method).toBe("DELETE");
    expect(calls[0].url).toBe("http://search.local/search/documents?doc_id=a%20b");
    expect(calls[1].url).toBe("http://search.local/search/documents");
  });

  it("searchStatus() maps the status payload", async () => {
    mockFetch({ documents: 5, mode: "bm25", alpha: 0.5, dense_available: false, ngram: 2 });
    expect(await client().searchStatus()).toEqual({
      documents: 5,
      mode: "bm25",
      alpha: 0.5,
      denseAvailable: false,
      ngram: 2,
    });
  });

  it("defaults the search base URL to endpointUrl", async () => {
    const calls = mockFetch({ documents: 0, mode: "bm25", alpha: 0.5, dense_available: false, ngram: 2 });
    await new NeuroQuantumClient({ endpointUrl: "http://only.local", maxRetries: 0 }).searchStatus();
    expect(calls[0].url).toBe("http://only.local/search/status");
  });
});
