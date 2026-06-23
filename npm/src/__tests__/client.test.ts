import { NeuroQuantumClient } from "../client.js";

describe("NeuroQuantumClient", () => {
  it("constructs with default config", () => {
    const client = new NeuroQuantumClient();
    expect(client).toBeDefined();
  });

  it("constructs with custom config", () => {
    const client = new NeuroQuantumClient({
      endpointUrl: "https://example.com/api",
      hfToken: "test-token",
      timeoutMs: 5000,
      maxRetries: 3,
    });
    expect(client).toBeDefined();
  });

  it("throws on HTTP error without retry", async () => {
    const client = new NeuroQuantumClient({
      endpointUrl: "https://httpstat.us/500",
      maxRetries: 1,
      timeoutMs: 10_000,
    });
    await expect(client.generate("test")).rejects.toThrow();
  });
});
