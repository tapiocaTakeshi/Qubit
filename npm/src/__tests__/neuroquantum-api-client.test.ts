/**
 * Tests for NeuroQuantumAPIClient
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { NeuroQuantumAPIClient } from "../neuroquantum-api-client.js";

// Mock fetch globally
global.fetch = vi.fn();

describe("NeuroQuantumAPIClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("initialization", () => {
    it("should use default configuration", () => {
      const client = new NeuroQuantumAPIClient();
      // Check that client can be created with defaults
      expect(client).toBeDefined();
    });

    it("should accept custom configuration", () => {
      const client = new NeuroQuantumAPIClient({
        baseUrl: "http://example.com:8000",
        timeout: 60000,
        maxRetries: 5,
        retryDelayMs: 2000,
      });

      expect(client).toBeDefined();
    });
  });

  describe("judge", () => {
    it("should make POST request to /api/v1/judge", async () => {
      const mockResponse = {
        decision: "Yes" as const,
        score: 75,
        reasoning: "Test reasoning",
        confidence: "high" as const,
        factors: ["factor1"],
        timestamp: "2024-01-01T00:00:00Z",
        system: "neuroquantum",
        processing_time_ms: 100,
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });

      const client = new NeuroQuantumAPIClient();
      const result = await client.judge("action", "context", "safety", false);

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:5000/api/v1/judge",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "action",
            context: "context",
            judgment_type: "safety",
            strict_mode: false,
          }),
        })
      );
    });

    it("should throw on HTTP error", async () => {
      (global.fetch as any).mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => "Internal Server Error",
      });

      const client = new NeuroQuantumAPIClient({ maxRetries: 1 });

      await expect(client.judge("action", "context")).rejects.toThrow();
    });

    it("should retry on network error", async () => {
      const mockResponse = {
        decision: "Yes" as const,
        score: 75,
        reasoning: "Test",
        confidence: "high" as const,
        factors: [],
        timestamp: "2024-01-01T00:00:00Z",
        system: "neuroquantum",
        processing_time_ms: 100,
      };

      (global.fetch as any)
        .mockRejectedValueOnce(new Error("ERR_NETWORK"))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse,
        });

      const client = new NeuroQuantumAPIClient({ maxRetries: 3, retryDelayMs: 10 });
      const result = await client.judge("action", "context");

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });

  describe("batchJudge", () => {
    it("should handle batch requests", async () => {
      const mockResponse = {
        results: [
          {
            decision: "Yes" as const,
            score: 80,
            reasoning: "Batch item 1",
            confidence: "high" as const,
            factors: [],
            timestamp: "2024-01-01T00:00:00Z",
            system: "neuroquantum",
            processing_time_ms: 100,
          },
        ],
        count: 1,
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });

      const client = new NeuroQuantumAPIClient();
      const result = await client.batchJudge([
        { action: "action", context: "context" },
      ]);

      expect(result.count).toBe(1);
      expect(result.results).toHaveLength(1);
    });
  });

  describe("safetyCheck", () => {
    it("should check safety with risks", async () => {
      const mockResult = {
        decision: "No" as const,
        score: 25,
        reasoning: "Unsafe action",
        confidence: "high" as const,
        factors: ["risk1"],
        timestamp: "2024-01-01T00:00:00Z",
        system: "neuroquantum",
        processing_time_ms: 100,
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => ({
          safe: false,
          result: mockResult,
        }),
      });

      const client = new NeuroQuantumAPIClient();
      const result = await client.safetyCheck("action", "context", {
        risks: ["data loss"],
      });

      expect(result.safe).toBe(false);
      expect(result.result.decision).toBe("No");
    });
  });

  describe("ethicsCheck", () => {
    it("should check ethical implications", async () => {
      const mockResult = {
        decision: "Yes" as const,
        score: 70,
        reasoning: "Ethically sound",
        confidence: "medium" as const,
        factors: ["stakeholders"],
        timestamp: "2024-01-01T00:00:00Z",
        system: "neuroquantum",
        processing_time_ms: 100,
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockResult,
      });

      const client = new NeuroQuantumAPIClient();
      const result = await client.ethicsCheck("action", ["users"], ["harm"]);

      expect(result.decision).toBe("Yes");
    });
  });

  describe("healthCheck", () => {
    it("should check API health", async () => {
      const mockHealth = {
        status: "healthy",
        version: "1.0.0",
        neuroquantum_available: true,
      };

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockHealth,
      });

      const client = new NeuroQuantumAPIClient();
      const health = await client.healthCheck();

      expect(health.status).toBe("healthy");
      expect(health.neuroquantum_available).toBe(true);
    });
  });

  describe("isAvailable", () => {
    it("should return true when API is healthy", async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => ({
          status: "healthy",
          version: "1.0.0",
          neuroquantum_available: true,
        }),
      });

      const client = new NeuroQuantumAPIClient();
      const available = await client.isAvailable();

      expect(available).toBe(true);
    });

    it("should return false when API is unreachable", async () => {
      (global.fetch as any).mockRejectedValue(new Error("Connection refused"));

      const client = new NeuroQuantumAPIClient({ maxRetries: 1 });
      const available = await client.isAvailable();

      expect(available).toBe(false);
    });
  });

  describe("waitForAvailable", () => {
    it("should resolve when API becomes available", async () => {
      const mockHealth = {
        status: "healthy",
        version: "1.0.0",
        neuroquantum_available: true,
      };

      (global.fetch as any)
        .mockRejectedValueOnce(new Error("Unavailable"))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => mockHealth,
        });

      const client = new NeuroQuantumAPIClient();
      const promise = client.waitForAvailable(2000);

      // Give some time for the polling to work
      await new Promise((resolve) => setTimeout(resolve, 100));

      await expect(promise).resolves.toBeUndefined();
    });

    it("should timeout when API is unavailable", async () => {
      (global.fetch as any).mockRejectedValue(new Error("Connection refused"));

      const client = new NeuroQuantumAPIClient({ maxRetries: 1 });

      await expect(client.waitForAvailable(100)).rejects.toThrow(
        /NeuroQuantum API.*within 100ms/
      );
    });
  });

  describe("timeout handling", () => {
    it("should abort request on timeout", async () => {
      let abortController: AbortController | null = null;

      (global.fetch as any).mockImplementation(
        (_url: string, options: any) => {
          abortController = new AbortController();
          // Simulate the timeout by not resolving
          return new Promise(() => {
            // Never resolves - timeout will abort
          });
        }
      );

      const client = new NeuroQuantumAPIClient({
        timeout: 50,
        maxRetries: 1,
      });

      const promise = client.judge("action", "context");

      // Give fetch time to be called
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Should have timed out
      await expect(promise).rejects.toThrow();
    });
  });
});
