/**
 * Integration tests for QubitAI with NeuroQuantum backend
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { QubitAI } from "../qubit_ai.js";
import { NeuroQuantumAPIClient } from "../neuroquantum-api-client.js";
import type { NeuroQuantumResponse } from "../neuroquantum-api-client.js";

// Mock the NeuroQuantumAPIClient
vi.mock("../neuroquantum-api-client.js");

describe("QubitAI with NeuroQuantum backend", () => {
  let mockClient: any;
  let qubit: QubitAI;

  const mockResponse: NeuroQuantumResponse = {
    decision: "Yes",
    score: 75,
    reasoning: "Quantum-inspired reasoning result",
    confidence: "high",
    factors: ["neural_analysis", "quantum_inspired", "entanglement"],
    timestamp: "2024-01-01T00:00:00Z",
    system: "neuroquantum",
    processing_time_ms: 150,
  };

  beforeEach(() => {
    // Create mock client with all required methods
    mockClient = {
      judge: vi.fn().mockResolvedValue(mockResponse),
      safetyCheck: vi.fn().mockResolvedValue({
        safe: mockResponse.decision === "Yes",
        result: mockResponse,
      }),
      ethicsCheck: vi.fn().mockResolvedValue(mockResponse),
      qualityEval: vi.fn().mockResolvedValue(mockResponse),
      batchJudge: vi.fn().mockResolvedValue({
        results: [mockResponse],
        count: 1,
      }),
      healthCheck: vi.fn().mockResolvedValue({
        status: "healthy",
        version: "1.0.0",
        neuroquantum_available: true,
      }),
      getConfig: vi.fn().mockResolvedValue({}),
      getStatus: vi.fn().mockResolvedValue({}),
      isAvailable: vi.fn().mockResolvedValue(true),
      waitForAvailable: vi.fn().mockResolvedValue(undefined),
    };

    // Mock the NeuroQuantumAPIClient constructor
    (NeuroQuantumAPIClient as any).mockImplementation(() => mockClient);

    // Create QubitAI with NeuroQuantum backend enabled
    qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: "http://localhost:5000",
      },
    });
  });

  describe("judge", () => {
    it("should use NeuroQuantum backend for judgment", async () => {
      const result = await qubit.judge("test action", "test context", "safety");

      expect(result).toEqual({
        decision: "Yes",
        score: 75,
        reasoning: "Quantum-inspired reasoning result",
        confidence: "high",
        factors: ["neural_analysis", "quantum_inspired", "entanglement"],
        timestamp: "2024-01-01T00:00:00Z",
      });

      expect(mockClient.judge).toHaveBeenCalledWith(
        "test action",
        "test context",
        "safety",
        false
      );
    });

    it("should support different judgment types", async () => {
      await qubit.judge("test action", "test context", "ethics");

      expect(mockClient.judge).toHaveBeenCalledWith(
        "test action",
        "test context",
        "ethics",
        false
      );
    });

    it("should apply strict mode", async () => {
      await qubit.judge("test action", "test context", "safety", true);

      expect(mockClient.judge).toHaveBeenCalledWith(
        "test action",
        "test context",
        "safety",
        true
      );
    });
  });

  describe("safetyCheck", () => {
    it("should return safety decision tuple", async () => {
      const [safe, result] = await qubit.safetyCheck(
        "delete database",
        "production",
        { risks: ["data loss"] }
      );

      expect(safe).toBe(true);
      expect(result.decision).toBe("Yes");
      expect(mockClient.safetyCheck).toHaveBeenCalled();
    });

    it("should mark unsafe decisions correctly", async () => {
      mockClient.safetyCheck.mockResolvedValue({
        safe: false,
        result: { ...mockResponse, decision: "No", score: 25 },
      });

      const [safe, result] = await qubit.safetyCheck(
        "unsafe action",
        "context"
      );

      expect(safe).toBe(false);
      expect(result.decision).toBe("No");
    });
  });

  describe("evaluateQuality", () => {
    it("should evaluate content quality", async () => {
      const result = await qubit.evaluateQuality("test content", {
        requirements: ["clarity", "accuracy"],
      });

      expect(result.decision).toBe("Yes");
      expect(result.score).toBe(75);
      expect(mockClient.qualityEval).toHaveBeenCalled();
    });
  });

  describe("ethicsCheck", () => {
    it("should evaluate ethical implications", async () => {
      const result = await qubit.ethicsCheck("share user data", [
        "users",
        "regulators",
      ]);

      expect(result.decision).toBe("Yes");
      expect(mockClient.ethicsCheck).toHaveBeenCalled();
    });
  });

  describe("prioritize", () => {
    it("should rank items by priority", async () => {
      mockClient.batchJudge.mockResolvedValue({
        results: [
          { ...mockResponse, score: 90 },
          { ...mockResponse, score: 60 },
          { ...mockResponse, score: 75 },
        ],
        count: 3,
      });

      const items = [
        { name: "Task 1", description: "Important" },
        { name: "Task 2", description: "Low priority" },
        { name: "Task 3", description: "Medium priority" },
      ];

      const result = await qubit.prioritize(items);

      expect(result).toHaveLength(3);
      // Results should be sorted by score (highest first)
      expect(result[0][1]).toBeGreaterThanOrEqual(result[1][1]);
      expect(result[1][1]).toBeGreaterThanOrEqual(result[2][1]);
    });
  });

  describe("history tracking", () => {
    it("should record judgment history", async () => {
      await qubit.judge("action", "context", "safety");
      await qubit.judge("another action", "context", "ethics");

      const history = qubit.getHistory();

      expect(history).toHaveLength(2);
      expect(history[0].judgmentType).toBe("safety");
      expect(history[1].judgmentType).toBe("ethics");
    });

    it("should clear history", async () => {
      await qubit.judge("action", "context", "safety");

      qubit.clearHistory();

      expect(qubit.getHistory()).toHaveLength(0);
    });
  });

  describe("info and status", () => {
    it("should return product info", () => {
      const info = qubit.getInfo();

      expect(info.product).toBeDefined();
      expect(info.version).toBeDefined();
      expect(info.sessionId).toBeDefined();
    });

    it("should return system status", () => {
      const status = qubit.getStatus();

      expect(status.status).toBe("operational");
      expect(status.frontalEngineAvailable).toBe(true);
      expect(status.judgmentHistorySize).toBeGreaterThanOrEqual(0);
    });
  });
});
