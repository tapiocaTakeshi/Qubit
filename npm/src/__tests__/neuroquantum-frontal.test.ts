/**
 * Tests for NeuroQuantumFrontalEngine
 */

import { NeuroQuantumFrontalEngine } from "../neuroquantum-frontal.js";
import type { NeuroQuantumResponse } from "../neuroquantum-api-client.js";

declare const jest: any;

describe("NeuroQuantumFrontalEngine", () => {
  let mockClient: any;
  let engine: NeuroQuantumFrontalEngine;

  const mockResponse: NeuroQuantumResponse = {
    decision: "Yes",
    score: 75,
    reasoning: "Action is safe based on quantum analysis",
    confidence: "high",
    factors: ["quantum_analysis", "neural_layers", "entanglement"],
    timestamp: "2024-01-01T00:00:00Z",
    system: "neuroquantum",
    processing_time_ms: 234,
  };

  beforeEach(() => {
    // Create mock client
    mockClient = {
      judge: jest.fn().mockResolvedValue(mockResponse),
      safetyCheck: jest.fn().mockResolvedValue({
        safe: true,
        result: mockResponse,
      }),
      ethicsCheck: jest.fn().mockResolvedValue(mockResponse),
      qualityEval: jest.fn().mockResolvedValue(mockResponse),
      batchJudge: jest.fn().mockResolvedValue({
        results: [mockResponse],
        count: 1,
      }),
      healthCheck: jest.fn().mockResolvedValue({
        status: "healthy",
        version: "1.0.0",
        neuroquantum_available: true,
      }),
      waitForAvailable: jest.fn().mockResolvedValue(undefined),
    };

    engine = new NeuroQuantumFrontalEngine(mockClient);
  });

  describe("judge", () => {
    it("should call client.judge and return formatted result", async () => {
      const result = await engine.judge("test action", "test context", {
        type: "safety",
        strictMode: false,
      });

      expect(mockClient.judge).toHaveBeenCalledWith(
        "test action",
        "test context",
        "safety",
        false
      );

      expect(result).toEqual({
        decision: "Yes",
        score: 75,
        reasoning: "Action is safe based on quantum analysis",
        confidence: "high",
        keyFactors: ["quantum_analysis", "neural_layers", "entanglement"],
        timestamp: "2024-01-01T00:00:00Z",
        system: "neuroquantum",
        processingTimeMs: 234,
      });
    });

    it("should handle different judgment types", async () => {
      await engine.judge("action", "context", {
        type: "ethics",
        strictMode: true,
      });

      expect(mockClient.judge).toHaveBeenCalledWith(
        "action",
        "context",
        "ethics",
        true
      );
    });
  });

  describe("checkSafety", () => {
    it("should call client.safetyCheck and return formatted result", async () => {
      const result = await engine.checkSafety("delete data", "production", {
        risks: ["data loss"],
      });

      expect(mockClient.safetyCheck).toHaveBeenCalledWith(
        "delete data",
        "production",
        { risks: ["data loss"] }
      );

      expect(result.decision).toBe("Yes");
      expect(result.system).toBe("neuroquantum");
    });
  });

  describe("evaluateQuality", () => {
    it("should call client.qualityEval", async () => {
      const result = await engine.evaluateQuality("test content", {
        requirements: ["clarity"],
      });

      expect(mockClient.qualityEval).toHaveBeenCalledWith("test content", {
        requirements: ["clarity"],
      });

      expect(result.decision).toBe("Yes");
      expect(result.confidence).toBe("high");
    });
  });

  describe("evaluateEthics", () => {
    it("should call client.ethicsCheck", async () => {
      const result = await engine.evaluateEthics(
        "release data",
        "Stakeholders: users\nPotential harms: privacy loss"
      );

      expect(mockClient.ethicsCheck).toHaveBeenCalled();
      expect(result.system).toBe("neuroquantum");
    });
  });

  describe("prioritize", () => {
    it("should rank tasks by score", async () => {
      mockClient.batchJudge.mockResolvedValue({
        results: [
          {
            ...mockResponse,
            score: 80,
          },
          {
            ...mockResponse,
            score: 60,
          },
          {
            ...mockResponse,
            score: 75,
          },
        ],
        count: 3,
      });

      const result = await engine.prioritize(
        ["task1", "task2", "task3"],
        "context"
      );

      expect(result.rankedTasks).toEqual(["task1", "task3", "task2"]);
      expect(result.scores).toEqual([80, 75, 60]);
    });
  });

  describe("getStatus", () => {
    it("should return health check status", async () => {
      const status = await engine.getStatus();

      expect(status.available).toBe(true);
      expect(status.version).toBe("1.0.0");
      expect(status.neuroquantumAvailable).toBe(true);
    });

    it("should return unavailable on error", async () => {
      mockClient.healthCheck.mockRejectedValue(new Error("Connection failed"));

      const status = await engine.getStatus();

      expect(status.available).toBe(false);
    });
  });

  describe("waitForAvailable", () => {
    it("should wait for API availability", async () => {
      await engine.waitForAvailable(5000);

      expect(mockClient.waitForAvailable).toHaveBeenCalledWith(5000);
    });
  });
});
