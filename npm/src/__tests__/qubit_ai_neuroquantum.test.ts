/**
 * Integration tests for QubitAI with NeuroQuantum backend
 */

import { describe, it, expect } from "@jest/globals";
import { QubitAI } from "../qubit_ai.js";

describe("QubitAI with NeuroQuantum backend", () => {
  it("should construct with NeuroQuantum backend enabled", () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: "http://localhost:5000",
      },
    });

    const info = qubit.getInfo();
    expect(info.product).toBe("Qubit.ai");
    expect(info.version).toBeDefined();
    expect(info.sessionId).toMatch(/^qubit-ai-/);
  });

  it("should return valid info when NeuroQuantum is enabled", () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
    });

    const info = qubit.getInfo();
    expect(info).toHaveProperty("product");
    expect(info).toHaveProperty("version");
    expect(info).toHaveProperty("sessionId");
    expect(info).toHaveProperty("status");
    expect(info.status).toBe("operational");
  });

  it("should return operational status with NeuroQuantum backend", () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: "http://localhost:5000",
        timeout: 5000,
      },
    });

    const status = qubit.getStatus();
    expect(status.status).toBe("operational");
    expect(status.frontalEngineAvailable).toBe(true);
  });

  it("should track history with NeuroQuantum backend", async () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
    });

    const history = qubit.getHistory();
    expect(Array.isArray(history)).toBe(true);
    expect(history).toHaveLength(0);
  });

  it("should allow custom NeuroQuantum config", () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: "http://custom-api:5000",
        timeout: 10000,
        maxRetries: 5,
        retryDelayMs: 500,
      },
    });

    const status = qubit.getStatus();
    expect(status.status).toBe("operational");
  });

  it("should support hybrid mode with NeuroQuantum and heuristics", () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      fallbackToHeuristics: true,
      neuroquantumConfig: {
        baseUrl: "http://localhost:5000",
      },
    });

    const status = qubit.getStatus();
    expect(status.status).toBe("operational");
    expect(status.frontalEngineAvailable).toBe(true);
  });
});
