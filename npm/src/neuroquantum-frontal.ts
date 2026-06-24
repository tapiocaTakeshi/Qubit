/**
 * NeuroQuantumFrontalEngine — Python-backed quantum-inspired reasoning
 *
 * Uses the Python neuroquantum_layered.py inference engine via REST API
 * for quantum-inspired neural network reasoning.
 */

import { NeuroQuantumAPIClient } from "./neuroquantum-api-client.js";
import type {
  JudgeOptions,
  JudgmentResult,
  SafetyCheckOptions,
  QualityEvalOptions,
} from "./types.js";

/**
 * NeuroQuantumFrontalEngine — Quantum-inspired reasoning via Python backend
 *
 * Delegates judgment requests to the Python REST API, which runs the
 * neuroquantum_layered.py inference engine with quantum-inspired neural layers.
 */
export class NeuroQuantumFrontalEngine {
  private client: NeuroQuantumAPIClient;

  constructor(apiClient?: NeuroQuantumAPIClient) {
    this.client = apiClient || new NeuroQuantumAPIClient();
  }

  /**
   * Execute a judgment using the NeuroQuantum backend
   */
  async judge(
    action: string,
    context: string,
    options: JudgeOptions = {}
  ): Promise<JudgmentResult> {
    const { type = "safety", strictMode = false } = options;

    const response = await this.client.judge(action, context, type, strictMode);

    // Convert API response to JudgmentResult
    return {
      decision: response.decision as "Yes" | "No",
      score: response.score,
      reasoning: response.reasoning,
      confidence: response.confidence as "high" | "medium" | "low",
      keyFactors: response.factors,
      timestamp: response.timestamp,
      system: "neuroquantum",
      processingTimeMs: response.processing_time_ms,
    };
  }

  /**
   * Check safety of an action
   */
  async checkSafety(
    action: string,
    context: string,
    options: SafetyCheckOptions = {}
  ): Promise<JudgmentResult> {
    const response = await this.client.safetyCheck(action, context, options);

    return {
      decision: response.result.decision as "Yes" | "No",
      score: response.result.score,
      reasoning: response.result.reasoning,
      confidence: response.result.confidence as "high" | "medium" | "low",
      keyFactors: response.result.factors,
      timestamp: response.result.timestamp,
      system: "neuroquantum",
      processingTimeMs: response.result.processing_time_ms,
    };
  }

  /**
   * Evaluate quality of content
   */
  async evaluateQuality(
    content: string,
    options: QualityEvalOptions = {}
  ): Promise<JudgmentResult> {
    const response = await this.client.qualityEval(content, options);

    return {
      decision: response.decision as "Yes" | "No",
      score: response.score,
      reasoning: response.reasoning,
      confidence: response.confidence as "high" | "medium" | "low",
      keyFactors: response.factors,
      timestamp: response.timestamp,
      system: "neuroquantum",
      processingTimeMs: response.processing_time_ms,
    };
  }

  /**
   * Evaluate ethics of an action
   */
  async evaluateEthics(
    action: string,
    context: string
  ): Promise<JudgmentResult> {
    const stakeholders: string[] = [];
    const harms: string[] = [];

    // Parse context for stakeholder and harm information if present
    const lines = context.split("\n");
    for (const line of lines) {
      if (line.includes("Stakeholders:")) {
        const stakeholdersText = line.split("Stakeholders:")[1];
        if (stakeholdersText) {
          stakeholders.push(
            ...stakeholdersText.split(",").map((s) => s.trim())
          );
        }
      }
      if (line.includes("Potential harms:")) {
        const harmsText = line.split("Potential harms:")[1];
        if (harmsText) {
          harms.push(...harmsText.split(",").map((h) => h.trim()));
        }
      }
    }

    const response = await this.client.ethicsCheck(action, stakeholders, harms);

    return {
      decision: response.decision as "Yes" | "No",
      score: response.score,
      reasoning: response.reasoning,
      confidence: response.confidence as "high" | "medium" | "low",
      keyFactors: response.factors,
      timestamp: response.timestamp,
      system: "neuroquantum",
      processingTimeMs: response.processing_time_ms,
    };
  }

  /**
   * Prioritize tasks using quantum-inspired reasoning
   */
  async prioritize(
    tasks: string[],
    context: string
  ): Promise<{ rankedTasks: string[]; scores: number[] }> {
    // Use batch judgment to score all tasks and rank them
    const requests = tasks.map((action) => ({
      action,
      context,
      judgment_type: "priority",
      strict_mode: false,
    }));

    const batchResult = await this.client.batchJudge(requests);

    // Sort by score descending
    const scored = batchResult.results.map((result, index) => ({
      task: tasks[index]!,
      score: result.score,
    }));

    scored.sort((a, b) => b.score - a.score);

    return {
      rankedTasks: scored.map((s) => s.task),
      scores: scored.map((s) => s.score),
    };
  }

  /**
   * Get health status of the NeuroQuantum API
   */
  async getStatus(): Promise<{
    available: boolean;
    version?: string;
    neuroquantumAvailable?: boolean;
    gpuInfo?: Record<string, unknown>;
  }> {
    try {
      const health = await this.client.healthCheck();
      return {
        available: true,
        version: health.version,
        neuroquantumAvailable: health.neuroquantum_available,
        gpuInfo: (health as any).gpu_info,
      };
    } catch (error) {
      return {
        available: false,
      };
    }
  }

  /**
   * Wait for NeuroQuantum API to become available
   */
  async waitForAvailable(timeoutMs?: number): Promise<void> {
    return this.client.waitForAvailable(timeoutMs);
  }
}
