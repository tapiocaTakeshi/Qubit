/**
 * HybridFrontalEngine — Combines LLM and heuristic judgment
 *
 * Blends results from LLM-based and keyword-based engines for safety
 */

import { QBNNFrontalEngine } from "./frontal.js";
import type { LLMFrontalEngine } from "./llm-frontal.js";
import type {
  JudgmentResult,
  JudgmentType,
  JudgeOptions,
  SafetyCheckOptions,
  QualityEvalOptions,
  RiskAssessmentOptions,
  PrioritizationResult,
  QubitAIConfig,
} from "./types.js";

/**
 * Blending strategy for combining scores
 */
type BlendingStrategy = "weighted" | "confidence-based" | "llm-primary";

/**
 * HybridFrontalEngine combines LLM and heuristic judgment
 *
 * Useful for production safety: if LLM fails or disagrees with heuristics,
 * the hybrid approach provides redundancy and safety.
 */
export class HybridFrontalEngine {
  private llmEngine: LLMFrontalEngine;
  private heuristicEngine: QBNNFrontalEngine;
  private strategy: BlendingStrategy;
  private llmWeight: number; // 0.0 to 1.0, default 0.7

  constructor(
    llmEngine: LLMFrontalEngine,
    heuristicEngine: QBNNFrontalEngine,
    config: QubitAIConfig = {}
  ) {
    this.llmEngine = llmEngine;
    this.heuristicEngine = heuristicEngine;
    this.strategy = (config.llmBlendStrategy as BlendingStrategy) ?? "weighted";
    this.llmWeight = 0.7; // 70% LLM, 30% heuristic by default
  }

  /**
   * Judge using both engines and blend results
   */
  async judge(
    action: string,
    context: string,
    options: JudgeOptions = {}
  ): Promise<JudgmentResult> {
    try {
      // Run both engines in parallel
      const [llmResult, heuristicResult] = await Promise.all([
        this.llmEngine.judge(action, context, options),
        this.heuristicEngine.judge(action, context, options),
      ]);

      // Blend results based on strategy
      const blended = this.blendResults(llmResult, heuristicResult);
      blended.system = "hybrid";

      return blended;
    } catch (error) {
      // If hybrid fails, try LLM only
      try {
        const llmResult = await this.llmEngine.judge(action, context, options);
        llmResult.system = "llm";
        return llmResult;
      } catch {
        // Fallback to heuristic
        const heuristicResult = await this.heuristicEngine.judge(action, context, options);
        heuristicResult.system = "heuristic";
        return heuristicResult;
      }
    }
  }

  /**
   * Safety check using hybrid approach
   */
  async checkSafety(
    action: string,
    context: string,
    opts: SafetyCheckOptions = {}
  ): Promise<JudgmentResult> {
    return this.judge(action, context, { ...opts, type: "safety" });
  }

  /**
   * Quality evaluation using hybrid approach
   */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<JudgmentResult> {
    // Build context from options
    let context = "Quality evaluation";
    if (opts.requirements && opts.requirements.length > 0) {
      context += `\nRequirements: ${opts.requirements.join(", ")}`;
    }
    if (opts.userIntent) {
      context += `\nIntent: ${opts.userIntent}`;
    }

    return this.judge(content, context, { type: "quality" });
  }

  /**
   * Ethics evaluation using hybrid approach
   */
  async evaluateEthics(action: string, context: string = "倫理的評価"): Promise<JudgmentResult> {
    return this.judge(action, context, { type: "ethics" });
  }

  /**
   * Risk assessment using hybrid approach
   */
  async assessRisk(
    action: string,
    context: string,
    opts: RiskAssessmentOptions = {}
  ): Promise<JudgmentResult> {
    let enhancedContext = context;
    if (opts.riskTolerance !== undefined) {
      enhancedContext += `\nRisk tolerance: ${opts.riskTolerance}/100`;
    }

    return this.judge(action, enhancedContext, { type: "risk" });
  }

  /**
   * Prioritize using hybrid approach
   */
  async prioritize(tasks: string[], context: string = "タスク優先順位付け"): Promise<PrioritizationResult> {
    try {
      // Both engines should produce similar rankings
      // Use LLM result with heuristic validation
      const llmResult = await this.llmEngine.prioritize(tasks, context);

      // Heuristic result for comparison
      const heuristicResult = await this.heuristicEngine.prioritize(tasks, context);

      // Blend: weight LLM scores more heavily
      const blendedScores = llmResult.scores.map((llmScore, i) => {
        const hScore = heuristicResult.scores[i] ?? 50;
        return Math.round(this.llmWeight * llmScore + (1 - this.llmWeight) * hScore);
      });

      // Re-sort by blended scores
      const indices = Array.from({ length: tasks.length }, (_, i) => i)
        .sort((a, b) => blendedScores[b]! - blendedScores[a]!);

      const rankedTasks = indices.map((i) => llmResult.rankedTasks[i] ?? tasks[i] ?? "");
      const scores = indices.map((i) => blendedScores[i] ?? 50);
      const reasonings = indices.map((i) => llmResult.reasonings[i] ?? "");

      return { rankedTasks, scores, reasonings };
    } catch (error) {
      // Fallback to LLM
      return this.llmEngine.prioritize(tasks, context);
    }
  }

  /**
   * Blend two judgment results
   */
  private blendResults(llmResult: JudgmentResult, heuristicResult: JudgmentResult): JudgmentResult {
    switch (this.strategy) {
      case "weighted":
        return this.blendWeighted(llmResult, heuristicResult);
      case "confidence-based":
        return this.blendConfidenceBased(llmResult, heuristicResult);
      case "llm-primary":
        return this.blendLLMPrimary(llmResult, heuristicResult);
      default:
        return this.blendWeighted(llmResult, heuristicResult);
    }
  }

  /**
   * Weighted blending: 70% LLM, 30% heuristic
   */
  private blendWeighted(llmResult: JudgmentResult, heuristicResult: JudgmentResult): JudgmentResult {
    // Blend scores
    const blendedScore = Math.round(this.llmWeight * llmResult.score + (1 - this.llmWeight) * heuristicResult.score);

    // Decision: whichever score is higher determines decision
    const decision = blendedScore >= 50 ? "Yes" : "No";

    // Reasoning: combine both
    const reasoning = `LLM: ${llmResult.reasoning} | Heuristic: ${heuristicResult.reasoning}`;

    // Confidence: if both agree, high; if they disagree, medium
    const llmYes = llmResult.decision === "Yes";
    const heurYes = heuristicResult.decision === "Yes";
    const confidence = llmYes === heurYes ? "high" : "medium";

    // Factors: combine unique factors from both
    const factors = Array.from(
      new Set([...llmResult.keyFactors, ...heuristicResult.keyFactors])
    ).slice(0, 5);

    return {
      decision,
      score: blendedScore,
      reasoning: reasoning.substring(0, 500),
      confidence,
      keyFactors: factors,
      timestamp: new Date().toISOString(),
      system: "hybrid",
      llmModelUsed: llmResult.llmModelUsed,
      processingTimeMs: (llmResult.processingTimeMs ?? 0) + (heuristicResult.processingTimeMs ?? 0),
    };
  }

  /**
   * Confidence-based blending: use LLM if high confidence, else heuristic
   */
  private blendConfidenceBased(
    llmResult: JudgmentResult,
    heuristicResult: JudgmentResult
  ): JudgmentResult {
    // Use LLM if high confidence, otherwise heuristic
    if (llmResult.confidence === "high") {
      return {
        ...llmResult,
        system: "hybrid",
        reasoning: `${llmResult.reasoning} (High confidence, using LLM)`,
      };
    }

    return {
      ...heuristicResult,
      system: "hybrid",
      reasoning: `${heuristicResult.reasoning} (LLM confidence low, using heuristic)`,
    };
  }

  /**
   * LLM-primary: use LLM, validate with heuristic for red flags
   */
  private blendLLMPrimary(llmResult: JudgmentResult, heuristicResult: JudgmentResult): JudgmentResult {
    // If LLM and heuristic strongly disagree on decision, flag as low confidence
    const llmYes = llmResult.decision === "Yes";
    const heurYes = heuristicResult.decision === "Yes";
    const scoreDiff = Math.abs(llmResult.score - heuristicResult.score);

    let confidence = llmResult.confidence;
    if (llmYes !== heurYes && scoreDiff > 40) {
      // Strong disagreement: reduce confidence
      confidence = "low";
    }

    return {
      ...llmResult,
      system: "hybrid",
      confidence,
      reasoning: `${llmResult.reasoning}${llmYes !== heurYes ? " ⚠️ Heuristic disagrees" : ""}`,
    };
  }
}
