/**
 * LLMFrontalEngine — LLM-based judgment engine
 *
 * Implements the same interface as QBNNFrontalEngine but uses LLM inference
 * for reasoning instead of keyword-based heuristics
 */

import { LLMProvider } from "./llm-provider.js";
import { PromptTemplates } from "./prompt-templates.js";
import { ResponseParser } from "./response-parser.js";
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
 * LLMFrontalEngine — LLM-based judgment implementation
 *
 * Methods mirror QBNNFrontalEngine but use LLM reasoning
 */
export class LLMFrontalEngine {
  private llmProvider: LLMProvider;
  private config: QubitAIConfig;

  constructor(llmProvider: LLMProvider, config: QubitAIConfig = {}) {
    this.llmProvider = llmProvider;
    this.config = config;
  }

  /**
   * Judge an action using LLM inference
   */
  async judge(
    action: string,
    context: string,
    options: JudgeOptions = {}
  ): Promise<JudgmentResult> {
    const type = options.type ?? "decision";
    const strictMode = options.strictMode ?? this.config.strictMode ?? false;

    try {
      // Build prompt from template
      const { system, user } = PromptTemplates.buildPrompt(
        type,
        action,
        context,
        options.criteria
      );

      // Create full prompt
      const fullPrompt = `${system}\n\n${user}`;

      // Generate LLM response
      const startTime = Date.now();
      const response = await this.llmProvider.generate(fullPrompt, {
        maxNewTokens: 500,
        temperature: this.config.llmConfig?.temperature ?? 0.7,
        maxTokens: this.config.llmConfig?.maxTokens ?? 500,
      });

      const processingTimeMs = Date.now() - startTime;

      // Parse response
      const result = ResponseParser.parse(response.generatedText);

      // Apply strict mode if configured
      if (strictMode && result.decision === "Yes" && result.score < 70) {
        result.decision = "No";
        result.reasoning = `${result.reasoning} [Strict mode: score ${result.score} < 70 threshold]`;
      }

      result.system = "llm";
      result.processingTimeMs = processingTimeMs;
      result.llmModelUsed = this.config.llmConfig?.model;

      return result;
    } catch (error) {
      // Return error result
      return {
        decision: "No",
        score: 0,
        reasoning: `LLM judgment error: ${error instanceof Error ? error.message : String(error)}`,
        confidence: "low",
        keyFactors: ["Error in LLM judgment"],
        timestamp: new Date().toISOString(),
        system: "llm",
      };
    }
  }

  /**
   * Safety check using LLM
   */
  async checkSafety(
    action: string,
    context: string,
    opts: SafetyCheckOptions = {}
  ): Promise<JudgmentResult> {
    // Build context with risks if provided
    let enhancedContext = context;
    if (opts.risks && opts.risks.length > 0) {
      enhancedContext += `\n\nKnown risks: ${opts.risks.join(", ")}`;
    }
    if (opts.constraints) {
      enhancedContext += `\n\nConstraints: ${JSON.stringify(opts.constraints)}`;
    }

    return this.judge(action, enhancedContext, {
      type: "safety",
      strictMode: this.config.strictMode,
    });
  }

  /**
   * Evaluate quality using LLM
   */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<JudgmentResult> {
    let context = "Quality evaluation";

    if (opts.requirements && opts.requirements.length > 0) {
      context += `\n\nRequirements:\n${opts.requirements.map((r) => `- ${r}`).join("\n")}`;
    }

    if (opts.userIntent) {
      context += `\n\nUser intent: ${opts.userIntent}`;
    }

    return this.judge(content, context, {
      type: "quality",
      strictMode: this.config.strictMode,
    });
  }

  /**
   * Evaluate ethics using LLM
   */
  async evaluateEthics(
    action: string,
    context: string = "倫理的評価"
  ): Promise<JudgmentResult> {
    return this.judge(action, context, {
      type: "ethics",
      strictMode: this.config.strictMode,
    });
  }

  /**
   * Assess risk using LLM
   */
  async assessRisk(
    action: string,
    context: string,
    opts: RiskAssessmentOptions = {}
  ): Promise<JudgmentResult> {
    let enhancedContext = context;
    if (opts.riskTolerance !== undefined) {
      enhancedContext += `\n\nRisk tolerance: ${opts.riskTolerance}/100`;
    }

    return this.judge(action, enhancedContext, {
      type: "risk",
      strictMode: this.config.strictMode,
    });
  }

  /**
   * Prioritize tasks using LLM
   *
   * Returns a list of scored tasks sorted by priority
   */
  async prioritize(tasks: string[], context: string = "Task prioritization"): Promise<PrioritizationResult> {
    try {
      // Build prompt for prioritization
      const { system, user } = PromptTemplates.buildPrompt(
        "priority",
        `Tasks:\n${tasks.map((t, i) => `${i + 1}. ${t}`).join("\n")}`,
        context,
        {}
      );

      const fullPrompt = `${system}\n\n${user}`;

      // Generate LLM response
      const response = await this.llmProvider.generate(fullPrompt, {
        maxNewTokens: 500,
        temperature: this.config.llmConfig?.temperature ?? 0.7,
      });

      // Parse prioritization response
      const rankingData = this.parsePrioritizationResponse(response.generatedText, tasks);

      return rankingData;
    } catch (error) {
      // Fallback: return tasks in original order with low scores
      return {
        rankedTasks: tasks,
        scores: tasks.map(() => 50),
        reasonings: tasks.map(() => "Error in prioritization"),
      };
    }
  }

  /**
   * Parse LLM response for prioritization
   */
  private parsePrioritizationResponse(response: string, tasks: string[]): PrioritizationResult {
    try {
      // Try to extract JSON from response
      const cleaned = response
        .replace(/```json\s*([\s\S]*?)\s*```/g, "$1")
        .replace(/```\s*([\s\S]*?)\s*```/g, "$1")
        .trim();

      const parsed = JSON.parse(cleaned);

      if (Array.isArray(parsed)) {
        // Response is array of ranked items
        const rankedTasks: string[] = [];
        const scores: number[] = [];
        const reasonings: string[] = [];

        parsed.forEach((item: any) => {
          rankedTasks.push(item.task ?? item.name ?? "");
          scores.push(Math.min(100, Math.max(0, Number(item.score) ?? 50)));
          reasonings.push(item.reasoning ?? "");
        });

        return { rankedTasks, scores, reasonings };
      }
    } catch {
      // Parsing failed, fall back
    }

    // Fallback: return tasks in original order
    return {
      rankedTasks: tasks,
      scores: tasks.map(() => 50),
      reasonings: tasks.map(() => "Prioritization failed, using original order"),
    };
  }
}
