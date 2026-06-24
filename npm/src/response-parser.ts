/**
 * ResponseParser — Robust extraction of structured judgment from LLM output
 *
 * Handles various LLM response formats and imperfections gracefully
 */

import type { JudgmentResult } from "./types.js";

/**
 * ResponseParser extracts and validates judgment results from LLM output
 */
export class ResponseParser {
  /**
   * Parse LLM output into a JudgmentResult
   *
   * Multi-layer approach:
   * 1. Strip markdown code blocks
   * 2. Try JSON parsing
   * 3. Fallback to regex extraction
   * 4. Fill missing fields with defaults
   * 5. Validate and normalize
   *
   * @param llmOutput - Raw text from LLM
   * @returns Parsed and validated JudgmentResult
   */
  static parse(llmOutput: string): JudgmentResult {
    let parsed: Record<string, unknown>;

    try {
      // Step 1: Clean markdown
      const cleaned = this.stripMarkdown(llmOutput);

      // Step 2: Try standard JSON parsing
      try {
        parsed = JSON.parse(cleaned);
      } catch {
        // Step 3: Try regex extraction as fallback
        parsed = this.extractWithRegex(cleaned);
      }

      // Step 4: Validate and normalize fields
      const result = this.validateAndNormalize(parsed);

      return result;
    } catch (error) {
      // Ultimate fallback: return minimal valid result
      return this.createDefaultResult(
        `Failed to parse LLM response: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  /**
   * Strip markdown code blocks from output
   */
  private static stripMarkdown(text: string): string {
    // Remove markdown code blocks
    let cleaned = text.replace(/```json\s*([\s\S]*?)\s*```/g, "$1");
    cleaned = cleaned.replace(/```\s*([\s\S]*?)\s*```/g, "$1");

    // Also remove single backticks
    cleaned = cleaned.replace(/`([^`]+)`/g, "$1");

    return cleaned.trim();
  }

  /**
   * Extract JSON from text using regex patterns
   */
  private static extractWithRegex(text: string): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    // Try to extract decision
    const decisionMatch = text.match(/"decision"\s*:\s*"(Yes|No)"/i);
    if (decisionMatch) {
      result.decision = decisionMatch[1].charAt(0).toUpperCase() + decisionMatch[1].slice(1);
    }

    // Try to extract score
    const scoreMatch = text.match(/"score"\s*:\s*(\d+)/);
    if (scoreMatch) {
      result.score = parseInt(scoreMatch[1], 10);
    }

    // Try to extract reasoning
    const reasoningMatch = text.match(/"reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"/) ||
                           text.match(/"reasoning"\s*:\s*"([^"]{0,500})"/);
    if (reasoningMatch) {
      result.reasoning = reasoningMatch[1];
    }

    // Try to extract confidence
    const confidenceMatch = text.match(/"confidence"\s*:\s*"(high|medium|low)"/i);
    if (confidenceMatch) {
      result.confidence = confidenceMatch[1].toLowerCase();
    }

    // Try to extract factors array
    const factorsMatch = text.match(/"factors"\s*:\s*\[(.*?)\]/);
    if (factorsMatch) {
      const factorsStr = factorsMatch[1];
      const factors = factorsStr
        .split(",")
        .map((f) => f.trim().replace(/^["']|["']$/g, ""))
        .filter((f) => f.length > 0);
      if (factors.length > 0) {
        result.factors = factors;
      }
    }

    return result;
  }

  /**
   * Validate and normalize parsed data
   */
  private static validateAndNormalize(parsed: Record<string, unknown>): JudgmentResult {
    // Validate decision
    let decision: string = typeof parsed.decision === "string" ? parsed.decision : "No";
    decision = decision.toLowerCase() === "yes" ? "Yes" : "No";

    // Validate and normalize score
    let score: number = 50;
    if (typeof parsed.score === "string") {
      score = parseFloat(parsed.score);
    } else if (typeof parsed.score === "number") {
      score = parsed.score;
    }
    if (isNaN(score)) {
      score = 50;
    }
    score = Math.max(0, Math.min(100, Math.round(score)));

    // Validate reasoning
    let reasoning: string = typeof parsed.reasoning === "string"
      ? parsed.reasoning
      : `Decision: ${decision}`;
    reasoning = reasoning.trim().substring(0, 500);

    // Validate confidence
    let confidence: "high" | "medium" | "low" = "medium";
    const confidenceStr = String(parsed.confidence).toLowerCase();
    if (["high", "medium", "low"].includes(confidenceStr)) {
      confidence = confidenceStr as "high" | "medium" | "low";
    } else {
      // Infer from score
      if (score >= 80 || score <= 20) {
        confidence = "high";
      } else if (score >= 65 || score <= 35) {
        confidence = "medium";
      } else {
        confidence = "low";
      }
    }

    // Validate factors
    let factors: string[] = [];
    if (Array.isArray(parsed.factors)) {
      factors = parsed.factors
        .filter((f): f is string => typeof f === "string")
        .slice(0, 5) // Max 5 factors
        .map((f) => f.trim())
        .filter((f) => f.length > 0);
    }

    if (factors.length === 0) {
      // Generate default factors
      if (decision === "Yes") {
        factors = ["Positive assessment", "Meets criteria"];
      } else {
        factors = ["Negative assessment", "Does not meet criteria"];
      }
    }

    return {
      decision: decision as "Yes" | "No",
      score,
      reasoning,
      confidence: confidence as "high" | "medium" | "low",
      keyFactors: factors,
      timestamp: new Date().toISOString(),
      system: "llm",
    };
  }

  /**
   * Create a default result when parsing fails
   */
  private static createDefaultResult(error: string): JudgmentResult {
    return {
      decision: "No",
      score: 25,
      reasoning: error,
      confidence: "low",
      keyFactors: ["Parsing error", "Invalid LLM output format"],
      timestamp: new Date().toISOString(),
      system: "llm",
    };
  }

  /**
   * Normalize score from alternative ranges to [0, 100]
   *
   * @param rawScore - Score in any range
   * @param inputRange - Input range [min, max], default [0, 1]
   * @returns Normalized score in [0, 100]
   */
  static normalizeScore(rawScore: number, inputRange: [number, number] = [0, 1]): number {
    const [min, max] = inputRange;
    if (min === max) return 50;

    const normalized = ((rawScore - min) / (max - min)) * 100;
    return Math.max(0, Math.min(100, Math.round(normalized)));
  }

  /**
   * Map qualitative confidence to one of three levels
   *
   * @param raw - Any confidence representation
   * @returns Standardized confidence level
   */
  static normalizeConfidence(raw: unknown): "high" | "medium" | "low" {
    const str = String(raw).toLowerCase();

    // Direct matches
    if (["high", "certain", "confident", "very high", "sure", "definitely"].some((s) => str.includes(s))) {
      return "high";
    }
    if (["medium", "moderate", "somewhat", "maybe", "uncertain"].some((s) => str.includes(s))) {
      return "medium";
    }
    if (["low", "uncertain", "unsure", "low confidence", "barely"].some((s) => str.includes(s))) {
      return "low";
    }

    // Default
    return "medium";
  }

  /**
   * Extract factors from various formats
   */
  static extractFactors(raw: unknown): string[] {
    if (Array.isArray(raw)) {
      return raw
        .filter((f) => typeof f === "string")
        .map((f) => f.trim())
        .filter((f) => f.length > 0)
        .slice(0, 5);
    }

    if (typeof raw === "string") {
      // Try to parse as JSON array
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return this.extractFactors(parsed);
        }
      } catch {
        // Not JSON, split by common delimiters
      }

      // Split by common delimiters
      const factors = raw
        .split(/[,;]/)
        .map((f) => f.trim().replace(/^[-•*]\s*/, ""))
        .filter((f) => f.length > 0)
        .slice(0, 5);

      return factors;
    }

    return [];
  }

  /**
   * Validate that a JudgmentResult meets all requirements
   */
  static validate(result: JudgmentResult): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check decision
    if (!["Yes", "No"].includes(result.decision)) {
      errors.push(`Invalid decision: ${result.decision}`);
    }

    // Check score range
    if (typeof result.score !== "number" || result.score < 0 || result.score > 100) {
      errors.push(`Invalid score: ${result.score}`);
    }

    // Check reasoning
    if (!result.reasoning || result.reasoning.length === 0) {
      errors.push("Missing reasoning");
    }

    // Check confidence
    if (!["high", "medium", "low"].includes(result.confidence)) {
      errors.push(`Invalid confidence: ${result.confidence}`);
    }

    // Check factors
    if (!Array.isArray(result.keyFactors) || result.keyFactors.length === 0) {
      errors.push("Missing factors");
    }

    // Check timestamp
    if (!result.timestamp || !/\d{4}-\d{2}-\d{2}T/.test(result.timestamp)) {
      errors.push("Invalid timestamp format");
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}
