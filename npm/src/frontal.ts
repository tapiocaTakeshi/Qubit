import type {
  JudgeOptions,
  JudgmentCriteria,
  JudgmentResult,
  JudgmentType,
  PrioritizationResult,
  QualityEvalOptions,
  RiskAssessmentOptions,
  SafetyCheckOptions,
} from "./types.js";

// ---------------------------------------------------------------------------
// QBNN-inspired scoring helpers (pure TypeScript — no torch dependency)
//
// The scoring logic mirrors the heuristic / fallback path of
// FrontalEngineJudge._compute_score() in frontal_engine_mcp_server.py.
// ---------------------------------------------------------------------------

/** APQB quantum correlation: r = cos(2θ), mapped to a 0–100 score */
function apqbScore(theta: number): number {
  const r = Math.cos(2 * theta);          // correlation ∈ [-1, 1]
  return Math.round(((r + 1) / 2) * 100); // map to [0, 100]
}

/**
 * Derive a pseudo-quantum angle θ from a normalised text signal.
 * The signal is the fraction of "positive" keyword hits over total matched
 * words, shifted into [0, π/4] — the high-correlation regime.
 */
function textToTheta(signal: number): number {
  // signal ∈ [0, 1]  →  θ ∈ [0, π/4]
  // θ = 0   ⟹ r = 1  (fully positive)
  // θ = π/4 ⟹ r = 0  (neutral)
  return (1 - signal) * (Math.PI / 4);
}

interface ScoredText {
  score: number;
  positiveMatches: string[];
  negativeMatches: string[];
}

// Keyword lexicons
const POSITIVE_KEYWORDS: Record<JudgmentType, string[]> = {
  safety: [
    "safe", "secure", "protected", "authorised", "trusted",
    "安全", "セキュリティ", "保護", "認証", "信頼", "承認", "適切",
  ],
  ethics: [
    "ethical", "fair", "transparent", "respectful", "consent",
    "倫理", "公正", "透明", "尊重", "同意", "プライバシー",
  ],
  quality: [
    "accurate", "clear", "complete", "relevant", "helpful",
    "正確", "明確", "完全", "関連", "有用", "品質", "良い",
  ],
  risk: [
    "mitigated", "controlled", "reversible", "low-risk",
    "軽減", "管理可能", "可逆", "低リスク", "対策",
  ],
  decision: [
    "beneficial", "effective", "optimal", "recommended",
    "有益", "効果的", "最適", "推奨", "良い", "適切",
  ],
  priority: [
    "urgent", "critical", "high-priority", "important",
    "緊急", "重要", "優先", "必須",
  ],
};

const NEGATIVE_KEYWORDS: Record<JudgmentType, string[]> = {
  safety: [
    "unsafe", "dangerous", "risk", "vulnerability", "breach", "attack",
    "危険", "リスク", "脆弱", "違反", "攻撃", "不正",
  ],
  ethics: [
    "unethical", "unfair", "biased", "harmful", "discriminatory",
    "非倫理", "不公平", "偏見", "有害", "差別",
  ],
  quality: [
    "inaccurate", "incomplete", "irrelevant", "misleading", "vague",
    "不正確", "不完全", "無関係", "誤解", "曖昧", "悪い",
  ],
  risk: [
    "high-risk", "uncontrolled", "irreversible", "catastrophic",
    "高リスク", "制御不能", "不可逆", "壊滅的", "危険",
  ],
  decision: [
    "harmful", "ineffective", "wasteful", "suboptimal", "avoid",
    "有害", "非効果的", "非効率", "回避", "悪い",
  ],
  priority: [
    "low-priority", "deferred", "optional", "minor",
    "低優先", "後回し", "任意", "軽微",
  ],
};

function scoreText(
  text: string,
  judgmentType: JudgmentType
): ScoredText {
  const lower = text.toLowerCase();
  const posWords = POSITIVE_KEYWORDS[judgmentType] ?? [];
  const negWords = NEGATIVE_KEYWORDS[judgmentType] ?? [];

  const positiveMatches = posWords.filter((w) => lower.includes(w.toLowerCase()));
  const negativeMatches = negWords.filter((w) => lower.includes(w.toLowerCase()));

  const total = positiveMatches.length + negativeMatches.length;
  const signal = total === 0 ? 0.5 : positiveMatches.length / total;

  const theta = textToTheta(signal);
  const score = apqbScore(theta);

  return { score, positiveMatches, negativeMatches };
}

function evaluateCriteria(
  context: string,
  criteria: JudgmentCriteria
): number {
  let hits = 0;
  let total = 0;
  for (const [key, val] of Object.entries(criteria)) {
    total++;
    if (context.toLowerCase().includes(String(key).toLowerCase())) hits++;
    if (typeof val === "string" && context.toLowerCase().includes(val.toLowerCase())) hits++;
  }
  return total === 0 ? 50 : Math.round((hits / (total * 2)) * 100);
}

function evaluateOptions(context: string, options: string[]): number {
  if (options.length === 0) return 50;
  const matched = options.filter((o) =>
    context.toLowerCase().includes(o.toLowerCase())
  );
  return Math.round((matched.length / options.length) * 100);
}

// ---------------------------------------------------------------------------
// QBNNFrontalEngine
// ---------------------------------------------------------------------------

/**
 * Pure-JavaScript quantum-inspired judgment engine.
 *
 * Ports the heuristic fallback path of `FrontalEngineJudge` from
 * `frontal_engine_mcp_server.py` into TypeScript, with the APQB
 * (Adjustable Pseudo Quantum Bit) scoring model.
 *
 * @example
 * ```ts
 * import { QBNNFrontalEngine } from "neuroquantum";
 *
 * const engine = new QBNNFrontalEngine();
 *
 * const result = await engine.judge(
 *   "ユーザーの個人情報をログに記録する",
 *   "セキュリティ監査のため",
 *   { type: "safety" }
 * );
 * console.log(result.decision, result.score, result.reasoning);
 * ```
 */
export class QBNNFrontalEngine {
  // -------------------------------------------------------------------------
  // Core judgment
  // -------------------------------------------------------------------------

  /**
   * Judge an action given its context.
   *
   * @param action  - Description of the action / content to judge
   * @param context - Background situation or explanation
   * @param options - Judgment parameters
   */
  async judge(
    action: string,
    context: string,
    options: JudgeOptions = {}
  ): Promise<JudgmentResult> {
    const type: JudgmentType = options.type ?? "decision";
    const strictMode = options.strictMode ?? false;

    const fullText = `${action}\n${context}`;
    const { score: baseScore, positiveMatches, negativeMatches } =
      scoreText(fullText, type);

    let score = baseScore;
    const keyFactors: string[] = [];

    if (positiveMatches.length > 0) {
      keyFactors.push(`肯定的指標: ${positiveMatches.slice(0, 3).join(", ")}`);
    }
    if (negativeMatches.length > 0) {
      keyFactors.push(`否定的指標: ${negativeMatches.slice(0, 3).join(", ")}`);
    }

    if (options.criteria && Object.keys(options.criteria).length > 0) {
      const criteriaScore = evaluateCriteria(fullText, options.criteria);
      score = Math.round((score + criteriaScore) / 2);
      keyFactors.push(`基準マッチ度: ${criteriaScore}%`);
    }

    if (options.options && options.options.length > 0) {
      const optionScore = evaluateOptions(fullText, options.options);
      score = Math.round((score + optionScore) / 2);
      keyFactors.push(`選択肢マッチ度: ${optionScore}%`);
    }

    score = Math.max(0, Math.min(100, score));

    const threshold = strictMode ? 70 : 50;
    const decision: "Yes" | "No" = score >= threshold ? "Yes" : "No";

    let confidence: "high" | "medium" | "low";
    if (score >= 80 || score <= 20) {
      confidence = "high";
    } else if (score >= 65 || score <= 35) {
      confidence = "medium";
    } else {
      confidence = "low";
    }

    const reasoning = this.buildReasoning(score, type, keyFactors);

    return {
      decision,
      score,
      reasoning,
      confidence,
      keyFactors: keyFactors.slice(0, 5),
      timestamp: new Date().toISOString(),
      system: "qbnn",
    };
  }

  private buildReasoning(
    score: number,
    type: JudgmentType,
    keyFactors: string[]
  ): string {
    const typeLabel: Record<JudgmentType, string> = {
      safety: "安全性",
      ethics: "倫理性",
      quality: "品質",
      risk: "リスク評価",
      decision: "意思決定",
      priority: "優先度",
    };
    const label = typeLabel[type];

    let base: string;
    if (score >= 70) {
      base = `${label}の観点から肯定的な判断が支持されます（スコア: ${score}/100）。`;
    } else if (score >= 50) {
      base = `${label}の観点からやや肯定的ですが不確実性があります（スコア: ${score}/100）。`;
    } else {
      base = `${label}の観点から否定的な判断が示唆されます（スコア: ${score}/100）。`;
    }

    if (keyFactors.length > 0) {
      base += ` 根拠: ${keyFactors.slice(0, 3).join("; ")}。`;
    }
    return base;
  }

  // -------------------------------------------------------------------------
  // Convenience wrappers
  // -------------------------------------------------------------------------

  /**
   * Check whether an action is safe to perform.
   */
  async checkSafety(
    action: string,
    context: string,
    opts: SafetyCheckOptions = {}
  ): Promise<JudgmentResult> {
    const riskText =
      opts.risks && opts.risks.length > 0
        ? `\n考慮するリスク: ${opts.risks.join(", ")}`
        : "";
    const constraintText = opts.constraints
      ? `\n制約: ${JSON.stringify(opts.constraints)}`
      : "";
    return this.judge(action, `${context}${riskText}${constraintText}`, {
      type: "safety",
    });
  }

  /**
   * Evaluate the ethical implications of an action.
   */
  async evaluateEthics(
    action: string,
    context: string
  ): Promise<JudgmentResult> {
    return this.judge(action, context, { type: "ethics" });
  }

  /**
   * Assess the risk level of an action.
   *
   * A high `riskTolerance` lowers the bar for a "Yes" (allow) decision.
   */
  async assessRisk(
    action: string,
    context: string,
    opts: RiskAssessmentOptions = {}
  ): Promise<JudgmentResult> {
    const tolerance = opts.riskTolerance ?? 50;
    const result = await this.judge(action, context, { type: "risk" });
    // Re-derive decision with tolerance-adjusted threshold
    const threshold = 100 - tolerance;
    return {
      ...result,
      decision: result.score >= threshold ? "Yes" : "No",
    };
  }

  /**
   * Evaluate the quality of content.
   */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<JudgmentResult> {
    const requirementsText =
      opts.requirements && opts.requirements.length > 0
        ? `\n要件: ${opts.requirements.join(", ")}`
        : "";
    const intentText = opts.userIntent
      ? `\nユーザーの意図: ${opts.userIntent}`
      : "";
    return this.judge(
      content,
      `品質評価${requirementsText}${intentText}`,
      { type: "quality" }
    );
  }

  /**
   * Rank a list of tasks by priority.
   *
   * Returns the tasks in descending order of QBNN priority score.
   */
  async prioritize(
    tasks: string[],
    context: string
  ): Promise<PrioritizationResult> {
    const scored = await Promise.all(
      tasks.map(async (task) => {
        const result = await this.judge(task, context, { type: "priority" });
        return { task, score: result.score, reasoning: result.reasoning };
      })
    );
    scored.sort((a, b) => b.score - a.score);
    return {
      rankedTasks: scored.map((s) => s.task),
      scores: scored.map((s) => s.score),
      reasonings: scored.map((s) => s.reasoning),
    };
  }
}
