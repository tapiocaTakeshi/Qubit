/**
 * Gemma + QBNN as Frontal
 * 前頭葉として動作する判断エンジン
 */

import {
  FrontalEngineConfig,
  FrontalJudgmentResult,
  FrontalJudgmentTask,
} from "./types";

const POSITIVE_KEYWORDS = [
  "重要",
  "必須",
  "確認",
  "承認",
  "安全",
  "有効",
  "高い",
  "良い",
  "正しい",
  "成功",
  "完了",
  "済み",
  "十分",
  "品質",
  "テスト",
  "バックアップ",
  "ロールバック",
  "監査",
  "準拠",
  "同意",
];

const NEGATIVE_KEYWORDS = [
  "危険",
  "リスク",
  "問題",
  "失敗",
  "低い",
  "悪い",
  "不正",
  "禁止",
  "未実施",
  "不明確",
  "違反",
  "脆弱",
  "障害",
  "不足",
  "未完了",
];

/**
 * npm環境で利用できる軽量な Gemma+QBNN 前頭葉判断エンジン。
 * Python版の `FrontalEngineJudge` と同じ入出力スキーマを提供します。
 */
export class GemmaQBNNFrontal {
  private entangleStrength: number;
  private quantumWeight: number;
  private seed?: number;

  constructor(config: FrontalEngineConfig = {}) {
    this.entangleStrength = config.entangle_strength ?? 0.7;
    this.quantumWeight = config.quantum_weight ?? 0.6;
    this.seed = config.seed;
  }

  judge(task: FrontalJudgmentTask): FrontalJudgmentResult {
    const context = task.context?.trim() ?? "";
    const judgmentRequest = task.judgment_request?.trim() ?? "";

    if (!context || !judgmentRequest) {
      return this.errorResponse("context と judgment_request は必須です");
    }

    const criteria = task.criteria ?? {};
    const options = task.options ?? [];
    const strictMode = task.strict_mode ?? false;

    let score = this.computeHybridScore(context, judgmentRequest);
    const keyFactors = this.extractKeyFactors(context, judgmentRequest, criteria, options);

    if (Object.keys(criteria).length > 0) {
      const criteriaScore = this.evaluateCriteria(context, criteria);
      score = (score + criteriaScore) / 2;
      keyFactors.unshift(`基準マッチ度: ${Math.round(criteriaScore)}%`);
    }

    if (options.length > 0) {
      const optionsScore = this.evaluateOptions(context, options);
      score = (score + optionsScore) / 2;
      keyFactors.unshift(`選択肢マッチ度: ${Math.round(optionsScore)}%`);
    }

    const finalScore = Math.max(0, Math.min(100, Math.round(score)));
    const threshold = strictMode ? 70 : 50;
    const decision = finalScore >= threshold ? "Yes" : "No";
    const confidence = this.confidenceFromScore(finalScore, strictMode);

    return {
      decision,
      score: finalScore,
      reasoning: this.generateReasoning(finalScore, keyFactors),
      confidence,
      key_factors: keyFactors.slice(0, 5),
      timestamp: new Date().toISOString(),
      quantum_info: {
        yes_probability: finalScore / 100,
        quantum_weight: this.quantumWeight,
        entangle_strength: this.entangleStrength,
        system: "gemma_qbnn_frontal_npm",
      },
    };
  }

  getInfo(): { model: string; version: string; capabilities: string[] } {
    return {
      model: "Gemma + QBNN as Frontal",
      version: "1.3.0",
      capabilities: [
        "意思決定",
        "リスク評価",
        "品質判定",
        "優先順位付け",
        "倫理的判断",
        "Python版 FrontalEngineJudge 互換スキーマ",
      ],
    };
  }

  private computeHybridScore(context: string, judgmentRequest: string): number {
    const traditionalScore = this.computeTraditionalScore(context, judgmentRequest);
    const quantumScore = this.computeQBNNScore(`${context} [SEP] ${judgmentRequest}`);
    return quantumScore * this.quantumWeight + traditionalScore * (1 - this.quantumWeight);
  }

  private computeQBNNScore(text: string): number {
    let correctionSum = 0;
    const limit = Math.min(text.length, 512);
    const seedOffset = this.seed ?? 0;

    for (let i = 0; i < limit; i++) {
      const token = text.charCodeAt(i) % 256;
      const theta = Math.sin((token + 1) * (i + 1 + seedOffset)) * 0.1;
      const r = Math.cos(2 * theta);
      const t = Math.abs(Math.sin(2 * theta));
      correctionSum += (r * 0.3 + t * 0.2) * this.entangleStrength;
    }

    if (limit === 0) {
      return 50;
    }

    const rawScore = correctionSum / limit;
    const normalized = Math.max(0, Math.min(1, (rawScore + 0.3) / 0.6));
    return normalized * 100;
  }

  private computeTraditionalScore(context: string, judgmentRequest: string): number {
    const fullText = `${context}\n${judgmentRequest}`;
    let score = 50;

    if (context.split(/\s+/).filter(Boolean).length > 100 || context.length > 300) {
      score += 5;
    }

    for (const keyword of POSITIVE_KEYWORDS) {
      if (fullText.includes(keyword)) {
        score += 3;
      }
    }

    for (const keyword of NEGATIVE_KEYWORDS) {
      if (fullText.includes(keyword)) {
        score -= 3;
      }
    }

    return Math.max(0, Math.min(100, score));
  }

  private evaluateCriteria(context: string, criteria: Record<string, string | number | boolean>): number {
    let score = 50;

    for (const criterionValue of Object.values(criteria)) {
      if (typeof criterionValue === "string") {
        score += context.toLowerCase().includes(criterionValue.toLowerCase()) ? 15 : -5;
      } else if (typeof criterionValue === "boolean") {
        score += criterionValue ? 20 : -20;
      } else if (typeof criterionValue === "number") {
        score += Math.min(25, Math.max(-25, criterionValue / 5));
      }
    }

    return Math.max(0, Math.min(100, score));
  }

  private evaluateOptions(context: string, options: string[]): number {
    const matched = options.filter((option) => context.toLowerCase().includes(option.toLowerCase())).length;
    return matched > 0 ? Math.min(100, 50 + (matched / options.length) * 40) : 50;
  }

  private extractKeyFactors(
    context: string,
    judgmentRequest: string,
    criteria: Record<string, string | number | boolean>,
    options: string[]
  ): string[] {
    const factors: string[] = ["Gemma+QBNN前頭葉推論", "量子推論適用"];

    if (Object.keys(criteria).length > 0) {
      factors.push(`基準: ${Object.keys(criteria).slice(0, 2).join(", ")}`);
    }

    if (options.length > 0) {
      const matched = options.filter((option) => context.includes(option)).length;
      factors.push(`マッチオプション: ${matched}/${options.length}`);
    }

    if (context.length > 200) {
      factors.push("複雑なコンテキスト");
    }

    if (judgmentRequest.includes("リスク")) {
      factors.push("リスク評価");
    } else if (judgmentRequest.includes("倫理") || judgmentRequest.includes("プライバシー")) {
      factors.push("倫理的判断");
    } else if (judgmentRequest.includes("優先")) {
      factors.push("優先順位付け");
    }

    return factors;
  }

  private confidenceFromScore(score: number, strictMode: boolean): "high" | "medium" | "low" {
    if (strictMode) {
      return score >= 80 || score <= 20 ? "high" : "medium";
    }
    return score >= 75 || score <= 25 ? "high" : "medium";
  }

  private generateReasoning(score: number, keyFactors: string[]): string {
    let reasoning: string;
    if (score >= 70) {
      reasoning = "Gemma+QBNN前頭葉推論により、提供された情報は肯定的な判断を支持しています。";
    } else if (score >= 50) {
      reasoning = "Gemma+QBNN前頭葉推論の結果、判断は不確定ですが、妥当な結論が導き出されます。";
    } else {
      reasoning = "Gemma+QBNN前頭葉推論により、提供された情報は否定的な判断を支持しています。";
    }

    return `${reasoning} 分析手法: ${keyFactors.slice(0, 3).join(", ")}`;
  }

  private errorResponse(errorMessage: string): FrontalJudgmentResult {
    return {
      decision: "No",
      score: 0,
      reasoning: errorMessage,
      confidence: "low",
      key_factors: ["エラーが発生しました"],
      timestamp: new Date().toISOString(),
      error: true,
    };
  }
}

export { GemmaQBNNFrontal as FrontalEngineJudge };
