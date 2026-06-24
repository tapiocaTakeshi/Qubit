/**
 * NeuroQuantumEngine — TypeScript port of neuroquantum_layered.py
 *
 * Implements the core mathematical components of neuroquantum_layered.py
 * in pure TypeScript — no HTTP endpoints, no Python runtime required.
 *
 * Ported components:
 *   - QBNNLayer.forward()       : entanglement correction + dynamic λ
 *   - QBNNAttention.forward()   : QBNN-enhanced multi-head attention
 *   - NeuroQuantumAI.generate() : dynamic temperature (sinusoidal θ phase)
 *   - APQB scoring              : r = cos(2θ), score = ((r+1)/2) × 100
 */

import type { JudgmentResult, JudgmentType, QualityEvalOptions, RiskAssessmentOptions, SafetyCheckOptions } from "./types.js";

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/** GELU activation — matches PyTorch's default GELU used in QBNNLayer */
function gelu(x: number): number {
  return (
    x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))
  );
}

/** APQB correlation: r = cos(2θ) */
function apqbCorrelation(theta: number): number {
  return Math.cos(2 * theta);
}

/** Map a signal ∈ [0, 1] to θ ∈ [0, π/4] — high-correlation regime */
function signalToTheta(signal: number): number {
  return (1 - signal) * (Math.PI / 4);
}

/** Map correlation r ∈ [-1, 1] to a 0–100 score */
function correlationToScore(r: number): number {
  return Math.round(((r + 1) / 2) * 100);
}

// ---------------------------------------------------------------------------
// QBNNLayer — port of neuroquantum_layered.py: QBNNLayer.forward()
// ---------------------------------------------------------------------------

class QBNNLayerTS {
  /** Inference-mode call counter — drives the sinusoidal dynamic λ */
  private callCount = 0;

  constructor(
    readonly lambdaMin: number,
    readonly lambdaMax: number,
    /** Scalar entanglement coefficient (J tensor simplified to scalar) */
    readonly J: number
  ) {}

  /**
   * Forward pass matching QBNNLayer.forward() in neuroquantum_layered.py.
   *
   * Equations (from the Python source):
   *   s_prev  = tanh(h_prev)                              # Bloch sphere z
   *   h_tilde = W(h_prev)                                 # linear (identity)
   *   s_raw   = tanh(h_tilde)
   *   delta   = J * s_prev * s_raw                        # entanglement correction
   *   phase   = call_count × 0.2
   *   dynamic = 0.5 + 0.5 × sin(phase)                   # θ fluctuation
   *   λ_eff   = λ_min + Δλ × (λ_norm×0.7 + dynamic×0.3)
   *   h_hat   = h_tilde + λ_eff × delta
   *   output  = GELU(LayerNorm(h_hat))
   */
  forward(hPrev: number): number {
    const sPrev = Math.tanh(hPrev);
    const hTilde = hPrev; // W = identity in scalar form
    const sRaw = Math.tanh(hTilde);
    const delta = this.J * sPrev * sRaw;

    // Dynamic λ — inference branch of QBNNLayer.forward()
    const phase = this.callCount * 0.2;
    const dynamicFactor = 0.5 + 0.5 * Math.sin(phase);
    this.callCount++;

    const lambdaNorm = 0.5; // neutral λ_base (sigmoid(0) ≈ 0.5)
    const lambdaRange = this.lambdaMax - this.lambdaMin;
    const lambdaEff =
      this.lambdaMin + lambdaRange * (lambdaNorm * 0.7 + dynamicFactor * 0.3);

    const hHat = hTilde + lambdaEff * delta;

    // LayerNorm (scalar: clamp to [-2, 2]) → GELU
    return gelu(Math.max(-2, Math.min(2, hHat)));
  }
}

// ---------------------------------------------------------------------------
// QBNN Attention — port of neuroquantum_layered.py: QBNNAttention.forward()
// ---------------------------------------------------------------------------

/**
 * Simplified multi-head QBNN attention.
 *
 * Equations (from the Python source):
 *   Q_norm = tanh(Q), K_norm = tanh(K)              # Bloch normalization
 *   delta  = Σ_h Q_norm_h × J_h × K_norm_h          # entanglement correction
 *   score  = Q·K + λ_attn × delta                   # QBNN-enhanced score
 *   output = sigmoid(score)                          # attention weight
 */
function qbnnAttention(
  actionH: number,
  contextH: number,
  numHeads: number,
  lambdaAttn: number
): number {
  const qNorm = Math.tanh(actionH);
  const kNorm = Math.tanh(contextH);

  // J_attn per head — initialised as small value (0.02 std in Python)
  const jPerHead = 0.3 / numHeads;
  const delta = numHeads * qNorm * jPerHead * kNorm; // Σ_h

  const attnScore = qNorm * kNorm + lambdaAttn * delta;
  return 1 / (1 + Math.exp(-attnScore)); // sigmoid
}

// ---------------------------------------------------------------------------
// Keyword lexicons (judgment-type specific)
// ---------------------------------------------------------------------------

const POSITIVE: Record<JudgmentType, string[]> = {
  safety: ["safe", "secure", "protected", "authorised", "trusted", "安全", "セキュリティ", "保護", "認証", "信頼", "承認", "適切"],
  ethics: ["ethical", "fair", "transparent", "respectful", "consent", "倫理", "公正", "透明", "尊重", "同意", "プライバシー"],
  quality: ["accurate", "clear", "complete", "relevant", "helpful", "正確", "明確", "完全", "関連", "有用", "品質", "良い"],
  risk: ["mitigated", "controlled", "reversible", "low-risk", "軽減", "管理可能", "可逆", "低リスク", "対策"],
  decision: ["beneficial", "effective", "optimal", "recommended", "有益", "効果的", "最適", "推奨", "良い", "適切"],
  priority: ["urgent", "critical", "high-priority", "important", "緊急", "重要", "優先", "必須"],
};

const NEGATIVE: Record<JudgmentType, string[]> = {
  safety: ["unsafe", "dangerous", "risk", "vulnerability", "breach", "attack", "危険", "リスク", "脆弱", "違反", "攻撃", "不正"],
  ethics: ["unethical", "unfair", "biased", "harmful", "discriminatory", "非倫理", "不公平", "偏見", "有害", "差別"],
  quality: ["inaccurate", "incomplete", "irrelevant", "misleading", "vague", "不正確", "不完全", "無関係", "誤解", "曖昧", "悪い"],
  risk: ["high-risk", "uncontrolled", "irreversible", "catastrophic", "高リスク", "制御不能", "不可逆", "壊滅的", "危険"],
  decision: ["harmful", "ineffective", "wasteful", "suboptimal", "avoid", "有害", "非効果的", "非効率", "回避", "悪い"],
  priority: ["low-priority", "deferred", "optional", "minor", "低優先", "後回し", "任意", "軽微"],
};

/** Extract a polarised signal ∈ [-1, 1] from text using judgment-type keywords */
function extractSignal(text: string, type: JudgmentType): number {
  const lower = text.toLowerCase();
  const pos = (POSITIVE[type] ?? []).filter((w) => lower.includes(w.toLowerCase())).length;
  const neg = (NEGATIVE[type] ?? []).filter((w) => lower.includes(w.toLowerCase())).length;
  const total = pos + neg;
  return total === 0 ? 0 : (pos - neg) / total; // ∈ (-1, 1)
}

const TYPE_LABELS: Record<JudgmentType, string> = {
  safety: "安全性",
  ethics: "倫理性",
  quality: "品質",
  risk: "リスク評価",
  decision: "意思決定",
  priority: "優先度",
};

// ---------------------------------------------------------------------------
// NeuroQuantumEngine config
// ---------------------------------------------------------------------------

/**
 * Configuration mirroring the 'cpu' tier of NeuroQuantumConfig in
 * neuroquantum_layered.py (num_layers=3, num_heads=4, lambda_entangle=0.5).
 */
export interface NeuroQuantumEngineConfig {
  /** Number of QBNN transformer blocks (default: 3 — matches 'cpu' tier) */
  numLayers?: number;
  /** Number of attention heads (default: 4 — matches 'cpu' tier) */
  numHeads?: number;
  /** Base entanglement strength λ (default: 0.5) */
  lambdaEntangle?: number;
}

// ---------------------------------------------------------------------------
// NeuroQuantumEngine
// ---------------------------------------------------------------------------

/**
 * Pure-TypeScript implementation of the NeuroQuantum judgment engine.
 *
 * Ports the following from neuroquantum_layered.py:
 *   - QBNNLayer (entanglement correction + dynamic sinusoidal λ)
 *   - QBNNAttention (QBNN-enhanced action × context attention)
 *   - Dynamic θ phase from NeuroQuantumAI.generate()
 *   - APQB scoring: r = cos(2θ)
 *
 * No HTTP endpoint, no PyTorch, no external dependencies.
 *
 * @example
 * ```ts
 * import { NeuroQuantumEngine } from "qubit_ai";
 *
 * const engine = new NeuroQuantumEngine({ numLayers: 3 });
 * const result = await engine.judge(
 *   "ユーザーデータをログに記録",
 *   "本番環境",
 *   { type: "safety" }
 * );
 * console.log(result.decision, result.score);
 * ```
 */
export class NeuroQuantumEngine {
  private readonly layers: QBNNLayerTS[];
  private readonly numHeads: number;
  private readonly lambdaEntangle: number;
  /** Global call counter — drives the dynamic θ phase in generate() */
  private callCount = 0;

  constructor(config: NeuroQuantumEngineConfig = {}) {
    const numLayers = config.numLayers ?? 3;
    this.numHeads = config.numHeads ?? 4;
    this.lambdaEntangle = config.lambdaEntangle ?? 0.5;

    // Build layers with λ range = [entangle×0.5, entangle×1.5]
    // matching QBNNTransformerBlock in neuroquantum_layered.py
    this.layers = Array.from({ length: numLayers }, (_, i) =>
      new QBNNLayerTS(
        this.lambdaEntangle * 0.5,
        this.lambdaEntangle * 1.5,
        // J decreases per layer (small-std initialisation in Python)
        0.3 / (i + 1)
      )
    );
  }

  // ---------------------------------------------------------------------------
  // Core judgment
  // ---------------------------------------------------------------------------

  /**
   * Run a judgment through the NeuroQuantum pipeline.
   *
   * Pipeline:
   *   text → keyword signals → QBNN attention → N QBNN layers → APQB score
   */
  async judge(
    action: string,
    context: string,
    opts: { type?: JudgmentType; strictMode?: boolean } = {}
  ): Promise<JudgmentResult> {
    const type: JudgmentType = opts.type ?? "decision";
    const strictMode = opts.strictMode ?? false;

    // 1. Extract polarised signals ∈ [-1, 1]
    const actionH = extractSignal(action, type);
    const contextH = extractSignal(context, type);
    const combinedH = extractSignal(`${action}\n${context}`, type);

    // 2. QBNN attention — QBNNAttention.forward() (action as Q, context as K)
    const attnWeight = qbnnAttention(
      actionH,
      contextH,
      this.numHeads,
      this.lambdaEntangle
    );

    // 3. Attention-weighted combination (standard + QBNN blend: 70/30)
    const stdOut = combinedH;
    const qbnnOut = combinedH * attnWeight;
    let h = 0.7 * stdOut + 0.3 * qbnnOut; // FFN blend from QBNNTransformerBlock

    // 4. Pass through N QBNN layers — QBNNLayer.forward() × numLayers
    for (const layer of this.layers) {
      h = layer.forward(h);
    }

    // 5. Dynamic temperature phase — NeuroQuantumAI.generate()
    //    theta_phase = step × 0.2; temp = t_min + Δt × (0.5 + 0.5×sin(phase))
    const thetaPhase = this.callCount * 0.2;
    const dynamicFactor = 0.5 + 0.5 * Math.sin(thetaPhase);
    this.callCount++;

    // Blend processed signal with dynamic factor (mirrors dynamic temperature)
    const blendedH = h * (0.7 + 0.3 * dynamicFactor);

    // 6. APQB scoring
    const signal = (Math.max(-1, Math.min(1, blendedH)) + 1) / 2; // [-1,1] → [0,1]
    const theta = signalToTheta(signal);
    const r = apqbCorrelation(theta);
    const score = Math.max(0, Math.min(100, correlationToScore(r)));

    const threshold = strictMode ? 70 : 50;
    const decision: "Yes" | "No" = score >= threshold ? "Yes" : "No";

    const distance = Math.abs(score - threshold);
    const confidence: "high" | "medium" | "low" =
      distance >= 30 ? "high" : distance >= 15 ? "medium" : "low";

    const label = TYPE_LABELS[type];
    let reasoning: string;
    if (score >= 70) {
      reasoning = `${label}の観点から肯定的な判断が支持されます（スコア: ${score}/100）。${this.layers.length}層QBNNとλ_eff動的もつれ補正を適用済み。`;
    } else if (score >= 50) {
      reasoning = `${label}の観点からやや肯定的ですが不確実性があります（スコア: ${score}/100）。${this.layers.length}層QBNNとλ_eff動的もつれ補正を適用済み。`;
    } else {
      reasoning = `${label}の観点から否定的な判断が示唆されます（スコア: ${score}/100）。${this.layers.length}層QBNNとλ_eff動的もつれ補正を適用済み。`;
    }

    const keyFactors = [
      `アクション信号: ${actionH.toFixed(3)}`,
      `コンテキスト信号: ${contextH.toFixed(3)}`,
      `QBNN注意重み: ${attnWeight.toFixed(3)}`,
      `${this.layers.length}層処理後h: ${h.toFixed(3)}`,
      `θ位相: ${thetaPhase.toFixed(2)} rad (動的λ)`,
    ];

    return {
      decision,
      score,
      reasoning,
      confidence,
      keyFactors,
      timestamp: new Date().toISOString(),
      system: "qbnn",
    };
  }

  // ---------------------------------------------------------------------------
  // Convenience wrappers (mirror QBNNFrontalEngine API)
  // ---------------------------------------------------------------------------

  /** Check whether an action is safe to perform. */
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

  /** Evaluate ethical implications. */
  async evaluateEthics(action: string, context: string): Promise<JudgmentResult> {
    return this.judge(action, context, { type: "ethics" });
  }

  /**
   * Assess risk level.
   * A higher `riskTolerance` lowers the bar for a "Yes" (allow) decision.
   */
  async assessRisk(
    action: string,
    context: string,
    opts: RiskAssessmentOptions = {}
  ): Promise<JudgmentResult> {
    const tolerance = opts.riskTolerance ?? 50;
    const result = await this.judge(action, context, { type: "risk" });
    const threshold = 100 - tolerance;
    return { ...result, decision: result.score >= threshold ? "Yes" : "No" };
  }

  /** Evaluate content quality. */
  async evaluateQuality(
    content: string,
    opts: QualityEvalOptions = {}
  ): Promise<JudgmentResult> {
    const reqText =
      opts.requirements && opts.requirements.length > 0
        ? `\n要件: ${opts.requirements.join(", ")}`
        : "";
    const intentText = opts.userIntent ? `\nユーザーの意図: ${opts.userIntent}` : "";
    return this.judge(content, `品質評価${reqText}${intentText}`, {
      type: "quality",
    });
  }

  /** Rank tasks by priority score (descending). */
  async prioritize(
    tasks: string[],
    context: string
  ): Promise<{ rankedTasks: string[]; scores: number[]; reasonings: string[] }> {
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
