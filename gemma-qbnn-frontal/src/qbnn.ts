/**
 * QBNN判断層
 * APQB量子状態計算と判断処理
 */

import { LanguageUnderstanding, QBNNJudgmentResult } from "./types";

export class QBNNJudgment {
  private theta: number[];
  private entangle_strength: number;

  constructor(entangle_strength: number = 0.7, seed?: number) {
    // パラメータ検証
    if (entangle_strength < 0 || entangle_strength > 1) {
      throw new Error(
        `entangle_strength must be between 0 and 1, got ${entangle_strength}`
      );
    }

    if (seed !== undefined && !Number.isFinite(seed)) {
      throw new Error(`seed must be a finite number, got ${seed}`);
    }

    this.entangle_strength = entangle_strength;
    this.theta = this.initializeTheta(256, seed);
  }

  /**
   * Thetaパラメータを初期化
   */
  private initializeTheta(size: number, seed?: number): number[] {
    const theta: number[] = [];
    const rng = seed !== undefined ? this.seededRandom(seed) : Math.random;

    for (let i = 0; i < size; i++) {
      // Box-Muller変換でガウス分布を生成
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      theta.push(z * 0.1);
    }
    return theta;
  }

  /**
   * シード付きランダム数生成器
   */
  private seededRandom(seed: number): () => number {
    let x = Math.sin(seed) * 10000;
    return () => {
      x = Math.sin(x) * 10000;
      return x - Math.floor(x);
    };
  }

  /**
   * テキストをトークン化
   */
  private tokenize(text: string): number[] {
    const tokens: number[] = [];
    const text_slice = text.substring(0, 256);

    for (let i = 0; i < text_slice.length; i++) {
      tokens.push(text_slice.charCodeAt(i) % 256);
    }
    return tokens;
  }

  /**
   * 課題に対して判断を実行
   */
  judgeIssues(
    understanding: LanguageUnderstanding,
    issues: string[]
  ): QBNNJudgmentResult {
    const { raw_text } = understanding;

    // テキストを数値化
    const tokens = this.tokenize(raw_text);

    // APQB計算（量子状態）
    const theta = this.theta.slice(0, tokens.length);
    const r: number[] = [];
    const T: number[] = [];

    for (let i = 0; i < theta.length; i++) {
      r.push(Math.cos(2 * theta[i]));
      T.push(Math.abs(Math.sin(2 * theta[i])));
    }

    // 量子補正
    const quantum_correction: number[] = [];
    for (let i = 0; i < r.length; i++) {
      quantum_correction.push(
        (r[i] * 0.3 + T[i] * 0.2) * this.entangle_strength
      );
    }

    // 判断スコア計算
    let judgment_score =
      quantum_correction.length > 0
        ? quantum_correction.reduce((a, b) => a + b, 0) /
          quantum_correction.length
        : 0;

    // 正規化 (-0.3～+0.3 を 0～1に)
    let normalized_score = (judgment_score + 0.3) / 0.6;
    normalized_score = Math.max(0, Math.min(1, normalized_score));

    // 信頼度計算
    let confidence = 0;
    if (quantum_correction.length > 0) {
      const variance =
        quantum_correction.reduce((sum, val) => {
          return sum + Math.pow(val - judgment_score, 2);
        }, 0) / quantum_correction.length;
      confidence = Math.sqrt(variance);
    }

    // 判断結果の生成
    const decision = normalized_score > 0.5 ? "Yes" : "No";
    const tendency = judgment_score > 0 ? "positive" : "negative";

    return {
      score: normalized_score * 100,
      decision,
      tendency,
      confidence,
      issues,
      quantum_info: {
        raw_score: judgment_score,
        quantum_correction_magnitude:
          quantum_correction.length > 0
            ? quantum_correction.reduce((sum, val) => sum + Math.abs(val), 0) /
              quantum_correction.length
            : 0,
        entangle_strength: this.entangle_strength,
      },
    };
  }
}
