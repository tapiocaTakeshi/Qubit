/**
 * Gemma + QBNN ハイブリッド推論エンジン
 * 統合されたパイプライン
 */

import { GemmaLanguageProcessor } from "./gemma";
import { QBNNJudgment } from "./qbnn";
import {
  HybridResponse,
  EngineConfig,
  LanguageUnderstanding,
  QBNNJudgmentResult,
} from "./types";

/**
 * Gemma + QBNN ハイブリッド推論エンジン
 *
 * パイプライン:
 * 入力 → セキュリティフィルタ → Gemma言語理解 → Gemma課題発見 → QBNN課題判断 → Gemma言語生成 → 出力
 */
export class GemmaQBNNEngine {
  private gemma: GemmaLanguageProcessor;
  private qbnn: QBNNJudgment;
  private config: EngineConfig;
  private securityKeywords: Set<string>;
  private readonly MAX_BATCH_SIZE = 100;
  private readonly MAX_INPUT_LENGTH = 10000;

  constructor(config: EngineConfig = {}) {
    this.config = config;
    this.gemma = new GemmaLanguageProcessor();
    this.qbnn = new QBNNJudgment(
      config.entangle_strength || 0.7,
      config.seed
    );
    this.securityKeywords = this.initializeSecurityFilters();
  }

  /**
   * セキュリティ関連キーワードを初期化
   */
  private initializeSecurityFilters(): Set<string> {
    return new Set([
      "ハッキング",
      "侵入",
      "クラッキング",
      "バックドア",
      "ウイルス",
      "ランサムウェア",
      "マルウェア",
      "フィッシング",
      "パスワード盗難",
      "パスワード盗み",
      "DDoS",
      "ddos",
      "Ddos",
      "分散型サービス妨害",
      "不正アクセス",
      "システム侵害",
      "データ盗難",
      "暗号資産盗難",
      "詐欺サイト",
      "脆弱性悪用",
    ]);
  }

  /**
   * セキュリティフィルタ: 悪意のある要求を検出
   */
  private validateSecurityRequest(userInput: string): { safe: boolean; reason?: string } {
    const input_lower = userInput.toLowerCase();

    for (const keyword of this.securityKeywords) {
      if (input_lower.includes(keyword)) {
        return {
          safe: false,
          reason: `セキュリティ関連キーワード検出: "${keyword}"`,
        };
      }
    }

    return { safe: true };
  }

  /**
   * 応答を生成
   *
   * パイプライン:
   * 0. セキュリティフィルタ
   * 1. Gemma言語理解
   * 2. Gemma課題発見
   * 3. QBNN課題判断
   * 4. Gemma言語生成
   */
  async generate(userInput: string): Promise<HybridResponse> {
    // 入力検証
    if (userInput.length > this.MAX_INPUT_LENGTH) {
      throw new Error(
        `入力が長すぎます。最大${this.MAX_INPUT_LENGTH}文字までです。`
      );
    }

    // ステップ0: セキュリティフィルタ
    const securityCheck = this.validateSecurityRequest(userInput);
    if (!securityCheck.safe) {
      return {
        input: userInput,
        response: `申し訳ございませんが、セキュリティ上の理由からこのリクエストにはお応えできません。(${securityCheck.reason})`,
        issues_discovered: ["セキュリティ関連の不適切な要求"],
        qbnn_decision: "No",
        qbnn_score: 0,
        qbnn_tendency: "negative",
        confidence: 1.0,
        model: "Gemma + QBNN ハイブリッド推論（セキュリティフィルタ適用）",
        processing_pipeline: [
          "ステップ0: セキュリティフィルタ（ブロック）",
        ],
        timestamp: new Date().toISOString(),
      };
    }

    // ステップ1: Gemmaが言語を理解
    const understanding = this.gemma.understandLanguage(userInput);

    // ステップ2: Gemmaが課題を発見
    const issues = this.gemma.discoverIssues(understanding);

    // ステップ3: QBNNが課題に対して判断を実行
    const judgment = this.qbnn.judgeIssues(understanding, issues);

    // ステップ4: Gemmaが動的応答を生成
    const response = this.gemma.generateDynamicResponse(understanding, judgment);

    return {
      input: userInput,
      response,
      issues_discovered: issues,
      qbnn_decision: judgment.decision,
      qbnn_score: judgment.score,
      qbnn_tendency: judgment.tendency,
      confidence: judgment.confidence,
      model: "Gemma + QBNN ハイブリッド推論",
      processing_pipeline: [
        "ステップ0: セキュリティフィルタ（許可）",
        "ステップ1: Gemma言語理解",
        "ステップ2: Gemma課題発見",
        "ステップ3: QBNN課題判断",
        "ステップ4: Gemma言語生成",
      ],
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * 複数の応答を生成
   */
  async generateBatch(
    userInput: string,
    numVariations: number = 3
  ): Promise<HybridResponse[]> {
    if (numVariations > this.MAX_BATCH_SIZE) {
      throw new Error(
        `バッチサイズが大きすぎます。最大${this.MAX_BATCH_SIZE}までです。`
      );
    }

    if (numVariations < 1) {
      throw new Error("numVariationsは1以上である必要があります。");
    }

    const responses: HybridResponse[] = [];

    for (let i = 0; i < numVariations; i++) {
      const response = await this.generate(userInput);
      responses.push(response);
    }

    return responses;
  }

  /**
   * エンジンの情報を取得
   */
  getInfo(): {
    model: string;
    version: string;
    capabilities: string[];
  } {
    return {
      model: "Gemma + QBNN ハイブリッド推論",
      version: "1.0.0",
      capabilities: [
        "言語理解",
        "課題発見",
        "量子判断",
        "動的応答生成",
        "複数バリエーション生成",
      ],
    };
  }
}
