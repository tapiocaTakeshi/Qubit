/**
 * NeuroQuantumAPIClient — Python REST API クライアント
 *
 * neuroquantum_layered.py が提供する REST API にアクセスし、
 * 量子インスパイアド推論の結果を取得します
 */

import type { JudgmentResult, SafetyCheckOptions, QualityEvalOptions } from "./types.js";

/**
 * NeuroQuantum REST API の応答型
 */
export interface NeuroQuantumResponse {
  decision: "Yes" | "No";
  score: number;
  reasoning: string;
  confidence: "high" | "medium" | "low";
  factors: string[];
  timestamp: string;
  system: string;
  processing_time_ms: number;
}

/**
 * NeuroQuantumAPIClient の設定
 */
export interface NeuroQuantumAPIClientConfig {
  /** REST API のベース URL (デフォルト: http://localhost:5000) */
  baseUrl?: string;
  /** リクエストタイムアウト (ミリ秒、デフォルト: 30000) */
  timeout?: number;
  /** リトライ回数 (デフォルト: 3) */
  maxRetries?: number;
  /** リトライ待機時間 (ミリ秒、デフォルト: 1000) */
  retryDelayMs?: number;
}

/**
 * NeuroQuantumAPIClient - Python neuroquantum_layered.py との連携
 */
export class NeuroQuantumAPIClient {
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;
  private retryDelayMs: number;

  constructor(config: NeuroQuantumAPIClientConfig = {}) {
    this.baseUrl = config.baseUrl ?? "http://localhost:5000";
    this.timeout = config.timeout ?? 30000;
    this.maxRetries = config.maxRetries ?? 3;
    this.retryDelayMs = config.retryDelayMs ?? 1000;
  }

  /**
   * API へのリクエストを実行（リトライ付き）
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeout);

        const response = await fetch(url, {
          method,
          headers: {
            "Content-Type": "application/json",
          },
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timer);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data: T = await response.json();
        return data;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // リトライ可能なエラーか判定
        if (attempt < this.maxRetries - 1) {
          const isRetryable =
            lastError.message.includes("ERR_NETWORK") ||
            lastError.message.includes("timeout") ||
            lastError.message.includes("503") ||
            lastError.message.includes("502") ||
            lastError.message.includes("500");

          if (isRetryable) {
            await this.sleep(this.retryDelayMs * Math.pow(2, attempt));
            continue;
          }
        }

        break;
      }
    }

    throw lastError || new Error("Request failed");
  }

  /**
   * スリープ utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * ヘルスチェック
   */
  async healthCheck(): Promise<{
    status: string;
    version: string;
    neuroquantum_available: boolean;
  }> {
    return this.request("GET", "/api/v1/health");
  }

  /**
   * 設定を取得
   */
  async getConfig(): Promise<Record<string, unknown>> {
    const response = await this.request<{ config: Record<string, unknown> }>(
      "GET",
      "/api/v1/config"
    );
    return response.config;
  }

  /**
   * ステータスを取得
   */
  async getStatus(): Promise<{
    status: string;
    version: string;
    model_config: Record<string, unknown>;
    neuroquantum_available: boolean;
  }> {
    return this.request("GET", "/api/v1/status");
  }

  /**
   * 判定を実行
   */
  async judge(
    action: string,
    context: string,
    judgmentType: string = "safety",
    strictMode: boolean = false
  ): Promise<NeuroQuantumResponse> {
    return this.request<NeuroQuantumResponse>("POST", "/api/v1/judge", {
      action,
      context,
      judgment_type: judgmentType,
      strict_mode: strictMode,
    });
  }

  /**
   * バッチ判定を実行
   */
  async batchJudge(
    requests: Array<{
      action: string;
      context: string;
      judgment_type?: string;
      strict_mode?: boolean;
    }>
  ): Promise<{
    results: NeuroQuantumResponse[];
    count: number;
  }> {
    return this.request("POST", "/api/v1/batch_judge", {
      requests,
    });
  }

  /**
   * 安全性チェック
   */
  async safetyCheck(
    action: string,
    context?: string,
    opts?: SafetyCheckOptions
  ): Promise<{
    safe: boolean;
    result: NeuroQuantumResponse;
  }> {
    return this.request("POST", "/api/v1/safety_check", {
      action,
      context: context ?? "",
      risks: opts?.risks ?? [],
    });
  }

  /**
   * 倫理チェック
   */
  async ethicsCheck(
    action: string,
    stakeholders?: string[],
    harms?: string[]
  ): Promise<NeuroQuantumResponse> {
    return this.request("POST", "/api/v1/ethics_check", {
      action,
      stakeholders: stakeholders ?? [],
      potential_harms: harms ?? [],
    });
  }

  /**
   * 品質評価
   */
  async qualityEval(
    content: string,
    opts?: QualityEvalOptions
  ): Promise<NeuroQuantumResponse> {
    return this.request("POST", "/api/v1/quality_eval", {
      content,
      requirements: opts?.requirements ?? [],
      user_intent: opts?.userIntent ?? "",
    });
  }

  /**
   * API の可用性をチェック
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * API がリッチャブルになるまで待機
   */
  async waitForAvailable(timeoutMs: number = 30000): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      if (await this.isAvailable()) {
        return;
      }
      await this.sleep(500);
    }

    throw new Error(`NeuroQuantum API は ${timeoutMs}ms 内に応答しませんでした`);
  }
}

/**
 * グローバル NeuroQuantum API クライアント (シングルトン)
 */
let globalClient: NeuroQuantumAPIClient | null = null;

/**
 * グローバル NeuroQuantum API クライアントを取得/作成
 */
export function getNeuroQuantumClient(
  config?: NeuroQuantumAPIClientConfig
): NeuroQuantumAPIClient {
  if (!globalClient) {
    globalClient = new NeuroQuantumAPIClient(config);
  }
  return globalClient;
}

/**
 * グローバル NeuroQuantum API クライアントをリセット
 */
export function resetNeuroQuantumClient(): void {
  globalClient = null;
}

/**
 * NeuroQuantum レスポンスを QubitAIResult に変換
 */
export function convertNeuroQuantumToQubitResult(
  nqResponse: NeuroQuantumResponse
): Record<string, unknown> {
  return {
    decision: nqResponse.decision,
    score: nqResponse.score,
    reasoning: nqResponse.reasoning,
    confidence: nqResponse.confidence,
    factors: nqResponse.factors,
    timestamp: nqResponse.timestamp,
    system: nqResponse.system,
    processingTimeMs: nqResponse.processing_time_ms,
  };
}
