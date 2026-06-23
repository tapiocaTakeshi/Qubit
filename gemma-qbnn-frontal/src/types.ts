/**
 * Gemma + QBNN ハイブリッド推論システム
 * TypeScript型定義
 */

/**
 * ユーザー入力の言語理解結果
 */
export interface LanguageUnderstanding {
  raw_text: string;
  is_question: boolean;
  is_request: boolean;
  is_decision: boolean;
  is_emotional: boolean;
  keywords: string[];
}

/**
 * 発見された課題の定義
 */
export interface DiscoveredIssue {
  keyword: string;
  label: string;
  confidence: number;
}

/**
 * QBNN判断の結果
 */
export interface QBNNJudgmentResult {
  score: number; // 0-100
  decision: "Yes" | "No";
  tendency: "positive" | "negative";
  confidence: number;
  issues: string[];
  quantum_info: {
    raw_score: number;
    quantum_correction_magnitude: number;
    entangle_strength: number;
  };
}

/**
 * 最終的な応答結果
 */
export interface HybridResponse {
  input: string;
  response: string;
  issues_discovered: string[];
  qbnn_decision: "Yes" | "No";
  qbnn_score: number;
  qbnn_tendency: "positive" | "negative";
  confidence: number;
  model: string;
  processing_pipeline: string[];
  timestamp: string;
}

/**
 * エンジン設定
 */
export interface EngineConfig {
  entangle_strength?: number;
  seed?: number;
}

/**
 * チャットbot用の会話メッセージ
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  timestamp: string;
}

/**
 * チャットbot設定
 */
export interface ChatbotConfig extends EngineConfig {
  systemPrompt?: string;
  maxHistory?: number;
  showDiagnostics?: boolean;
}

/**
 * チャットbotの1ターン分の結果
 */
export interface ChatbotTurn {
  user: string;
  assistant: string;
  raw_response: HybridResponse;
  history: ChatMessage[];
}

/**
 * スコア表現マップ
 */
export type ScoreExpressionRange = {
  min: number;
  max: number;
  expression: string;
};
