/**
 * Qubit AI 推論デモンストレーション
 * Gemma言語処理エンジンの実際の推論実行
 */

import { GemmaLanguageProcessor } from "../src/gemma";
import { QBNNJudgmentResult } from "../src/types";

// テスト用のジャッジメント結果を生成
const createJudgment = (score = 75, tendency: "positive" | "negative" = "positive"): QBNNJudgmentResult => ({
  score,
  decision: score > 60 ? "Yes" : "No",
  tendency,
  confidence: 0.85,
  issues: [],
  quantum_info: {
    raw_score: 0.65,
    quantum_correction_magnitude: 0.15,
    entangle_strength: 0.72,
  },
});

// テストケース
const testCases = [
  {
    input: "転職を検討しているのですが、今の会社と新しい会社のどちらを選ぶべきですか？",
    description: "キャリア変更の推論テスト",
    judgment: createJudgment(82, "positive"),
  },
  {
    input: "困っているのですが、プログラミングを学ぶのに最適な方法は何ですか？",
    description: "学習方法の推論テスト",
    judgment: createJudgment(78, "positive"),
  },
  {
    input: "怪しい投資話を持ちかけられたのですが、どう判断すれば良いですか？",
    description: "リスク評価・詐欺検知の推論テスト",
    judgment: createJudgment(45, "negative"),
  },
  {
    input: "データサイエンティストへのキャリアシフトについてアドバイスをください",
    description: "キャリア・スキル習得の複合推論テスト",
    judgment: createJudgment(88, "positive"),
  },
  {
    input: "最近気分が落ち込んでいるのですが、どうすればいいですか？",
    description: "感情・メンタルヘルスの推論テスト",
    judgment: createJudgment(55, "negative"),
  },
];

async function runReasoningDemo() {
  console.log("🧠 Qubit AI 推論デモンストレーション\n");
  console.log("=" .repeat(80) + "\n");

  const processor = new GemmaLanguageProcessor();

  for (const testCase of testCases) {
    console.log(`📋 テスト: ${testCase.description}`);
    console.log(`   入力: "${testCase.input}"\n`);

    try {
      // ステップ1: 言語理解
      console.log("   [ステップ1] 言語理解フェーズ");
      const understanding = processor.understandLanguage(testCase.input);
      console.log(`   ✓ 質問判定: ${understanding.is_question}`);
      console.log(`   ✓ リクエスト: ${understanding.is_request}`);
      console.log(`   ✓ 意思決定: ${understanding.is_decision}`);
      console.log(`   ✓ 感情的: ${understanding.is_emotional}`);
      console.log();

      // ステップ2: 課題発見
      console.log("   [ステップ2] 課題発見フェーズ");
      const issues = processor.discoverIssues(understanding);
      console.log(`   ✓ 検出課題: ${issues.join(", ")}`);
      console.log();

      // ステップ3: 安全性検知
      console.log("   [ステップ3] 安全性検知フェーズ");
      const safetyCategory = processor.detectSafetyCategory(testCase.input);
      console.log(`   ✓ 安全性リスク: ${safetyCategory || "なし"}`);
      console.log();

      // ステップ4: 動的応答生成
      console.log("   [ステップ4] 動的応答生成フェーズ");
      const judgment = createJudgment(...Object.values(testCase.judgment));
      judgment.issues = issues;
      const response = processor.generateDynamicResponse(understanding, judgment);
      console.log(`   ✓ スコア: ${judgment.score}点`);
      console.log(`   ✓ 傾向: ${judgment.tendency}`);
      console.log(`   ✓ 信頼度: ${(judgment.confidence * 100).toFixed(0)}%`);
      console.log(`   ✓ 量子補正: ${(judgment.quantum_info?.quantum_correction_magnitude || 0).toFixed(3)}`);
      console.log();

      console.log("   📢 生成応答:");
      console.log("   " + response.split("\n").join("\n   "));
      console.log();

    } catch (error) {
      console.error(`   ❌ エラー: ${error instanceof Error ? error.message : String(error)}`);
    }

    console.log("-".repeat(80) + "\n");
  }

  console.log("✨ 推論デモンストレーション完了！");
  console.log("=" .repeat(80));
}

runReasoningDemo().catch(console.error);
