/**
 * Gemma + QBNN ハイブリッド推論システム
 * 使用例
 */

import { GemmaQBNNEngine } from "./src/index";

async function main() {
  console.log("=".repeat(70));
  console.log("Gemma + QBNN ハイブリッド推論システム");
  console.log("=".repeat(70));

  // エンジンを初期化
  const engine = new GemmaQBNNEngine();

  // エンジン情報を表示
  const info = engine.getInfo();
  console.log(`\nモデル: ${info.model}`);
  console.log(`バージョン: ${info.version}`);
  console.log(`機能: ${info.capabilities.join(", ")}`);

  // 例1: プログラミング学習
  console.log("\n" + "=".repeat(70));
  console.log("例1: プログラミング学習");
  console.log("=".repeat(70));

  const response1 = await engine.generate(
    "プログラミングを学ぶコツは何ですか？"
  );

  console.log(`\n入力: ${response1.input}`);
  console.log(`\n課題: ${response1.issues_discovered.join(", ")}`);
  console.log(`\nQBNN判定: ${response1.qbnn_decision}`);
  console.log(`スコア: ${response1.qbnn_score.toFixed(1)}/100`);
  console.log(`傾向: ${response1.qbnn_tendency}`);
  console.log(`信頼度: ${response1.confidence.toFixed(3)}`);
  console.log(`\n応答:\n${response1.response}`);

  // 例2: キャリア決定
  console.log("\n" + "=".repeat(70));
  console.log("例2: キャリア決定");
  console.log("=".repeat(70));

  const response2 = await engine.generate(
    "転職すべきですか？給与は上がるけど、安定性が不安です。"
  );

  console.log(`\n入力: ${response2.input}`);
  console.log(`\n課題: ${response2.issues_discovered.join(", ")}`);
  console.log(`\nQBNN判定: ${response2.qbnn_decision}`);
  console.log(`スコア: ${response2.qbnn_score.toFixed(1)}/100`);
  console.log(`傾向: ${response2.qbnn_tendency}`);
  console.log(`\n応答:\n${response2.response}`);

  // 例3: 感情サポート
  console.log("\n" + "=".repeat(70));
  console.log("例3: 感情サポート");
  console.log("=".repeat(70));

  const response3 = await engine.generate(
    "今日の気分が落ち込んでいます。元気づけてください。"
  );

  console.log(`\n入力: ${response3.input}`);
  console.log(`\n課題: ${response3.issues_discovered.join(", ")}`);
  console.log(`\nQBNN判定: ${response3.qbnn_decision}`);
  console.log(`スコア: ${response3.qbnn_score.toFixed(1)}/100`);
  console.log(`傾向: ${response3.qbnn_tendency}`);
  console.log(`\n応答:\n${response3.response}`);

  // 例4: 複数応答の生成
  console.log("\n" + "=".repeat(70));
  console.log("例4: 複数応答の生成と一貫性分析");
  console.log("=".repeat(70));

  const responses = await engine.generateBatch(
    "起業に興味があります。アドバイスをください。",
    5
  );

  console.log(`\n入力: ${responses[0].input}`);
  console.log(`\n5回の実行結果:`);

  responses.forEach((r, i) => {
    console.log(
      `実行 ${i + 1}: スコア=${r.qbnn_score.toFixed(1)}/100, 判定=${r.qbnn_decision}, 傾向=${r.qbnn_tendency}`
    );
  });

  // QBNN判断の統計
  const scores = responses.map((r) => r.qbnn_score);
  console.log(`\n統計:`);
  console.log(
    `  平均スコア: ${(scores.reduce((a, b) => a + b) / scores.length).toFixed(1)}/100`
  );
  console.log(
    `  最大スコア: ${Math.max(...scores).toFixed(1)}/100`
  );
  console.log(
    `  最小スコア: ${Math.min(...scores).toFixed(1)}/100`
  );

  console.log("\n" + "=".repeat(70));
  console.log("実行完了");
  console.log("=".repeat(70));
}

main().catch(console.error);
