/**
 * Qubit AI v3.0.0 with Pyodide Backend Example
 *
 * Demonstrates quantum-inspired reasoning + HF dataset training
 * No external servers required - everything runs in WebAssembly
 */

import {
  QubitAIPyodide,
  getQubitAIPyodide,
} from "../npm/src/index.js";

/**
 * Example 1: Basic quantum-inspired judgment
 */
async function exampleBasicJudgment() {
  console.log("\n=== Basic Quantum-Inspired Judgment ===\n");

  const qubit = getQubitAIPyodide({
    productName: "Qubit AI (Pyodide)",
  });

  await qubit.initialize();

  const result = await qubit.judge(
    "データベース全体を削除する",
    "本番環境でのメンテナンス操作",
    "safety"
  );

  console.log("Decision:", result.decision);
  console.log("Score:", result.score);
  console.log("Confidence:", result.confidence);
  console.log("Reasoning:", result.reasoning);
  console.log("Factors:", result.factors);
  console.log("\nExplanation:");
  console.log(qubit.explain(result));
}

/**
 * Example 2: Safety checks with quantum reasoning
 */
async function exampleSafetyChecks() {
  console.log("\n=== Safety Checks ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  const checks = [
    {
      action: "APIキーをログ出力",
      context: "本番環境",
    },
    {
      action: "ユーザーデータをエクスポート",
      context: "デバッグ目的",
    },
    {
      action: "パスワードをハードコード",
      context: "開発環境",
    },
  ];

  for (const check of checks) {
    const [isSafe, details] = await qubit.safetyCheck(
      check.action,
      check.context
    );

    console.log(`✓ ${check.action}`);
    console.log(`  Safe: ${isSafe}`);
    console.log(`  Score: ${details.score}`);
    console.log();
  }
}

/**
 * Example 3: Ethics evaluation
 */
async function exampleEthicsEvaluation() {
  console.log("\n=== Ethics Evaluation ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  const result = await qubit.ethicsCheck(
    "ユーザーの位置情報を収集・分析",
    ["ユーザー", "企業", "規制当局"],
    ["プライバシー侵害", "信頼の喪失", "法的リスク"]
  );

  console.log("Action: User location data collection");
  console.log("Decision:", result.decision);
  console.log("Ethical Score:", result.score);
  console.log("Reasoning:", result.reasoning);
}

/**
 * Example 4: Quality assessment
 */
async function exampleQualityEvaluation() {
  console.log("\n=== Quality Assessment ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  const contents = [
    "これはAIが生成した正確で有用なテキストです。",
    "テキスト",
    "複雑な技術概念を分かりやすく説明した詳細なドキュメント。",
  ];

  for (const content of contents) {
    const result = await qubit.evaluateQuality(content);
    console.log(`Content: "${content.substring(0, 50)}..."`);
    console.log(`Quality Score: ${result.score}/100`);
    console.log(`Confidence: ${result.confidence}`);
    console.log();
  }
}

/**
 * Example 5: Task prioritization with quantum reasoning
 */
async function examplePrioritization() {
  console.log("\n=== Task Prioritization ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  const tasks = [
    {
      name: "セキュリティパッチ",
      description: "本番環境のセキュリティ脆弱性修正",
    },
    {
      name: "ドキュメント更新",
      description: "APIドキュメントの更新",
    },
    {
      name: "バグ修正",
      description: "エッジケースのバグ修正",
    },
    {
      name: "新機能開発",
      description: "ユーザーから要望のある機能開発",
    },
  ];

  const ranked = await qubit.prioritize(tasks);

  console.log("Quantum-Inspired Task Ranking:");
  ranked.forEach(([task, score], index) => {
    console.log(`${index + 1}. ${task.name}`);
    console.log(`   Priority: ${(score * 100).toFixed(1)}%`);
    console.log(`   Description: ${task.description}`);
  });
}

/**
 * Example 6: Train on HuggingFace dataset
 */
async function exampleHFDatasetTraining() {
  console.log("\n=== HuggingFace Dataset Training ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  try {
    console.log("Starting training on HF dataset...");

    const result = await qubit.trainOnHFDataset({
      dataset: "llm-jp/oasst2-33k-ja",
      judgmentType: "safety",
      maxExamples: 100,  // Small batch for demo
      onProgress: (progress) => {
        const percent = (
          (progress.processedExamples / progress.totalExamples) * 100
        ).toFixed(1);
        console.log(
          `Processing: ${progress.processedExamples}/${progress.totalExamples} (${percent}%)`
        );
      },
    });

    console.log("\nTraining Complete!");
    console.log(`Total Examples: ${result.totalExamples}`);
    console.log(`Batches: ${result.batches}`);
    console.log(`Duration: ${result.durationMs}ms`);
    console.log(`Status: ${result.status}`);

    if (result.errors.length > 0) {
      console.log("Errors:", result.errors);
    }
  } catch (error) {
    console.error("Training failed:", error);
  }
}

/**
 * Example 7: Strict mode (enforced threshold)
 */
async function exampleStrictMode() {
  console.log("\n=== Strict Mode Judgment ===\n");

  const qubit = getQubitAIPyodide({
    strictMode: true,  // Requires score >= 70 for "Yes"
  });

  await qubit.initialize();

  const testCases = [
    {
      action: "バックアップされたデータを削除",
      context: "確認済み",
    },
    {
      action: "テストデータをエクスポート",
      context: "開発環境",
    },
  ];

  for (const testCase of testCases) {
    const result = await qubit.judge(
      testCase.action,
      testCase.context,
      "safety"
    );

    console.log(`Action: ${testCase.action}`);
    console.log(`Score: ${result.score} (requires >= 70 for "Yes")`);
    console.log(
      `Decision: ${result.decision} ${
        result.score >= 70 ? "✓" : "(blocked by strict mode)"
      }`
    );
    console.log();
  }
}

/**
 * Example 8: Judgment history and system status
 */
async function exampleHistoryAndStatus() {
  console.log("\n=== History and Status Tracking ===\n");

  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  // Make several judgments
  await qubit.judge("action1", "context1", "safety");
  await qubit.judge("action2", "context2", "ethics");
  await qubit.safetyCheck("action3", "context3");

  // Get history
  const history = qubit.getHistory(10);
  console.log("Recent Judgments:");
  history.forEach((record, i) => {
    console.log(
      `${i + 1}. ${record.judgmentType}: ${record.decision} (${record.score})`
    );
  });

  // Get status
  const status = await qubit.getStatus();
  console.log("\nSystem Status:");
  console.log(`Product: ${status.product}`);
  console.log(`Status: ${status.status}`);
  console.log(`Engine Available: ${status.frontalEngineAvailable}`);
  console.log(`History Size: ${status.judgmentHistorySize}/${status.maxHistory}`);

  // Get info
  const info = qubit.getInfo();
  console.log("\nProduct Info:");
  console.log(`Product: ${info.product}`);
  console.log(`Version: ${info.version}`);
  console.log(`Session ID: ${info.sessionId}`);
}

/**
 * Example 9: Convenience functions with global singleton
 */
async function exampleConvenienceFunctions() {
  console.log("\n=== Convenience Functions (Global Singleton) ===\n");

  const {
    judge,
    safetyCheck,
    ethicsCheck,
    evaluateQuality,
  } = await import("../npm/src/index.js");

  // Initialize global instance
  const qubit = getQubitAIPyodide();
  await qubit.initialize();

  // Use module-level functions
  const judgmentResult = await judge(
    "ユーザーログイン情報を記録",
    "デバッグモード",
    "safety"
  );

  console.log("Global judge():", judgmentResult.decision);

  const [safe, details] = await safetyCheck(
    "APIキーをログに出力",
    "本番環境"
  );

  console.log("Global safetyCheck():", safe);

  const ethics = await ethicsCheck("ユーザーデータ分析", ["ユーザー"]);

  console.log("Global ethicsCheck():", ethics.decision);

  const quality = await evaluateQuality("品質評価対象のテキスト");

  console.log("Global evaluateQuality():", quality.score);
}

/**
 * Main execution
 */
async function main() {
  console.log("╔════════════════════════════════════════╗");
  console.log("║  Qubit AI v3.0.0 - Pyodide Examples   ║");
  console.log("║  Quantum + HF Training in WebAssembly ║");
  console.log("╚════════════════════════════════════════╝");

  try {
    await exampleBasicJudgment();
    await exampleSafetyChecks();
    await exampleEthicsEvaluation();
    await exampleQualityEvaluation();
    await examplePrioritization();
    await exampleStrictMode();
    await exampleHistoryAndStatus();
    await exampleConvenienceFunctions();

    // Training example (commented out - takes time)
    // await exampleHFDatasetTraining();

    console.log("\n╔════════════════════════════════════════╗");
    console.log("║     All Examples Completed! ✓          ║");
    console.log("╚════════════════════════════════════════╝\n");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
