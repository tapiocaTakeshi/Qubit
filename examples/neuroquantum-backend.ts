/**
 * Example: Using Qubit AI with NeuroQuantum Python Backend
 *
 * This example demonstrates how to use the quantum-inspired neural network
 * reasoning from Python's neuroquantum_layered.py via REST API integration.
 *
 * Prerequisites:
 * 1. Start the Python API server:
 *    python neuroquantum_api_server.py --host 127.0.0.1 --port 5000
 *
 * 2. Install dependencies:
 *    npm install qubit-ai
 */

import { QubitAI } from "../npm/src/index.js";

async function exampleBasicJudgment() {
  console.log("\n=== Basic Quantum-Inspired Judgment ===\n");

  // Create QubitAI with NeuroQuantum backend enabled
  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
      timeout: 30000,
    },
  });

  // Perform a safety judgment
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
}

async function exampleSafetyCheck() {
  console.log("\n=== Safety Check with NeuroQuantum ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  // Check if an action is safe
  const [isSafe, details] = await qubit.safetyCheck(
    "APIキーをログ出力",
    "本番環境",
    {
      risks: ["認証情報漏洩", "セキュリティ侵害"],
    }
  );

  console.log("Safe?:", isSafe);
  console.log("Score:", details.score);
  console.log("Explanation:", details.reasoning);
}

async function exampleEthicsEvaluation() {
  console.log("\n=== Ethics Evaluation ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  // Evaluate ethical implications
  const result = await qubit.ethicsCheck(
    "ユーザーの位置情報を収集・共有する",
    ["ユーザー", "規制当局"],
    ["プライバシー侵害", "信頼の喪失"]
  );

  console.log("Decision:", result.decision);
  console.log("Score:", result.score);
  console.log("Reasoning:", result.reasoning);
}

async function exampleQualityEvaluation() {
  console.log("\n=== Content Quality Evaluation ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  // Evaluate quality of content
  const result = await qubit.evaluateQuality(
    "これはAIが生成したテキストです。明確で有用な情報が含まれています。",
    {
      requirements: ["明確性", "正確性", "実用性"],
    }
  );

  console.log("Quality Assessment:");
  console.log("Decision:", result.decision);
  console.log("Score:", result.score);
  console.log("Confidence:", result.confidence);
}

async function examplePrioritization() {
  console.log("\n=== Task Prioritization with Quantum Reasoning ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  // Prioritize tasks
  const tasks = [
    { name: "Task 1", description: "セキュリティパッチの適用" },
    { name: "Task 2", description: "ドキュメント更新" },
    { name: "Task 3", description: "バグ修正" },
    { name: "Task 4", description: "新機能開発" },
  ];

  const prioritized = await qubit.prioritize(
    tasks,
    "緊急度と影響度を考慮した優先順位付け"
  );

  console.log("Prioritized Tasks:");
  prioritized.forEach(([task, score], index) => {
    console.log(`${index + 1}. ${task.name} (Priority: ${(score * 100).toFixed(1)}%)`);
  });
}

async function exampleHybridMode() {
  console.log("\n=== Hybrid Mode: NeuroQuantum + Heuristic Fallback ===\n");

  // Hybrid mode combines NeuroQuantum with fallback to keyword heuristics
  // if the API is unavailable
  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
      timeout: 5000, // Shorter timeout to trigger fallback faster if needed
    },
    fallbackToHeuristics: true, // Enable hybrid mode
  });

  try {
    const result = await qubit.judge("test action", "test context", "safety");

    console.log("Result:", result);
    // The result will include which system was used (neuroquantum or heuristic)
  } catch (error) {
    console.error("Error:", error);
  }
}

async function exampleWaitForAPI() {
  console.log("\n=== Wait for API to Become Available ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  try {
    console.log("Waiting for NeuroQuantum API...");
    // Wait up to 10 seconds for the API to become available
    const nqEngine = (qubit as any).engine;
    if (nqEngine.waitForAvailable) {
      await nqEngine.waitForAvailable(10000);
      console.log("✓ API is now available!");
    }
  } catch (error) {
    console.error("API did not become available:", error);
  }
}

async function exampleBatchJudgment() {
  console.log("\n=== Batch Judgment ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
  });

  // Get the underlying API client for batch operations
  // (Note: QubitAI doesn't expose batchJudge directly, but you can use the client)
  const { NeuroQuantumAPIClient } = await import("../npm/src/index.js");

  const client = new NeuroQuantumAPIClient({
    baseUrl: "http://localhost:5000",
  });

  const requests = [
    {
      action: "データ削除",
      context: "本番環境",
      judgment_type: "safety",
      strict_mode: false,
    },
    {
      action: "ログ出力",
      context: "デバッグモード",
      judgment_type: "safety",
      strict_mode: false,
    },
  ];

  const results = await client.batchJudge(requests);

  console.log(`Processed ${results.count} judgments:`);
  results.results.forEach((result, index) => {
    console.log(`${index + 1}. Decision: ${result.decision}, Score: ${result.score}`);
  });
}

async function exampleStrictMode() {
  console.log("\n=== Strict Mode Judgment ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
    strictMode: true, // Requires score >= 70 for "Yes" decision
  });

  const result = await qubit.judge(
    "ユーザーデータを外部APIに送信",
    "セキュリティ監査",
    "safety"
  );

  console.log("Strict Mode Result:");
  console.log("Decision:", result.decision, "(requires score >= 70)");
  console.log("Score:", result.score);
  console.log("Meets strict threshold?", result.score >= 70);
}

async function exampleJudgmentHistory() {
  console.log("\n=== Judgment History Tracking ===\n");

  const qubit = new QubitAI({
    neuroquantumEnabled: true,
    neuroquantumConfig: {
      baseUrl: "http://localhost:5000",
    },
    maxJudgmentHistory: 50,
  });

  // Make multiple judgments
  await qubit.judge("action1", "context1", "safety");
  await qubit.judge("action2", "context2", "ethics");
  await qubit.safetyCheck("action3", "context3");

  // Retrieve history
  const history = qubit.getHistory(10);

  console.log("Recent Judgments:");
  history.forEach((record, index) => {
    console.log(
      `${index + 1}. ${record.judgmentType}: ${record.decision} (${record.score})`
    );
  });

  // Get system status
  const status = qubit.getStatus();
  console.log(
    `\nTotal judgments in history: ${status.judgmentHistorySize}/${status.maxHistory}`
  );
}

// Main execution
async function main() {
  try {
    // Check if API is available first
    const client = new (await import("../npm/src/index.js")).NeuroQuantumAPIClient({
      baseUrl: "http://localhost:5000",
    });

    const isAvailable = await client.isAvailable();

    if (!isAvailable) {
      console.error("❌ NeuroQuantum API is not available at http://localhost:5000");
      console.error("Please start the server with:");
      console.error("  python neuroquantum_api_server.py --host 127.0.0.1 --port 5000");
      process.exit(1);
    }

    console.log("✓ NeuroQuantum API is available");

    // Run examples
    await exampleBasicJudgment();
    await exampleSafetyCheck();
    await exampleEthicsEvaluation();
    await exampleQualityEvaluation();
    await examplePrioritization();
    await exampleStrictMode();
    await exampleJudgmentHistory();

    console.log("\n=== All Examples Completed ===\n");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

// Run main if this is the entry point
main().catch(console.error);

// Export examples for use in other modules
export {
  exampleBasicJudgment,
  exampleSafetyCheck,
  exampleEthicsEvaluation,
  exampleQualityEvaluation,
  examplePrioritization,
  exampleHybridMode,
  exampleWaitForAPI,
  exampleBatchJudgment,
  exampleStrictMode,
  exampleJudgmentHistory,
};
