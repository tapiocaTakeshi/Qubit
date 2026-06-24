#!/usr/bin/env node

/**
 * QBNN-Only Chat CLI
 * Quantum-inspired Bidirectional Neural Network Interactive Chat
 */

import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import { NeuroQuantumClient } from "qubit_ai";

interface QBNNCLIState {
  rl: readline.Interface;
  messageCount: number;
  conversationHistory: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
  historyFile: string;
}

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  red: "\x1b[31m",
};

function log(message: string, color: string = colors.reset): void {
  console.log(`${color}${message}${colors.reset}`);
}

function logInfo(message: string): void {
  log(`ℹ️  ${message}`, colors.cyan);
}

function logSuccess(message: string): void {
  log(`✅ ${message}`, colors.green);
}

function logError(message: string): void {
  log(`❌ ${message}`, colors.red);
}

function logQBNN(message: string): void {
  log(`🧠 ${message}`, colors.blue);
}

function logUser(message: string): void {
  log(`👤 You: ${message}`, colors.magenta);
}

/**
 * Print welcome message
 */
function printWelcome(): void {
  console.clear();
  log("", colors.reset);
  log(
    "╔═══════════════════════════════════════════════════════════════╗",
    colors.bright
  );
  log(
    "║                                                               ║",
    colors.bright
  );
  log(
    "║   🧠 QBNN-Only Interactive Chat System 🧠                   ║",
    colors.bright
  );
  log(
    "║                                                               ║",
    colors.bright
  );
  log(
    "║  Quantum-inspired Bidirectional Neural Network Analysis      ║",
    colors.bright
  );
  log(
    "║                                                               ║",
    colors.bright
  );
  log(
    "╚═══════════════════════════════════════════════════════════════╝",
    colors.bright
  );
  log("", colors.reset);

  logInfo("Type 'help' for commands | 'exit' to quit\n");
}

/**
 * Print help message
 */
function printHelp(): void {
  log("\n📖 Available Commands:\n", colors.bright);
  log("  help              - Show this help message");
  log("  analyze <path>    - Analyze an image file with QBNN");
  log("  history           - View conversation history");
  log("  export            - Export conversation to JSON");
  log("  clear             - Clear conversation history");
  log("  exit / quit       - Exit the chat\n");

  log("💡 Tips:\n", colors.bright);
  log("  • Ask QBNN questions for deep logical analysis");
  log("  • QBNN excels at structured reasoning and synthesis");
  log("  • Use /analyze <image_path> for image analysis");
  log("  • Conversation history is automatically saved\n");
}

/**
 * Load and convert image to base64
 */
function loadImageAsBase64(imagePath: string): string {
  try {
    const absolutePath = path.resolve(imagePath);
    if (!fs.existsSync(absolutePath)) {
      throw new Error(`Image file not found: ${absolutePath}`);
    }

    const imageBuffer = fs.readFileSync(absolutePath);
    return imageBuffer.toString("base64");
  } catch (error) {
    throw new Error(
      `Failed to load image: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Process query with QBNN
 */
async function processWithQBNN(
  query: string,
  state: QBNNCLIState
): Promise<string> {
  try {
    const hfToken =
      process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || "";

    const client = new NeuroQuantumClient({
      hfToken,
      timeoutMs: 120000,
      maxRetries: 3,
    });

    // Use QBNN model with structured reasoning prompt
    const systemPrompt = `You are QBNN (Quantum-inspired Bidirectional Neural Network),
a sophisticated analytical system that specializes in:

1. **Logical Decomposition**: Breaking down complex problems into components
2. **Bidirectional Analysis**: Examining issues from multiple perspectives
3. **Structured Synthesis**: Creating coherent integrated conclusions
4. **Quantum Superposition Thinking**: Holding multiple valid interpretations simultaneously

Your analysis style:
- Begin with clear problem decomposition
- Explore multiple perspectives with equal validity
- Identify key relationships and dependencies
- Synthesize insights into actionable conclusions
- Present findings with confidence levels and alternative viewpoints

Temperature: 0.4 (analytical, structured, deterministic)`;

    // Build conversation context
    const conversationContext = state.conversationHistory
      .map((msg) => `${msg.role === "user" ? "User" : "QBNN"}: ${msg.content}`)
      .join("\n");

    const fullPrompt =
      conversationContext.length > 0
        ? `${conversationContext}\nUser: ${query}`
        : `User: ${query}`;

    logQBNN("Analyzing with quantum-inspired reasoning...\n");

    const startTime = Date.now();

    // Call QBNN model for analysis
    const response = await client.generateWithExamples(fullPrompt, [], {
      maxNewTokens: 1000,
      temperature: 0.4,
      topK: 40,
      topP: 0.9,
      repetitionPenalty: 1.2,
    });

    const duration = Date.now() - startTime;

    logInfo(`Processing time: ${duration}ms\n`);

    return response.generatedText;
  } catch (error) {
    const errorMsg =
      error instanceof Error ? error.message : String(error);
    const stack =
      error instanceof Error ? error.stack : "";

    // Log detailed error for debugging
    logError(`API Error: ${errorMsg}`);
    if (stack) {
      logError(`Stack: ${stack.split("\n").slice(0, 3).join(" → ")}`);
    }

    // Check if HF_TOKEN is set
    const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
    if (!token) {
      logError("HF_TOKEN environment variable is not set");
    } else {
      logError(
        `HF_TOKEN is set (length: ${token.length})`
      );
    }

    // Check proxy settings
    const proxy = process.env.HTTPS_PROXY || process.env.HTTP_PROXY;
    if (proxy) {
      logInfo(`Proxy configured: ${proxy}`);
    }

    // Fallback to simulation if API fails
    logInfo("Using simulation mode (API unavailable)\n");
    return generateQBNNSimulation(query);
  }
}

/**
 * Generate QBNN simulation response
 */
function generateQBNNSimulation(query: string): string {
  const analysisPoints = [
    "【問題分解】\n  ✓ 主要要素の特定\n  ✓ 依存関係のマッピング\n  ✓ 制約条件の明確化",
    "【双方向分析】\n  → フォワード視点: 原因から結果へ\n  → バックワード視点: 結果から原因へ\n  → 相互作用の特定",
    "【量子的重ね合わせ】\n  • 複数の有効な解釈を同時に保持\n  • 各視点の妥当性を評価\n  • コンテキストに応じた選別",
    "【統合的結論】\n  ◆ 各要素の相互関係を考慮\n  ◆ システム全体への影響を評価\n  ◆ 実装可能な推奨事項を提示",
  ];

  const randomPoints = analysisPoints.sort(
    () => Math.random() - 0.5
  ).slice(0, 3);

  return (
    `🧠 QBNN Analysis of: "${query.substring(0, 60)}..."\n\n` +
    randomPoints.join("\n\n") +
    "\n\n【信頼度】高 (構造化分析に基づく推論)\n【推奨アクション】詳細な掘り下げのため、特定の側面についてさらに質問してください"
  );
}

/**
 * Analyze image with QBNN
 */
async function analyzeImage(
  imagePath: string,
  state: QBNNCLIState
): Promise<void> {
  try {
    logInfo(`Loading image: ${imagePath}`);
    const base64Image = loadImageAsBase64(imagePath);
    const fileName = path.basename(imagePath);

    logUser(`Analyze image: ${fileName}`);
    logQBNN("Initiating quantum-inspired image analysis...\n");

    const analysisPrompt = `【画像分析リクエスト】

ファイル: ${fileName}
形式: ${path.extname(imagePath).toUpperCase()}

以下の観点から多次元分析を実施してください：

1. **ビジュアル構造分析**
   - 主要要素の識別と配置
   - 色彩・フォント・レイアウトの特徴
   - デザイン原則の適用

2. **目的・文脈理解**
   - このデザインの意図
   - ターゲットユーザー
   - 期待される機能

3. **UX/UI評価**
   - ユーザビリティの観点
   - アクセシビリティ
   - 心理学的効果

4. **改善提案**
   - 具体的な改善ポイント
   - 実装可能性の評価
   - 期待される効果

5. **技術的実装**
   - 使用すべき技術スタック
   - コード例（CSS/JS/TS）
   - パフォーマンス考慮

画像データ (Base64): ${base64Image.substring(0, 100)}...[画像バイナリデータ]`;

    const startTime = Date.now();
    const analysis = await processWithQBNN(analysisPrompt, state);
    const duration = Date.now() - startTime;

    log("\n" + analysis + "\n");

    logInfo(`Image analysis completed in ${duration}ms\n`);

    // Save to history
    state.conversationHistory.push({
      role: "user",
      content: `Image analysis: ${fileName}`,
    });
    state.conversationHistory.push({
      role: "assistant",
      content: analysis,
    });

    state.messageCount++;
    saveHistory(state);
  } catch (error) {
    const errorMsg =
      error instanceof Error ? error.message : String(error);
    logError(`Failed to analyze image: ${errorMsg}`);
    log("", colors.reset);
  }
}

/**
 * Handle special commands
 */
async function handleCommand(
  command: string,
  state: QBNNCLIState
): Promise<boolean> {
  const args = command.trim().split(/\s+/);
  const cmd = args[0].toLowerCase();

  switch (cmd) {
    case "help":
      printHelp();
      return true;

    case "analyze": {
      const imagePath = args.slice(1).join(" ");
      if (!imagePath) {
        logError("Please provide an image path: /analyze <path>");
        log("", colors.reset);
        return true;
      }
      await analyzeImage(imagePath, state);
      return true;
    }

    case "history": {
      if (state.conversationHistory.length === 0) {
        logInfo("No messages yet\n");
        return true;
      }

      log("\n📜 Conversation History:\n", colors.bright);
      state.conversationHistory.forEach((msg, i) => {
        const role = msg.role === "user" ? "👤 You" : "🧠 QBNN";
        log(`[${i + 1}] ${role}: ${msg.content.substring(0, 80)}...`);
      });
      log("", colors.reset);
      return true;
    }

    case "export": {
      const timestamp = new Date().toISOString().split("T")[0];
      const filename = `qbnn-chat-${timestamp}.json`;
      const conversation = JSON.stringify(
        {
          timestamp: new Date().toISOString(),
          messageCount: state.messageCount,
          messages: state.conversationHistory,
        },
        null,
        2
      );

      fs.writeFileSync(filename, conversation);
      logSuccess(`Conversation exported to ${filename}\n`);
      return true;
    }

    case "clear":
      state.conversationHistory = [];
      logSuccess("Conversation history cleared\n");
      return true;

    case "exit":
    case "quit":
      return false;

    default:
      return true;
  }
}

/**
 * Process user message
 */
async function processMessage(
  userInput: string,
  state: QBNNCLIState
): Promise<boolean> {
  if (!userInput.trim()) {
    return true;
  }

  // Check for commands
  if (userInput.trim().startsWith("/")) {
    const command = userInput.trim().slice(1);
    return await handleCommand(command, state);
  }

  // Process as regular message
  logUser(userInput);

  try {
    const response = await processWithQBNN(userInput, state);
    log("\n" + response + "\n");

    // Save to history
    state.conversationHistory.push({
      role: "user",
      content: userInput,
    });
    state.conversationHistory.push({
      role: "assistant",
      content: response,
    });

    state.messageCount++;
    saveHistory(state);

    return true;
  } catch (error) {
    const errorMsg =
      error instanceof Error ? error.message : String(error);
    logError(`Failed to process query: ${errorMsg}`);
    log("", colors.reset);
    return true;
  }
}

/**
 * Save conversation history
 */
function saveHistory(state: QBNNCLIState): void {
  try {
    const conversation = JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        messageCount: state.messageCount,
        messages: state.conversationHistory,
      },
      null,
      2
    );
    fs.writeFileSync(state.historyFile, conversation);
  } catch (error) {
    // Silently fail if history save fails
  }
}

/**
 * Start interactive mode
 */
async function startInteractiveMode(state: QBNNCLIState): Promise<void> {
  printWelcome();

  const askQuestion = (): void => {
    state.rl.question(
      `${colors.magenta}You: ${colors.reset}`,
      async (input) => {
        if (!input.trim()) {
          askQuestion();
          return;
        }

        const shouldContinue = await processMessage(input, state);

        if (shouldContinue) {
          askQuestion();
        } else {
          cleanup(state);
        }
      }
    );
  };

  askQuestion();
}

/**
 * Cleanup and exit
 */
function cleanup(state: QBNNCLIState): void {
  saveHistory(state);
  state.rl.close();

  log("\n", colors.reset);
  logSuccess(
    `QBNN chat session ended. (${state.messageCount} messages)\n`
  );
  process.exit(0);
}

/**
 * Main function
 */
async function main(): Promise<void> {
  try {
    // Setup readline
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const historyDir = path.join(process.cwd(), ".qubit-qbnn-history");
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    const state: QBNNCLIState = {
      rl,
      messageCount: 0,
      conversationHistory: [],
      historyFile: path.join(
        historyDir,
        `qbnn-${Date.now()}.json`
      ),
    };

    // Start interactive mode
    await startInteractiveMode(state);
  } catch (error) {
    console.error(
      "Fatal error:",
      error instanceof Error ? error.message : error
    );
    process.exit(1);
  }
}

// Handle signals
process.on("SIGINT", () => {
  log("\n\n👋 Interrupted by user\n", colors.yellow);
  process.exit(0);
});

process.on("SIGTERM", () => {
  log("\n\n👋 Terminated\n", colors.yellow);
  process.exit(0);
});

// Run main
main().catch((error) => {
  console.error("Unhandled error:", error);
  process.exit(1);
});
