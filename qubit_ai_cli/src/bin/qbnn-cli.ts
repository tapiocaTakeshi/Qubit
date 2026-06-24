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
 * Generate comprehensive QBNN simulation response
 */
function generateQBNNSimulation(query: string): string {
  const queryLower = query.toLowerCase();

  // AI/MLに関する質問
  if (
    queryLower.includes("ai") ||
    queryLower.includes("機械学習") ||
    queryLower.includes("ニューラル") ||
    queryLower.includes("ディープ")
  ) {
    return `【AIについてのQBNN分析】

【1】論理的分解
AIシステムは以下のコアコンポーネントで構成されます：

  層1: データ入力層
    └─ テキスト、画像、音声などの多次元データ処理
       └─ トークン化、正規化、特徴抽出

  層2: ニューラルネットワーク層
    └─ 深層学習モデル（Transformer、RNN、CNN）
       └─ 重み付けパラメータ（数百万～数十億個）
       └─ 活性化関数による非線形変換

  層3: 推論エンジン
    └─ 確率的テキスト生成（Sampling、Top-K、Top-P）
       └─ コンテキストウィンドウ管理
       └─ ビーム探索による最適シーケンス選択

【2】双方向分析

フォワードパス（入力→出力）：
  入力テキスト
    ↓ [エンべディング層]
  分散表現 (数千次元ベクトル)
    ↓ [注意機構: Attention]
  コンテキスト統合 (各トークン間の関連性計算)
    ↓ [フィードフォワードネット]
  特徴変換 (層ごとの抽象化レベル上昇)
    ↓ [出力層]
  確率分布 → トークン選択 → テキスト生成

バックワードパス（誤差逆伝播）：
  損失関数（交差エントロピー）から開始
    ↓ [勾配計算]
  各パラメータの重要度を定量化
    ↓ [確率的勾配降下法 SGD]
  パラメータ更新（学習率 α で調整）
    ↓ [複数エポック]
  モデルの精度向上、汎化性能改善

相互作用: 前向きと後ろ向きが交互に最適化を行い、モデル全体が段階的に改善される

【3】量子的重ね合わせ的視点
AIモデルは複数の解釈を同時に保持します：

  入力「昨日は雨だった」に対して：
    ｜過去のイベント⟩ + ｜天気現象⟩ + ｜情緒的文脈⟩の重ね合わせ

  生成時に「測定」（トークン選択）を行うことで、
  最も確率の高い解釈が確定される

【4】統合的結論

AIの本質：
  • パターン認識と確率的予測の融合
  • 大規模データからの自動特徴学習
  • スケーラブルで汎用的なコンピューティング

実装のポイント：
  ✓ Transformer（注意機構）による並列処理効率化
  ✓ トークン化による言語の数値化
  ✓ 温度パラメータによる出力の創造性/確定性制御
  ✓ コンテキストウィンドウの効率的管理

【信頼度】高（多くの学術論文と実装実績に基づく）
【次のステップ】具体的なアーキテクチャ（BERT、GPT、Llama）について詳しく知りたいですか？`;
  }

  // 画像処理に関する質問
  if (queryLower.includes("画像") || queryLower.includes("vision")) {
    return `【画像処理・ビジョンモデルのQBNN分析】

【1】問題分解 - 画像処理のコアタスク

分類タスク: 画像 → カテゴリ（犬、猫、建物など）
  └─ CNN (Convolutional Neural Network)で実装
  └─ 畳み込み層で局所パターン抽出

検出タスク: 画像 → 物体位置 + 境界ボックス
  └─ YOLO, Faster R-CNNなど
  └─ 複数の物体を同時検出

セグメンテーション: 画像 → ピクセルレベルの分類
  └─ U-Net, DeepLabなど
  └─ 詳細な領域分析

【2】双方向分析

フォワード処理:
  元画像 (RGB 3チャネル)
    ↓ [特徴抽出]
  低レベル特徴: エッジ、コーナー、テクスチャ
    ↓ [階層的抽出]
  中レベル特徴: 形状、パターン
    ↓ [高レベル特徴]
  高レベル特徴: 物体、シーン理解
    ↓ [決定層]
  クラス確率分布

バックワード処理:
  予測誤差 → 各層へ勾配逆伝播
    ↓ [ビジュアル解釈可能性分析]
  GradCAM: どのピクセルが決定に影響したか可視化

【3】実装パターン

転移学習の活用:
  ImageNetで事前学習済みモデル（ResNet, EfficientNet）
    ↓ [ファインチューニング]
  特定タスク用に少量データで調整

パイプライン例:
  入力画像 → 前処理（リサイズ、正規化）
          → モデル推論（0.1～1秒）
          → 後処理（NMS, スムージング）
          → 出力（クラス + 信頼度）

【4】統合的推奨

画像AIを使う際のポイント：
  ✓ 学習データの品質が最重要
  ✓ データ拡張（回転、反転、色変換）で汎化性向上
  ✓ 計算効率とモデル精度のトレードオフを考慮
  ✓ エッジデバイス対応には軽量モデル（MobileNet）を選択

【信頼度】高
【関連トピック】物体検出、顔認識、医療画像解析など`;
  }

  // Flutter/モバイル開発に関する質問
  if (queryLower.includes("flutter") || queryLower.includes("モバイル")) {
    return `【Flutter開発のQBNN分析】

【1】Flutterの本質的特徴

クロスプラットフォーム フレームワーク
  └─ 単一Dartコードベース
  └─ iOS, Android, Web, Desktop に配信
  └─ ネイティブパフォーマンス

設計思想:
  Everything is a Widget（ウィジェット中心設計）
    ↓ [階層的UI構成]
  ツリー構造で任意の複雑なUIを表現可能

【2】双方向アーキテクチャ

ホットリロード（開発体験向上）:
  コード変更 → 2秒以内にUI反映
    ↓ [ステートの保持]
  アプリを再起動しせず変更が適用

ビルドプロセス（本番環境）:
  Dart コード
    ↓ [AOT コンパイル / JIT]
  ネイティブコード (ARM64, x86)
    ↓ [最適化]
  実行可能バイナリ (APK/IPA)

【3】実装例: ゲーム開発

Flutterゲームの構成:
  GameState クラス
    ├─ ゲームロジック（スコア、衝突判定）
    └─ アクター管理（プレイヤー、敵）

GameScreen ウィジェット
    ├─ AnimationController (60FPS駆動)
    ├─ CustomPaint で描画
    └─ GestureDetector でタップ入力処理

衝突検出アルゴリズム:
  // 2つの矩形の衝突判定
  if (rect1.overlaps(rect2)) {
    // 衝突処理
    score += 10;
    particle.emit(); // パーティクル生成
  }

【4】パフォーマンス最適化

フレームレート管理:
  ✓ 60FPS 維持 (16.67ms/フレーム)
  ✓ UI 更新の最小化 (setState は必要なウィジェットのみ)
  ✓ 画像アセットの圧縮 (WebP形式推奨)

メモリ効率:
  ✓ Widget キャッシング (const constructor)
  ✓ 不要なリスナー削除
  ✓ 大規模リスト: ListView.builder で遅延生成

【信頼度】高（多くのプロダクションアプリで実証）
【次のステップ】Firebase 統合、アナリティクス、マネタイズなど`;
  }

  // デフォルト: 汎用的な詳しい応答
  return `【QBNNによる構造化分析】

質問: "${query.substring(0, 80)}${query.length > 80 ? "..." : ""}"

【1】問題分解と要素抽出

主要要素:
  • 中心概念: 質問の核となる概念
  • 周辺要素: サポーティング要素
  • 制約条件: 解決を制限する条件
  • 目標状態: 達成したい最終状態

依存関係マッピング:
  要素A ← [因果関係] → 要素B
  └─ 直接的な影響
  └─ 間接的な影響（2段階以上）

【2】双方向分析フレームワーク

→ フォワード視点（原因 → 結果）:
  初期状態 = {設定, パラメータ, 前提}
    ↓ [プロセス実行]
  結果 = {出力, 副作用, 影響}
  評価: 期待値との乖離度

← バックワード視点（結果 ← 原因）:
  望ましい結果から逆算
    ↓ [必要条件の特定]
  実現に必要な条件・リソース・時間
  現状とのギャップ分析

相互作用:
  • フォワード分析で予測される結果
  • バックワード分析で必要な条件
  → 両者を統合し最適なアプローチを導出

【3】量子的重ね合わせ思考

複数の有効な解釈を同時に保持:
  解釈A: {視点1の根拠}
  解釈B: {視点2の根拠}
  解釈C: {視点3の根拠}

「測定」（決定時点）までは複数の可能性を並存させ、
コンテキストに応じて最適な解釈を選別する

【4】統合的結論

推奨アプローチ:
  1. 短期: すぐに実施可能で効果が高い施策
  2. 中期: より大規模だが高いROIが期待される施策
  3. 長期: 基盤整備、システム構築

期待効果:
  • 定量的効果 (数値化可能)
  • 定性的効果 (体験・感覚的向上)
  • リスク低減効果

実装ロードマップ:
  ✓ フェーズ1: 前提条件確認 (1-2週)
  ✓ フェーズ2: パイロット実施 (2-4週)
  ✓ フェーズ3: スケール展開 (1-3ヶ月)
  ✓ フェーズ4: 継続改善 (運用段階)

【信頼度】高（QBNNの構造化分析フレームワークに基づく）
【推奨】より詳細な分析が必要な場合は、特定の側面についてさらに質問してください`;
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
