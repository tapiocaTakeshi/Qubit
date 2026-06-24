#!/usr/bin/env node

/**
 * Qubit AI CLI - Interactive Chat Simulation Demo
 *
 * This demonstrates what using the CLI would look like
 * with realistic conversation exchanges.
 */

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function simulateTyping(text, speed = 30) {
  for (const char of text) {
    process.stdout.write(char);
    await sleep(speed);
  }
  console.log();
}

async function printWelcome() {
  console.clear();
  log("\n", colors.reset);
  log("╔════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║         🤖 Qubit AI Interactive Chat CLI 🤖              ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║  Powered by quantum-inspired neural networks              ║", colors.bright);
  log("║  Real-time conversation with NeuroQuantum engine           ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("╚════════════════════════════════════════════════════════════╝", colors.bright);
  log("\n", colors.reset);

  log("ℹ️  Type 'help' for commands | 'exit' to quit\n", colors.cyan);
  await sleep(1000);
}

async function chatExchange(userMsg, assistantMsg, duration = 1200) {
  // User message
  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping(userMsg, 20);

  await sleep(500);

  // Thinking indicator
  log("🤔 Thinking...", colors.cyan);
  await sleep(duration);

  // Assistant response
  log(`${colors.blue}🤖 Assistant: ${colors.reset}`, colors.reset);
  await simulateTyping(assistantMsg, 15);

  log(`${colors.cyan}ℹ️  Generated in ${duration}ms${colors.reset}\n`);
  await sleep(800);
}

async function demo1_BasicConversation() {
  log("┌─ Demo 1: Basic Conversation\n", colors.bright);

  await chatExchange(
    "こんにちは。今日はどんな日ですか？",
    "こんにちは！今日は素晴らしい一日のようですね。何かお手伝いできることはありますか？",
    950
  );

  await chatExchange(
    "人工知能について教えてください",
    "人工知能（AI）は、コンピュータシステムが人間のような知能を持つように設計されたものです。機械学習、自然言語処理、コンピュータビジョンなどの技術を含みます。",
    1340
  );

  await chatExchange(
    "その応用例を教えてください",
    "AIの応用例は多くあります：医療診断、自動運転車、推奨システム、言語翻訳、画像認識などが挙げられます。ビジネスの効率化にも大きく貢献しています。",
    1280
  );

  log("", colors.reset);
}

async function demo2_Commands() {
  log("┌─ Demo 2: Using Commands\n", colors.bright);

  // Config command
  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping("/config", 40);
  await sleep(300);

  log("\n" + colors.bright + "⚙️  Current Configuration:\n", colors.reset);
  log("  Temperature: 0.7");
  log("  Max Tokens: 150");
  log("  Top K: 40");
  log("  Top P: 0.9");
  log("  Repetition Penalty: 1.2");
  log("", colors.reset);
  await sleep(800);

  // Change temperature
  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping("/temp 1.5", 40);
  await sleep(300);
  log(`${colors.green}✅ Temperature set to 1.5${colors.reset}\n`);
  await sleep(500);

  // Now chat with higher creativity
  await chatExchange(
    "創作的なストーリーの始まりを書いてください",
    "月光の下、古い館の塔から青い光が放たれていた。それは何世紀も前に失われた魔法の痕跡だった。謎めいた声が風に乗って聞こえた...",
    1500
  );

  log("", colors.reset);
}

async function demo3_Learning() {
  log("┌─ Demo 3: Learning & Research\n", colors.bright);

  await chatExchange(
    "機械学習の主要なアルゴリズムを説明してください",
    "機械学習の主要なアルゴリズムには以下のものがあります：\n1. 決定木 - 簡単で解釈可能\n2. ランダムフォレスト - 複数の決定木の集合\n3. サポートベクターマシン - 分類に有効\n4. ニューラルネットワーク - 複雑なパターン認識\n5. K-means - クラスタリング用",
    1580
  );

  await chatExchange(
    "それぞれの長所と短所は？",
    "長所：決定木は高速、ランダムフォレストは精度が高い、SVMは高次元でも効果的、ニューラルネットは複雑なパターンに対応、K-meansは単純で高速です。短所：決定木は過学習、ランダムフォレストは複雑、SVMはハイパーパラメータ調整が難しい、ニューラルネットは解釈困難です。",
    1650
  );

  log("", colors.reset);
}

async function demo4_Export() {
  log("┌─ Demo 4: Export Conversation\n", colors.bright);

  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping("/export", 40);
  await sleep(300);

  log(`${colors.green}✅ Conversation exported to qubit-chat-2024-06-24.json${colors.reset}\n`);
  await sleep(500);

  log(`${colors.yellow}📋 Exported file contains:${colors.reset}`);
  log("  • Session metadata (ID, timestamps, message count)");
  log("  • Configuration used");
  log("  • All messages with timestamps");
  log("  • Role information (user/assistant)\n");
  await sleep(800);

  log("", colors.reset);
}

async function demo5_Help() {
  log("┌─ Demo 5: Help Command\n", colors.bright);

  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping("/help", 40);
  await sleep(300);

  log("\n" + colors.bright + "📖 Available Commands:\n", colors.reset);
  log("  help          - Show this help message");
  log("  clear         - Clear conversation history");
  log("  history       - Show conversation history");
  log("  export        - Export conversation to JSON");
  log("  config        - Show current configuration");
  log("  temp <value>  - Set temperature (0.0-2.0)");
  log("  tokens <num>  - Set max tokens to generate");
  log("  exit / quit   - Exit the chat\n");

  log(colors.bright + "💡 Tips:\n", colors.reset);
  log("  • Start questions with 'Q:' for better responses");
  log("  • Use complete sentences for better context");
  log("  • Lower temperature (0.3) = more factual");
  log("  • Higher temperature (1.2) = more creative\n");

  await sleep(1000);
  log("", colors.reset);
}

async function demo6_Exit() {
  log("┌─ Demo 6: Exiting Chat\n", colors.bright);

  log(`${colors.magenta}You: ${colors.reset}`, colors.reset);
  await simulateTyping("/exit", 40);
  await sleep(300);

  log("\n", colors.reset);
}

async function printSummary() {
  log("\n╔════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                    Chat Session Ended                     ║", colors.bright);
  log("╚════════════════════════════════════════════════════════════╝\n", colors.bright);

  log("📊 Session Summary:", colors.bright);
  log(`  Total messages: 9`);
  log(`  User queries: 4`);
  log(`  Commands used: 3`);
  log(`  Session duration: ~45 seconds\n`);

  log("💾 Files saved:", colors.bright);
  log(`  • .qubit-history/history-1719230000123.json`);
  log(`  • qubit-chat-2024-06-24.json (manual export)\n`);

  log("✨ Features demonstrated:", colors.bright);
  log(`  ✓ Natural conversation`);
  log(`  ✓ Parameter adjustment (/temp)`);
  log(`  ✓ Configuration viewing (/config)`);
  log(`  ✓ Conversation export (/export)`);
  log(`  ✓ Help system (/help)`);
  log(`  ✓ Graceful exit (/exit)\n`);

  log("━".repeat(60) + "\n", colors.reset);

  log("🚀 Next Steps:\n", colors.bright);
  log("  1. Get HuggingFace token: https://huggingface.co/settings/tokens");
  log("  2. Set environment variable: export HF_TOKEN='hf_...'");
  log("  3. Run interactive mode: npm start");
  log("  4. Or single query: npm start 'Your question here'\n");

  log("📖 For more information, see:", colors.bright);
  log("  • README.md - Complete documentation");
  log("  • USAGE.md - Detailed usage examples\n");

  log("═".repeat(60) + "\n", colors.reset);
}

async function main() {
  try {
    await printWelcome();

    await demo1_BasicConversation();
    log("━".repeat(60) + "\n", colors.reset);
    await sleep(500);

    await demo2_Commands();
    log("━".repeat(60) + "\n", colors.reset);
    await sleep(500);

    await demo3_Learning();
    log("━".repeat(60) + "\n", colors.reset);
    await sleep(500);

    await demo4_Export();
    log("━".repeat(60) + "\n", colors.reset);
    await sleep(500);

    await demo5_Help();
    log("━".repeat(60) + "\n", colors.reset);
    await sleep(500);

    await demo6_Exit();

    await printSummary();

    log(colors.green + "✅ Demo completed successfully!\n" + colors.reset);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
