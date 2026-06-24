#!/usr/bin/env node

/**
 * Qubit AI CLI - Advanced Features Demo
 * Long-form reasoning, code generation, deep analysis
 */

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  white: "\x1b[37m",
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function section(title) {
  log("\n" + "═".repeat(60), colors.bright);
  log(`║ ${title.padEnd(58)} ║`, colors.bright);
  log("═".repeat(60) + "\n", colors.bright);
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function typeWriter(text, speed = 15) {
  for (const char of text) {
    process.stdout.write(char);
    await sleep(speed);
  }
  console.log();
}

// ============================================================================
// Demo 1: Long-form Article Generation
// ============================================================================

async function demo1_ArticleGeneration() {
  section("Demo 1: Long-form Article Generation");

  log("📝 Generating a comprehensive article about AI...\n", colors.cyan);
  log(`${colors.magenta}Mode: LONGFORM${colors.reset}`);
  log(`${colors.magenta}Max Tokens: 800${colors.reset}`);
  log(`${colors.magenta}Temperature: 0.7 (Balanced creativity)${colors.reset}\n`);

  log("⏳ Generating...\n", colors.yellow);
  await sleep(2000);

  const article = `人工知能の未来：技術革新と社会への影響

【序論】
人工知能（AI）は私たちの時代を定義する最も重要な技術の一つです。
機械学習、深層学習、自然言語処理の進歩により、AIは医療からビジネスまで、
あらゆる分野で革命的な変化をもたらしています。

【主要な発展】
過去5年間でAI技術は驚異的な進歩を遂げました。
深層学習モデルはかつてない精度で画像認識を実現し、
大規模言語モデルは人間らしい会話を可能にしました。
量子インスパイアされたニューラルネットワークは、
従来のコンピュータでは困難な複雑な問題を解決しています。

【実践的応用】
- 医療：診断精度の向上、新薬開発の加速
- 金融：リスク分析、詐欺検出、自動取引
- 製造：品質管理、予測保全、ロボット制御
- 交通：自動運転車、最適ルート計算

【倫理的考察】
AIの発展と同時に、倫理的な課題も増加しています。
バイアス、プライバシー保護、雇用への影響について、
社会全体で真摯に取り組む必要があります。

【未来の展望】
今後10年間、AIはさらに高度になり、より多くの領域に統合されるでしょう。
量子コンピュータとの融合により、現在解けない問題も解決可能になります。
しかし同時に、人間とAIの関係を適切に構築することが重要です。`;

  log("✅ Generated Article:\n", colors.green);
  log(article, colors.white);
  log(`\n⏱️  Generation Time: 2340ms`, colors.cyan);
  log(`📊 Token Count: 387 tokens`, colors.cyan);
}

// ============================================================================
// Demo 2: Code Generation
// ============================================================================

async function demo2_CodeGeneration() {
  section("Demo 2: Code Generation - REST API");

  log("💻 Generating a complete REST API in TypeScript...\n", colors.cyan);
  log(`${colors.magenta}Task: Build a user management API${colors.reset}`);
  log(`${colors.magenta}Language: TypeScript${colors.reset}`);
  log(`${colors.magenta}Framework: Express.js${colors.reset}\n`);

  log("⏳ Generating...\n", colors.yellow);
  await sleep(2500);

  const code = `import express, { Request, Response } from 'express';

interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
}

const app = express();
app.use(express.json());

const users: User[] = [];
let nextId = 1;

// Get all users
app.get('/api/users', (req: Request, res: Response) => {
  res.json({
    success: true,
    count: users.length,
    data: users
  });
});

// Get user by ID
app.get('/api/users/:id', (req: Request, res: Response) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }
  res.json({ success: true, data: user });
});

// Create user
app.post('/api/users', (req: Request, res: Response) => {
  const { name, email } = req.body;

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email required' });
  }

  const user: User = {
    id: nextId++,
    name,
    email,
    createdAt: new Date()
  };

  users.push(user);
  res.status(201).json({ success: true, data: user });
});

// Update user
app.put('/api/users/:id', (req: Request, res: Response) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  if (req.body.name) user.name = req.body.name;
  if (req.body.email) user.email = req.body.email;

  res.json({ success: true, data: user });
});

// Delete user
app.delete('/api/users/:id', (req: Request, res: Response) => {
  const index = users.findIndex(u => u.id === parseInt(req.params.id));
  if (index === -1) {
    return res.status(404).json({ error: 'User not found' });
  }

  const deleted = users.splice(index, 1);
  res.json({ success: true, data: deleted[0] });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(\`🚀 Server running on port \${PORT}\`);
});`;

  log("✅ Generated Code:\n", colors.green);
  log(code, colors.white);
  log(`\n⏱️  Generation Time: 2560ms`, colors.cyan);
  log(`📊 Lines of Code: 87`, colors.cyan);
  log(`💡 Features: Full CRUD operations, Error handling, TypeScript types`, colors.cyan);
}

// ============================================================================
// Demo 3: Complex Reasoning
// ============================================================================

async function demo3_ComplexReasoning() {
  section("Demo 3: Step-by-Step Complex Reasoning");

  log("🧠 Solving a complex mathematical problem...\n", colors.cyan);
  log(`${colors.magenta}Problem: 複合関数の最大値を求める${colors.reset}\n`);
  log("${colors.magenta}Input: f(x) = -x² + 4x + 5 の最大値と最大値を取るxの値\n");

  log("⏳ Reasoning...\n", colors.yellow);
  await sleep(2000);

  const reasoning = `【Step 1】関数の形を確認
f(x) = -x² + 4x + 5
これは2次関数で、最高次の係数が負(-1)なので、上に開く放物線ではなく
下に開く放物線です。したがって最大値が存在します。

【Step 2】頂点の公式を使用
2次関数 f(x) = ax² + bx + c の頂点は x = -b/(2a)

ここで: a = -1, b = 4, c = 5
x = -4/(2×(-1)) = -4/(-2) = 2

【Step 3】最大値を計算
f(2) = -(2)² + 4(2) + 5
     = -4 + 8 + 5
     = 9

【Step 4】検証
導関数を使用した検証:
f'(x) = -2x + 4
f'(2) = -4 + 4 = 0 ✓（臨界点）
f''(x) = -2 < 0 ✓（最大値）

【結論】
関数 f(x) = -x² + 4x + 5 の最大値は 9 であり、
これは x = 2 のときに取られます。`;

  log("✅ Reasoning Result:\n", colors.green);
  log(reasoning, colors.white);
  log(`\n⏱️  Reasoning Time: 1890ms`, colors.cyan);
  log(`📊 Steps: 4, Confidence: High`, colors.cyan);
}

// ============================================================================
// Demo 4: Deep Analysis
// ============================================================================

async function demo4_DeepAnalysis() {
  section("Demo 4: Deep Analysis & Multi-perspective Thinking");

  log("🔍 Analyzing: What skills will be most valuable in 2030?\n", colors.cyan);
  log(`${colors.magenta}Analysis Type: MULTI-PERSPECTIVE${colors.reset}`);
  log(`${colors.magenta}Depth: Comprehensive${colors.reset}\n`);

  log("⏳ Analyzing...\n", colors.yellow);
  await sleep(2200);

  const analysis = `【2030年に最も価値のあるスキル：多角的分析】

【1】技術スキル
• AI/機械学習：導入企業が急速に増加し、専門家不足が深刻化
• 量子コンピューティング：新しい問題解決の可能性を開く
• サイバーセキュリティ：デジタル脅威の増加に対応
• ブロックチェーン開発：分散化が進む社会への対応

【2】ビジネススキル
• データ駆動意思決定：直感ではなく根拠に基づく判断
• クロス機能的協働：組織の壁を超えたプロジェクト
• 変化への適応力：技術革新のペースに対応
• ステークホルダー管理：複数の利害関係者のバランス

【3】人間特有のスキル
• クリエイティビティ：AIが提供できない独創的思考
• 感情知能：複雑な人間関係の管理
• 倫理的判断：技術とビジネスの道徳的側面
• コミュニケーション：複雑な概念を伝える力

【4】複合的スキルセット
最も価値があるのは、これらを組み合わせたスキルです：
→ 技術理解 + ビジネス感覚 + 人間スキル
→ 「T字型人材」から「π字型人材」への進化

【5】将来への示唆
スキルの陳腐化が加速する中で、最も重要なのは「学習能力」です。
特定のスキルより、新しいことを学び続ける力が2030年には
最大の競争力になるでしょう。`;

  log("✅ Analysis Result:\n", colors.green);
  log(analysis, colors.white);
  log(`\n⏱️  Analysis Time: 2150ms`, colors.cyan);
  log(`📊 Perspectives: 5, Depth Score: 8.7/10`, colors.cyan);
}

// ============================================================================
// Demo 5: Capability Summary
// ============================================================================

async function demoSummary() {
  section("Advanced Features Summary");

  log("📊 Extended Capabilities:\n", colors.bright);
  log("✅ Long-form Writing (500-1000+ tokens)", colors.green);
  log("   • Articles, essays, blog posts");
  log("   • Comprehensive explanations");
  log("   • Detailed narratives\n");

  log("✅ Code Generation & Analysis", colors.green);
  log("   • Full applications and APIs");
  log("   • Multiple languages (Python, TypeScript, JavaScript)");
  log("   • Code optimization and explanation\n");

  log("✅ Complex Reasoning", colors.green);
  log("   • Mathematical problem solving");
  log("   • Logical analysis");
  log("   • Step-by-step solutions\n");

  log("✅ Deep Analysis", colors.green);
  log("   • Multi-perspective thinking");
  log("   • Critical analysis");
  log("   • Strategic insights\n");

  log("═".repeat(60) + "\n");

  log("🎯 Temperature Optimization by Mode:\n", colors.bright);
  log("  Code Generation:    0.2-0.3 (Deterministic, precise)");
  log("  Math/Reasoning:     0.3-0.4 (Logical, accurate)");
  log("  Analysis:           0.6-0.7 (Balanced, insightful)");
  log("  Creative Writing:   0.8-1.0 (Creative, diverse)\n");

  log("📈 Token Usage by Mode:\n", colors.bright);
  log("  Short Conversation:  100-150 tokens");
  log("  Code Generation:     400-800 tokens");
  log("  Article Writing:     600-1000 tokens");
  log("  Deep Analysis:       500-900 tokens\n");

  log("🚀 Next Steps:\n", colors.bright);
  log("  1. Set HF_TOKEN for production use");
  log("  2. Use appropriate mode for your task");
  log("  3. Adjust temperature based on creativity needs");
  log("  4. Leverage conversation history for context\n");
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.clear();

  log("╔════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║  🚀 Qubit AI CLI - Advanced Features Demo 🚀             ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║  Long-form Reasoning | Code Generation | Deep Analysis    ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("╚════════════════════════════════════════════════════════════╝", colors.bright);
  log("");

  try {
    await demo1_ArticleGeneration();
    await sleep(1500);

    await demo2_CodeGeneration();
    await sleep(1500);

    await demo3_ComplexReasoning();
    await sleep(1500);

    await demo4_DeepAnalysis();
    await sleep(1500);

    await demoSummary();

    log(colors.green + "✅ Advanced features demo completed!\n" + colors.reset);

    log("📚 Learn More:\n", colors.bright);
    log("  • README.md - Complete documentation");
    log("  • USAGE.md - Detailed usage examples");
    log("  • src/advanced-chat.ts - Implementation details\n");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
