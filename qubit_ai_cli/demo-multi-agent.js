#!/usr/bin/env node

/**
 * Multi-Agent Chat System Demo
 * Claude, ChatGPT, Gemini, Perplexity collaborative analysis
 */

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

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function section(title) {
  log("\n" + "═".repeat(75), colors.bright);
  log(`║ ${title.padEnd(73)} ║`, colors.bright);
  log("═".repeat(75) + "\n", colors.bright);
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ============================================================================
// Overview
// ============================================================================

function overview() {
  section("Multi-Agent Chat System Overview");

  log("🤖 Four Specialized AI Agents Working Together\n", colors.bright);

  const agents = [
    {
      name: "Claude",
      icon: "🧠",
      role: "Analyzer",
      specialty: "Deep Analysis & Logical Reasoning",
      temp: "0.3°C (Analytical)",
      strength: "Complex problem decomposition",
    },
    {
      name: "ChatGPT",
      icon: "✍️",
      role: "Writer",
      specialty: "Natural Communication & Explanation",
      temp: "0.6°C (Balanced)",
      strength: "Clear, engaging explanations",
    },
    {
      name: "Gemini",
      icon: "🔄",
      role: "Synthesizer",
      specialty: "Multi-Perspective Integration",
      temp: "0.7°C (Creative)",
      strength: "Holistic understanding",
    },
    {
      name: "Perplexity",
      icon: "🔍",
      role: "Researcher",
      specialty: "Research & Verification",
      temp: "0.4°C (Analytical)",
      strength: "Evidence-based insights",
    },
  ];

  agents.forEach((agent, i) => {
    log(`\n${i + 1}. ${agent.icon} ${colors.magenta}${agent.name}${colors.reset} (${agent.role})`);
    log(`   Specialty: ${agent.specialty}`);
    log(`   Temperature: ${agent.temp}`);
    log(`   Strength: ${agent.strength}`);
  });

  log("\n" + "─".repeat(75), colors.dim);
  log("\n📊 How It Works:\n", colors.bright);
  log("1. User asks a complex question");
  log("2. Question is distributed to all 4 agents in parallel");
  log("3. Each agent analyzes from their perspective");
  log("4. Responses are synthesized into a comprehensive answer");
  log("5. Final output combines all insights\n");
}

// ============================================================================
// Use Case 1: Technology Analysis
// ============================================================================

async function useCase1_TechAnalysis() {
  section("Use Case 1: Technology Question");

  const query =
    "クラウドコンピューティングがビジネスに与える影響は？";

  log("📝 User Question:\n", colors.cyan);
  log(`"${query}"\n`);
  log('(Impact of cloud computing on business)\n');

  log("═".repeat(75), colors.dim);
  log("\n🧠 CLAUDE (Analyzer) - Deep Logical Analysis\n", colors.magenta);
  log("Temperature: 0.3°C (Analytical)");
  log("─".repeat(75));

  await sleep(600);

  log("\nAnalytical Framework:");
  log("  1. Cost Structure Changes");
  log("     • CapEx → OpEx model transformation");
  log("     • Reduced infrastructure investment");
  log("     • Scalable cost per usage");
  log("");
  log("  2. Operational Impact");
  log("     • Reduced IT maintenance burden");
  log("     • Enhanced system reliability");
  log("     • Geographic distribution capabilities");
  log("");
  log("  3. Strategic Implications");
  log("     • Accelerated time-to-market");
  log("     • Competitive differentiation");
  log("     • Innovation acceleration");

  log("\n" + "═".repeat(75), colors.dim);
  log("\n✍️  CHATGPT (Writer) - Clear Communication\n", colors.yellow);
  log("Temperature: 0.6°C (Balanced)");
  log("─".repeat(75));

  await sleep(700);

  log("\nSimple Explanation:");
  log(
    "クラウドコンピューティングは、ビジネスの運営方法を根本的に変えています。"
  );
  log(
    "従来、企業は高額なサーバーを購入して維持する必要がありました。"
  );
  log("今では、インターネット経由でコンピュータ能力を借りることができます。");
  log("");
  log("主な利点：");
  log("• 初期投資が減少");
  log("• 柔軟な拡張が可能");
  log("• セキュリティの専門家が対応");
  log("• どこからでもアクセス可能");

  log("\n" + "═".repeat(75), colors.dim);
  log("\n🔄 GEMINI (Synthesizer) - Multi-Perspective View\n", colors.cyan);
  log("Temperature: 0.7°C (Creative)");
  log("─".repeat(75));

  await sleep(750);

  log("\nIntegrated Perspectives:");
  log("  Technical Dimension:");
  log("    • Infrastructure abstraction enables agility");
  log("  Business Dimension:");
  log("    • Cost optimization and capital reallocation");
  log("  Human Dimension:");
  log("    • Skill shift from infrastructure to innovation");
  log("  Ecosystem Dimension:");
  log("    • New business models and services emerge");

  log("\n" + "═".repeat(75), colors.dim);
  log("\n🔍 PERPLEXITY (Researcher) - Evidence & Verification\n", colors.green);
  log("Temperature: 0.4°C (Analytical)");
  log("─".repeat(75));

  await sleep(600);

  log("\nEvidence-Based Findings:");
  log("  ✓ Verified: 77% of enterprises use cloud services (IDC 2024)");
  log("  ✓ Verified: 30-40% cost reduction typical for cloud migration");
  log("  ✓ Verified: Faster deployment cycles (weeks → days)");
  log("  ✓ Verified: Enhanced disaster recovery capabilities");
  log("  ? Needs monitoring: Data security and compliance evolution");
  log("  ? Needs monitoring: Vendor lock-in risks");

  log("\n" + "═".repeat(75) + "\n", colors.bright);
  log("✅ FINAL SYNTHESIZED RESPONSE\n", colors.green);
  log("─".repeat(75));
  log(
    "\nクラウドコンピューティングはビジネスに多層的な影響を与えています。"
  );
  log(
    "論理的には、資本支出から運用支出へのシフトが起こり、"
  );
  log("コスト構造が根本的に変わります。実務的には、");
  log(
    "デプロイメント速度が向上し、イノベーション能力が高まります。"
  );
  log(
    "多角的には、技術、ビジネス、人的、生態系の各次元で"
  );
  log(
    "変化が起こっており、これらは相互に影響し合っています。"
  );
  log(
    "証拠に基づけば、これらの効果は実際に企業で実現されており、"
  );
  log("クラウド採用企業では平均30-40%のコスト削減が確認されています。\n");
}

// ============================================================================
// Use Case 2: Strategic Business Question
// ============================================================================

async function useCase2_StrategyQuestion() {
  section("Use Case 2: Strategic Business Question");

  const query = "AIスタートアップが成功するための戦略は？";

  log("📝 User Question:\n", colors.cyan);
  log(`"${query}"\n`);
  log('(Strategy for AI startup success)\n');

  log("🧠 Claude Analysis + ✍️ ChatGPT Explanation +");
  log("🔄 Gemini Synthesis + 🔍 Perplexity Research\n");

  await sleep(1000);

  log("═".repeat(75), colors.yellow);
  log("\n📊 Multi-Agent Coordination Results:\n");
  log("─".repeat(75));

  log("\n🎯 Strategic Framework (from 4 perspectives):\n");

  log("1️⃣  ANALYTICAL FOUNDATION (Claude)");
  log("   ├─ Market positioning analysis");
  log("   ├─ Competitive advantage identification");
  log("   └─ Resource allocation strategy\n");

  log("2️⃣  COMMUNICATION EXCELLENCE (ChatGPT)");
  log("   ├─ Investor pitch clarity");
  log("   ├─ Product messaging");
  log("   └─ Team narrative\n");

  log("3️⃣  SYSTEMIC INTEGRATION (Gemini)");
  log("   ├─ Market + Product + Team alignment");
  log("   ├─ Ecosystem partnerships");
  log("   └─ Scaling pathways\n");

  log("4️⃣  EVIDENCE-BASED (Perplexity)");
  log("   ├─ Success metrics from industry data");
  log("   ├─ Risk factors documented");
  log("   └─ Best practices confirmed\n");

  log("─".repeat(75));
  log("\n✅ Integrated Recommendations:\n");

  log("Phase 1: Foundation (Months 1-6)");
  log("  • Define clear AI problem-solution fit");
  log("  • Secure initial seed funding");
  log("  • Build minimum viable team");
  log("  • Achieve first 100 customers\n");

  log("Phase 2: Validation (Months 6-18)");
  log("  • Demonstrate product-market fit");
  log("  • Series A funding round");
  log("  • Build production infrastructure");
  log("  • Achieve revenue milestone\n");

  log("Phase 3: Scaling (18+ months)");
  log("  • Expand market coverage");
  log("  • Develop ecosystem partnerships");
  log("  • Build enterprise sales team");
  log("  • Plan Series B funding\n");

  log("Success Indicators:");
  log("  ✓ Customer acquisition cost < $500");
  log("  ✓ Churn rate < 5% monthly");
  log("  ✓ Net retention ratio > 120%");
  log("  ✓ Monthly growth rate > 10%");
}

// ============================================================================
// System Architecture
// ============================================================================

function systemArchitecture() {
  section("Multi-Agent System Architecture");

  log("📐 Data Flow Diagram\n", colors.bright);

  log(`
┌──────────────────────────────────────────────────────────────────┐
│                       User Question                              │
│                         Input                                    │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Query Distribution                            │
│           (QBNN Coordination Engine)                             │
└──┬──────────────┬──────────────┬──────────────┬────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│ Claude  │ │ChatGPT │ │  Gemini  │ │Perplexity│
│ Analyzer│ │ Writer │ │Synthesizer│ │Researcher│
│  T:0.3  │ │ T:0.6  │ │  T:0.7   │ │  T:0.4   │
└────┬────┘ └───┬────┘ └─────┬────┘ └────┬─────┘
     │          │            │           │
     └──────────┴────────────┴───────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  Response Aggregation       │
        │  (Parallel Processing)      │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   Synthesis Engine          │
        │  (Integration & Formatting) │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │   Final Response            │
        │   (Comprehensive Answer)    │
        └─────────────────────────────┘
  `);

  log("\n⚡ Performance Characteristics:\n", colors.bright);
  log("  • Parallel Agent Processing: 800-1200ms each");
  log("  • Aggregation: ~100ms");
  log("  • Synthesis: ~200ms");
  log("  • Total Time: ~2.0 seconds");
  log("  • Average Output: 400+ tokens");
  log("  • Coverage: 4 perspectives in one response\n");
}

// ============================================================================
// Commands & Features
// ============================================================================

function commands() {
  section("CLI Commands & Features");

  log("🎮 Interactive Commands:\n", colors.bright);

  const cmds = [
    { cmd: "/help", desc: "Show all available commands" },
    { cmd: "/agents", desc: "List all agents and their roles" },
    { cmd: "/details", desc: "Toggle detailed agent output display" },
    { cmd: "/history", desc: "View conversation history" },
    { cmd: "/export", desc: "Save conversation to JSON" },
    { cmd: "/clear", desc: "Clear conversation history" },
    { cmd: "/exit", desc: "Exit the chat" },
  ];

  cmds.forEach((c) => {
    log(`  ${c.cmd.padEnd(10)} - ${c.desc}`);
  });

  log("\n📊 Features:\n", colors.bright);
  log("  ✓ Parallel agent processing");
  log("  ✓ Automatic response synthesis");
  log("  ✓ Detailed agent insights (toggle with /details)");
  log("  ✓ Conversation persistence");
  log("  ✓ JSON export capability");
  log("  ✓ Role-based specialization\n");
}

// ============================================================================
// Summary
// ============================================================================

function summary() {
  section("Multi-Agent System Summary");

  log("🌟 Why This Approach Excels:\n", colors.bright);

  log("1. 📊 Comprehensive Coverage");
  log("   Each agent brings specialized expertise,");
  log("   ensuring no perspective is missed\n");

  log("2. 🚀 Parallel Efficiency");
  log("   All agents work simultaneously,");
  log("   providing results in ~2 seconds\n");

  log("3. 🎯 Integrated Intelligence");
  log("   Responses combine logical analysis,");
  log("   clear communication, and evidence\n");

  log("4. 🔄 Continuous Improvement");
  log("   Each agent learns from the synthesis,");
  log("   improving future collaborative responses\n");

  log("5. 👥 Role Specialization");
  log("   Claude → Analysis");
  log("   ChatGPT → Communication");
  log("   Gemini → Integration");
  log("   Perplexity → Verification\n");

  log("═".repeat(75), colors.bright);
  log("\n🚀 Quick Start:\n", colors.bright);
  log("  npm run multi-agent\n");

  log("📚 Inside a session:\n", colors.bright);
  log("  • Type your complex question");
  log("  • Use /agents to see all roles");
  log("  • Use /details to see individual agent responses");
  log("  • Use /export to save conversation\n");
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.clear();

  log("╔═══════════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                                   ║", colors.bright);
  log("║    🤖 Multi-Agent Collaborative Chat System Demo 🤖             ║", colors.bright);
  log("║                                                                   ║", colors.bright);
  log("║  Claude | ChatGPT | Gemini | Perplexity Working Together        ║", colors.bright);
  log("║                                                                   ║", colors.bright);
  log("╚═══════════════════════════════════════════════════════════════════╝", colors.bright);

  try {
    overview();
    await sleep(1000);

    useCase1_TechAnalysis();
    await sleep(1000);

    useCase2_StrategyQuestion();
    await sleep(1000);

    systemArchitecture();
    await sleep(500);

    commands();
    await sleep(500);

    summary();

    log(colors.green + "✅ Demo Complete!\n" + colors.reset);
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
