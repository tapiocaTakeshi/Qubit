#!/usr/bin/env node

/**
 * Qubit AI Hybrid Chat Demo
 * Demonstrates Gemma + QBNN Frontal System
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
  log("\n" + "═".repeat(65), colors.bright);
  log(`║ ${title.padEnd(63)} ║`, colors.bright);
  log("═".repeat(65) + "\n", colors.bright);
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ============================================================================
// Demo 1: System Architecture
// ============================================================================

function demo1_Architecture() {
  section("Demo 1: Hybrid System Architecture");

  log("🧠 Gemma + QBNN Frontal System\n", colors.bright);

  log("This system combines two AI models in a sophisticated two-phase approach:\n");

  log("┌─ Phase 1: QBNN Reasoning (Frontal Analysis)\n", colors.cyan);
  log("│  • Quantum-inspired neural network analyzes the query");
  log("│  • Identifies key concepts and requirements");
  log("│  • Plans the best approach to answer");
  log("│  • Temperature: 0.4 (logical, analytical)\n");

  log("└─ Phase 2: Gemma Generation (High-Quality Output)\n", colors.cyan);
  log("   • Google's Gemma model generates the response");
  log("   • Uses QBNN's reasoning to guide generation");
  log("   • Produces coherent, well-structured text");
  log("   • Temperature: 0.6 (balanced, natural)\n");

  log("Result: Intelligent reasoning + Quality writing\n", colors.green);
}

// ============================================================================
// Demo 2: Comparison with Single Models
// ============================================================================

function demo2_Comparison() {
  section("Demo 2: Comparison with Single Models");

  const examples = [
    {
      model: "QBNN Only",
      pros: ["Excellent logical reasoning", "Structured outputs", "Fast reasoning"],
      cons: ["Less natural text", "Can be terse", "Limited creativity"],
      bestFor: ["Math problems", "Logic puzzles", "Code analysis"],
    },
    {
      model: "Gemma Only",
      pros: ["High-quality text", "Natural language", "Creative responses"],
      cons: ["Less structured reasoning", "Can miss details", "May be verbose"],
      bestFor: ["Content creation", "General conversation", "Writing tasks"],
    },
    {
      model: "Gemma + QBNN Frontal",
      pros: [
        "Intelligent reasoning + Quality writing",
        "Best of both worlds",
        "Structured yet natural",
        "Comprehensive answers",
      ],
      cons: ["Slightly longer processing time", "More tokens used"],
      bestFor: ["Complex questions", "Analysis tasks", "Technical writing"],
    },
  ];

  examples.forEach((ex, i) => {
    log(`\n${colors.magenta}${i + 1}. ${ex.model}${colors.reset}`);
    log("Strengths:");
    ex.pros.forEach((pro) => log(`  ✓ ${pro}`));
    log("Limitations:");
    ex.cons.forEach((con) => log(`  • ${con}`));
    log("Best for:");
    ex.bestFor.forEach((use) => log(`  → ${use}`));
  });
}

// ============================================================================
// Demo 3: Example Interaction Flow
// ============================================================================

function demo3_InteractionFlow() {
  section("Demo 3: Interaction Flow Example");

  log("📝 Example Question:\n", colors.cyan);
  log("What are the main differences between machine learning and deep learning?\n");

  log("🧠 Phase 1 - QBNN Reasoning Analysis:\n", colors.magenta);
  log("  1. Identifies: 2 AI concepts, requires comparison");
  log("  2. Key differences: Data requirements, architecture, interpretability");
  log("  3. Best approach: Structure as clear comparison with examples");
  log("  4. Important context: Both are subsets of AI\n");

  log("💬 Phase 2 - Gemma Generation:\n", colors.blue);
  log("  Based on QBNN's structured analysis, Gemma generates:");
  log("  Machine learning and deep learning are both subfields of artificial");
  log("  intelligence, but they differ in several key ways:\n");
  log("  Key Differences:\n");
  log("  1. Data Requirements: ML works with smaller datasets, DL needs massive...");
  log("  2. Architecture: ML uses simpler algorithms, DL uses neural networks...");
  log("  3. Computational Power: ML can run on regular computers, DL requires GPU...\n");
  log("  Applications include...\n");

  log("✅ Final Output: Intelligent structure + Natural presentation\n", colors.green);
}

// ============================================================================
// Demo 4: Use Cases
// ============================================================================

function demo4_UseCases() {
  section("Demo 4: Ideal Use Cases");

  const useCases = [
    {
      category: "Technical Analysis",
      examples: [
        "Explaining complex algorithms",
        "Comparing programming languages",
        "System architecture discussions",
      ],
      why: "Reasoning for structure + Gemma for clarity",
    },
    {
      category: "Business Problems",
      examples: [
        "Market analysis questions",
        "Strategy evaluation",
        "Risk assessment",
      ],
      why: "Logical analysis + Professional writing",
    },
    {
      category: "Educational Content",
      examples: [
        "Detailed explanations",
        "Conceptual breakdowns",
        "Teaching approaches",
      ],
      why: "Structured reasoning + Engaging presentation",
    },
    {
      category: "Problem Solving",
      examples: [
        "Complex troubleshooting",
        "Decision making frameworks",
        "Root cause analysis",
      ],
      why: "Analytical thinking + Actionable output",
    },
  ];

  useCases.forEach((useCase, i) => {
    log(`\n${colors.yellow}${i + 1}. ${useCase.category}${colors.reset}`);
    useCase.examples.forEach((ex) => {
      log(`   • ${ex}`);
    });
    log(`   💡 ${useCase.why}`);
  });
}

// ============================================================================
// Demo 5: Configuration Options
// ============================================================================

function demo5_Configuration() {
  section("Demo 5: Configuration & Control");

  log("⚙️  Hybrid Chat Configuration:\n", colors.bright);

  log("Temperature Settings:\n", colors.cyan);
  log("  • QBNN Reasoning Temperature: 0.4");
  log("    Controls analytical thinking (lower = more logical)\n");
  log("  • Gemma Generation Temperature: 0.6");
  log("    Controls output creativity (higher = more creative)\n");

  log("Control Commands:\n", colors.cyan);
  log("  /reasoning        Toggle reasoning display (see QBNN's analysis)");
  log("  /temp <0-2>      Adjust Gemma generation temperature");
  log("  /tokens <num>    Set maximum output length");
  log("  /config          View current hybrid configuration\n");

  log("Performance:\n", colors.cyan);
  log("  • QBNN Reasoning: 500-1500ms");
  log("  • Gemma Generation: 1000-3000ms");
  log("  • Total: 1500-4500ms per query\n");

  log("💡 Tip: Complex questions benefit most from hybrid processing\n", colors.green);
}

// ============================================================================
// Demo 6: Starting the Hybrid Chat
// ============================================================================

function demo6_StartingHybrid() {
  section("Demo 6: Starting the Hybrid Chat");

  log("🚀 Launch the Hybrid Chat:\n", colors.bright);

  log("Interactive Mode:\n", colors.cyan);
  log("  npm run hybrid\n");

  log("Development Mode:\n", colors.cyan);
  log("  npm run dev:hybrid\n");

  log("Example Session:\n", colors.cyan);
  log("  $ npm run hybrid");
  log("  🧠 Qubit AI Hybrid Chat - Gemma + QBNN");
  log("  ℹ️  Type 'help' for commands\n");
  log("  You: /reasoning            # Toggle reasoning display");
  log("  You: /config               # View configuration");
  log("  You: What is quantum computing?");
  log("  [QBNN analyzes query...]");
  log("  [Gemma generates response...]");
  log("  🤖 Assistant: ...\n");

  log("Tips:\n", colors.yellow);
  log("  • Use /reasoning ON to see QBNN's analysis");
  log("  • Ask complex questions for best results");
  log("  • Use /temp 0.3 for factual responses");
  log("  • Use /temp 0.8 for more creative output\n");
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.clear();

  log("╔═════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                             ║", colors.bright);
  log("║    🧠 Gemma + QBNN Frontal Hybrid Chat System Demo 🧠     ║", colors.bright);
  log("║                                                             ║", colors.bright);
  log("║  Intelligent Reasoning + Quality Writing                   ║", colors.bright);
  log("║                                                             ║", colors.bright);
  log("╚═════════════════════════════════════════════════════════════╝", colors.bright);

  try {
    demo1_Architecture();
    await sleep(800);

    demo2_Comparison();
    await sleep(800);

    demo3_InteractionFlow();
    await sleep(800);

    demo4_UseCases();
    await sleep(800);

    demo5_Configuration();
    await sleep(800);

    demo6_StartingHybrid();
    await sleep(800);

    log("═".repeat(65), colors.bright);
    log("✅ Hybrid Chat System Demo Complete!", colors.green);
    log("═".repeat(65) + "\n", colors.bright);

    log("🎯 Next Steps:\n", colors.bright);
    log("  1. Try the hybrid chat: npm run hybrid");
    log("  2. Ask a complex question");
    log("  3. Enable reasoning display: /reasoning");
    log("  4. View configuration: /config");
    log("  5. Experiment with temperature settings\n");

    log("📊 When to use each model:\n", colors.bright);
    log("  • Standard chat: npm start (default model)");
    log("  • Model selection: npm start -- --model gemma-2");
    log("  • Hybrid reasoning: npm run hybrid\n");

    log("📖 For more information:\n", colors.bright);
    log("  • README.md - Complete CLI documentation");
    log("  • demo-model-selection.js - Model comparison");
    log("  • demo-advanced-features.js - Advanced capabilities\n");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
