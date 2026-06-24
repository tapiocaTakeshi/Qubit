#!/usr/bin/env node

/**
 * Qubit AI CLI - Model Selection Demo
 * Shows how to use different models (QBNN, Gemma-2, Gemma-7)
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

function section(title) {
  log("\n" + "═".repeat(65), colors.bright);
  log(`║ ${title.padEnd(63)} ║`, colors.bright);
  log("═".repeat(65) + "\n", colors.bright);
}

// ============================================================================
// Model Information
// ============================================================================

const MODELS = {
  qbnn: {
    name: "QBNN",
    fullName: "Quantum-inspired Bidirectional Neural Network",
    endpoint: "neuroq-ai/quantum-llm",
    advantages: [
      "Quantum-inspired neural architecture",
      "Excellent for logical reasoning",
      "Good balance of speed and quality",
      "Specialized for structured outputs",
    ],
    use_cases: ["Logical puzzles", "Structured data analysis", "Code generation"],
    token_speed: "~100-150ms per generation",
  },
  "gemma-2": {
    name: "Gemma 2",
    fullName: "Google Gemma 2 9B Instruct",
    endpoint: "google/gemma-2-9b-it",
    advantages: [
      "Latest Google language model",
      "High quality text generation",
      "Instruction-following capability",
      "Balanced performance",
    ],
    use_cases: ["General conversation", "Content creation", "Instruction following"],
    token_speed: "~150-200ms per generation",
  },
  "gemma-7": {
    name: "Gemma 7B",
    fullName: "Google Gemma 7B Instruct",
    endpoint: "google/gemma-7b-it",
    advantages: [
      "Lightweight model",
      "Fast inference",
      "Good for mobile/edge deployment",
      "Lower latency",
    ],
    use_cases: ["Real-time chat", "Edge computing", "Latency-sensitive tasks"],
    token_speed: "~80-120ms per generation",
  },
};

// ============================================================================
// Demo 1: Model Overview
// ============================================================================

function demo1_ModelOverview() {
  section("Demo 1: Available Models & Endpoints");

  log("🤖 Qubit AI CLI supports three models:\n", colors.bright);

  Object.entries(MODELS).forEach(([key, model]) => {
    log(`\n${colors.bright}${model.fullName} (${key})${colors.reset}`);
    log(`Endpoint: ${colors.cyan}${model.endpoint}${colors.reset}`);
    log("\n  Advantages:");
    model.advantages.forEach((adv) => {
      log(`    • ${adv}`);
    });
    log("\n  Best for:");
    model.use_cases.forEach((use) => {
      log(`    • ${use}`);
    });
    log(`\n  Typical latency: ${colors.yellow}${model.token_speed}${colors.reset}`);
  });
}

// ============================================================================
// Demo 2: Command-line Model Selection
// ============================================================================

function demo2_CommandlineSelection() {
  section("Demo 2: Starting CLI with Specific Model");

  log("📝 You can specify a model when starting the CLI:\n", colors.magenta);

  const examples = [
    {
      command: "npm start",
      description: "Start with default model (QBNN)",
      result: "🤖 Using model: Quantum-inspired Bidirectional Neural Network (QBNN)",
    },
    {
      command: "npm start -- --model gemma-2",
      description: "Start with Gemma 2 model",
      result: "🤖 Using model: Google Gemma 2 9B Instruct",
    },
    {
      command: "npm start -- --model gemma-7",
      description: "Start with Gemma 7B model",
      result: "🤖 Using model: Google Gemma 7B Instruct",
    },
    {
      command: "npm start -- --model qbnn",
      description: "Explicitly specify QBNN model",
      result: "🤖 Using model: Quantum-inspired Bidirectional Neural Network (QBNN)",
    },
  ];

  examples.forEach((ex, i) => {
    log(`\n${i + 1}. ${ex.description}`, colors.green);
    log(`   Command: ${colors.cyan}${ex.command}${colors.reset}`);
    log(`   Output: ${colors.yellow}${ex.result}${colors.reset}`);
  });
}

// ============================================================================
// Demo 3: Runtime Model Switching
// ============================================================================

function demo3_RuntimeSwitching() {
  section("Demo 3: Switching Models During Chat");

  log("💬 During an interactive chat session, use /model command:\n", colors.bright);

  const steps = [
    { command: "/model", description: "List all available models", output: "Shows 3 models with checkmark on current" },
    {
      command: "/model gemma-2",
      description: "Switch to Gemma 2",
      output: "✅ Model switched to: Google Gemma 2 9B Instruct",
    },
    { command: "/config", description: "View current configuration", output: "Shows selected model in config" },
  ];

  steps.forEach((step, i) => {
    log(`\n${colors.cyan}Step ${i + 1}: ${step.description}${colors.reset}`);
    log(`Input:  ${colors.magenta}${step.command}${colors.reset}`);
    log(`Output: ${colors.green}${step.output}${colors.reset}`);
  });

  log(
    "\n\n📌 Note: Switching models changes the model configuration for subsequent messages.",
    colors.yellow
  );
}

// ============================================================================
// Demo 4: Model Comparison
// ============================================================================

function demo4_ModelComparison() {
  section("Demo 4: Model Comparison & Selection Guide");

  log("📊 Choosing the Right Model:\n", colors.bright);

  const comparisons = [
    {
      scenario: "Fastest response needed",
      recommendation: "Gemma 7B",
      reason: "Lowest latency (~80-120ms)",
    },
    {
      scenario: "Best quality text generation",
      recommendation: "Gemma 2",
      reason: "Latest Google model with highest quality",
    },
    {
      scenario: "Logical reasoning & structured output",
      recommendation: "QBNN",
      reason: "Quantum-inspired architecture excels at reasoning",
    },
    {
      scenario: "Default general-purpose chat",
      recommendation: "QBNN",
      reason: "Good balance of quality, speed, and reliability",
    },
    {
      scenario: "Code generation",
      recommendation: "QBNN or Gemma 2",
      reason: "Both have strong instruction-following abilities",
    },
  ];

  comparisons.forEach((comp, i) => {
    log(`\n${colors.yellow}${i + 1}. ${comp.scenario}${colors.reset}`);
    log(`   → Recommended: ${colors.green}${comp.recommendation}${colors.reset}`);
    log(`   • ${comp.reason}`);
  });
}

// ============================================================================
// Demo 5: Practical Examples
// ============================================================================

function demo5_PracticalExamples() {
  section("Demo 5: Practical Usage Examples");

  log("🎯 Common Use Cases:\n", colors.bright);

  const examples = [
    {
      title: "Quick Chat Session with QBNN",
      command: "npm start",
      description: "Starts interactive chat with QBNN (default)",
    },
    {
      title: "Switch to Gemma 2 for Better Content",
      steps: [
        "npm start",
        "/model",
        "/model gemma-2",
        "Ask your question",
      ],
    },
    {
      title: "Test Different Models",
      steps: [
        "npm start -- --model qbnn",
        "Ask a logic question",
        "Exit and compare with: npm start -- --model gemma-2",
      ],
    },
    {
      title: "Single Query with Specific Model",
      command: 'npm start -- --model gemma-7 "Your question here"',
      description: "Run a single query with Gemma 7B and exit",
    },
  ];

  examples.forEach((ex, i) => {
    log(`\n${colors.cyan}Example ${i + 1}: ${ex.title}${colors.reset}`);
    if (ex.command) {
      log(`Command: ${colors.magenta}${ex.command}${colors.reset}`);
      if (ex.description) {
        log(`${ex.description}`);
      }
    }
    if (ex.steps) {
      log("Steps:");
      ex.steps.forEach((step, j) => {
        log(`  ${j + 1}. ${colors.magenta}${step}${colors.reset}`);
      });
    }
  });
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.clear();

  log("╔═════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                             ║", colors.bright);
  log("║  🤖 Qubit AI CLI - Model Selection Demo 🤖                ║", colors.bright);
  log("║                                                             ║", colors.bright);
  log("║  Learn how to use QBNN, Gemma 2, and Gemma 7 models       ║", colors.bright);
  log("║                                                             ║", colors.bright);
  log("╚═════════════════════════════════════════════════════════════╝", colors.bright);

  try {
    demo1_ModelOverview();
    await new Promise((r) => setTimeout(r, 1000));

    demo2_CommandlineSelection();
    await new Promise((r) => setTimeout(r, 1000));

    demo3_RuntimeSwitching();
    await new Promise((r) => setTimeout(r, 1000));

    demo4_ModelComparison();
    await new Promise((r) => setTimeout(r, 1000));

    demo5_PracticalExamples();
    await new Promise((r) => setTimeout(r, 1000));

    log("\n" + "═".repeat(65), colors.bright);
    log("✅ Model Selection Demo Complete!", colors.green);
    log("═".repeat(65) + "\n", colors.bright);

    log("🚀 Next Steps:\n", colors.bright);
    log("  1. Try different models: npm start -- --model gemma-2");
    log("  2. Use /model command to switch during chat");
    log("  3. Use /config to see current model details");
    log("  4. Compare model outputs for your use case\n");

    log("📖 For more information:\n", colors.bright);
    log("  • README.md - Complete documentation");
    log("  • USAGE.md - Detailed usage examples");
    log("  • ADVANCED_FEATURES.md - Advanced capabilities\n");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
