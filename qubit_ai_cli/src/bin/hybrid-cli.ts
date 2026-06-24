#!/usr/bin/env node

/**
 * Qubit AI Hybrid Chat CLI - Gemma + QBNN Frontal
 *
 * QBNN provides reasoning analysis
 * Gemma provides high-quality generation
 */

import * as readline from "readline";
import { createHybridChat, HybridChat } from "../hybrid-chat.js";
import * as fs from "fs";
import * as path from "path";

interface HybridCLIState {
  chat: HybridChat;
  rl: readline.Interface;
  messageCount: number;
  showReasoning: boolean;
  saveHistory: boolean;
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
  log(`❌ ${message}`, colors.yellow);
}

function logAssistant(message: string): void {
  log(`🤖 Assistant: ${message}`, colors.blue);
}

function logReasoning(message: string): void {
  log(`🧠 Reasoning: ${message}`, colors.magenta);
}

function logUser(message: string): void {
  log(`👤 You: ${message}`, colors.cyan);
}

/**
 * Print welcome message
 */
function printWelcome(): void {
  console.clear();
  log("", colors.reset);
  log("╔════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║      🧠 Qubit AI Hybrid Chat - Gemma + QBNN 🧠          ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║  QBNN Reasoning + Gemma Generation                        ║", colors.bright);
  log("║  Advanced reasoning with high-quality text output          ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("╚════════════════════════════════════════════════════════════╝", colors.bright);
  log("", colors.reset);

  logInfo("Type 'help' for commands | 'exit' to quit\n");
}

/**
 * Print help message
 */
function printHelp(): void {
  log("\n📖 Available Commands:\n", colors.bright);
  log("  help          - Show this help message");
  log("  clear         - Clear conversation history");
  log("  reasoning     - Toggle reasoning display (on/off)");
  log("  history       - Show conversation history");
  log("  export        - Export conversation to JSON");
  log("  config        - Show current configuration");
  log("  temp <value>  - Set Gemma temperature (0.0-2.0)");
  log("  tokens <num>  - Set max tokens to generate");
  log("  exit / quit   - Exit the chat\n");

  log("💡 Tips:\n", colors.bright);
  log("  • Ask complex questions for better reasoning");
  log("  • Use /reasoning to see QBNN's analysis");
  log("  • Lower temperature (0.3) = more factual");
  log("  • Higher temperature (1.5) = more creative\n");
}

/**
 * Handle special commands
 */
async function handleCommand(
  command: string,
  state: HybridCLIState
): Promise<boolean> {
  const args = command.trim().split(/\s+/);
  const cmd = args[0].toLowerCase();

  switch (cmd) {
    case "help":
      printHelp();
      return true;

    case "clear":
      state.chat.clearHistory();
      logSuccess("Conversation history cleared\n");
      return true;

    case "reasoning": {
      state.showReasoning = !state.showReasoning;
      const status = state.showReasoning ? "enabled" : "disabled";
      logSuccess(`Reasoning display ${status}\n`);
      return true;
    }

    case "history": {
      const messages = state.chat.getHistory();
      if (messages.length === 0) {
        logInfo("No messages yet\n");
        return true;
      }

      log("\n📜 Conversation History:\n", colors.bright);
      messages.forEach((msg, i) => {
        const role = msg.role === "user" ? "👤 You" : "🤖 Assistant";
        log(`[${i + 1}] ${role}: ${msg.content}`);
      });
      log("", colors.reset);
      return true;
    }

    case "export": {
      const timestamp = new Date().toISOString().split("T")[0];
      const filename = `qubit-hybrid-${timestamp}.json`;
      const conversation = state.chat.exportConversation(true);

      fs.writeFileSync(filename, conversation);
      logSuccess(`Conversation exported to ${filename}\n`);
      return true;
    }

    case "config": {
      const config = state.chat.getConfig();
      log("\n⚙️  Hybrid Chat Configuration:\n", colors.bright);
      log(`  Model: Gemma + QBNN Frontal`);
      log(`  Max Tokens: ${config.maxTokens}`);
      log(`  Gemma Temperature: ${config.temperature}`);
      log(`  QBNN Reasoning Temperature: ${config.reasoningTemperature}`);
      log(`  Reasoning Enabled: ${config.enableReasoning ? "Yes" : "No"}`);
      log(`  Reasoning Display: ${state.showReasoning ? "On" : "Off"}`);
      log("", colors.reset);
      return true;
    }

    case "temp": {
      const temp = parseFloat(args[1]);
      if (isNaN(temp) || temp < 0 || temp > 2) {
        logError("Temperature must be between 0.0 and 2.0\n");
        return true;
      }

      state.chat.updateConfig({ temperature: temp });
      logSuccess(`Gemma temperature set to ${temp}\n`);
      return true;
    }

    case "tokens": {
      const tokens = parseInt(args[1], 10);
      if (isNaN(tokens) || tokens < 10 || tokens > 1000) {
        logError("Max tokens must be between 10 and 1000\n");
        return true;
      }

      state.chat.updateConfig({ maxTokens: tokens });
      logSuccess(`Max tokens set to ${tokens}\n`);
      return true;
    }

    case "exit":
    case "quit":
      return false;

    default:
      return true;
  }
}

/**
 * Process user input and generate response
 */
async function processMessage(
  userInput: string,
  state: HybridCLIState
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
    const startTime = Date.now();

    // Show processing message
    logInfo("🔄 Processing (QBNN reasoning + Gemma generation)...\n");

    // Generate response with reasoning
    const { response, reasoning } = await state.chat.sendMessage(userInput);
    const duration = Date.now() - startTime;

    // Show reasoning if enabled
    if (state.showReasoning && reasoning) {
      const reasoningPreview = reasoning.split("\n").slice(0, 3).join("\n");
      logReasoning(reasoningPreview);
      log("", colors.reset);
    }

    // Show response
    logAssistant(response);
    logInfo(`Generated in ${duration}ms (Reasoning + Generation)\n`);

    state.messageCount++;

    // Auto-save history
    if (state.saveHistory) {
      saveHistory(state);
    }

    return true;
  } catch (error) {
    const errorMsg =
      error instanceof Error ? error.message : String(error);

    logError(`Failed to generate response: ${errorMsg}`);

    if (errorMsg.includes("HF_TOKEN")) {
      logInfo("Set your HuggingFace token: export HF_TOKEN='hf_...'");
    }

    log("", colors.reset);
    return true;
  }
}

/**
 * Save conversation history to file
 */
function saveHistory(state: HybridCLIState): void {
  try {
    const conversation = state.chat.exportConversation();
    fs.writeFileSync(state.historyFile, conversation);
  } catch (error) {
    // Silently fail if history save fails
  }
}

/**
 * Start interactive chat mode
 */
async function startInteractiveMode(state: HybridCLIState): Promise<void> {
  printWelcome();

  const askQuestion = (): void => {
    state.rl.question(
      `${colors.cyan}You: ${colors.reset}`,
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
function cleanup(state: HybridCLIState): void {
  if (state.saveHistory) {
    saveHistory(state);
  }

  state.rl.close();

  log("\n", colors.reset);
  logSuccess(`Hybrid chat session ended. (${state.messageCount} messages)\n`);
  process.exit(0);
}

/**
 * Main function
 */
async function main(): Promise<void> {
  try {
    // Create hybrid chat instance
    const chat = await createHybridChat({
      maxTokens: 300,
      temperature: 0.6,
      reasoningTemperature: 0.4,
      enableReasoning: true,
    });

    // Setup readline
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const historyDir = path.join(process.cwd(), ".qubit-history");
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    const state: HybridCLIState = {
      chat,
      rl,
      messageCount: 0,
      showReasoning: true, // Show reasoning by default
      saveHistory: true,
      historyFile: path.join(
        historyDir,
        `hybrid-${Date.now()}.json`
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
