#!/usr/bin/env node

/**
 * Qubit AI Interactive Chat CLI
 *
 * Usage:
 *   npx qubit-chat                    # Start interactive chat
 *   npx qubit-chat "Your question"    # Single query mode
 */

import * as readline from "readline";
import { createChat, QubitAIChat } from "../chat.js";
import * as fs from "fs";
import * as path from "path";

interface CLIState {
  chat: QubitAIChat;
  rl: readline.Interface;
  quiet: boolean;
  saveHistory: boolean;
  historyFile: string;
  messageCount: number;
  currentModel: string;
}

// Color codes for terminal output
const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
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

function logUser(message: string): void {
  log(`👤 You: ${message}`, colors.magenta);
}

/**
 * Print welcome message
 */
function printWelcome(chat: QubitAIChat): void {
  console.clear();
  log("", colors.reset);
  log("╔════════════════════════════════════════════════════════════╗", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║         🤖 Qubit AI Interactive Chat CLI 🤖              ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("║  Powered by quantum-inspired neural networks              ║", colors.bright);
  log("║  Real-time conversation with NeuroQuantum engine           ║", colors.bright);
  log("║                                                            ║", colors.bright);
  log("╚════════════════════════════════════════════════════════════╝", colors.bright);
  log("", colors.reset);

  const model = chat.getCurrentModel();
  logInfo(`Using model: ${model.description}`);
  logInfo("Type 'help' for commands | 'exit' to quit\n");
}

/**
 * Print help message
 */
function printHelp(): void {
  log("\n📖 Available Commands:\n", colors.bright);
  log("  help          - Show this help message");
  log("  model         - Show available models and switch models");
  log("  clear         - Clear conversation history");
  log("  history       - Show conversation history");
  log("  export        - Export conversation to JSON");
  log("  config        - Show current configuration");
  log("  temp <value>  - Set temperature (0.0-2.0)");
  log("  tokens <num>  - Set max tokens to generate");
  log("  exit / quit   - Exit the chat\n");

  log("💡 Tips:\n", colors.bright);
  log("  • Start questions with 'Q:' for better responses");
  log("  • Use complete sentences for better context");
  log("  • Lower temperature (0.3) = more factual");
  log("  • Higher temperature (1.2) = more creative");
  log("  • Try different models with /model command\n");
}

/**
 * Handle special commands
 */
async function handleCommand(
  command: string,
  state: CLIState
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
      const filename = `qubit-chat-${timestamp}.json`;
      const conversation = state.chat.exportConversation();

      fs.writeFileSync(filename, conversation);
      logSuccess(`Conversation exported to ${filename}\n`);
      return true;
    }

    case "config": {
      const config = state.chat.getConfig();
      const model = state.chat.getCurrentModel();
      log("\n⚙️  Current Configuration:\n", colors.bright);
      log(`  Model: ${model.description}`);
      log(`  Endpoint: ${model.endpoint}`);
      log(`  Temperature: ${config.temperature}`);
      log(`  Max Tokens: ${config.maxTokens}`);
      log(`  Top K: ${config.topK}`);
      log(`  Top P: ${config.topP}`);
      log(`  Repetition Penalty: ${config.repetitionPenalty}`);
      log("", colors.reset);
      return true;
    }

    case "model": {
      const availableModels = state.chat.getAvailableModels();
      const currentModel = state.chat.getCurrentModel();
      const modelName = args[1];

      // If model name provided, try to switch
      if (modelName) {
        if (state.chat.setModel(modelName)) {
          state.currentModel = modelName;
          const newModel = state.chat.getCurrentModel();
          logSuccess(`Model switched to: ${newModel.description}`);
          logInfo(`Endpoint: ${newModel.endpoint}\n`);
          return true;
        } else {
          logError(`Unknown model: ${modelName}`);
          log(`Available models: ${Object.keys(availableModels).join(", ")}\n`, colors.yellow);
          return true;
        }
      }

      // Otherwise, show available models
      log("\n🤖 Available Models:\n", colors.bright);

      Object.entries(availableModels).forEach(([key, model]) => {
        const indicator = key === state.currentModel ? "  ✓ " : "    ";
        log(`${indicator}${key.padEnd(12)} - ${model.description}`);
        log(`              Endpoint: ${model.endpoint}`);
      });

      log("\nUsage: /model <name>", colors.cyan);
      log(`Current model: ${currentModel.description}\n`, colors.green);
      return true;
    }

    case "temp": {
      const temp = parseFloat(args[1]);
      if (isNaN(temp) || temp < 0 || temp > 2) {
        logError("Temperature must be between 0.0 and 2.0\n");
        return true;
      }

      state.chat.updateConfig({ temperature: temp });
      logSuccess(`Temperature set to ${temp}\n`);
      return true;
    }

    case "tokens": {
      const tokens = parseInt(args[1], 10);
      if (isNaN(tokens) || tokens < 10 || tokens > 500) {
        logError("Max tokens must be between 10 and 500\n");
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
  state: CLIState
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
    const response = await state.chat.sendMessage(userInput);
    const duration = Date.now() - startTime;

    logAssistant(response);
    logInfo(`Generated in ${duration}ms\n`);

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
function saveHistory(state: CLIState): void {
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
async function startInteractiveMode(state: CLIState): Promise<void> {
  printWelcome(state.chat);

  const askQuestion = (): void => {
    state.rl.question(`${colors.magenta}You: ${colors.reset}`, async (input) => {
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
    });
  };

  askQuestion();
}

/**
 * Single query mode
 */
async function singleQueryMode(
  query: string,
  state: CLIState
): Promise<void> {
  log("", colors.reset);
  logUser(query);

  try {
    const startTime = Date.now();
    const response = await state.chat.sendMessage(query);
    const duration = Date.now() - startTime;

    logAssistant(response);
    logInfo(`Generated in ${duration}ms\n`);
  } catch (error) {
    const errorMsg =
      error instanceof Error ? error.message : String(error);
    logError(`Failed: ${errorMsg}`);
  }

  cleanup(state);
}

/**
 * Cleanup and exit
 */
function cleanup(state: CLIState): void {
  if (state.saveHistory) {
    saveHistory(state);
  }

  state.rl.close();

  log("\n", colors.reset);
  logSuccess(`Chat session ended. (${state.messageCount} messages)\n`);
  process.exit(0);
}

/**
 * Parse command-line arguments
 */
function parseArguments(args: string[]): { model: string; query: string } {
  let model = "qbnn";
  const queryParts: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--model" || arg === "-m") {
      if (i + 1 < args.length) {
        model = args[i + 1];
        i++;
      }
    } else {
      queryParts.push(arg);
    }
  }

  return {
    model,
    query: queryParts.join(" ").trim(),
  };
}

/**
 * Main function
 */
async function main(): Promise<void> {
  try {
    // Parse arguments
    const args = process.argv.slice(2);
    const { model: selectedModel, query } = parseArguments(args);

    // Create chat instance with selected model
    const chat = await createChat(
      {
        generation: {
          maxTokens: 150,
          temperature: 0.7,
          topK: 40,
          topP: 0.9,
          repetitionPenalty: 1.2,
        },
        contextWindowSize: 5,
        enableHistory: true,
      },
      selectedModel
    );

    // Setup readline for interactive mode
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const historyDir = path.join(process.cwd(), ".qubit-history");
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    const state: CLIState = {
      chat,
      rl,
      quiet: false,
      saveHistory: true,
      historyFile: path.join(
        historyDir,
        `history-${Date.now()}.json`
      ),
      messageCount: 0,
      currentModel: selectedModel,
    };

    // Run in appropriate mode
    if (query) {
      // Single query mode
      await singleQueryMode(query, state);
    } else {
      // Interactive mode
      await startInteractiveMode(state);
    }
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
