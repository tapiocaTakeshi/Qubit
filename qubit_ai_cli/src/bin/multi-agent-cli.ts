#!/usr/bin/env node

/**
 * Multi-Agent Chat CLI
 * Claude, ChatGPT, Gemini, Perplexity role-sharing system
 */

import * as readline from "readline";
import { createMultiAgentChat, MultiAgentChat } from "../multi-agent-chat.js";
import * as fs from "fs";
import * as path from "path";

interface MultiAgentCLIState {
  chat: MultiAgentChat;
  rl: readline.Interface;
  messageCount: number;
  showAgentDetails: boolean;
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
  log(`❌ ${message}`, colors.red);
}

function logAgent(message: string): void {
  log(`🤖 ${message}`, colors.blue);
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
    "╔════════════════════════════════════════════════════════════╗",
    colors.bright
  );
  log(
    "║                                                            ║",
    colors.bright
  );
  log(
    "║   🤖 Multi-Agent Collaborative Chat System 🤖            ║",
    colors.bright
  );
  log(
    "║                                                            ║",
    colors.bright
  );
  log(
    "║  Claude (Analysis) | ChatGPT (Writing)                   ║",
    colors.bright
  );
  log(
    "║  Gemini (Synthesis) | Perplexity (Research)              ║",
    colors.bright
  );
  log(
    "║                                                            ║",
    colors.bright
  );
  log(
    "╚════════════════════════════════════════════════════════════╝",
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
  log("  agents            - Show all agents and their roles");
  log("  details           - Toggle detailed agent output display");
  log("  analyze <path>    - Analyze an image file with multi-agent system");
  log("  history           - View conversation history");
  log("  export            - Export conversation to JSON");
  log("  clear             - Clear conversation history");
  log("  exit / quit       - Exit the chat\n");

  log("💡 Tips:\n", colors.bright);
  log("  • Ask complex questions to see all agents work");
  log("  • Use /details to see individual agent responses");
  log("  • Each agent specializes in different aspects");
  log("  • Responses are synthesized into a final answer");
  log("  • Use /analyze <image_path> for image analysis\n");
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
 * Analyze image with multi-agent system
 */
async function analyzeImage(
  imagePath: string,
  state: MultiAgentCLIState
): Promise<void> {
  try {
    logInfo(`Loading image: ${imagePath}`);
    const base64Image = loadImageAsBase64(imagePath);
    const fileName = path.basename(imagePath);

    logUser(`Analyze image: ${fileName}`);
    logInfo("🔄 Coordinating agents for image analysis...\n");

    const startTime = Date.now();
    const analysisPrompt = `You are analyzing an image. The image is provided as base64 data.
Please analyze the following image and provide insights:
- Visual description
- Design elements
- Purpose and context
- Notable features
- Potential improvements

Image (base64): ${base64Image.substring(0, 100)}... [truncated for brevity]
File: ${fileName}`;

    const result = await state.chat.processQuery(analysisPrompt);
    const duration = Date.now() - startTime;

    // Show agent details if enabled
    if (state.showAgentDetails) {
      log("\n" + "═".repeat(70), colors.dim);
      log("📊 INDIVIDUAL AGENT RESPONSES", colors.yellow);
      log("═".repeat(70), colors.dim);

      result.agentResponses.forEach((response) => {
        log(`\n🤖 ${response.agentName}`, colors.bright);
        log(`   Role: ${response.role}`);
        log(`   Processing time: ${response.processingTime}ms`);
        log(`   Response:\n${response.output}\n`);
      });

      log("═".repeat(70), colors.dim);
    }

    // Show synthesized response
    log("\n" + result.finalSynthesis + "\n");

    logInfo(`Image analysis completed in ${duration}ms\n`);

    state.messageCount++;

    // Auto-save history
    if (state.saveHistory) {
      saveHistory(state);
    }
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
  state: MultiAgentCLIState
): Promise<boolean> {
  const args = command.trim().split(/\s+/);
  const cmd = args[0].toLowerCase();

  switch (cmd) {
    case "help":
      printHelp();
      return true;

    case "agents": {
      const agents = state.chat.getAvailableAgents();
      log("\n🤖 Available Agents:\n", colors.bright);

      Object.entries(agents).forEach(([key, agent]) => {
        log(`\n  ${agent.name}`);
        log(`  Role: ${agent.role}`);
        log(`  Description: ${agent.description}`);
        log(`  Temperature: ${agent.temperature}`);
      });

      log("", colors.reset);
      return true;
    }

    case "details": {
      state.showAgentDetails = !state.showAgentDetails;
      const status = state.showAgentDetails ? "enabled" : "disabled";
      logSuccess(`Detailed agent output display ${status}\n`);
      return true;
    }

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
      const messages = state.chat.getHistory();
      if (messages.length === 0) {
        logInfo("No messages yet\n");
        return true;
      }

      log("\n📜 Conversation History:\n", colors.bright);
      messages.forEach((msg, i) => {
        const role = msg.role === "user" ? "👤 You" : "🤖 Assistant";
        log(`[${i + 1}] ${role}: ${msg.content.substring(0, 100)}...`);
      });
      log("", colors.reset);
      return true;
    }

    case "export": {
      const timestamp = new Date().toISOString().split("T")[0];
      const filename = `qubit-multi-agent-${timestamp}.json`;
      const conversation = state.chat.exportConversation();

      fs.writeFileSync(filename, conversation);
      logSuccess(`Conversation exported to ${filename}\n`);
      return true;
    }

    case "clear":
      state.chat.clearHistory();
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
 * Process user input and generate response
 */
async function processMessage(
  userInput: string,
  state: MultiAgentCLIState
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
    logInfo(
      "🔄 Coordinating agents (Claude, ChatGPT, Gemini, Perplexity)...\n"
    );

    const startTime = Date.now();
    const result = await state.chat.processQuery(userInput);
    const duration = Date.now() - startTime;

    // Show agent details if enabled
    if (state.showAgentDetails) {
      log("\n" + "═".repeat(70), colors.dim);
      log("📊 INDIVIDUAL AGENT RESPONSES", colors.yellow);
      log("═".repeat(70), colors.dim);

      result.agentResponses.forEach((response) => {
        log(`\n🤖 ${response.agentName}`, colors.bright);
        log(`   Role: ${response.role}`);
        log(`   Processing time: ${response.processingTime}ms`);
        log(`   Response:\n${response.output}\n`);
      });

      log("═".repeat(70), colors.dim);
    }

    // Show synthesized response
    log("\n" + result.finalSynthesis + "\n");

    logInfo(`Total coordination time: ${duration}ms\n`);

    state.messageCount++;

    // Auto-save history
    if (state.saveHistory) {
      saveHistory(state);
    }

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
 * Save conversation history to file
 */
function saveHistory(state: MultiAgentCLIState): void {
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
async function startInteractiveMode(state: MultiAgentCLIState): Promise<void> {
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
function cleanup(state: MultiAgentCLIState): void {
  if (state.saveHistory) {
    saveHistory(state);
  }

  state.rl.close();

  log("\n", colors.reset);
  logSuccess(`Multi-agent chat session ended. (${state.messageCount} messages)\n`);
  process.exit(0);
}

/**
 * Main function
 */
async function main(): Promise<void> {
  try {
    // Create multi-agent chat instance
    const chat = await createMultiAgentChat();

    // Setup readline
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    const historyDir = path.join(process.cwd(), ".qubit-history");
    if (!fs.existsSync(historyDir)) {
      fs.mkdirSync(historyDir, { recursive: true });
    }

    const state: MultiAgentCLIState = {
      chat,
      rl,
      messageCount: 0,
      showAgentDetails: false,
      saveHistory: true,
      historyFile: path.join(
        historyDir,
        `multi-agent-${Date.now()}.json`
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
