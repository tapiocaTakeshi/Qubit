#!/usr/bin/env node

/**
 * Qubit AI Interactive Chat CLI
 *
 * Usage:
 *   qbnn                     # Launch the full-screen chat TUI
 *   qbnn "Your question"     # One-shot query, print answer and exit
 *   qbnn --help             # Show usage
 */

import React from "react";
import { render } from "ink";
import { createChat } from "../chat.js";
import { App } from "../ui/app.js";

const DEFAULT_CONFIG = {
  generation: {
    maxTokens: 150,
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.2,
  },
  contextWindowSize: 5,
  enableHistory: true,
};

function printUsage(): void {
  console.log(`
Qubit AI — quantum-inspired chat CLI

Usage:
  qbnn                   Launch the interactive chat interface
  qbnn "your question"   Ask a single question and exit
  qbnn --help            Show this help

Inside the chat, type /help to list commands.
`);
}

/**
 * One-shot mode: answer a single query without entering the TUI.
 */
async function singleQuery(query: string): Promise<void> {
  const chat = await createChat(DEFAULT_CONFIG);
  try {
    const start = Date.now();
    const response = await chat.sendMessage(query);
    const duration = Date.now() - start;
    console.log(`\n› You: ${query}`);
    console.log(`⏺ Qubit: ${response}`);
    console.log(`\x1b[2m  generated in ${duration}ms\x1b[0m`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`\x1b[31m✖ Failed: ${message}\x1b[0m`);
    if (message.includes("HF_TOKEN")) {
      console.error("  Set your token: export HF_TOKEN='hf_...'");
    }
    process.exit(1);
  }
}

/**
 * Interactive mode: render the full-screen Ink chat application.
 */
async function interactive(): Promise<void> {
  const chat = await createChat(DEFAULT_CONFIG);

  const { waitUntilExit } = render(
    <App
      chat={chat}
      onExit={(count) => {
        // Restore a clean prompt after the TUI tears down.
        process.stdout.write(
          `\n\x1b[32m✓ Chat session ended (${Math.round(count)} messages).\x1b[0m\n`
        );
      }}
    />
  );

  await waitUntilExit();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.includes("--help") || args.includes("-h")) {
    printUsage();
    return;
  }

  const query = args.join(" ").trim();

  if (query) {
    await singleQuery(query);
  } else {
    await interactive();
  }
}

main().catch((error) => {
  console.error("Unhandled error:", error);
  process.exit(1);
});
