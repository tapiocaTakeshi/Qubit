#!/usr/bin/env node
/**
 * gemma-qbnn-frontal CLI チャットbot
 */

import readline from "node:readline";
import { GemmaQBNNChatbot } from "./chatbot";

interface CliOptions {
  debug: boolean;
  json: boolean;
  help: boolean;
  prompt?: string;
}

function parseArgs(argv: string[]): CliOptions {
  const promptParts: string[] = [];
  const options: CliOptions = {
    debug: false,
    json: false,
    help: false,
  };

  for (const arg of argv) {
    if (arg === "--debug") {
      options.debug = true;
    } else if (arg === "--json") {
      options.json = true;
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else {
      promptParts.push(arg);
    }
  }

  if (promptParts.length > 0) {
    options.prompt = promptParts.join(" ");
  }

  return options;
}

function printHelp(): void {
  console.log(`Gemma + QBNN チャットbot\n\n使い方:\n  gemma-qbnn-chatbot                  対話モードで起動\n  gemma-qbnn-chatbot "相談内容"       1回だけ推論して終了\n  gemma-qbnn-chatbot --json "相談内容" JSONで推論結果を出力\n\nオプション:\n  --debug  QBNN判定、スコア、検出課題を応答に表示\n  --json   ワンショット推論結果をJSONで出力\n  --help   ヘルプを表示\n\n対話モードのコマンド:\n  /exit   終了\n  /reset  会話履歴をリセット`);
}

async function runOneShot(prompt: string, bot: GemmaQBNNChatbot, json: boolean): Promise<void> {
  const turn = await bot.infer(prompt);

  if (json) {
    console.log(JSON.stringify(turn, null, 2));
    return;
  }

  console.log(turn.assistant);
}

function runInteractive(bot: GemmaQBNNChatbot): void {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "あなた> ",
  });
  let isClosed = false;

  console.log("Gemma + QBNN チャットbot");
  console.log("終了するには /exit、履歴を消すには /reset と入力してください。\n");
  rl.prompt();

  rl.on("line", async (line) => {
    const input = line.trim();

    if (["/exit", "exit", "quit"].includes(input.toLowerCase())) {
      rl.close();
      return;
    }

    if (input === "/reset") {
      bot.reset();
      console.log("Bot> 会話履歴をリセットしました。");
      rl.prompt();
      return;
    }

    if (!input) {
      rl.prompt();
      return;
    }

    try {
      const turn = await bot.send(input);
      console.log(`Bot> ${turn.assistant}\n`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`Bot> エラー: ${message}`);
    } finally {
      if (!isClosed) {
        rl.prompt();
      }
    }
  });

  rl.on("close", () => {
    isClosed = true;
    console.log("\nGemma + QBNN チャットbotを終了します。");
  });
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    printHelp();
    return;
  }

  const bot = new GemmaQBNNChatbot({
    showDiagnostics: options.debug,
  });

  if (options.prompt) {
    await runOneShot(options.prompt, bot, options.json);
    return;
  }

  runInteractive(bot);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`エラー: ${message}`);
  process.exitCode = 1;
});
