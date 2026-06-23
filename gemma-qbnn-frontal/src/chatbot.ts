/**
 * Gemma + QBNN チャットbot
 * 会話履歴を保持しながら GemmaQBNNEngine の応答を扱う薄いラッパー
 */

import { GemmaQBNNEngine } from "./engine";
import { ChatMessage, ChatbotConfig, ChatbotTurn, EngineConfig } from "./types";

const DEFAULT_SYSTEM_PROMPT =
  "あなたは Gemma + QBNN ハイブリッド推論で相談に乗る日本語チャットbotです。";

export class GemmaQBNNChatbot {
  private engine: GemmaQBNNEngine;
  private history: ChatMessage[];
  private maxHistory: number;
  private showDiagnostics: boolean;

  constructor(config: ChatbotConfig = {}) {
    const engineConfig: EngineConfig = {
      entangle_strength: config.entangle_strength,
      seed: config.seed,
    };

    this.engine = new GemmaQBNNEngine(engineConfig);
    this.maxHistory = config.maxHistory ?? 12;
    this.showDiagnostics = config.showDiagnostics ?? false;
    this.history = [
      {
        role: "system",
        content: config.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
        timestamp: new Date().toISOString(),
      },
    ];
  }

  async send(userInput: string): Promise<ChatbotTurn> {
    const trimmedInput = userInput.trim();

    if (!trimmedInput) {
      throw new Error("メッセージを入力してください。");
    }

    const contextualInput = this.buildContextualInput(trimmedInput);

    this.history.push({
      role: "user",
      content: trimmedInput,
      timestamp: new Date().toISOString(),
    });

    const result = await this.engine.generate(contextualInput);
    const assistantMessage = this.formatAssistantMessage(result.response, result);

    this.history.push({
      role: "assistant",
      content: assistantMessage,
      timestamp: result.timestamp,
    });
    this.trimHistory();

    return {
      user: trimmedInput,
      assistant: assistantMessage,
      raw_response: result,
      history: this.getHistory(),
    };
  }

  async infer(userInput: string): Promise<ChatbotTurn> {
    return this.send(userInput);
  }

  reset(): void {
    const systemMessage = this.history.find((message) => message.role === "system");
    this.history = systemMessage ? [systemMessage] : [];
  }

  getHistory(): ChatMessage[] {
    return this.history.map((message) => ({ ...message }));
  }

  getInfo(): ReturnType<GemmaQBNNEngine["getInfo"]> {
    return this.engine.getInfo();
  }

  private buildContextualInput(userInput: string): string {
    const recentMessages = this.history
      .filter((message) => message.role !== "system")
      .slice(-this.maxHistory)
      .map((message) => `${message.role === "user" ? "ユーザー" : "Bot"}: ${message.content}`)
      .join("\n");

    if (!recentMessages) {
      return userInput;
    }

    return `以下の会話文脈を踏まえて、最後のユーザー発話に回答してください。\n${recentMessages}\n最後のユーザー発話: ${userInput}`;
  }

  private formatAssistantMessage(
    response: string,
    result: Awaited<ReturnType<GemmaQBNNEngine["generate"]>>
  ): string {
    if (!this.showDiagnostics) {
      return response;
    }

    return `${response}\n\n---\nQBNN: ${result.qbnn_decision} / score=${result.qbnn_score.toFixed(
      1
    )} / confidence=${result.confidence.toFixed(3)}\n課題: ${result.issues_discovered.join(
      ", "
    )}`;
  }

  private trimHistory(): void {
    const systemMessages = this.history.filter((message) => message.role === "system");
    const dialogueMessages = this.history
      .filter((message) => message.role !== "system")
      .slice(-this.maxHistory);

    this.history = [...systemMessages, ...dialogueMessages];
  }
}
