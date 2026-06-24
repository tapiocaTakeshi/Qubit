/**
 * Chat functionality using Qubit AI
 */

import { NeuroQuantumClient, HFDatasetLoader } from "qubit_ai";
import type {
  ChatMessage,
  ChatSession,
  GenerationConfig,
  ChatConfig,
} from "./types.js";

const DEFAULT_GENERATION_CONFIG: GenerationConfig = {
  maxTokens: 150,
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.2,
};

const DEFAULT_CHAT_CONFIG: ChatConfig = {
  generation: DEFAULT_GENERATION_CONFIG,
  contextWindowSize: 5,
  enableHistory: true,
};

export class QubitAIChat {
  private client: NeuroQuantumClient;
  private session: ChatSession;
  private config: ChatConfig;
  private fewShotExamples: Array<{ prompt: string; completion: string }> = [];

  constructor(config: Partial<ChatConfig> = {}) {
    const hfToken =
      process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || "";

    this.client = new NeuroQuantumClient({
      hfToken,
      timeoutMs: 60000,
      maxRetries: 3,
    });

    this.config = {
      ...DEFAULT_CHAT_CONFIG,
      ...config,
    };

    this.session = {
      id: this.generateSessionId(),
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  /**
   * Add few-shot examples to improve generation quality
   */
  async loadFewShotExamples(
    datasetName?: string,
    numExamples: number = 3
  ): Promise<void> {
    try {
      if (datasetName) {
        const loader = new HFDatasetLoader({
          hfToken: process.env.HF_TOKEN,
        });

        const examples = await loader.preview(datasetName, numExamples);
        this.fewShotExamples = examples;
        console.log(`📚 Loaded ${examples.length} examples from dataset`);
      }
    } catch (error) {
      console.warn(
        "⚠️  Could not load dataset examples:",
        error instanceof Error ? error.message : error
      );
    }
  }

  /**
   * Send a message and get a response
   */
  async sendMessage(userMessage: string): Promise<string> {
    // Add user message to history
    const userMsg: ChatMessage = {
      role: "user",
      content: userMessage,
      timestamp: new Date(),
    };
    this.session.messages.push(userMsg);

    try {
      // Build context from conversation history
      const context = this.buildContext();

      // Generate response
      const response = await this.generateResponse(userMessage, context);

      // Add assistant message to history
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: response,
        timestamp: new Date(),
      };
      this.session.messages.push(assistantMsg);

      // Update session
      this.session.updatedAt = new Date();

      return response;
    } catch (error) {
      // Remove user message if generation failed
      this.session.messages.pop();
      throw error;
    }
  }

  /**
   * Generate response using NeuroQuantum
   */
  private async generateResponse(
    userMessage: string,
    context: string
  ): Promise<string> {
    const prompt = context ? `${context}\nUser: ${userMessage}\nAssistant:` : `User: ${userMessage}\nAssistant:`;

    const result = await this.client.generateWithExamples(
      prompt,
      this.fewShotExamples,
      {
        numExamples: Math.min(3, this.fewShotExamples.length),
        exampleTemplate: "{prompt}",
        queryTemplate: "{prompt}",
        maxNewTokens: this.config.generation.maxTokens,
        temperature: this.config.generation.temperature,
        topK: this.config.generation.topK,
        topP: this.config.generation.topP,
        repetitionPenalty: this.config.generation.repetitionPenalty,
      }
    );

    // Extract just the response part
    const response = result.generatedText
      .split("\n")[0]
      .trim();

    return response || "I'm not sure how to respond to that.";
  }

  /**
   * Build context from recent messages
   */
  private buildContext(): string {
    if (!this.config.enableHistory || this.session.messages.length === 0) {
      return "";
    }

    const recentMessages = this.session.messages.slice(
      -this.config.contextWindowSize * 2
    );

    return recentMessages
      .map((msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`)
      .join("\n");
  }

  /**
   * Get conversation history
   */
  getHistory(): ChatMessage[] {
    return [...this.session.messages];
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.session.messages = [];
    this.session.updatedAt = new Date();
  }

  /**
   * Get session info
   */
  getSessionInfo(): ChatSession {
    return {
      ...this.session,
      messages: [...this.session.messages],
    };
  }

  /**
   * Update generation config
   */
  updateConfig(config: Partial<GenerationConfig>): void {
    this.config.generation = {
      ...this.config.generation,
      ...config,
    };
  }

  /**
   * Export conversation to JSON
   */
  exportConversation(): string {
    return JSON.stringify(
      {
        session: {
          id: this.session.id,
          createdAt: this.session.createdAt.toISOString(),
          updatedAt: this.session.updatedAt.toISOString(),
          messageCount: this.session.messages.length,
        },
        config: this.config,
        messages: this.session.messages.map((msg) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp.toISOString(),
        })),
      },
      null,
      2
    );
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Factory function to create chat instance
 */
export async function createChat(
  config?: Partial<ChatConfig>
): Promise<QubitAIChat> {
  const chat = new QubitAIChat(config);

  // Try to load few-shot examples
  try {
    await chat.loadFewShotExamples("llm-jp/oasst2-33k-ja", 2);
  } catch {
    // Continue without examples if loading fails
  }

  return chat;
}
