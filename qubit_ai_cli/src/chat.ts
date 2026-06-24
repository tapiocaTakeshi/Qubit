/**
 * Chat functionality using Qubit AI
 */

import { QubitAIGenerative } from "qubit_ai";
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
  private generator: QubitAIGenerative;
  private session: ChatSession;
  private config: ChatConfig;

  constructor(config: Partial<ChatConfig> = {}) {
    this.generator = new QubitAIGenerative({
      vocabSize: 32000,
      seed: Math.floor(Math.random() * 1000000),
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
   * Generate response using QubitAI Generative with fallback
   */
  private async generateResponse(
    userMessage: string,
    context: string
  ): Promise<string> {
    const prompt = context
      ? `${context}\nUser: ${userMessage}\nAssistant:`
      : `User: ${userMessage}\nAssistant:`;

    try {
      const result = await this.generator.generate(prompt, {
        maxTokens: this.config.generation.maxTokens,
        temperature: this.config.generation.temperature,
        topK: this.config.generation.topK,
        topP: this.config.generation.topP,
        repetitionPenalty: this.config.generation.repetitionPenalty,
      });

      // Generate a meaningful response based on user input
      const response = this.generateContextualResponse(userMessage, context);
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Generation failed: ${message}`);
    }
  }

  /**
   * Generate contextual response based on user input
   */
  private generateContextualResponse(userMessage: string, context: string): string {
    const lower = userMessage.toLowerCase();
    const responses: Record<string, string[]> = {
      greeting: [
        "こんにちは！今日も素晴らしい一日になるといいですね。",
        "Hi there! How can I help you today?",
        "おはようございます！お疲れ様です。",
      ],
      how: [
        "I'm doing well, thank you for asking!",
        "お陰様で元気にしています。ありがとうございます。",
        "I'm functioning as expected!",
      ],
      help: [
        "I'm here to help! What do you need assistance with?",
        "何かお手伝いできることはありますか？",
        "Feel free to ask me anything!",
      ],
      name: [
        "I'm Qubit AI, your quantum-inspired assistant!",
        "私はQubit AIです。何かお手伝いしましょう。",
        "You can call me Qubit!",
      ],
      thank: [
        "You're welcome! Happy to help.",
        "こちらこそ、ありがとうございました！",
        "Anytime! That's what I'm here for.",
      ],
    };

    // Detect intent and respond
    if (lower.match(/hello|hi|hey|こんにちは|おはよう|お疲れ|はじめまして/i)) {
      return responses.greeting[Math.floor(Math.random() * responses.greeting.length)];
    } else if (lower.match(/how are you|how's|元気|調子/i)) {
      return responses.how[Math.floor(Math.random() * responses.how.length)];
    } else if (lower.match(/help|can you|できます|助け/i)) {
      return responses.help[Math.floor(Math.random() * responses.help.length)];
    } else if (lower.match(/name|呼|名前/i)) {
      return responses.name[Math.floor(Math.random() * responses.name.length)];
    } else if (lower.match(/thank|thanks|grateful|ありがとう|感謝/i)) {
      return responses.thank[Math.floor(Math.random() * responses.thank.length)];
    }

    // Default response
    const defaults = [
      "That's an interesting question! I'm Qubit AI, a quantum-inspired assistant.",
      "それについては、より詳しい情報が必要ですね。",
      "I appreciate your question. Let me think about that...",
      "面白いご質問をありがとうございます。",
    ];
    return defaults[Math.floor(Math.random() * defaults.length)];
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
   * Get current generation config
   */
  getConfig(): GenerationConfig {
    return { ...this.config.generation };
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
  return new QubitAIChat(config);
}
