/**
 * Chat functionality using Qubit AI - Local Generative Model
 */

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

/**
 * Local generative AI using pattern-based response generation
 */
class LocalGenerativeAI {
  private responsePatterns: Record<string, string[][]> = {};
  private contextTemplates: Map<string, string> = new Map();

  constructor() {
    this.initializePatterns();
  }

  private initializePatterns(): void {
    this.responsePatterns = {
      greeting: [
        ["Hello", "wonderful", "to", "meet", "you!"],
        ["Hi", "there", "how", "can", "I", "help", "you", "today?"],
        ["Greetings", "I'm", "delighted", "to", "assist", "you"],
        ["こんにちは", "何かお手伝いできることはありますか？"],
      ],
      inquiry: [
        ["That's", "an", "interesting", "question", "let", "me", "think", "about", "it"],
        ["I", "appreciate", "that", "question", "here's", "what", "I", "think"],
        ["That", "makes", "sense", "I", "understand", "what", "you're", "asking"],
        ["それは良い質問ですね", "考えてみます"],
      ],
      affirmation: [
        ["Yes", "absolutely", "I", "completely", "agree", "with", "you"],
        ["That's", "a", "great", "point", "I", "support", "that", "idea"],
        ["I", "understand", "and", "I", "think", "you're", "right"],
        ["そうですね", "その通りです", "素晴らしい考えです"],
      ],
      assistance: [
        ["I'd", "be", "happy", "to", "help", "you", "with", "that"],
        ["Of", "course", "I'm", "here", "to", "assist", "you"],
        ["Let", "me", "help", "you", "with", "this", "important", "matter"],
        ["お手伝いさせていただきます", "ぜひお任せください"],
      ],
      closing: [
        ["I", "hope", "that", "helps", "please", "let", "me", "know", "if", "you", "need", "anything", "else"],
        ["Is", "there", "anything", "else", "I", "can", "assist", "you", "with"],
        ["Feel", "free", "to", "ask", "me", "anytime", "I'm", "always", "here"],
        ["何かご不明な点がございましたらお知らせください"],
      ],
    };

    // Context-aware templates
    this.contextTemplates.set("hello", "I'm Qubit AI, a quantum-inspired conversational assistant. I'm here to help you with information, analysis, and creative discussions!");
    this.contextTemplates.set("name", "My name is Qubit AI. I'm a generative AI assistant powered by quantum-inspired neural networks, designed for natural and helpful conversations.");
    this.contextTemplates.set("help", "I can help you with a wide range of tasks including answering questions, analyzing information, brainstorming ideas, writing assistance, and much more. What would you like help with?");
  }

  generate(userMessage: string, maxTokens: number = 25): string {
    const lower = userMessage.toLowerCase();

    // Check for context-specific responses
    for (const [key, response] of this.contextTemplates.entries()) {
      if (lower.includes(key) || lower.includes(key.substring(0, 3))) {
        return response;
      }
    }

    // Detect intent and generate response
    let category = "inquiry";
    if (lower.match(/hello|hi|hey|greet|おはよう|こんにちは|はじめまして/i)) {
      category = "greeting";
    } else if (lower.match(/yes|yeah|right|agree|good|that|that's|その通り|はい|そう/i)) {
      category = "affirmation";
    } else if (lower.match(/help|can you|assist|できます|手伝って|お願い/i)) {
      category = "assistance";
    } else if (lower.match(/thanks|thank|grateful|thx|ありがとう|感謝/i)) {
      category = "closing";
    }

    // Generate response from patterns
    const patterns = this.responsePatterns[category] || this.responsePatterns["inquiry"];
    const selectedPattern = patterns[Math.floor(Math.random() * patterns.length)];

    // Create response by combining words with some randomness
    let response = selectedPattern.join(" ");

    // Apply smoothing and fix formatting
    response = response
      .replace(/\s+([.!?])/g, "$1")  // Remove space before punctuation
      .replace(/\s+([,])/g, "$1")     // Remove space before comma
      .replace(/([.!?])\s+([a-z])/g, "$1 $2");  // Capitalize after punctuation

    return response;
  }
}

export class QubitAIChat {
  private model: LocalGenerativeAI;
  private session: ChatSession;
  private config: ChatConfig;

  constructor(config: Partial<ChatConfig> = {}) {
    this.model = new LocalGenerativeAI();

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
   * Generate response using local language model
   */
  private async generateResponse(
    userMessage: string,
    context: string
  ): Promise<string> {
    try {
      // Create prompt for the model
      const prompt = userMessage;

      // Generate using the local model
      const response = this.model.generate(
        prompt,
        Math.min(this.config.generation.maxTokens, 25)
      );

      return response || "I'm here to help. How can I assist you?";
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Generation failed: ${message}`);
    }
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
  const chat = new QubitAIChat(config);
  return chat;
}
