/**
 * Hybrid Chat - Gemma + QBNN Frontal System
 * QBNN provides reasoning/judgment, Gemma provides generation
 */

import { NeuroQuantumClient, HFDatasetLoader } from "qubit_ai";

export interface HybridChatConfig {
  maxTokens: number;
  temperature: number;
  reasoningTemperature: number;
  enableReasoning: boolean;
}

const DEFAULT_HYBRID_CONFIG: HybridChatConfig = {
  maxTokens: 300,
  temperature: 0.6, // Gemma generation
  reasoningTemperature: 0.4, // QBNN reasoning (more logical)
  enableReasoning: true,
};

export class HybridChat {
  private client: NeuroQuantumClient;
  private config: HybridChatConfig;
  private conversationHistory: Array<{ role: string; content: string }> = [];

  constructor(config: Partial<HybridChatConfig> = {}) {
    const hfToken =
      process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN || "";

    this.client = new NeuroQuantumClient({
      hfToken,
      timeoutMs: 120000, // Extended for hybrid processing
      maxRetries: 3,
    });

    this.config = {
      ...DEFAULT_HYBRID_CONFIG,
      ...config,
    };
  }

  /**
   * QBNN Reasoning Phase - Analyze and reason about the input
   */
  private async reasoningPhase(userInput: string): Promise<string> {
    const prompt = `You are a quantum-inspired reasoning system. Analyze this query deeply:

Query: "${userInput}"

Provide concise reasoning about:
1. What the user is really asking
2. Key concepts to address
3. Best approach to answer
4. Important context to consider

Keep reasoning brief and focused.`;

    const result = await this.client.generateWithExamples(prompt, [], {
      maxNewTokens: 150,
      temperature: this.config.reasoningTemperature,
      topK: 30,
      topP: 0.9,
      repetitionPenalty: 1.1,
    });

    return result.generatedText;
  }

  /**
   * Gemma Generation Phase - Generate response based on reasoning
   */
  private async generationPhase(
    userInput: string,
    reasoning: string,
    context: string
  ): Promise<string> {
    const gemmaPrompt = `You are an expert assistant powered by Google Gemma.

User Query: "${userInput}"

Reasoning from analysis:
${reasoning}

${context ? `Previous context:\n${context}\n` : ""}

Provide a clear, helpful, and comprehensive response:`;

    const result = await this.client.generateWithExamples(gemmaPrompt, [], {
      maxNewTokens: this.config.maxTokens,
      temperature: this.config.temperature,
      topK: 50,
      topP: 0.95,
      repetitionPenalty: 1.2,
    });

    return result.generatedText;
  }

  /**
   * Hybrid Generate - QBNN Reasoning + Gemma Generation
   */
  async sendMessage(userInput: string): Promise<{ response: string; reasoning: string }> {
    // Add user message to history
    this.conversationHistory.push({
      role: "user",
      content: userInput,
    });

    try {
      // Phase 1: QBNN Reasoning
      const reasoning = this.config.enableReasoning
        ? await this.reasoningPhase(userInput)
        : "";

      // Phase 2: Gemma Generation
      const context = this.buildContext();
      const response = await this.generationPhase(userInput, reasoning, context);

      // Add to history
      this.conversationHistory.push({
        role: "assistant",
        content: response,
      });

      return { response, reasoning };
    } catch (error) {
      // Remove user message if generation failed
      this.conversationHistory.pop();
      throw error;
    }
  }

  /**
   * Build context from conversation history
   */
  private buildContext(): string {
    if (this.conversationHistory.length <= 1) {
      return "";
    }

    const recent = this.conversationHistory.slice(-4);
    return recent
      .map((msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`)
      .join("\n\n");
  }

  /**
   * Get conversation history
   */
  getHistory(): Array<{ role: string; content: string }> {
    return [...this.conversationHistory];
  }

  /**
   * Clear history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /**
   * Get config
   */
  getConfig(): HybridChatConfig {
    return { ...this.config };
  }

  /**
   * Update config
   */
  updateConfig(config: Partial<HybridChatConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
  }

  /**
   * Export conversation with reasoning
   */
  exportConversation(includeReasoning: boolean = true): string {
    return JSON.stringify(
      {
        type: "hybrid-chat",
        timestamp: new Date().toISOString(),
        config: this.config,
        model: "Gemma (generation) + QBNN (frontal reasoning)",
        messageCount: this.conversationHistory.length,
        messages: this.conversationHistory.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
      },
      null,
      2
    );
  }
}

/**
 * Factory function to create hybrid chat instance
 */
export async function createHybridChat(
  config?: Partial<HybridChatConfig>
): Promise<HybridChat> {
  return new HybridChat(config);
}
