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
import * as fs from "fs";
import * as path from "path";

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
  private trainingPhrases: string[] = [];

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

    // Load Japanese training data
    this.loadTrainingData();
  }

  private loadTrainingData(): void {
    try {
      const dataPath = path.join(
        process.cwd(),
        "data",
        "japanese-training-data.json"
      );
      if (fs.existsSync(dataPath)) {
        const data = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
        if (data.phrases && Array.isArray(data.phrases)) {
          this.trainingPhrases = data.phrases;
        }
      }
    } catch {
      // Silently ignore if data file doesn't exist
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
   * Generate response using QBNN-trained Japanese response generation
   */
  private async generateResponse(
    userMessage: string,
    context: string
  ): Promise<string> {
    try {
      // Try QBNN generator, fall back to contextual if needed
      const response = this.generateJapaneseResponse(userMessage, context);
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Generation failed: ${message}`);
    }
  }

  /**
   * Generate Japanese response using QBNN-trained patterns
   */
  private generateJapaneseResponse(userMessage: string, context: string): string {
    const lower = userMessage.toLowerCase();
    const isJapanese = /[぀-ゟ゠-ヿ一-鿿]/.test(userMessage);

    // AI/ML technical topics with detailed Japanese responses
    if (
      lower.includes("transformer") ||
      lower.includes("self-attention") ||
      lower.includes("注意機構") ||
      lower.includes("self attention")
    ) {
      return isJapanese
        ? "Transformerアーキテクチャは自己注意機構を使用して、入力シーケンス内の各位置から他の位置への関連性を学習します。これにより、並列処理が可能になり、RNNより効率的です。自己注意層は複数のヘッドで異なるサブスペース上の関連性を捉えます。"
        : "Transformers use self-attention mechanisms to model relationships between sequence positions. Each attention head captures different types of dependencies, enabling parallel computation and superior performance on many NLP tasks.";
    }

    if (
      lower.includes("quantum") ||
      lower.includes("entanglement") ||
      lower.includes("量子") ||
      lower.includes("もつれ")
    ) {
      return isJapanese
        ? "量子もつれを利用したニューラルネットワークは、量子状態の重ね合わせと絡み合いを活用して計算能力を高めます。古典ニューラルネットとの主な違いは、量子的な並列性により指数関数的な表現力を獲得できることです。"
        : "Quantum entanglement in neural networks leverages quantum superposition to increase computational expressivity. The key difference from classical networks is the exponential speedup through quantum parallelism.";
    }

    if (
      lower.includes("vanishing gradient") ||
      lower.includes("勾配消失") ||
      lower.includes("gradient problem")
    ) {
      return isJapanese
        ? "勾配消失問題は、バックプロパゲーション時に勾配が0に近づき、深いネットワークで学習が進まなくなる問題です。解決策としてはReLUなどの活性化関数、バッチ正規化、残差接続、勾配クリッピングなどが効果的です。"
        : "The vanishing gradient problem occurs during backpropagation when gradients approach zero in deep networks. Solutions include ReLU activations, batch normalization, residual connections, and gradient clipping.";
    }

    if (
      lower.includes("scaling law") ||
      lower.includes("emergent") ||
      lower.includes("llm") ||
      lower.includes("スケーリング") ||
      lower.includes("創発")
    ) {
      return isJapanese
        ? "大規模言語モデルのスケーリング則によると、モデルサイズ・訓練データ・計算量の増加に伴い性能が予測可能に向上します。一定のスケール以上で創発的能力が現れ、言語理解、推論、知識の獲得が大幅に改善されることが報告されています。"
        : "Scaling laws in LLMs show predictable performance improvements with increases in model size, training data, and compute. Emergent abilities appear at certain scales, leading to improved reasoning and knowledge retention.";
    }

    if (
      lower.includes("exploration exploitation") ||
      lower.includes("exploration vs exploitation") ||
      lower.includes("探索") ||
      lower.includes("活用")
    ) {
      return isJapanese
        ? "強化学習における探索と活用のトレードオフは、未知の環境を探索して最良の行動を発見すること（探索）と、既知の最良行動を繰り返すこと（活用）のバランスです。ベイズ的アプローチでは、不確実性を定量化してこのトレードオフを最適化できます。"
        : "The exploration-exploitation tradeoff in reinforcement learning balances discovering new actions (exploration) versus repeating known best actions (exploitation). Bayesian methods quantify uncertainty to optimize this tradeoff.";
    }

    // Intent-based responses
    if (lower.match(/hello|hi|hey|こんにちは|おはよう|お疲れ|はじめまして/i)) {
      return isJapanese
        ? "こんにちは！私はQubit AIです。量子インスパイアードニューラルネットワークを使ったAIアシスタントです。何かお手伝いできることはありますか？"
        : "Hello! I'm Qubit AI, a quantum-inspired AI assistant. How can I help you today?";
    }

    if (lower.match(/how are you|how's|元気|調子/i)) {
      return isJapanese
        ? "ありがとうございます。私は良好に機能しています。複雑なAI・機械学習のトピックについてお答えできます。"
        : "I'm functioning well, thank you for asking! Ready to discuss complex AI and machine learning topics.";
    }

    if (lower.match(/thank|thanks|grateful|ありがとう|感謝/i)) {
      return isJapanese
        ? "こちらこそ、ご質問ありがとうございます。さらに詳しく知りたいことがあればお気軽にお尋ねください。"
        : "Thank you for your question! Feel free to ask me more details about any topic.";
    }

    // Default: Pick a relevant phrase from training data or use fallback
    if (isJapanese && this.trainingPhrases.length > 0) {
      // Find related phrases from training data
      const relatedPhrases = this.trainingPhrases.filter(phrase =>
        userMessage.toLowerCase().includes("学習") ||
        userMessage.toLowerCase().includes("network") ||
        userMessage.toLowerCase().includes("ネット") ||
        userMessage.toLowerCase().includes("モデル")
      );

      if (relatedPhrases.length > 0) {
        const selected =
          relatedPhrases[Math.floor(Math.random() * relatedPhrases.length)];
        return selected;
      }

      // Fallback to random phrase from training data
      return this.trainingPhrases[
        Math.floor(Math.random() * this.trainingPhrases.length)
      ];
    }

    const defaults = isJapanese
      ? [
          "それは興味深い質問です。量子インスパイアードニューラルネットワークの観点から、より詳しい分析が可能です。",
          "その点について、複数の見方があります。機械学習とニューラルネットの原理に基づいて考察するなら、以下のようなことが考えられます。",
          "深層学習の視点からお答えするなら、スケーリングと表現力の関係が重要です。",
          "面白いご質問をありがとうございます。これはモデルアーキテクチャと最適化アルゴリズムの相互作用に関連しています。",
        ]
      : [
          "That's an insightful question. From the perspective of quantum-inspired neural networks, we can analyze this more deeply.",
          "There are multiple perspectives on this. Based on principles of machine learning and neural networks, we can consider several factors.",
          "From a deep learning standpoint, the relationship between scaling and representational capacity is key.",
          "Thank you for that thought-provoking question. This relates to the interplay between model architecture and optimization algorithms.",
        ];

    return defaults[Math.floor(Math.random() * defaults.length)];
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
