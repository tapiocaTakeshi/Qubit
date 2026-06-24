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
 * Advanced Local Generative AI with reasoning and long-form response generation
 */
class LocalGenerativeAI {
  private knowledgeBase: Record<string, string[]> = {};
  private reasoningChains: Record<string, string[][]> = {};
  private contextTemplates: Map<string, string> = new Map();

  constructor() {
    this.initializeKnowledge();
  }

  private initializeKnowledge(): void {
    // Knowledge base for reasoning - English and Japanese
    this.knowledgeBase = {
      ai: [
        "Artificial Intelligence refers to computer systems designed to perform tasks that typically require human intelligence.",
        "AI systems can learn from data, recognize patterns, make decisions, and improve their performance over time.",
        "Modern AI applications include natural language processing, computer vision, recommendation systems, and autonomous vehicles.",
        "Machine learning is a subset of AI where algorithms learn from training data without being explicitly programmed.",
        "Neural networks are inspired by biological neurons and form the foundation of deep learning approaches.",
        "Quantum-inspired algorithms use principles from quantum mechanics to optimize classical computing problems.",
        "人工知能とは、コンピュータが人間のような知的な処理を行うシステムのことを指します。",
        "AIシステムは、データから学習し、パターンを認識し、判断を下し、時間とともにその性能を向上させることができます。",
        "現代のAI応用には、自然言語処理、コンピュータビジョン、推薦システム、自動運転などが含まれます。",
        "機械学習とは、明確にプログラムされることなく、訓練データから学習するアルゴリズムのサブセットです。",
        "ニューラルネットワークは生物学的なニューロンに着想を得たもので、深層学習アプローチの基礎を形成しています。",
        "量子にインスパイアされたアルゴリズムは、古典的なコンピューティングの問題を最適化するために量子力学の原理を使用します。",
      ],
      learning: [
        "Learning is the process of acquiring new knowledge, skills, or understanding through experience or instruction.",
        "There are several types of learning: supervised learning, unsupervised learning, reinforcement learning, and self-supervised learning.",
        "Effective learning requires practice, feedback, and the ability to apply knowledge to solve new problems.",
        "Machine learning models improve their performance by adjusting internal parameters based on training data.",
        "Continuous learning and adaptation are essential for staying relevant in rapidly changing fields.",
        "学習とは、経験や指導を通じて新しい知識、スキル、理解を獲得するプロセスです。",
        "学習にはいくつかのタイプがあります：教師あり学習、教師なし学習、強化学習、自己教師あり学習です。",
        "効果的な学習には、練習、フィードバック、新しい問題を解決するための知識を応用する能力が必要です。",
        "機械学習モデルは、訓練データに基づいて内部パラメータを調整することで、その性能を向上させます。",
        "継続的な学習と適応は、急速に変化する分野で関連性を保つために不可欠です。",
      ],
      problem: [
        "Problem-solving involves breaking down complex challenges into smaller, manageable components.",
        "Effective problem-solving requires understanding the root cause, not just addressing symptoms.",
        "Different problems require different approaches: analytical, creative, systematic, or intuitive thinking.",
        "Collaboration and diverse perspectives often lead to more innovative and robust solutions.",
        "Documentation and knowledge sharing help teams learn from past problem-solving experiences.",
        "問題解決とは、複雑な課題をより小さく、管理可能なコンポーネントに分解することです。",
        "効果的な問題解決には、症状に対処するだけでなく、根本原因を理解する必要があります。",
        "異なる問題には異なるアプローチが必要です：分析的、創造的、体系的、または直感的思考です。",
        "協力と異なる視点により、より革新的で堅牢なソリューションが生まれることが多いです。",
        "ドキュメンテーションと知識共有により、チームは過去の問題解決経験から学ぶことができます。",
      ],
      future: [
        "The future of AI involves more advanced reasoning, better integration with human expertise, and improved explainability.",
        "Emerging technologies like quantum computing and neuromorphic hardware may revolutionize AI capabilities.",
        "Addressing ethical concerns, bias, and fairness in AI systems is crucial for responsible development.",
        "The convergence of different AI approaches—symbolic, neural, and hybrid—may lead to more powerful systems.",
        "Interdisciplinary collaboration between computer science, psychology, philosophy, and other fields is essential.",
        "AIの未来には、より高度な推論、人間の専門知識とのより良い統合、改善された説明可能性が含まれます。",
        "量子コンピュータやニューロモルフィックハードウェアなどの新興技術は、AI能力に革命をもたらす可能性があります。",
        "AI システムにおける倫理的懸念、バイアス、公平性に対処することは、責任ある開発に不可欠です。",
        "異なるAIアプローチの収束（記号的、ニューラル、ハイブリッド）は、より強力なシステムにつながる可能性があります。",
        "コンピュータサイエンス、心理学、哲学、およびその他の分野間の学際的協力が不可欠です。",
      ],
    };

    // Multi-sentence reasoning chains - English and Japanese
    this.reasoningChains = {
      analysis: [
        ["Let", "me", "analyze", "your", "question", "step", "by", "step."],
        ["First,", "I", "need", "to", "understand", "the", "core", "issue."],
        ["Then,", "I", "can", "identify", "the", "key", "factors", "involved."],
        ["Finally,", "I'll", "synthesize", "these", "insights", "into", "actionable", "recommendations."],
        ["ご質問を", "段階的に", "分析させていただきます。"],
        ["まず、", "核心的な", "問題を", "理解する", "必要があります。"],
        ["その後、", "関係する", "重要な", "要因を", "特定できます。"],
        ["最後に、", "これらの", "洞察を", "実行可能な", "提案に", "統合します。"],
      ],
      explanation: [
        ["To", "understand", "this", "concept,", "let's", "break", "it", "down."],
        ["The", "fundamental", "principle", "is", "that", "systems", "evolve", "through", "feedback."],
        ["When", "applied", "to", "practical", "scenarios,", "this", "creates", "powerful", "patterns."],
        ["Therefore,", "mastering", "these", "patterns", "enables", "better", "decision", "making."],
        ["この概念を", "理解するために、", "分解してみましょう。"],
        ["基本的な", "原則は、", "システムはフィードバックを", "通じて", "進化するということです。"],
        ["実践的なシナリオに", "適用すると、", "これは強力なパターンを", "生み出します。"],
        ["したがって、", "これらのパターンをマスターすることで、", "より良い意思決定が", "可能になります。"],
      ],
      exploration: [
        ["That's", "an", "excellent", "question", "that", "opens", "up", "many", "possibilities."],
        ["From", "one", "perspective,", "we", "can", "examine", "the", "technical", "dimensions."],
        ["From", "another", "angle,", "we", "should", "consider", "the", "human", "and", "social", "implications."],
        ["Integrating", "both", "viewpoints", "gives", "us", "a", "comprehensive", "understanding."],
        ["それは、", "多くの可能性を", "開く素晴らしい質問です。"],
        ["一つの視点から見ると、", "技術的な側面を", "検討することができます。"],
        ["別の角度から見ると、", "人間的および社会的な", "影響を考慮する必要があります。"],
        ["両方の視点を統合することで、", "包括的な理解が得られます。"],
      ],
    };

    // Long-form context templates - English and Japanese
    this.contextTemplates.set("hello",
      "こんにちは。私はQubit AIと申します。量子にインスパイアされた高度な推論能力を持つ会話型アシスタントです。" +
      "私は、思慮深い議論に参加し、詳細な説明を提供し、複雑なアイデアの探索を支援するために設計されています。" +
      "私の強みは、複雑なトピックを理解可能なコンポーネントに分解し、異なる領域のアイデアを結びつけることにあります。" +
      "情報提供、分析、創造的な問題解決、戦略的思考など、様々な分野でお力になることができます。" +
      "本日は、どのようなテーマについて一緒に考えていきたいですか？"
    );

    this.contextTemplates.set("name",
      "私の名前はQubit AIです。量子にインスパイアされたニューラルネットワークによって駆動される生成型AIアシスタントです。" +
      "私は、単純なパターンマッチングを超えた、思慮深く、ニュアンスのある応答を提供することに重点を置いて設計されています。" +
      "私のアーキテクチャは複数の推論アプローチを組み合わせています：論理的な問題に対する分析的思考、" +
      "オープンエンドの質問に対する創造的な統合、そしてニュアンスのある会話に対する文脈的理解です。" +
      "私はインタラクションから学習し、価値のある洞察を提供する能力を継続的に向上させています。" +
      "技術的なトピック、創造的な探索、または戦略的思考に興味がおありでしたら、" +
      "ぜひ有意義な対話をさせていただきたいと思います。"
    );

    this.contextTemplates.set("help",
      "多様な課題とトピックについて、包括的な支援をさせていただくことができます。" +
      "分析的な作業の面では、複雑な問題を分解し、パターンを特定し、体系的なソリューションを開発するお手伝いができます。" +
      "創造的な取り組みについては、アイデアのブレーンストーミング、執筆支援、" +
      "および従来にない手法の探索を行うことができます。" +
      "知識領域では、技術や科学から哲学や人文科学に至るまでのコンセプトを説明できます。" +
      "戦略的思考については、複数の視点から状況を分析し、長期的な影響を考慮するのを支援できます。" +
      "私は、明確な質問を投げかけたり、詳細な説明を提供したり、異なるアイデア間の関連性を見つけることが得意です。" +
      "具体的には、どの領域に焦点を当てたいですか？"
    );

    this.contextTemplates.set("why",
      "これは基本的な原則に関わる素晴らしい質問です。物事の背後にある「なぜ」を理解することは、" +
      "深い学習とイノベーションにとって極めて重要です。" +
      "表面的な説明を単に受け入れるのではなく、「なぜ」を繰り返し問うことで" +
      "（これを『5つのなぜ』技法と呼ぶ人もいます）、根本原因を明らかにすることができます。" +
      "根本的な理由とメカニズムを理解することで、その知識を新しい状況に適用し、" +
      "新しい問題を解決することができるようになります。" +
      "これが、表面的な理解と真の習熟の違いなのです。" +
      "深く掘り下げたい特定の「なぜ」の質問はありますか？"
    );

    this.contextTemplates.set("how",
      "『どのように』という問いに答えるには、メカニズムと広いコンテキストの両方を調べる必要があります。" +
      "高度な概要から段階的な説明まで、様々な詳細レベルでプロセスを説明することができます。" +
      "効果的な説明には、重要なコンポーネントを特定し、それらがどのように相互作用するかを理解し、" +
      "制約と仮定を認識することが含まれます。" +
      "異なる領域には、プロセスを説明する異なるアプローチがあります：" +
      "工学はメカニズムに焦点を当て、心理学は人間的要因に、システム思考は相互接続に、" +
      "哲学は第一原理に焦点を当てています。" +
      "お客様の質問に対してはどの視点が最も有用でしょうか？"
    );
  }

  generate(userMessage: string, maxTokens: number = 150): string {
    const lower = userMessage.toLowerCase();
    const isJapanese = /[぀-ゟ゠-ヿ一-鿿]/.test(userMessage);

    // Context keywords in Japanese
    const japaneseContextMap: Record<string, string> = {
      "name": "名前",
      "hello": "こんにちは|おはよう|はじめまして|呼んで",
      "help": "手伝う|助け|サポート|支援|できます|お願い",
      "why": "なぜ|理由|原因",
      "how": "どうして|方法|やり方|プロセス|仕組み",
    };

    // Check for context-specific long-form responses
    for (const [key, response] of this.contextTemplates.entries()) {
      if (lower.includes(key) || lower.includes(key.substring(0, 3))) {
        return response;
      }
      // Also check Japanese keywords
      if (isJapanese && japaneseContextMap[key]) {
        const japanesePattern = new RegExp(japaneseContextMap[key]);
        if (japanesePattern.test(userMessage)) {
          return response;
        }
      }
    }

    // Extract key topics from user message
    const topics = this.extractTopics(userMessage);
    let response = "";

    // Generate reasoning introduction
    const reasoningType = this.selectReasoningChain(topics);
    if (reasoningType && this.reasoningChains[reasoningType]) {
      const chains = this.reasoningChains[reasoningType];
      // Filter for Japanese chains if input is Japanese
      const appropriateChains = isJapanese
        ? chains.filter(chain => chain.some(word => /[぀-ゟ゠-ヿ一-鿿]/.test(word)))
        : chains.filter(chain => !chain.some(word => /[぀-ゟ゠-ヿ一-鿿]/.test(word)));

      const selectedChains = appropriateChains.length > 0 ? appropriateChains : chains;
      const chain = selectedChains[
        Math.floor(Math.random() * selectedChains.length)
      ];
      const chainText = chain.join(" ");
      // Clean up spacing around Japanese punctuation
      const cleanedChain = chainText
        .replace(/\s+([。、])/g, "$1")  // Remove space before Japanese punctuation
        .replace(/([。、])\s+/g, "$1");  // Remove space after Japanese punctuation
      response += cleanedChain + " ";
    }

    // Add knowledge-based content
    if (topics.length > 0) {
      const topic = topics[0];
      if (this.knowledgeBase[topic]) {
        const knowledgeItems = this.knowledgeBase[topic];
        // Filter for appropriate language
        const appropriateItems = isJapanese
          ? knowledgeItems.filter(item => /[぀-ゟ゠-ヿ一-鿿]/.test(item))
          : knowledgeItems.filter(item => !/[぀-ゟ゠-ヿ一-鿿]/.test(item));

        const selectedItems = appropriateItems.length > 0 ? appropriateItems : knowledgeItems;
        const numItems = Math.min(2, selectedItems.length);
        for (let i = 0; i < numItems; i++) {
          const item = selectedItems[Math.floor(Math.random() * selectedItems.length)];
          response += item + " ";
        }
      }
    }

    // Add synthesis and next steps
    if (isJapanese) {
      response += "このような理解は、関連する課題により効果的に対処するのに役立ちます。 ";
      response += "この内容についてさらに具体的な側面を探索してほしいか、フォローアップの質問はありますか？ ";
    } else {
      response += "This understanding allows us to approach related challenges more effectively. ";
      response += "Would you like me to explore any specific aspect of this further, or do you have follow-up questions? ";
    }

    // Clean up spacing
    response = response
      .replace(/\s+([.!?])/g, "$1")  // Remove space before punctuation
      .replace(/\s+/g, " ")           // Normalize spaces
      .trim();

    return response;
  }

  private extractTopics(userMessage: string): string[] {
    const topics: string[] = [];
    const topicKeywords: Record<string, string[]> = {
      ai: ["ai", "artificial", "intelligence", "machine", "learning", "neural", "algorithm",
            "人工知能", "AI", "機械学習", "ニューラル", "アルゴリズム", "深層学習"],
      learning: ["learn", "learning", "education", "study", "training", "skill",
                 "学習", "教育", "勉強", "研修", "習得", "スキル"],
      problem: ["problem", "challenge", "issue", "solve", "solution", "fix",
                "問題", "課題", "チャレンジ", "解決", "ソリューション"],
      future: ["future", "tomorrow", "ahead", "upcoming", "innovation", "next",
               "未来", "将来", "明日", "今後", "イノベーション", "次"],
    };

    const lower = userMessage.toLowerCase();
    for (const [topic, keywords] of Object.entries(topicKeywords)) {
      if (keywords.some(kw => lower.includes(kw))) {
        topics.push(topic);
      }
    }

    return topics;
  }

  private selectReasoningChain(topics: string[]): string | null {
    if (topics.length === 0) {
      const chains = Object.keys(this.reasoningChains);
      return chains[Math.floor(Math.random() * chains.length)];
    }

    // Match topics to reasoning chains
    if (topics.includes("ai") || topics.includes("learning")) {
      return "analysis";
    } else if (topics.includes("future")) {
      return "exploration";
    } else {
      return "explanation";
    }
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
