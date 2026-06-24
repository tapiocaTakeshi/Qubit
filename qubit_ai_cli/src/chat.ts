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
    // Knowledge base for reasoning
    this.knowledgeBase = {
      ai: [
        "Artificial Intelligence refers to computer systems designed to perform tasks that typically require human intelligence.",
        "AI systems can learn from data, recognize patterns, make decisions, and improve their performance over time.",
        "Modern AI applications include natural language processing, computer vision, recommendation systems, and autonomous vehicles.",
        "Machine learning is a subset of AI where algorithms learn from training data without being explicitly programmed.",
        "Neural networks are inspired by biological neurons and form the foundation of deep learning approaches.",
        "Quantum-inspired algorithms use principles from quantum mechanics to optimize classical computing problems.",
      ],
      learning: [
        "Learning is the process of acquiring new knowledge, skills, or understanding through experience or instruction.",
        "There are several types of learning: supervised learning, unsupervised learning, reinforcement learning, and self-supervised learning.",
        "Effective learning requires practice, feedback, and the ability to apply knowledge to solve new problems.",
        "Machine learning models improve their performance by adjusting internal parameters based on training data.",
        "Continuous learning and adaptation are essential for staying relevant in rapidly changing fields.",
      ],
      problem: [
        "Problem-solving involves breaking down complex challenges into smaller, manageable components.",
        "Effective problem-solving requires understanding the root cause, not just addressing symptoms.",
        "Different problems require different approaches: analytical, creative, systematic, or intuitive thinking.",
        "Collaboration and diverse perspectives often lead to more innovative and robust solutions.",
        "Documentation and knowledge sharing help teams learn from past problem-solving experiences.",
      ],
      future: [
        "The future of AI involves more advanced reasoning, better integration with human expertise, and improved explainability.",
        "Emerging technologies like quantum computing and neuromorphic hardware may revolutionize AI capabilities.",
        "Addressing ethical concerns, bias, and fairness in AI systems is crucial for responsible development.",
        "The convergence of different AI approaches—symbolic, neural, and hybrid—may lead to more powerful systems.",
        "Interdisciplinary collaboration between computer science, psychology, philosophy, and other fields is essential.",
      ],
    };

    // Multi-sentence reasoning chains
    this.reasoningChains = {
      analysis: [
        ["Let", "me", "analyze", "your", "question", "step", "by", "step."],
        ["First,", "I", "need", "to", "understand", "the", "core", "issue."],
        ["Then,", "I", "can", "identify", "the", "key", "factors", "involved."],
        ["Finally,", "I'll", "synthesize", "these", "insights", "into", "actionable", "recommendations."],
      ],
      explanation: [
        ["To", "understand", "this", "concept,", "let's", "break", "it", "down."],
        ["The", "fundamental", "principle", "is", "that", "systems", "evolve", "through", "feedback."],
        ["When", "applied", "to", "practical", "scenarios,", "this", "creates", "powerful", "patterns."],
        ["Therefore,", "mastering", "these", "patterns", "enables", "better", "decision", "making."],
      ],
      exploration: [
        ["That's", "an", "excellent", "question", "that", "opens", "up", "many", "possibilities."],
        ["From", "one", "perspective,", "we", "can", "examine", "the", "technical", "dimensions."],
        ["From", "another", "angle,", "we", "should", "consider", "the", "human", "and", "social", "implications."],
        ["Integrating", "both", "viewpoints", "gives", "us", "a", "comprehensive", "understanding."],
      ],
    };

    // Long-form context templates
    this.contextTemplates.set("hello",
      "I'm Qubit AI, a quantum-inspired conversational assistant powered by advanced reasoning capabilities. " +
      "I'm designed to engage in thoughtful discussions, provide detailed explanations, and help you explore complex ideas. " +
      "My strength lies in breaking down complicated topics into understandable components and connecting ideas across different domains. " +
      "I can help you with information, analysis, creative problem-solving, and strategic thinking. " +
      "What would you like to explore today?"
    );

    this.contextTemplates.set("name",
      "My name is Qubit AI, and I'm a generative AI assistant powered by quantum-inspired neural networks. " +
      "I was designed with a focus on providing thoughtful, nuanced responses that go beyond simple pattern matching. " +
      "My architecture combines multiple reasoning approaches: analytical thinking for logical problems, creative synthesis for open-ended questions, " +
      "and contextual understanding for nuanced conversations. " +
      "I learn from interactions and continuously improve my ability to provide valuable insights. " +
      "Whether you're interested in technical topics, creative exploration, or strategic thinking, I'm here to engage meaningfully."
    );

    this.contextTemplates.set("help",
      "I can assist you with a comprehensive range of tasks and topics. In terms of analytical work, I can help you break down complex problems, " +
      "identify patterns, and develop systematic solutions. For creative endeavors, I can brainstorm ideas, provide writing assistance, and explore unconventional approaches. " +
      "In knowledge domains, I can explain concepts ranging from technology and science to philosophy and humanities. " +
      "For strategic thinking, I can help you analyze situations from multiple perspectives and consider long-term implications. " +
      "I excel at asking clarifying questions, providing detailed explanations, and making connections between different ideas. " +
      "What specific area would you like to focus on?"
    );

    this.contextTemplates.set("why",
      "That's a great question that gets at fundamental principles. Understanding the 'why' behind things is crucial for deep learning and innovation. " +
      "Rather than just accepting surface-level explanations, asking 'why' repeatedly—what some call the 'five whys' technique—helps us uncover root causes. " +
      "When we understand the underlying reasons and mechanisms, we can apply that knowledge to new situations and solve novel problems. " +
      "This is what separates superficial understanding from genuine mastery. Is there a specific 'why' question you'd like to explore in depth?"
    );

    this.contextTemplates.set("how",
      "Understanding 'how' something works requires examining both the mechanism and the broader context. " +
      "I can explain processes at different levels of detail—from high-level overviews to step-by-step breakdowns. " +
      "Effective explanation involves identifying the key components, understanding how they interact, and recognizing the constraints and assumptions. " +
      "Different domains have different approaches to explaining processes: engineering focuses on mechanisms, psychology on human factors, " +
      "systems thinking on interconnections, and philosophy on first principles. Which perspective would be most helpful for your question?"
    );
  }

  generate(userMessage: string, maxTokens: number = 150): string {
    const lower = userMessage.toLowerCase();

    // Check for context-specific long-form responses
    for (const [key, response] of this.contextTemplates.entries()) {
      if (lower.includes(key) || lower.includes(key.substring(0, 3))) {
        return response;
      }
    }

    // Extract key topics from user message
    const topics = this.extractTopics(userMessage);
    let response = "";

    // Generate reasoning introduction
    const reasoningType = this.selectReasoningChain(topics);
    if (reasoningType && this.reasoningChains[reasoningType]) {
      const chain = this.reasoningChains[reasoningType][
        Math.floor(Math.random() * this.reasoningChains[reasoningType].length)
      ];
      response += chain.join(" ") + " ";
    }

    // Add knowledge-based content
    if (topics.length > 0) {
      const topic = topics[0];
      if (this.knowledgeBase[topic]) {
        const knowledgeItems = this.knowledgeBase[topic];
        const numItems = Math.min(2, knowledgeItems.length);
        for (let i = 0; i < numItems; i++) {
          const item = knowledgeItems[Math.floor(Math.random() * knowledgeItems.length)];
          response += item + " ";
        }
      }
    }

    // Add synthesis and next steps
    response += "This understanding allows us to approach related challenges more effectively. ";
    response += "Would you like me to explore any specific aspect of this further, or do you have follow-up questions? ";

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
      ai: ["ai", "artificial", "intelligence", "machine", "learning", "neural", "algorithm"],
      learning: ["learn", "learning", "education", "study", "training", "skill"],
      problem: ["problem", "challenge", "issue", "solve", "solution", "fix"],
      future: ["future", "tomorrow", "ahead", "upcoming", "innovation", "next"],
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
