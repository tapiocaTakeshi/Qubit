/**
 * Advanced Chat - Long-form reasoning and code generation
 * Extended version with Gemma + QBNN for complex tasks
 */

import { NeuroQuantumClient } from "qubit_ai";

export interface AdvancedChatConfig {
  model: "qbnn" | "gemma-qbnn" | "gemma-large";
  maxTokens: number;
  temperature: number;
  enableLongForm: boolean;
  enableCodeGeneration: boolean;
}

export interface GenerationMode {
  type: "conversation" | "longform" | "code" | "reasoning" | "analysis";
  description: string;
}

const GENERATION_MODES: Record<string, GenerationMode> = {
  conversation: {
    type: "conversation",
    description: "通常の対話",
  },
  longform: {
    type: "longform",
    description: "長文生成（エッセイ、説明、記事など）",
  },
  code: {
    type: "code",
    description: "コード生成（Python、TypeScript、JavaScriptなど）",
  },
  reasoning: {
    type: "reasoning",
    description: "複雑な推論（数学、論理、問題解決）",
  },
  analysis: {
    type: "analysis",
    description: "深い分析と洞察",
  },
};

export class AdvancedChat {
  private client: NeuroQuantumClient;
  private config: AdvancedChatConfig;
  private conversationHistory: Array<{ role: string; content: string }> = [];

  constructor(config: Partial<AdvancedChatConfig> = {}) {
    const hfToken = process.env.HF_TOKEN || "";

    this.client = new NeuroQuantumClient({
      hfToken,
      timeoutMs: 120000, // 2分タイムアウト（長文生成用）
      maxRetries: 5,
    });

    this.config = {
      model: "qbnn",
      maxTokens: 500, // より長い出力
      temperature: 0.7,
      enableLongForm: true,
      enableCodeGeneration: true,
      ...config,
    };
  }

  /**
   * Generate response based on mode
   */
  async generateWithMode(
    userInput: string,
    mode: GenerationMode
  ): Promise<string> {
    const prompt = this.buildPromptForMode(userInput, mode);

    const result = await this.client.generateWithExamples(
      prompt,
      this.getFewShotExamplesForMode(mode),
      {
        maxNewTokens: this.config.maxTokens,
        temperature: this.adjustTemperatureForMode(mode),
        topK: 50,
        topP: 0.95,
        repetitionPenalty: 1.2,
      }
    );

    this.conversationHistory.push({
      role: "user",
      content: userInput,
    });

    this.conversationHistory.push({
      role: "assistant",
      content: result.generatedText,
    });

    return result.generatedText;
  }

  /**
   * Build context-aware prompt
   */
  private buildPromptForMode(userInput: string, mode: GenerationMode): string {
    let systemPrompt = "";

    switch (mode.type) {
      case "longform":
        systemPrompt = `You are an expert writer. Provide detailed, well-structured explanations with multiple paragraphs.
Focus on:
- Clear introduction
- Detailed body with examples
- Logical flow and connections
- Comprehensive conclusion`;
        break;

      case "code":
        systemPrompt = `You are an expert programmer. Write clean, efficient, well-documented code.
Include:
- Clear function/class definitions
- Helpful comments
- Error handling
- Usage examples`;
        break;

      case "reasoning":
        systemPrompt = `You are a logical thinker. Solve problems step by step.
For each problem:
1. Understand what is asked
2. Identify key information
3. Work through the logic
4. Verify your solution
5. Explain your reasoning`;
        break;

      case "analysis":
        systemPrompt = `You are an analytical expert. Provide deep insights and analysis.
Include:
- Multiple perspectives
- Critical thinking
- Evidence-based reasoning
- Implications and recommendations`;
        break;

      default:
        systemPrompt = "You are a helpful AI assistant.";
    }

    // Include recent context
    const context = this.buildContext();

    return `${systemPrompt}\n\n${context}\n\nUser: ${userInput}\n\nAssistant:`;
  }

  /**
   * Get few-shot examples for mode
   */
  private getFewShotExamplesForMode(
    mode: GenerationMode
  ): Array<{ prompt: string; completion: string }> {
    switch (mode.type) {
      case "code":
        return [
          {
            prompt:
              "Write a Python function to calculate fibonacci numbers with memoization",
            completion: `def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Usage example
print(fibonacci(10))  # Output: 55`,
          },
          {
            prompt: "Write TypeScript code for a simple REST API using Express",
            completion: `import express, { Request, Response } from 'express';

const app = express();
app.use(express.json());

interface User {
  id: number;
  name: string;
  email: string;
}

const users: User[] = [];

app.get('/users', (req: Request, res: Response) => {
  res.json(users);
});

app.post('/users', (req: Request, res: Response) => {
  const user: User = {
    id: users.length + 1,
    ...req.body
  };
  users.push(user);
  res.status(201).json(user);
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});`,
          },
        ];

      case "longform":
        return [
          {
            prompt: "Explain how machine learning works",
            completion: `Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

Foundation: Machine learning is based on the principle that systems can learn from data,
identify patterns, and make decisions with minimal human intervention.

Key Components:
1. Data: The foundation of all machine learning systems
2. Features: Relevant characteristics extracted from data
3. Algorithm: The mathematical model that learns from data
4. Training: The process of fitting the algorithm to data
5. Evaluation: Measuring performance on unseen data

Learning Paradigms:
- Supervised Learning: Learning from labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error

Applications span healthcare, finance, transportation, and more,
fundamentally transforming how we solve complex problems.`,
          },
        ];

      case "reasoning":
        return [
          {
            prompt: "If a train travels at 100 km/h for 2 hours, then 150 km/h for 3 hours, what is the average speed?",
            completion: `Step 1: Calculate distance for first segment
Distance₁ = Speed × Time = 100 km/h × 2 h = 200 km

Step 2: Calculate distance for second segment
Distance₂ = Speed × Time = 150 km/h × 3 h = 450 km

Step 3: Calculate total distance
Total Distance = 200 km + 450 km = 650 km

Step 4: Calculate total time
Total Time = 2 h + 3 h = 5 h

Step 5: Calculate average speed
Average Speed = Total Distance / Total Time = 650 km / 5 h = 130 km/h

Answer: The average speed is 130 km/h.`,
          },
        ];

      default:
        return [];
    }
  }

  /**
   * Adjust temperature based on generation mode
   */
  private adjustTemperatureForMode(mode: GenerationMode): number {
    switch (mode.type) {
      case "code":
      case "reasoning":
        return 0.3; // More deterministic
      case "longform":
      case "analysis":
        return 0.7; // Balanced
      default:
        return 0.7;
    }
  }

  /**
   * Build conversation context
   */
  private buildContext(): string {
    if (this.conversationHistory.length === 0) {
      return "";
    }

    const recent = this.conversationHistory.slice(-6);
    return recent
      .map((msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`)
      .join("\n\n");
  }

  /**
   * Get available modes
   */
  getAvailableModes(): GenerationMode[] {
    return Object.values(GENERATION_MODES);
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
}

/**
 * Code generation specialist
 */
export class CodeGenerator {
  private chat: AdvancedChat;

  constructor(config?: Partial<AdvancedChatConfig>) {
    this.chat = new AdvancedChat({
      ...config,
      maxTokens: 800, // Longer for code
      enableCodeGeneration: true,
    });
  }

  /**
   * Generate code in specified language
   */
  async generateCode(
    description: string,
    language: string = "Python"
  ): Promise<string> {
    const prompt = `Generate ${language} code for: ${description}`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.code);
  }

  /**
   * Explain code
   */
  async explainCode(code: string): Promise<string> {
    const prompt = `Explain this ${code.split("\n")[0].includes("def") ? "Python" : "JavaScript"} code in detail:\n\n${code}`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.analysis);
  }

  /**
   * Optimize code
   */
  async optimizeCode(code: string): Promise<string> {
    const prompt = `Optimize this code for performance and readability:\n\n${code}\n\nProvide the optimized version with explanations.`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.code);
  }
}

/**
 * Reasoning specialist
 */
export class ReasoningEngine {
  private chat: AdvancedChat;

  constructor(config?: Partial<AdvancedChatConfig>) {
    this.chat = new AdvancedChat({
      ...config,
      maxTokens: 600,
    });
  }

  /**
   * Solve mathematical problem
   */
  async solveMath(problem: string): Promise<string> {
    const prompt = `Solve this mathematical problem step by step:\n${problem}`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.reasoning);
  }

  /**
   * Analyze logical problem
   */
  async analyzeLogic(problem: string): Promise<string> {
    const prompt = `Analyze this logical problem:\n${problem}`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.reasoning);
  }

  /**
   * Generate explanations
   */
  async explain(topic: string): Promise<string> {
    const prompt = `Provide a comprehensive explanation of: ${topic}`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.longform);
  }
}

/**
 * Content writer specialist
 */
export class ContentWriter {
  private chat: AdvancedChat;

  constructor(config?: Partial<AdvancedChatConfig>) {
    this.chat = new AdvancedChat({
      ...config,
      maxTokens: 1000, // Longest for content
    });
  }

  /**
   * Write article
   */
  async writeArticle(topic: string, wordCount: number = 500): Promise<string> {
    const prompt = `Write a comprehensive article about "${topic}" (approximately ${wordCount} words).
Include introduction, body, and conclusion.`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.longform);
  }

  /**
   * Write blog post
   */
  async writeBlogPost(title: string): Promise<string> {
    const prompt = `Write an engaging blog post titled "${title}".
Include:
- Hook introduction
- Main points with examples
- Personal insights
- Call to action`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.longform);
  }

  /**
   * Analyze topic
   */
  async analyzeTopicDeeply(topic: string): Promise<string> {
    const prompt = `Provide a deep analysis of: ${topic}
Consider multiple perspectives, implications, and future outlook.`;
    return await this.chat.generateWithMode(prompt, GENERATION_MODES.analysis);
  }
}
