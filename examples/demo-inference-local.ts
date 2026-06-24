/**
 * Demo: Local Inference Examples (Mock)
 *
 * This demo shows how the inference examples work without requiring
 * an external API or HuggingFace token. It demonstrates the patterns
 * and interfaces used by the real examples.
 */

interface Example {
  prompt: string;
  completion: string;
}

interface GenerateResult {
  generatedText: string;
  raw?: unknown[];
}

// Mock client that simulates the real NeuroQuantumClient
class MockNeuroQuantumClient {
  private examples = new Map<string, string[]>();

  async generateWithExamples(
    prompt: string,
    examples: Example[] = [],
    opts: any = {}
  ): Promise<GenerateResult> {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, Math.random() * 500 + 200));

    // Generate mock response based on prompt
    const responses = this.getMockResponses();
    const response = responses[prompt] || this.generatePlaceholder(prompt);

    return {
      generatedText: response,
      raw: [{ generated_text: response }],
    };
  }

  private getMockResponses(): Record<string, string> {
    return {
      "人工知能は": "様々な分野で急速に応用が進んでおり、生産性向上と新しい価値創造を実現しています。",
      "量子コンピュータとは": "古典的なコンピュータとは異なる計算原理に基づいており、特定の問題において圧倒的な計算速度を実現する可能性を持っています。",
      "機械学習の利点は": "大量のデータから自動的にパターンを学習でき、人間では発見困難な複雑な関係性を明らかにすることができます。",
      "English: Good night\nJapanese:": " おやすみなさい",
      "English: See you tomorrow\nJapanese:": " また明日会いましょう",
      "Text: This product is amazing!\nSentiment:": " Positive",
      "Text: I'm very disappointed\nSentiment:": " Negative",
      "Q: What is the capital of Japan?\nA:": " Tokyo is the capital of Japan.",
      "Q: Who invented the telephone?\nA:": " Alexander Graham Bell invented the telephone.",
      "Text: Machine learning is a subset of AI\nSummary:": " ML is a branch of AI that learns from data automatically.",
      "Code: const doubled = arr.map(x => x * 2);\nExplanation:": " This code doubles each element in an array using the map function.",
    };
  }

  private generatePlaceholder(prompt: string): string {
    const templates = [
      `${prompt}ことで、多くの利点が得られます。`,
      `${prompt}ため、様々な応用が可能です。`,
      `${prompt}という点が重要です。`,
      `${prompt}ため、今後さらに発展が期待されます。`,
    ];
    return templates[Math.floor(Math.random() * templates.length)];
  }
}

// Demo Functions

async function demoBasicGeneration() {
  console.log("🔬 Demo: Basic Text Generation\n");

  const client = new MockNeuroQuantumClient();

  const prompts = [
    "人工知能は",
    "量子コンピュータとは",
    "機械学習の利点は",
  ];

  for (const prompt of prompts) {
    console.log(`📝 Prompt: "${prompt}"`);
    try {
      const result = await client.generateWithExamples(prompt, [], {
        maxNewTokens: 50,
        temperature: 0.7,
      });

      console.log(`✅ Generated: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed: ${error}\n`);
    }
  }
}

async function demoFewShotTranslation() {
  console.log("🌐 Demo: Few-shot Language Translation\n");

  const client = new MockNeuroQuantumClient();

  const examples: Example[] = [
    { prompt: "Hello, how are you?", completion: "こんにちは、お元気ですか？" },
    { prompt: "Good morning", completion: "おはようございます" },
    { prompt: "Thank you very much", completion: "本当にありがとうございました" },
  ];

  const testPrompts = ["Good night", "See you tomorrow"];

  console.log("📚 Examples provided:");
  examples.slice(0, 2).forEach((ex, i) => {
    console.log(`  ${i + 1}. "${ex.prompt}" → "${ex.completion}"`);
  });
  console.log();

  for (const testPrompt of testPrompts) {
    console.log(`📝 English: "${testPrompt}"`);
    try {
      const result = await client.generateWithExamples(
        testPrompt,
        examples,
        {
          numExamples: 2,
          exampleTemplate: "English: {prompt}\nJapanese: {completion}",
          queryTemplate: "English: {prompt}\nJapanese:",
          maxNewTokens: 40,
          temperature: 0.3,
        }
      );

      console.log(`✅ Japanese: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed: ${error}\n`);
    }
  }
}

async function demoSentimentAnalysis() {
  console.log("💭 Demo: Sentiment Analysis\n");

  const client = new MockNeuroQuantumClient();

  const sentimentExamples: Example[] = [
    { prompt: "This product is amazing!", completion: "Positive" },
    { prompt: "I'm very disappointed with this service", completion: "Negative" },
    { prompt: "It's okay, nothing special", completion: "Neutral" },
  ];

  const reviews = [
    "I absolutely love this app!",
    "This is the worst experience ever",
    "It works as expected",
  ];

  console.log("📚 Example sentiments:");
  sentimentExamples.slice(0, 2).forEach((ex, i) => {
    console.log(`  ${i + 1}. "${ex.prompt}" → ${ex.completion}`);
  });
  console.log();

  for (const review of reviews) {
    console.log(`📝 Review: "${review}"`);
    try {
      const result = await client.generateWithExamples(
        review,
        sentimentExamples,
        {
          numExamples: 3,
          exampleTemplate: "Text: {prompt}\nSentiment: {completion}",
          queryTemplate: "Text: {prompt}\nSentiment:",
          maxNewTokens: 20,
          temperature: 0.2,
        }
      );

      console.log(`✅ Sentiment: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed: ${error}\n`);
    }
  }
}

async function demoBatchProcessing() {
  console.log("📦 Demo: Batch Text Processing\n");

  const client = new MockNeuroQuantumClient();

  const prompts = [
    "深層学習の基本は",
    "自然言語処理の課題は",
    "コンピュータビジョンの応用は",
  ];

  const results: Array<{ prompt: string; output: string; time: number }> = [];
  const startTotal = Date.now();

  console.log(`📝 Processing ${prompts.length} prompts...\n`);

  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];
    console.log(`[${i + 1}/${prompts.length}] ${prompt}`);

    try {
      const startTime = Date.now();
      const result = await client.generateWithExamples(prompt, [], {
        maxNewTokens: 50,
        temperature: 0.7,
      });
      const duration = Date.now() - startTime;

      results.push({
        prompt,
        output: result.generatedText,
        time: duration,
      });

      console.log(`  ✅ Done in ${duration}ms\n`);
    } catch (error) {
      console.error(`  ❌ Failed: ${error}\n`);
    }
  }

  const totalDuration = Date.now() - startTotal;

  console.log("📊 Batch Results Summary:");
  console.log("━".repeat(60));
  results.forEach((r, i) => {
    console.log(`\n${i + 1}. Prompt: "${r.prompt}"`);
    console.log(`   Output: "${r.output}"`);
    console.log(`   Time: ${r.time}ms`);
  });

  console.log("\n📈 Statistics:");
  console.log(`  Total items: ${results.length}`);
  console.log(`  Total time: ${totalDuration}ms`);
  if (results.length > 0) {
    const avgTime = results.reduce((sum, r) => sum + r.time, 0) / results.length;
    console.log(`  Average time per item: ${avgTime.toFixed(0)}ms`);
  }
}

async function demoChainOfThought() {
  console.log("🧠 Demo: Chain-of-Thought Reasoning\n");

  const client = new MockNeuroQuantumClient();

  const examples: Example[] = [
    {
      prompt: "If I have 3 apples and get 2 more, how many do I have?",
      completion:
        "Let me think step by step:\n1. I start with 3 apples\n2. I get 2 more apples\n3. Total: 3 + 2 = 5 apples",
    },
  ];

  const problem =
    "If a car travels 60km per hour, how far will it travel in 3.5 hours?";

  console.log(`📝 Problem: "${problem}"\n`);
  console.log("📚 Expecting step-by-step reasoning...\n");

  try {
    const result = await client.generateWithExamples(problem, examples, {
      numExamples: 1,
      exampleTemplate: "Q: {prompt}\n{completion}",
      queryTemplate: "Q: {prompt}\nA: Let me think step by step:",
      maxNewTokens: 120,
      temperature: 0.5,
    });

    console.log("✅ Reasoning:");
    console.log(`A: Let me think step by step:${result.generatedText}\n`);
  } catch (error) {
    console.error(`❌ Failed: ${error}\n`);
  }
}

async function main() {
  console.log("╔════════════════════════════════════════════════════════════╗");
  console.log("║         Qubit AI - Local Inference Demo                   ║");
  console.log("║  (Using Mock Responses - No API Required)                 ║");
  console.log("╚════════════════════════════════════════════════════════════╝\n");

  try {
    await demoBasicGeneration();
    console.log("━".repeat(60) + "\n");

    await demoFewShotTranslation();
    console.log("━".repeat(60) + "\n");

    await demoSentimentAnalysis();
    console.log("━".repeat(60) + "\n");

    await demoBatchProcessing();
    console.log("━".repeat(60) + "\n");

    await demoChainOfThought();

    console.log("✨ All demos completed successfully!\n");
    console.log("💡 Next steps:");
    console.log("  1. Set HF_TOKEN environment variable");
    console.log("  2. Run real examples with actual API:");
    console.log("     npx ts-node examples/qubit-inference-basics.ts");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
