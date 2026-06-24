#!/usr/bin/env node

/**
 * Simple Demo: Qubit AI Inference Examples
 * Run without TypeScript compilation
 */

// Mock client simulation
class MockNeuroQuantumClient {
  async generateWithExamples(prompt, examples = [], opts = {}) {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, Math.random() * 300 + 100));

    const responses = {
      "人工知能は": "様々な分野で急速に応用が進んでおり、生産性向上と新しい価値創造を実現しています。",
      "量子コンピュータとは": "古典的なコンピュータとは異なる計算原理に基づいており、特定の問題において圧倒的な計算速度を実現する可能性を持っています。",
      "機械学習の利点は": "大量のデータから自動的にパターンを学習でき、人間では発見困難な複雑な関係性を明らかにすることができます。",
    };

    return {
      generatedText: responses[prompt] || `${prompt}という点が重要です。`,
      raw: [{ generated_text: responses[prompt] || `${prompt}という点が重要です。` }],
    };
  }
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function demo1_BasicGeneration() {
  console.log("\n🔬 Demo 1: Basic Text Generation\n");
  console.log("━".repeat(60));

  const client = new MockNeuroQuantumClient();

  const prompts = [
    "人工知能は",
    "量子コンピュータとは",
    "機械学習の利点は",
  ];

  for (const prompt of prompts) {
    console.log(`\n📝 Prompt: "${prompt}"`);
    try {
      const startTime = Date.now();
      const result = await client.generateWithExamples(prompt, [], {
        maxNewTokens: 50,
        temperature: 0.7,
      });
      const duration = Date.now() - startTime;

      console.log(`✅ Generated (${duration}ms):`);
      console.log(`   "${result.generatedText}"`);
    } catch (error) {
      console.error(`❌ Failed: ${error.message}`);
    }
  }
}

async function demo2_FewShotTranslation() {
  console.log("\n\n🌐 Demo 2: Few-shot Language Translation\n");
  console.log("━".repeat(60));

  const client = new MockNeuroQuantumClient();

  const examples = [
    { prompt: "Hello, how are you?", completion: "こんにちは、お元気ですか？" },
    { prompt: "Good morning", completion: "おはようございます" },
  ];

  console.log("\n📚 Few-shot Examples:");
  examples.forEach((ex, i) => {
    console.log(`  ${i + 1}. "${ex.prompt}" → "${ex.completion}"`);
  });

  const testPrompts = ["Good afternoon", "Thank you"];

  for (const testPrompt of testPrompts) {
    console.log(`\n📝 English: "${testPrompt}"`);
    try {
      const startTime = Date.now();
      const result = await client.generateWithExamples(
        testPrompt,
        examples,
        {
          numExamples: 2,
          maxNewTokens: 40,
          temperature: 0.3,
        }
      );
      const duration = Date.now() - startTime;

      console.log(`✅ Japanese (${duration}ms):`);
      console.log(`   "様々な翻訳が可能です"`);
    } catch (error) {
      console.error(`❌ Failed: ${error.message}`);
    }
  }
}

async function demo3_SentimentAnalysis() {
  console.log("\n\n💭 Demo 3: Batch Sentiment Analysis\n");
  console.log("━".repeat(60));

  const client = new MockNeuroQuantumClient();

  const sentimentExamples = [
    { prompt: "This product is amazing!", completion: "Positive" },
    { prompt: "I'm very disappointed", completion: "Negative" },
    { prompt: "It's okay", completion: "Neutral" },
  ];

  console.log("\n📚 Sentiment Examples:");
  sentimentExamples.forEach((ex, i) => {
    console.log(`  ${i + 1}. "${ex.prompt}" → ${ex.completion}`);
  });

  const reviews = [
    "I absolutely love this app!",
    "Terrible experience, never again",
    "It works fine",
  ];

  const sentiments = ["Positive", "Negative", "Neutral"];

  console.log("\n🔍 Analyzing reviews:");
  for (let i = 0; i < reviews.length; i++) {
    const review = reviews[i];
    const sentiment = sentiments[i];

    console.log(`\n📝 Review: "${review}"`);
    try {
      const startTime = Date.now();
      await client.generateWithExamples(review, sentimentExamples, {
        numExamples: 3,
        maxNewTokens: 20,
        temperature: 0.2,
      });
      const duration = Date.now() - startTime;

      console.log(`✅ Sentiment (${duration}ms): ${sentiment}`);
    } catch (error) {
      console.error(`❌ Failed: ${error.message}`);
    }
  }
}

async function demo4_BatchProcessing() {
  console.log("\n\n📦 Demo 4: Batch Text Processing\n");
  console.log("━".repeat(60));

  const client = new MockNeuroQuantumClient();

  const prompts = [
    "深層学習の基本は",
    "自然言語処理の課題は",
    "コンピュータビジョンの応用は",
    "強化学習では",
  ];

  console.log(`\n📝 Processing ${prompts.length} prompts in batch:\n`);

  const results = [];
  const startTotal = Date.now();

  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];
    console.log(`[${i + 1}/${prompts.length}] Processing: "${prompt}"`);

    try {
      const startTime = Date.now();
      const result = await client.generateWithExamples(prompt, [], {
        maxNewTokens: 50,
        temperature: 0.7,
      });
      const duration = Date.now() - startTime;

      results.push({ prompt, duration });
      console.log(`  ✅ Completed in ${duration}ms\n`);
    } catch (error) {
      console.error(`  ❌ Failed: ${error.message}\n`);
    }
  }

  const totalDuration = Date.now() - startTotal;

  console.log("📊 Batch Statistics:");
  console.log(`  Total items processed: ${results.length}/${prompts.length}`);
  console.log(`  Total time: ${totalDuration}ms`);
  if (results.length > 0) {
    const avgTime = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    console.log(`  Average time per item: ${avgTime.toFixed(0)}ms`);
    const minTime = Math.min(...results.map((r) => r.duration));
    const maxTime = Math.max(...results.map((r) => r.duration));
    console.log(`  Min/Max time: ${minTime}ms / ${maxTime}ms`);
  }
}

async function demo5_ChainOfThought() {
  console.log("\n\n🧠 Demo 5: Chain-of-Thought Reasoning\n");
  console.log("━".repeat(60));

  const client = new MockNeuroQuantumClient();

  const examples = [
    {
      prompt: "If I have 3 apples and get 2 more, how many do I have?",
      completion:
        "Let me think step by step:\n1. I start with 3 apples\n2. I get 2 more apples\n3. Total: 3 + 2 = 5 apples",
    },
  ];

  const problems = [
    "If a car travels 60km per hour, how far will it travel in 3.5 hours?",
    "A store sells items at $5 each. If I buy 4 items, how much do I spend?",
  ];

  for (const problem of problems) {
    console.log(`\n❓ Problem: "${problem}"`);
    console.log("📚 Expecting step-by-step reasoning...\n");

    try {
      const startTime = Date.now();
      const result = await client.generateWithExamples(problem, examples, {
        numExamples: 1,
        maxNewTokens: 120,
        temperature: 0.5,
      });
      const duration = Date.now() - startTime;

      console.log(`✅ Reasoning (${duration}ms):`);
      console.log("  Let me think step by step:");
      console.log("  1. Identify the given information");
      console.log("  2. Determine what we need to calculate");
      console.log("  3. Apply the formula and compute");
      console.log("  4. Verify the result\n");
    } catch (error) {
      console.error(`❌ Failed: ${error.message}`);
    }
  }
}

async function main() {
  console.log("╔════════════════════════════════════════════════════════════╗");
  console.log("║                                                            ║");
  console.log("║         🚀 Qubit AI - Inference Examples Demo 🚀          ║");
  console.log("║                                                            ║");
  console.log("║     Local Mock Demonstration (No API Required)             ║");
  console.log("║                                                            ║");
  console.log("╚════════════════════════════════════════════════════════════╝");

  try {
    await demo1_BasicGeneration();
    await sleep(500);

    await demo2_FewShotTranslation();
    await sleep(500);

    await demo3_SentimentAnalysis();
    await sleep(500);

    await demo4_BatchProcessing();
    await sleep(500);

    await demo5_ChainOfThought();

    console.log("\n\n" + "━".repeat(60));
    console.log("✨ All demos completed successfully!\n");

    console.log("📚 Example Features Demonstrated:");
    console.log("  ✅ Basic text generation with parameter tuning");
    console.log("  ✅ Few-shot learning for specialized tasks");
    console.log("  ✅ Batch processing with performance metrics");
    console.log("  ✅ Sentiment analysis from text");
    console.log("  ✅ Chain-of-thought reasoning patterns");

    console.log("\n🔗 Next Steps to Use Real API:");
    console.log("  1. Get HuggingFace token: https://huggingface.co/settings/tokens");
    console.log("  2. Set environment variable:");
    console.log("     export HF_TOKEN='hf_...'");
    console.log("  3. Run real inference examples:");
    console.log("     npx ts-node examples/qubit-inference-basics.ts");
    console.log("     npx ts-node examples/qubit-few-shot-inference.ts");
    console.log("     npx ts-node examples/qubit-batch-inference.ts");
    console.log("     npx ts-node examples/qubit-advanced-prompting.ts");
    console.log("     npx ts-node examples/qubit-inference-pipeline.ts");

    console.log("\n📖 Documentation: See examples/README.md for full guide");
    console.log("━".repeat(60) + "\n");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
