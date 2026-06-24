/**
 * Example: Batch Inference with Qubit AI
 *
 * Process multiple texts efficiently using batch processing.
 * Useful for analyzing large collections of texts or documents.
 *
 * Prerequisites:
 * 1. Set HF_TOKEN environment variable:
 *    export HF_TOKEN="hf_..."
 *
 * 2. Install dependencies:
 *    npm install qubit_ai
 *
 * 3. Run:
 *    npx ts-node examples/qubit-batch-inference.ts
 */

import { NeuroQuantumClient, HFDatasetLoader } from "qubit_ai";

interface BatchResult {
  input: string;
  output: string;
  duration: number;
}

async function batchTextGeneration() {
  console.log("📦 Batch Text Generation\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
    timeoutMs: 30000,
  });

  const prompts = [
    "深層学習の基本は",
    "自然言語処理の課題は",
    "コンピュータビジョンの応用は",
    "強化学習では",
    "転移学習の利点は",
  ];

  const results: BatchResult[] = [];
  const startTotal = Date.now();

  console.log(`📝 Processing ${prompts.length} prompts...\n`);

  for (let i = 0; i < prompts.length; i++) {
    const prompt = prompts[i];
    console.log(`[${i + 1}/${prompts.length}] Processing: "${prompt}"`);

    try {
      const startTime = Date.now();

      const result = await client.generateWithExamples(
        prompt,
        [],
        {
          maxNewTokens: 50,
          temperature: 0.7,
        }
      );

      const duration = Date.now() - startTime;
      results.push({
        input: prompt,
        output: result.generatedText,
        duration,
      });

      console.log(`  ✅ Completed in ${duration}ms\n`);
    } catch (error) {
      console.error(
        `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
      );
    }
  }

  const totalDuration = Date.now() - startTotal;

  console.log("\n📊 Batch Results Summary:");
  console.log("━".repeat(60));
  results.forEach((r, i) => {
    console.log(`\n${i + 1}. Input: "${r.input}"`);
    console.log(`   Output: "${r.output}"`);
    console.log(`   Time: ${r.duration}ms`);
  });

  console.log("\n📈 Statistics:");
  console.log(`  Total items: ${results.length}`);
  console.log(`  Successful: ${results.length}/${prompts.length}`);
  console.log(`  Total time: ${totalDuration}ms`);
  if (results.length > 0) {
    const avgTime = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    console.log(`  Average time per item: ${avgTime.toFixed(0)}ms`);
  }
}

async function batchSentimentAnalysis() {
  console.log("\n💬 Batch Sentiment Analysis\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  const texts = [
    "This product exceeded all my expectations!",
    "I'm very unhappy with this purchase.",
    "It's fine, nothing remarkable.",
    "Absolutely terrible experience, worst ever!",
    "Pretty good, would recommend to friends.",
    "Meh, could be better.",
  ];

  const examples = [
    { prompt: "Amazing product!", completion: "Positive" },
    { prompt: "Hate this", completion: "Negative" },
    { prompt: "It's okay", completion: "Neutral" },
  ];

  const results: Array<{ text: string; sentiment: string }> = [];

  console.log(`📝 Analyzing ${texts.length} texts...\n`);

  for (let i = 0; i < texts.length; i++) {
    const text = texts[i];
    console.log(`[${i + 1}/${texts.length}] "${text}"`);

    try {
      const result = await client.generateWithExamples(
        text,
        examples,
        {
          numExamples: 3,
          exampleTemplate: "Text: {prompt}\nSentiment: {completion}",
          queryTemplate: "Text: {prompt}\nSentiment:",
          maxNewTokens: 15,
          temperature: 0.2,
        }
      );

      const sentiment = result.generatedText.trim().split("\n")[0];
      results.push({ text, sentiment });
      console.log(`  → ${sentiment}\n`);
    } catch (error) {
      console.error(
        `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
      );
    }
  }

  console.log("\n📊 Sentiment Distribution:");
  const counts = results.reduce(
    (acc, r) => {
      const sentiment = r.sentiment.toLowerCase();
      acc[sentiment] = (acc[sentiment] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  Object.entries(counts).forEach(([sentiment, count]) => {
    const percentage = ((count / results.length) * 100).toFixed(1);
    console.log(`  ${sentiment.toUpperCase()}: ${count} (${percentage}%)`);
  });
}

async function batchTopicClassification() {
  console.log("\n🏷️  Batch Topic Classification\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  const documents = [
    "The stock market reached an all-time high today.",
    "Scientists discover new species in the Amazon rainforest.",
    "New AI model shows promising results in medical imaging.",
    "Sports teams compete for the championship title.",
    "Economic data suggests inflation is stabilizing.",
  ];

  const classifyExamples = [
    { prompt: "Tech company releases new smartphone", completion: "Technology" },
    { prompt: "Team wins the playoff game", completion: "Sports" },
    { prompt: "Interest rates are expected to rise", completion: "Finance" },
  ];

  console.log(`📄 Classifying ${documents.length} documents...\n`);

  const classifications: Array<{ doc: string; topic: string }> = [];

  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    console.log(`[${i + 1}/${documents.length}] "${doc}"`);

    try {
      const result = await client.generateWithExamples(
        doc,
        classifyExamples,
        {
          numExamples: 3,
          exampleTemplate: "Document: {prompt}\nTopic: {completion}",
          queryTemplate: "Document: {prompt}\nTopic:",
          maxNewTokens: 20,
          temperature: 0.3,
        }
      );

      const topic = result.generatedText.trim().split("\n")[0];
      classifications.push({ doc, topic });
      console.log(`  → ${topic}\n`);
    } catch (error) {
      console.error(
        `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
      );
    }
  }

  console.log("\n📊 Topic Distribution:");
  const topicCounts = classifications.reduce(
    (acc, c) => {
      const topic = c.topic.toLowerCase();
      acc[topic] = (acc[topic] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  Object.entries(topicCounts).forEach(([topic, count]) => {
    console.log(`  ${topic.toUpperCase()}: ${count}`);
  });
}

async function batchProcessingWithDataset() {
  console.log("\n📚 Batch Processing with Dataset Examples\n");

  try {
    const loader = new HFDatasetLoader({
      hfToken: process.env.HF_TOKEN,
    });

    console.log("📥 Loading dataset examples...");
    const examples = await loader.preview("llm-jp/oasst2-33k-ja", 2);
    console.log(`✅ Loaded ${examples.length} examples\n`);

    const client = new NeuroQuantumClient({
      hfToken: process.env.HF_TOKEN,
    });

    const queries = [
      "Pythonプログラミングとは何ですか？",
      "Webアプリケーション開発の流れを説明してください",
    ];

    console.log(`📝 Processing ${queries.length} queries with dataset context...\n`);

    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      console.log(`[${i + 1}/${queries.length}] "${query}"`);

      try {
        const result = await client.generateWithExamples(
          query,
          examples,
          {
            numExamples: 2,
            exampleTemplate: "質問: {prompt}\n回答: {completion}",
            queryTemplate: "質問: {prompt}\n回答:",
            maxNewTokens: 80,
            temperature: 0.6,
          }
        );

        const response = result.generatedText.trim().substring(0, 100);
        console.log(`  → ${response}...\n`);
      } catch (error) {
        console.error(
          `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
        );
      }
    }
  } catch (error) {
    console.error(
      "❌ Dataset loading failed:",
      error instanceof Error ? error.message : error
    );
  }
}

async function main() {
  try {
    if (!process.env.HF_TOKEN) {
      console.error("❌ Error: HF_TOKEN environment variable is not set");
      console.error("Please set your HuggingFace token:");
      console.error("  export HF_TOKEN='hf_...'");
      process.exit(1);
    }

    await batchTextGeneration();
    console.log("\n" + "━".repeat(60));

    await batchSentimentAnalysis();
    console.log("\n" + "━".repeat(60));

    await batchTopicClassification();
    console.log("\n" + "━".repeat(60));

    await batchProcessingWithDataset();

    console.log("\n✨ All batch inference examples completed!");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
