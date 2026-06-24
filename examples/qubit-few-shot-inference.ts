/**
 * Example: Few-shot Learning with Qubit AI
 *
 * Few-shot learning allows the model to learn from examples provided in the prompt.
 * This is more powerful than zero-shot generation for specific tasks.
 *
 * Prerequisites:
 * 1. Set HF_TOKEN environment variable:
 *    export HF_TOKEN="hf_..."
 *
 * 2. Install dependencies:
 *    npm install qubit_ai
 *
 * 3. Run:
 *    npx ts-node examples/qubit-few-shot-inference.ts
 */

import { NeuroQuantumClient, HFDatasetLoader } from "qubit_ai";

interface Example {
  prompt: string;
  completion: string;
}

async function languageTranslation() {
  console.log("🌐 Few-shot Language Translation\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Define examples for English-to-Japanese translation
  const translationExamples: Example[] = [
    { prompt: "Hello, how are you?", completion: "こんにちは、お元気ですか？" },
    { prompt: "Good morning", completion: "おはようございます" },
    { prompt: "Thank you very much", completion: "本当にありがとうございました" },
  ];

  const testPrompts = [
    "Good night",
    "See you tomorrow",
    "I love this place",
  ];

  console.log("📚 Examples provided to model:");
  translationExamples.forEach((ex, i) => {
    console.log(`  ${i + 1}. "${ex.prompt}" → "${ex.completion}"`);
  });
  console.log();

  for (const testPrompt of testPrompts) {
    console.log(`📝 Input: "${testPrompt}"`);
    try {
      const result = await client.generateWithExamples(
        testPrompt,
        translationExamples,
        {
          numExamples: 3,
          exampleTemplate: "English: {prompt}\nJapanese: {completion}",
          queryTemplate: "English: {prompt}\nJapanese:",
          maxNewTokens: 40,
          temperature: 0.3, // Low temp for consistent translations
        }
      );

      console.log(`✅ Output: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed:`, error instanceof Error ? error.message : error);
    }
  }
}

async function sentimentAnalysis() {
  console.log("💭 Few-shot Sentiment Analysis\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Define examples for sentiment analysis
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

  console.log("📚 Examples provided to model:");
  sentimentExamples.forEach((ex, i) => {
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
      console.error(`❌ Failed:`, error instanceof Error ? error.message : error);
    }
  }
}

async function questionAnswering() {
  console.log("❓ Few-shot Question Answering\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Define examples for factual QA
  const qaExamples: Example[] = [
    { prompt: "What is the capital of France?", completion: "The capital of France is Paris." },
    { prompt: "Who wrote 'Romeo and Juliet'?", completion: "William Shakespeare wrote 'Romeo and Juliet'." },
    { prompt: "What is the chemical symbol for gold?", completion: "The chemical symbol for gold is Au." },
  ];

  const questions = [
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "What is the largest planet in our solar system?",
  ];

  console.log("📚 Example QA pairs:");
  qaExamples.forEach((ex, i) => {
    console.log(`  ${i + 1}. Q: "${ex.prompt}"`);
    console.log(`     A: "${ex.completion}"`);
  });
  console.log();

  for (const question of questions) {
    console.log(`❓ Question: "${question}"`);
    try {
      const result = await client.generateWithExamples(
        question,
        qaExamples,
        {
          numExamples: 3,
          exampleTemplate: "Q: {prompt}\nA: {completion}",
          queryTemplate: "Q: {prompt}\nA:",
          maxNewTokens: 60,
          temperature: 0.4,
        }
      );

      console.log(`✅ Answer: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed:`, error instanceof Error ? error.message : error);
    }
  }
}

async function generationFromDataset() {
  console.log("📊 Few-shot Generation from HuggingFace Dataset\n");

  try {
    const loader = new HFDatasetLoader({
      hfToken: process.env.HF_TOKEN,
    });

    // Load a few examples from a public dataset
    console.log("📥 Loading examples from HuggingFace dataset...");
    const examples = await loader.preview("llm-jp/oasst2-33k-ja", 3);

    console.log(`✅ Loaded ${examples.length} examples\n`);

    const client = new NeuroQuantumClient({
      hfToken: process.env.HF_TOKEN,
    });

    // Use the loaded examples for generation
    const testPrompt = "AIアシスタントについて説明してください";

    console.log(`📝 Prompt: "${testPrompt}"\n`);
    console.log("📚 Using examples from dataset:");
    examples.slice(0, 2).forEach((ex, i) => {
      console.log(`  ${i + 1}. Q: "${ex.prompt}" → A: "${ex.completion}"`);
    });
    console.log();

    const result = await client.generateWithExamples(
      testPrompt,
      examples,
      {
        numExamples: 2,
        exampleTemplate: "質問: {prompt}\n回答: {completion}",
        queryTemplate: "質問: {prompt}\n回答:",
        maxNewTokens: 100,
        temperature: 0.7,
      }
    );

    console.log(`✅ Generated response:\n${result.generatedText}\n`);
  } catch (error) {
    console.error(
      "❌ Dataset loading failed:",
      error instanceof Error ? error.message : error
    );
    console.log("(This may fail without proper HF_TOKEN for private datasets)\n");
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

    await languageTranslation();
    console.log("━".repeat(60) + "\n");

    await sentimentAnalysis();
    console.log("━".repeat(60) + "\n");

    await questionAnswering();
    console.log("━".repeat(60) + "\n");

    await generationFromDataset();

    console.log("✨ All few-shot examples completed!");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
