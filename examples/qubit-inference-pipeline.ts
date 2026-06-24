/**
 * Example: Complete Inference Pipeline with Dataset Integration
 *
 * Demonstrates a realistic workflow that combines:
 * - Loading datasets from HuggingFace
 * - Generating predictions on multiple texts
 * - Evaluating results
 * - Saving outputs
 *
 * Prerequisites:
 * 1. Set HF_TOKEN environment variable:
 *    export HF_TOKEN="hf_..."
 *
 * 2. Install dependencies:
 *    npm install qubit_ai
 *
 * 3. Run:
 *    npx ts-node examples/qubit-inference-pipeline.ts
 */

import { NeuroQuantumClient, HFDatasetLoader } from "qubit_ai";
import * as fs from "fs";
import * as path from "path";

interface PipelineConfig {
  datasetName?: string;
  maxExamples?: number;
  temperature?: number;
  maxTokens?: number;
  batchSize?: number;
  outputDir?: string;
}

interface InferenceResult {
  input: string;
  output: string;
  duration: number;
  timestamp: string;
}

/**
 * Pipeline for text summarization
 */
async function summarizationPipeline(config: PipelineConfig = {}) {
  console.log("📝 Summarization Inference Pipeline\n");

  const {
    temperature = 0.5,
    maxTokens = 100,
    outputDir = "./output",
  } = config;

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Define summarization examples
  const summaryExamples = [
    {
      prompt: "Artificial intelligence is transforming industries by automating tasks and enabling new capabilities. From healthcare to finance, AI is being deployed to improve efficiency and decision-making.",
      completion: "AI is revolutionizing multiple industries through automation and enhanced decision-making.",
    },
    {
      prompt: "Climate change poses unprecedented challenges to global society. Rising temperatures, changing precipitation patterns, and extreme weather events threaten ecosystems and human communities worldwide.",
      completion: "Climate change presents global threats through rising temperatures, altered weather patterns, and ecosystem damage.",
    },
  ];

  // Texts to summarize
  const textsToSummarize = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions.",
    "Cloud computing has revolutionized how organizations manage their IT infrastructure. By leveraging cloud services, companies can scale their operations, reduce costs, and improve flexibility.",
  ];

  console.log(`📥 Input: ${textsToSummarize.length} texts\n`);

  const results: InferenceResult[] = [];
  const startPipeline = Date.now();

  for (let i = 0; i < textsToSummarize.length; i++) {
    const text = textsToSummarize[i];
    console.log(`[${i + 1}/${textsToSummarize.length}] Summarizing text...`);

    try {
      const startTime = Date.now();

      const result = await client.generateWithExamples(
        text,
        summaryExamples,
        {
          numExamples: 2,
          exampleTemplate: "Text: {prompt}\nSummary: {completion}",
          queryTemplate: "Text: {prompt}\nSummary:",
          maxNewTokens: maxTokens,
          temperature,
        }
      );

      const duration = Date.now() - startTime;
      results.push({
        input: text,
        output: result.generatedText,
        duration,
        timestamp: new Date().toISOString(),
      });

      console.log(`  ✅ Generated summary (${duration}ms)\n`);
    } catch (error) {
      console.error(
        `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
      );
    }
  }

  const totalDuration = Date.now() - startPipeline;

  // Output results
  console.log("📊 Pipeline Results:");
  console.log("━".repeat(60));
  results.forEach((r, i) => {
    console.log(`\n${i + 1}. Input (${r.input.length} chars):`);
    console.log(`   "${r.input.substring(0, 70)}..."`);
    console.log(`\n   Summary (${r.duration}ms):`);
    console.log(`   "${r.output}"`);
  });

  console.log("\n📈 Pipeline Statistics:");
  console.log(`  Texts processed: ${results.length}`);
  console.log(`  Total time: ${totalDuration}ms`);
  if (results.length > 0) {
    const avgTime = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    console.log(`  Average time per text: ${avgTime.toFixed(0)}ms`);
  }

  return results;
}

/**
 * Pipeline using dataset examples
 */
async function datasetIntegratedPipeline(config: PipelineConfig = {}) {
  console.log("\n🗂️  Dataset-Integrated Inference Pipeline\n");

  const {
    datasetName = "llm-jp/oasst2-33k-ja",
    maxExamples = 2,
    temperature = 0.6,
    maxTokens = 80,
  } = config;

  try {
    const loader = new HFDatasetLoader({
      hfToken: process.env.HF_TOKEN,
    });

    console.log(`📥 Loading examples from dataset: "${datasetName}"...`);
    const examples = await loader.preview(datasetName, maxExamples);
    console.log(`✅ Loaded ${examples.length} examples\n`);

    const client = new NeuroQuantumClient({
      hfToken: process.env.HF_TOKEN,
    });

    // Use dataset examples for QA
    const questions = [
      "プログラミングの初心者向けのアドバイスはありますか？",
      "大規模言語モデルの仕組みについて説明してください",
    ];

    const results: InferenceResult[] = [];

    console.log(`📝 Processing ${questions.length} questions...\n`);

    for (let i = 0; i < questions.length; i++) {
      const question = questions[i];
      console.log(`[${i + 1}/${questions.length}] "${question}"`);

      try {
        const startTime = Date.now();

        const result = await client.generateWithExamples(
          question,
          examples,
          {
            numExamples: Math.min(maxExamples, examples.length),
            exampleTemplate: "Q: {prompt}\nA: {completion}",
            queryTemplate: "Q: {prompt}\nA:",
            maxNewTokens: maxTokens,
            temperature,
          }
        );

        const duration = Date.now() - startTime;
        results.push({
          input: question,
          output: result.generatedText,
          duration,
          timestamp: new Date().toISOString(),
        });

        const preview = result.generatedText.substring(0, 60);
        console.log(`  ✅ Generated (${duration}ms): "${preview}..."\n`);
      } catch (error) {
        console.error(
          `  ❌ Failed: ${error instanceof Error ? error.message : error}\n`
        );
      }
    }

    console.log("📊 Results:");
    results.forEach((r, i) => {
      console.log(`\n${i + 1}. Q: "${r.input}"`);
      console.log(`   A: "${r.output}"\n   ⏱️  ${r.duration}ms`);
    });

    return results;
  } catch (error) {
    console.error(
      "❌ Pipeline failed:",
      error instanceof Error ? error.message : error
    );
    return [];
  }
}

/**
 * Complete end-to-end pipeline with caching and error handling
 */
async function robustInferencePipeline(config: PipelineConfig = {}) {
  console.log("\n🛡️  Robust Inference Pipeline with Error Handling\n");

  const {
    temperature = 0.7,
    maxTokens = 100,
    outputDir = "./results",
  } = config;

  // Create output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
    timeoutMs: 60000,
    maxRetries: 3,
  });

  // Example tasks: Code explanation
  const codeExamples = [
    {
      prompt: "const arr = [1, 2, 3]; const doubled = arr.map(x => x * 2);",
      completion: "This JavaScript code creates an array [1, 2, 3], then uses map() to create a new array [2, 4, 6] where each element is doubled.",
    },
  ];

  const codeSnippets = [
    "function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); }",
    "const fibonacci = (n) => n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);",
  ];

  const results: InferenceResult[] = [];
  const errors: Array<{ input: string; error: string }> = [];

  console.log(`⚙️  Processing ${codeSnippets.length} code snippets...\n`);
  console.log(`💾 Output directory: ${outputDir}\n`);

  for (let i = 0; i < codeSnippets.length; i++) {
    const snippet = codeSnippets[i];
    console.log(`[${i + 1}/${codeSnippets.length}] Explaining code...`);

    try {
      const startTime = Date.now();

      const result = await client.generateWithExamples(
        snippet,
        codeExamples,
        {
          numExamples: 1,
          exampleTemplate: "Code: {prompt}\nExplanation: {completion}",
          queryTemplate: "Code: {prompt}\nExplanation:",
          maxNewTokens: maxTokens,
          temperature,
        }
      );

      const duration = Date.now() - startTime;
      results.push({
        input: snippet,
        output: result.generatedText,
        duration,
        timestamp: new Date().toISOString(),
      });

      console.log(`  ✅ Complete (${duration}ms)\n`);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      errors.push({ input: snippet, error: errorMsg });
      console.error(`  ❌ Error: ${errorMsg}\n`);
    }
  }

  // Save results to file
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputFile = path.join(outputDir, `inference-results-${timestamp}.json`);

  const output = {
    pipeline: "Code Explanation",
    timestamp: new Date().toISOString(),
    config: { temperature, maxTokens },
    summary: {
      totalProcessed: results.length,
      totalErrors: errors.length,
      successRate: `${((results.length / (results.length + errors.length)) * 100).toFixed(1)}%`,
    },
    results,
    errors: errors.length > 0 ? errors : undefined,
  };

  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));
  console.log(`💾 Results saved to: ${outputFile}\n`);

  // Display summary
  console.log("📊 Pipeline Summary:");
  console.log(`  Successful: ${results.length}`);
  console.log(`  Errors: ${errors.length}`);
  if (results.length > 0) {
    const avgTime = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
    console.log(`  Average time: ${avgTime.toFixed(0)}ms`);
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

    // Run pipelines
    await summarizationPipeline();
    console.log("\n" + "━".repeat(60));

    await datasetIntegratedPipeline();
    console.log("\n" + "━".repeat(60));

    await robustInferencePipeline();

    console.log("\n✨ All inference pipelines completed successfully!");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
