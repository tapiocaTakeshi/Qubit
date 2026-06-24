/**
 * Example: Basic Text Generation with Qubit AI
 *
 * This example demonstrates simple text generation using the NeuroQuantumClient.
 * No complex setup required — just provide your HuggingFace token.
 *
 * Prerequisites:
 * 1. Set HF_TOKEN environment variable:
 *    export HF_TOKEN="hf_..."
 *
 * 2. Install dependencies:
 *    npm install qubit_ai
 *
 * 3. Run:
 *    npx ts-node examples/qubit-inference-basics.ts
 */

import { NeuroQuantumClient } from "qubit_ai";

async function basicGeneration() {
  console.log("🔬 Basic Text Generation Example\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Example 1: Simple Japanese sentence continuation
  console.log("📝 Generating text from prompt: '人工知能は'\n");
  try {
    const result = await client.generateWithExamples(
      "人工知能は",
      [], // No examples for basic generation
      {
        maxNewTokens: 50,
        temperature: 0.7,
        topP: 0.9,
      }
    );

    console.log("✅ Generated text:");
    console.log(result.generatedText);
    console.log();
  } catch (error) {
    console.error("❌ Generation failed:", error instanceof Error ? error.message : error);
  }
}

async function configuredGeneration() {
  console.log("⚙️ Generation with Custom Parameters\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
    timeoutMs: 30000, // 30 second timeout
  });

  const prompts = [
    "量子コンピュータは",
    "機械学習の利点は",
    "自然言語処理では",
  ];

  for (const prompt of prompts) {
    console.log(`📝 Prompt: "${prompt}"`);
    try {
      const result = await client.generateWithExamples(
        prompt,
        [],
        {
          maxNewTokens: 60,
          temperature: 0.5, // Lower temperature = more deterministic
          topK: 40,
          topP: 0.85,
          repetitionPenalty: 1.2, // Avoid repetition
        }
      );

      console.log(`✅ Result: ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed:`, error instanceof Error ? error.message : error);
    }
  }
}

async function variationGeneration() {
  console.log("🎲 Generation with Different Temperatures\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  const prompt = "データサイエンティストの仕事は";
  const temperatures = [0.3, 0.7, 1.0]; // low, medium, high

  for (const temp of temperatures) {
    console.log(`🌡️  Temperature: ${temp} (${
      temp < 0.5 ? "conservative" : temp < 0.8 ? "balanced" : "creative"
    })`);

    try {
      const result = await client.generateWithExamples(
        prompt,
        [],
        {
          maxNewTokens: 50,
          temperature: temp,
          topP: 0.9,
        }
      );

      console.log(`   ${result.generatedText}\n`);
    } catch (error) {
      console.error(`❌ Failed:`, error instanceof Error ? error.message : error);
    }
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

    await basicGeneration();
    console.log("━".repeat(60) + "\n");

    await configuredGeneration();
    console.log("━".repeat(60) + "\n");

    await variationGeneration();

    console.log("✨ All examples completed successfully!");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
