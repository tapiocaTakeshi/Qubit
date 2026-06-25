#!/usr/bin/env node

/**
 * Train qubit_ai on OASST-1 dataset
 *
 * Usage:
 *   node train-oasst1.js
 *   HF_TOKEN=hf_... node train-oasst1.js
 *
 * Note: This uses qubit_ai@4.0.5 Pyodide backend for training.
 */

import { trainOnHFDataset } from "qubit_ai";

const HF_TOKEN = process.env.HF_TOKEN;

async function main() {
  console.log("🧠 Starting OASST-1 training on qubit_ai...");
  console.log("📚 Dataset: kunishou/oasst1-chat-44k-ja (44k samples)");

  if (!HF_TOKEN) {
    console.warn("⚠️  HF_TOKEN not set. Using default/public access (may be rate-limited).");
  }

  try {
    console.log("\n⏱️  Training started...\n");

    const result = await trainOnHFDataset({
      hfToken: HF_TOKEN,
      dataset: "kunishou/oasst1-chat-44k-ja",
      split: "train",
      maxSamples: 100, // Start small for testing (full: 44042)
      epochs: 1,
      batchSize: 4,
      learningRate: 5e-5,
      validateInterval: 25,
      onProgress: (progress) => {
        if (progress.samplesProcessed % 50 === 0 || progress.samplesProcessed === progress.totalSamples) {
          const pct = Math.round((progress.samplesProcessed / progress.totalSamples) * 100);
          const loss = progress.currentLoss?.toFixed(4) ?? "?";
          console.log(
            `  [${progress.epoch}/${progress.totalEpochs}] ` +
            `${progress.samplesProcessed}/${progress.totalSamples} (${pct}%) loss=${loss}`,
          );
        }
      },
    });

    console.log("\n✅ Training complete!");
    console.log(`📊 Final loss: ${result.finalLoss?.toFixed(4)}`);
    console.log(`⏱️  Duration: ${(result.trainingDuration / 1000).toFixed(1)}s`);

    if (result.checkpoint) {
      console.log("💾 Checkpoint:", result.checkpoint);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error("\n❌ Training failed:", msg);
    process.exit(1);
  }
}

main();
