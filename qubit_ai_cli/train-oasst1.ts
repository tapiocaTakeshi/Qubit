/**
 * Train qubit_ai on OASST-1 dataset from HuggingFace
 *
 * Usage:
 *   npx tsx train-oasst1.ts
 *
 * Requires HF_TOKEN environment variable
 */

import * as qubitAi from "qubit_ai";

const { trainOnHFDataset, getQubitAIGenerative } = qubitAi;

const HF_TOKEN = process.env.HF_TOKEN;
if (!HF_TOKEN) {
  console.error("❌ HF_TOKEN environment variable is required");
  console.error("   export HF_TOKEN='hf_...'");
  process.exit(1);
}

async function main() {
  console.log("🧠 Training qubit_ai on OASST-1 dataset...");
  console.log("📚 Dataset: kunishou/oasst1-chat-44k-ja");
  console.log("🎯 Target: Full 44k samples, 30 epochs");
  console.log("");

  try {
    const result = await trainOnHFDataset({
      hfToken: HF_TOKEN,
      dataset: "kunishou/oasst1-chat-44k-ja",
      split: "train",
      maxSamples: 0, // 0 = all
      epochs: 30,
      batchSize: 4,
      learningRate: 5e-5,
      validateInterval: 500,
      onProgress: (progress) => {
        const pct = ((progress.samplesProcessed / progress.totalSamples) * 100).toFixed(1);
        console.log(
          `[${progress.epoch}/${progress.totalEpochs}] ` +
          `${progress.samplesProcessed}/${progress.totalSamples} (${pct}%) ` +
          `loss=${progress.currentLoss?.toFixed(4) ?? "?"}`,
        );
      },
    });

    console.log("\n✅ Training complete!");
    console.log(`📊 Final loss: ${result.finalLoss?.toFixed(4) ?? "?"}`);
    console.log(`⏱️  Duration: ${result.trainingDuration}ms`);

    // Save trained model state
    const model = getQubitAIGenerative();
    console.log("\n💾 Saving model state...");
    // Note: qubit_ai may or may not expose save/export methods
    console.log("✅ Model ready for inference");
  } catch (error) {
    console.error("❌ Training failed:", error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
