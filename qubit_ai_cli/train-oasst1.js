#!/usr/bin/env node

/**
 * Train qubit_ai_cli's underlying qubit_ai model on OASST-1 dataset.
 *
 * Usage:
 *   node train-oasst1.js [maxSamples]
 *
 * Loads pre-extracted OASST-1 Japanese conversation texts and trains the
 * qubit_ai NeuroQuantum generator on them via the train(texts) API.
 *
 * The texts JSON is produced from kunishou/oasst1-chat-44k-ja
 * (see scratchpad extraction or regenerate with the Python dataset utils).
 */

import { readFileSync } from "node:fs";
import { getQubitAIGenerative } from "qubit_ai";

const TEXTS_PATH =
  process.env.OASST_TEXTS_PATH ||
  "/tmp/claude-0/-home-user-Qubit/d5f7f6b3-6374-5267-b637-339ca5e2b148/scratchpad/oasst_texts.json";

async function main() {
  const maxSamples = parseInt(process.argv[2] || "0", 10); // 0 = all

  console.log("🧠 Training qubit_ai on OASST-1 (kunishou/oasst1-chat-44k-ja)");

  let texts;
  try {
    texts = JSON.parse(readFileSync(TEXTS_PATH, "utf-8"));
  } catch (e) {
    console.error(`❌ Could not read texts from ${TEXTS_PATH}`);
    console.error("   Set OASST_TEXTS_PATH or regenerate the JSON.");
    process.exit(1);
  }

  if (maxSamples > 0) {
    texts = texts.slice(0, maxSamples);
  }
  console.log(`📚 Loaded ${texts.length} training texts`);

  const gen = getQubitAIGenerative();

  console.log("⏱️  Training started...");
  const start = Date.now();

  await gen.train(texts);

  const duration = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`\n✅ Training complete in ${duration}s`);
  console.log("📊 Status:", JSON.stringify(gen.getStatus()));

  // Quick sanity-check generation after training
  console.log("\n=== Post-training sample generation ===");
  for (const prompt of ["量子コンピュータとは", "こんにちは"]) {
    try {
      const out = await gen.generate(prompt, { maxNewTokens: 40, temperature: 0.7 });
      const text = typeof out === "string" ? out : out?.generatedText ?? JSON.stringify(out);
      console.log(`  「${prompt}」→ ${text}`);
    } catch (e) {
      console.log(`  「${prompt}」→ (生成エラー: ${e.message})`);
    }
  }
}

main().catch((e) => {
  console.error("❌ Failed:", e instanceof Error ? e.message : e);
  process.exit(1);
});
