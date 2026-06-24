/**
 * Fetch ALL Japanese (lang=ja) messages from the OpenAssistant/oasst1 dataset
 * via the HuggingFace datasets-server, across train + validation splits.
 *
 * Builds prompter -> assistant Q&A pairs (matched by parent_id) so the chat
 * model can answer with real human-written Japanese responses.
 */

import fetch from "node-fetch";
import fs from "fs";
import path from "path";

const DATASET = "OpenAssistant/oasst1";
const CONFIG = "default";
const SPLITS = ["train", "validation"];
const PAGE = 100; // datasets-server filter max limit

const WHERE = encodeURIComponent("\"lang\"='ja'");

async function fetchSplit(split) {
  const all = [];
  let offset = 0;
  let total = Infinity;

  while (offset < total) {
    const url =
      `https://datasets-server.huggingface.co/filter?dataset=${encodeURIComponent(DATASET)}` +
      `&config=${CONFIG}&split=${split}&where=${WHERE}&limit=${PAGE}&offset=${offset}`;

    const res = await fetch(url);
    if (!res.ok) {
      console.log(`  ${split}: HTTP ${res.status} at offset ${offset}, stopping.`);
      break;
    }
    const json = await res.json();
    total = json.num_rows_total ?? all.length;
    const rows = (json.rows || []).map((r) => r.row);
    all.push(...rows);
    console.log(`  ${split}: fetched ${all.length}/${total}`);
    if (rows.length === 0) break;
    offset += PAGE;
  }
  return all;
}

function buildPairs(messages) {
  // Index messages by id and group children by parent_id
  const byId = new Map();
  for (const m of messages) byId.set(m.message_id, m);

  const pairs = [];
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    if (!m.text || !m.text.trim()) continue;
    const parent = m.parent_id ? byId.get(m.parent_id) : null;
    if (parent && parent.role === "prompter" && parent.text) {
      pairs.push({
        prompt: parent.text.trim(),
        response: m.text.trim(),
        rank: typeof m.rank === "number" ? m.rank : 0,
      });
    }
  }
  return pairs;
}

async function main() {
  console.log(`Fetching all Japanese messages from ${DATASET}...`);
  let messages = [];
  for (const split of SPLITS) {
    try {
      const rows = await fetchSplit(split);
      messages = messages.concat(rows);
    } catch (e) {
      console.log(`  ${split}: error ${e.message}`);
    }
  }

  // Keep only non-deleted messages
  messages = messages.filter((m) => !m.deleted);
  console.log(`Total Japanese messages: ${messages.length}`);

  const pairs = buildPairs(messages);

  // Prefer the best-ranked answer per prompt (lowest rank = best)
  const bestByPrompt = new Map();
  for (const p of pairs) {
    const existing = bestByPrompt.get(p.prompt);
    if (!existing || p.rank < existing.rank) {
      bestByPrompt.set(p.prompt, p);
    }
  }
  const uniquePairs = Array.from(bestByPrompt.values()).map((p) => ({
    prompt: p.prompt,
    response: p.response,
  }));

  // Also collect all standalone assistant phrases (for fallback variety)
  const phrases = messages
    .filter((m) => m.role === "assistant" && m.text && m.text.trim().length > 5)
    .map((m) => m.text.trim());

  const data = {
    dataset: DATASET,
    source: "huggingface datasets-server (lang=ja)",
    timestamp: new Date().toISOString(),
    message_count: messages.length,
    pair_count: uniquePairs.length,
    phrase_count: phrases.length,
    pairs: uniquePairs,
    phrases,
  };

  const outputPath = path.join(process.cwd(), "data", "oasst1-ja.json");
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));

  console.log(
    `✓ Saved ${data.pair_count} Q&A pairs and ${data.phrase_count} phrases to ${outputPath}`
  );
}

main().catch((e) => {
  console.error("Failed:", e);
  process.exit(1);
});
