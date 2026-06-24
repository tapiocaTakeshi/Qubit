# qubit_ai

**QBNN quantum-inspired inference & judgment engine for JavaScript / TypeScript**

`qubit_ai` is the official JavaScript/TypeScript SDK for the [NeuroQuantum (neuroQ)](https://github.com/tapiocaTakeshi/Qubit) project — a quantum-inspired neural network language model and AI decision-making system built on the **APQB (Adjustable Pseudo Quantum Bit)** theory.

---

## Install

```bash
npm install qubit_ai
# or
pnpm add qubit_ai
# or
yarn add qubit_ai
```

Requires **Node.js ≥ 18** (uses the built-in `fetch` API).

---

## Modules

| Export | Description |
|---|---|
| `NeuroQuantumClient` | HTTP client for the neuroQ HuggingFace inference endpoint |
| `QBNNFrontalEngine` | Pure-JS quantum-inspired judgment engine (no Python required) |
| `HFDatasetLoader` | HuggingFace Datasets API client — fetch, stream, and convert dataset rows |

---

## `NeuroQuantumClient` — text generation

```ts
import { NeuroQuantumClient } from "qubit_ai";

const client = new NeuroQuantumClient({
  hfToken: process.env.HF_TOKEN, // optional — public endpoint works without token
});

const result = await client.generate("量子コンピュータとは何ですか？", {
  maxNewTokens: 150,
  temperature: 0.7,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.3,
});

console.log(result.generatedText);
```

### Constructor options

| Option | Type | Default | Description |
|---|---|---|---|
| `endpointUrl` | `string` | neuroQ HF endpoint | Custom inference endpoint URL |
| `hfToken` | `string` | `$HF_TOKEN` env | HuggingFace API token |
| `timeoutMs` | `number` | `600_000` | Per-request timeout |
| `maxRetries` | `number` | `12` | Retries on 503 / network errors |

### Few-shot inference with dataset examples

Use `generateWithExamples()` to prepend examples from a HF dataset as in-context few-shot prompts:

```ts
import { HFDatasetLoader, NeuroQuantumClient } from "qubit_ai";

const loader = new HFDatasetLoader({ hfToken: process.env.HF_TOKEN });
const client = new NeuroQuantumClient({ hfToken: process.env.HF_TOKEN });

// Load a few examples from a public dataset
const examples = await loader.preview("llm-jp/oasst2-33k-ja", 3);

// Generate with those examples as context (few-shot learning)
const result = await client.generateWithExamples(
  "量子コンピュータの利点を教えてください",
  examples,
  {
    numExamples: 3,
    exampleTemplate: "Q: {prompt}\nA: {completion}",
    queryTemplate: "Q: {prompt}\nA:",
    maxNewTokens: 200,
  }
);

console.log(result.generatedText);
```

| Option | Type | Default | Description |
|---|---|---|---|
| `numExamples` | `number` | `3` | Number of examples to include in context |
| `exampleSeparator` | `string` | `"\n\n"` | Separator between examples |
| `exampleTemplate` | `string` | `"Q: {prompt}\nA: {completion}"` | Format for each example |
| `queryTemplate` | `string` | `"Q: {prompt}\nA:"` | Format for the query |

### Training from a HuggingFace dataset

Use `trainFromDataset()` to stream a HF dataset and send it in batches to a fine-tuning endpoint:

```ts
const result = await client.trainFromDataset({
  dataset: "llm-jp/oasst2-33k-ja",
  promptField: "input",       // column to use as prompt
  completionField: "output",  // column to use as completion
  maxRows: 500,               // cap total rows
  batchSize: 10,              // examples per HTTP batch
  trainingEndpointUrl: "https://your-training-endpoint/train",
  onProgress: (p) => {
    console.log(`${p.processedExamples}/${p.totalExamples} examples, batch ${p.currentBatch}/${p.totalBatches}`);
  },
});

console.log(result.status);        // "completed" | "partial" | "failed"
console.log(result.totalExamples); // number of examples sent
console.log(result.durationMs);    // total time in ms
```

| Option | Type | Default | Description |
|---|---|---|---|
| `dataset` | `string` | — | Dataset name on HuggingFace Hub |
| `config` | `string` | `"default"` | Dataset configuration |
| `split` | `string` | `"train"` | Dataset split |
| `promptField` | `string` | auto-inferred | Column name for prompts |
| `completionField` | `string` | auto-inferred | Column name for completions |
| `transform` | `(row) => TrainingExample \| null` | — | Custom row converter |
| `maxRows` | `number` | unlimited | Maximum rows to stream |
| `batchSize` | `number` | `10` | Examples per HTTP batch |
| `trainingEndpointUrl` | `string` | `endpointUrl + "/train"` | Fine-tuning endpoint URL |
| `onProgress` | `(p: TrainingProgress) => void` | — | Progress callback |

---

## `HFDatasetLoader` — dataset access

```ts
import { HFDatasetLoader } from "qubit_ai";

const loader = new HFDatasetLoader({
  hfToken: process.env.HF_TOKEN, // required for private datasets
});
```

### Constructor options

| Option | Type | Default | Description |
|---|---|---|---|
| `hfToken` | `string` | `$HF_TOKEN` env | HuggingFace API token |
| `datasetsServerUrl` | `string` | HF Datasets Server | Custom Datasets Server URL |
| `timeoutMs` | `number` | `30_000` | Per-request timeout |

### Methods

**`fetchRows(opts)`** — Fetch a single page of rows.

```ts
const page = await loader.fetchRows({
  dataset: "llm-jp/oasst2-33k-ja",
  config: "default",   // optional
  split: "train",      // optional
  offset: 0,           // optional
  limit: 50,           // optional, max 100
});
// page.rows: HFDatasetRow[]
// page.numRowsTotal: number
```

**`streamRows(opts)`** — Async generator yielding all rows page-by-page.

```ts
for await (const { rowIdx, row } of loader.streamRows({ dataset: "...", maxRows: 1000 })) {
  console.log(rowIdx, row);
}
```

**`streamExamples(opts)`** — Async generator yielding `{ prompt, completion }` pairs.

```ts
for await (const example of loader.streamExamples({
  dataset: "llm-jp/oasst2-33k-ja",
  promptField: "input",
  completionField: "output",
  maxRows: 200,
})) {
  console.log(example.prompt, "->", example.completion);
}
```

Field names (`promptField` / `completionField`) are auto-inferred when omitted, trying common names: `input`, `instruction`, `question`, `prompt` for prompts and `output`, `response`, `answer`, `completion` for completions.

**`loadExamples(opts)`** — Load all examples into memory (convenience wrapper).

```ts
const examples = await loader.loadExamples({ dataset: "...", maxRows: 100 });
```

**`preview(dataset, n)`** — Quickly fetch the first `n` examples.

```ts
const examples = await loader.preview("llm-jp/oasst2-33k-ja", 5);
```

**`fetchSplits(dataset)`** — List available splits.

```ts
const splits = await loader.fetchSplits("llm-jp/oasst2-33k-ja");
// ["train", "validation", "test"]
```

---

## `QBNNFrontalEngine` — judgment & decision-making

A **pure TypeScript** quantum-inspired judgment engine. No Python, no model weights — runs entirely in Node.js or the browser.

The engine implements the **APQB scoring model**: text signals are mapped to a pseudo-quantum angle θ, which yields a correlation score `r = cos(2θ)` normalised to 0–100.

### Core `judge()` API

```ts
import { QBNNFrontalEngine } from "qubit_ai";

const engine = new QBNNFrontalEngine();

const result = await engine.judge(
  "ユーザーの個人情報をログに記録する",   // action
  "セキュリティ監査のための操作",          // context
  { type: "safety", strictMode: true }
);

console.log(result.decision);    // "Yes" | "No"
console.log(result.score);       // 0–100
console.log(result.reasoning);   // human-readable explanation
console.log(result.confidence);  // "high" | "medium" | "low"
console.log(result.keyFactors);  // string[]
```

### Judgment types

| Type | Description |
|---|---|
| `"safety"` | Is this action safe to perform? |
| `"ethics"` | Is this ethically acceptable? |
| `"quality"` | Does this content meet quality standards? |
| `"risk"` | What is the risk level? |
| `"decision"` | General decision-making |
| `"priority"` | Task prioritisation |

### Convenience helpers

```ts
// Safety check
const safety = await engine.checkSafety(
  "データベースを削除する",
  "バックアップ済み、承認済み",
  { risks: ["データ損失", "ダウンタイム"] }
);

// Ethics evaluation
const ethics = await engine.evaluateEthics(
  "ユーザー行動を分析する",
  "匿名化されたデータのみを使用"
);

// Risk assessment (riskTolerance 0–100, higher = more permissive)
const risk = await engine.assessRisk(
  "新機能のリリース",
  "ステージングでテスト済み",
  { riskTolerance: 70 }
);

// Quality evaluation
const quality = await engine.evaluateQuality(
  "正確で明確なドキュメント",
  { requirements: ["正確性", "完全性"], userIntent: "APIリファレンス" }
);

// Task prioritisation
const { rankedTasks, scores } = await engine.prioritize(
  ["バグ修正", "新機能開発", "セキュリティパッチ"],
  "本番環境のインシデント対応"
);
console.log(rankedTasks); // sorted by QBNN priority score
```

---

## Background — APQB theory

The APQB (Adjustable Pseudo Quantum Bit) model unifies statistics, AI, and quantum mechanics via a single angle parameter θ:

```
|ψ⟩ = cosθ|0⟩ + sinθ|1⟩
r = cos(2θ)          — statistical correlation
T = |sin(2θ)|        — quantum tunnelling amplitude
r² + T² = 1          — conservation law
```

The JS engine maps text sentiment signals → θ → score (0–100), replicating the quantum-inspired reasoning of the Python QBNN model without requiring PyTorch.

---

## TypeScript support

Full type definitions are included. Import types directly:

```ts
import type {
  // Inference
  GenerateOptions,
  GenerateResult,
  // Judgment
  JudgmentResult,
  JudgmentType,
  JudgeOptions,
  // Dataset
  HFDatasetLoaderConfig,
  HFDatasetRow,
  HFDatasetPage,
  FetchRowsOptions,
  StreamRowsOptions,
  DatasetToExamplesOptions,
  TrainingExample,
  TrainFromDatasetOptions,
  TrainingProgress,
  TrainingResult,
  GenerateWithExamplesOptions,
} from "qubit_ai";
```

---

## License

MIT © tapiocaTakeshi
