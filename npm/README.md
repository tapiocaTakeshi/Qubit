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

Requires **Node.js ≥ 18**.

---

## Modules

| Export | Description |
|---|---|
| `QubitAI` | High-level judgment API — TypeScript port of `qubit_ai.py` |
| `NeuroQuantumEngine` | Pure-TS port of `neuroquantum_layered.py` math (QBNN layers + APQB) |
| `QBNNFrontalEngine` | Lightweight heuristic judgment engine (no external deps) |
| `NeuroQuantumClient` | HTTP client for the neuroQ HuggingFace inference endpoint |
| `HFDatasetLoader` | HuggingFace Datasets API client — fetch, stream, and convert dataset rows |

---

## `QubitAI` — primary judgment API

`QubitAI` is the TypeScript port of `qubit_ai.py`. It uses **`NeuroQuantumEngine`** internally — a pure-TypeScript implementation of the QBNN math from `neuroquantum_layered.py`. No HTTP endpoints, no Python runtime required.

```ts
import { QubitAI } from "qubit_ai";

const qubit = new QubitAI();

// Judge an action
const result = await qubit.judge("APIキーをログに記録", "本番環境", "safety");
console.log(result.decision);    // "Yes" | "No"
console.log(result.score);       // 0–100
console.log(result.reasoning);   // human-readable explanation (Japanese)
console.log(result.confidence);  // "high" | "medium" | "low"
console.log(result.factors);     // string[] — key contributing factors

// Explain in natural language
console.log(qubit.explain(result));
```

### Constructor options

| Option | Type | Default | Description |
|---|---|---|---|
| `version` | `string` | `"1.1.0"` | Library version string |
| `productName` | `string` | `"Qubit.ai"` | Product name |
| `description` | `string` | `"Claude's Quantum Prefrontal Cortex"` | Product description |
| `strictMode` | `boolean` | `false` | Require score ≥ 70 for a "Yes" decision |
| `enableLogging` | `boolean` | `true` | Enable internal logging |
| `maxJudgmentHistory` | `number` | `100` | Maximum judgment records to keep |

### Judgment methods

**`judge(action, context, judgmentType?, strict?)`** — Core judgment call.

```ts
const result = await qubit.judge(
  "ユーザーの個人情報を収集",   // action
  "GDPRが適用される本番環境",   // context
  "safety",                     // type: "safety" | "ethics" | "quality" | "risk" | "decision" | "priority"
  true,                         // strictMode override (optional)
);
```

**`safetyCheck(action, context, opts?)`** — Returns `[boolean, QubitAIResult]`.

```ts
const [safe, result] = await qubit.safetyCheck(
  "外部APIにデータを送信",
  "本番環境",
  { risks: ["情報漏洩", "プライバシー違反"] }
);
if (!safe) console.log("安全でない:", result.reasoning);
```

**`evaluateQuality(content, opts?)`**

```ts
const result = await qubit.evaluateQuality("正確で明確なドキュメント", {
  requirements: ["正確性", "完全性"],
  userIntent: "APIリファレンス",
});
```

**`ethicsCheck(action, stakeholders?, potentialHarms?)`**

```ts
const result = await qubit.ethicsCheck(
  "ユーザー行動を分析する",
  ["ユーザー", "社会"],
  ["プライバシー侵害"],
);
```

**`prioritize(items, constraints?)`** — Returns items sorted by priority score (descending).

```ts
const ranked = await qubit.prioritize(
  [
    { name: "セキュリティパッチ", description: "本番環境の脆弱性を修正" },
    { name: "新機能開発", description: "UIコンポーネントの追加" },
    { name: "ドキュメント更新", description: "APIリファレンスの整備" },
  ],
  "リソース制限あり、本番稼働中"
);
for (const [item, score] of ranked) {
  console.log(item.name, score.toFixed(2)); // score: 0–1
}
```

### Status & history

```ts
qubit.getInfo();    // product, version, sessionId, status
qubit.getStatus();  // frontalEngineAvailable, judgmentHistorySize, ...
qubit.getHistory(10);  // last N JudgmentRecord entries
qubit.clearHistory();
```

### Singleton & convenience functions

```ts
import { getQubitAI, resetQubitAI, judge, safetyCheck, evaluateQuality, ethicsCheck } from "qubit_ai";

// Module-level convenience (use shared singleton)
const result  = await judge("行動", "コンテキスト", "ethics");
const [safe]  = await safetyCheck("操作", "環境");
const quality = await evaluateQuality("コンテンツ", { requirements: ["正確性"] });
const ethics  = await ethicsCheck("行動", ["ユーザー"]);

// Reset the singleton (e.g. between tests)
resetQubitAI();
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

---

## `NeuroQuantumEngine` — pure-TS QBNN engine

`NeuroQuantumEngine` is a faithful TypeScript port of the mathematical core of `neuroquantum_layered.py`:

- **`QBNNLayer.forward()`** — entanglement correction with dynamic sinusoidal λ
- **`QBNNAttention.forward()`** — QBNN-enhanced multi-head attention (action × context)
- **Dynamic θ phase** from `NeuroQuantumAI.generate()` — `0.5 + 0.5×sin(step×0.2)`
- **APQB scoring** — `r = cos(2θ)`, `score = ((r+1)/2) × 100`

Defaults match the `'cpu'` tier of `NeuroQuantumConfig`: 3 layers, 4 heads, λ = 0.5.

```ts
import { NeuroQuantumEngine } from "qubit_ai";

const engine = new NeuroQuantumEngine({
  numLayers:      3,    // QBNN transformer blocks (default: 3)
  numHeads:       4,    // attention heads (default: 4)
  lambdaEntangle: 0.5,  // entanglement strength λ (default: 0.5)
});

const result = await engine.judge(
  "ユーザーデータをログに記録",
  "本番環境",
  { type: "safety", strictMode: true }
);

console.log(result.decision);   // "Yes" | "No"
console.log(result.score);      // 0–100 (APQB score)
console.log(result.system);     // "qbnn"
console.log(result.keyFactors); // action signal, context signal, attention weight, …
```

### Convenience wrappers

```ts
// Safety check (with optional risks / constraints)
const safety = await engine.checkSafety("操作", "環境", { risks: ["データ損失"] });

// Ethics evaluation
const ethics = await engine.evaluateEthics("行動", "コンテキスト");

// Risk assessment (riskTolerance 0–100, higher = more permissive)
const risk = await engine.assessRisk("リリース", "ステージング環境", { riskTolerance: 70 });

// Quality evaluation
const quality = await engine.evaluateQuality("コンテンツ", { requirements: ["正確性"] });

// Task prioritisation — returns sorted { rankedTasks, scores, reasonings }
const { rankedTasks, scores } = await engine.prioritize(
  ["バグ修正", "新機能開発", "セキュリティパッチ"],
  "本番環境のインシデント対応"
);
```

---

## `QBNNFrontalEngine` — lightweight heuristic engine

A lightweight pure-TypeScript judgment engine based on keyword scoring and APQB theory. Useful when minimal overhead is needed and the full QBNN layer math is not required.

```ts
import { QBNNFrontalEngine } from "qubit_ai";

const engine = new QBNNFrontalEngine();

const result = await engine.judge(
  "ユーザーの個人情報をログに記録する",
  "セキュリティ監査のための操作",
  { type: "safety", strictMode: true }
);

console.log(result.decision);    // "Yes" | "No"
console.log(result.score);       // 0–100
console.log(result.confidence);  // "high" | "medium" | "low"
```

---

## `NeuroQuantumClient` — text generation via HuggingFace endpoint

HTTP client for the neuroQ HuggingFace inference endpoint. Requires an `HF_TOKEN` for authenticated access.

```ts
import { NeuroQuantumClient } from "qubit_ai";

const client = new NeuroQuantumClient({
  hfToken: process.env.HF_TOKEN,
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

```ts
import { HFDatasetLoader, NeuroQuantumClient } from "qubit_ai";

const loader = new HFDatasetLoader({ hfToken: process.env.HF_TOKEN });
const client = new NeuroQuantumClient({ hfToken: process.env.HF_TOKEN });

const examples = await loader.preview("llm-jp/oasst2-33k-ja", 3);

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

### Training from a HuggingFace dataset

```ts
const result = await client.trainFromDataset({
  dataset: "llm-jp/oasst2-33k-ja",
  promptField: "input",
  completionField: "output",
  maxRows: 500,
  batchSize: 10,
  trainingEndpointUrl: "https://your-training-endpoint/train",
  onProgress: (p) => {
    console.log(`${p.processedExamples}/${p.totalExamples} examples`);
  },
});

console.log(result.status);        // "completed" | "partial" | "failed"
console.log(result.totalExamples);
console.log(result.durationMs);
```

---

## `HFDatasetLoader` — dataset access

```ts
import { HFDatasetLoader } from "qubit_ai";

const loader = new HFDatasetLoader({
  hfToken: process.env.HF_TOKEN, // required for private datasets
});
```

**`fetchRows(opts)`** — single page of rows.

```ts
const page = await loader.fetchRows({
  dataset: "llm-jp/oasst2-33k-ja",
  offset: 0,
  limit: 50,
});
```

**`streamRows(opts)`** — async generator yielding all rows.

```ts
for await (const { rowIdx, row } of loader.streamRows({ dataset: "...", maxRows: 1000 })) {
  console.log(rowIdx, row);
}
```

**`streamExamples(opts)`** — async generator of `{ prompt, completion }` pairs.

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

**`preview(dataset, n)`** — first `n` examples.

```ts
const examples = await loader.preview("llm-jp/oasst2-33k-ja", 5);
```

**`fetchSplits(dataset)`** — available splits.

```ts
const splits = await loader.fetchSplits("llm-jp/oasst2-33k-ja");
// ["train", "validation", "test"]
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

`NeuroQuantumEngine` ports the full QBNN pipeline from `neuroquantum_layered.py`:
text signals → keyword extraction → QBNN attention (action × context) → N QBNN layers (dynamic λ entanglement) → dynamic θ phase → APQB score 0–100.

---

## TypeScript support

Full type definitions are included:

```ts
import type {
  // QubitAI
  QubitAIConfig,
  QubitAIResult,
  QubitAIInfo,
  QubitAIStatus,
  JudgmentRecord,
  PriorityItem,
  PriorityItemResult,
  // NeuroQuantumEngine
  NeuroQuantumEngineConfig,
  // Judgment
  JudgmentResult,
  JudgmentType,
  JudgeOptions,
  // Inference
  GenerateOptions,
  GenerateResult,
  NeuroQuantumClientConfig,
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
