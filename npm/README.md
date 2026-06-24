# qubit_ai

**General-Purpose Quantum-Inspired Large Language Model (LLM) for JavaScript / TypeScript**

`qubit_ai` is the official JavaScript/TypeScript SDK for [Qubit AI](https://github.com/tapiocaTakeshi/Qubit) — a **next-generation general-purpose language model** that combines the conversational capabilities of ChatGPT with a quantum-inspired neural network architecture (APQB/QBNN).

## What is Qubit AI?

Qubit AI is a ChatGPT-like large language model built on **quantum-inspired computational principles**:

- 🧠 **General-Purpose Conversational AI**: Natural dialogue, code generation, summarization, translation, reasoning, and more
- ⚛️ **Quantum-Inspired Architecture**: Uses APQB (Adjustable Pseudo Quantum Bit) theory for more efficient and theoretically-grounded AI
- 🎯 **Judgment & Decision Engine**: Built-in specialized judgments (safety, ethics, quality, risk, decision, priority)
- 🔧 **Multiple Backends**: Choose from pure QBNN, LLM providers (Claude, OpenAI, HuggingFace), Python NeuroQuantum, or hybrid mode
- ⚡ **Production-Ready**: Retry logic, fallback mechanisms, history tracking, and fine-tuning capabilities

### Multiple Backend Support

- **QBNN Engine**: Pure JavaScript quantum-inspired inference (fast, offline)
- **LLM Providers**: Claude, OpenAI, HuggingFace (state-of-the-art generative AI)
- **NeuroQuantum**: Python backend quantum-inspired neural networks via REST API
- **Hybrid**: Automatic fallback and load balancing across backends

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

## Key Features

### 🤖 As a General-Purpose LLM
- ✅ **Natural Conversation**: Chat-based dialogue like ChatGPT
- ✅ **Code Generation**: Generate and explain code in multiple languages
- ✅ **Text Processing**: Summarization, translation, paraphrasing, creative writing
- ✅ **Complex Reasoning**: Mathematical problem-solving, logical analysis, research synthesis
- ✅ **Controllable Creativity**: APQB θ parameter for fine-tuned output diversity

### 🎯 As a Judgment Engine
- ✅ **6 Judgment Types**: safety, ethics, quality, risk, decision, priority
- ✅ **Multiple Backends**: Switch between QBNN (fast), LLM (accurate), NeuroQuantum (quantum-inspired), or hybrid mode
- ✅ **Generative AI Providers**: Claude, OpenAI, HuggingFace
- ✅ **Quantum-Inspired Inference**: Python NeuroQuantum neural networks via REST API

### ⚙️ Production Features
- ✅ **Reliability**: Automatic retry logic, fallback mechanisms, strict mode
- ✅ **History Tracking**: Judge history with configurable limits
- ✅ **Fine-tuning**: Train on HuggingFace datasets for custom tasks
- ✅ **Backward Compatible**: Upgrade from v1 without code changes
- ✅ **TypeScript Support**: Full type safety and intellisense

## Modules

| Export | Description |
|---|---|
| `QubitAI` | High-level judgment engine with backend selection |
| `QBNNFrontalEngine` | Pure-JS quantum-inspired QBNN engine (low-level) |
| `LLMFrontalEngine` | LLM-based judgment with multiple providers |
| `NeuroQuantumFrontalEngine` | Python REST API quantum-inspired backend |
| `HybridFrontalEngine` | Combines LLM and heuristic with blending |
| `LLMProvider` | Abstract base for pluggable LLM providers |
| `ClaudeProvider` / `OpenAIProvider` / `HuggingFaceProvider` | LLM implementations |
| `NeuroQuantumAPIClient` | REST client for Python backend |
| `LLMTrainer` | HuggingFace dataset fine-tuning |
| `NeuroQuantumClient` | HuggingFace inference endpoint client |
| `HFDatasetLoader` | HuggingFace Datasets API client |

---

## `QubitAI` — main high-level API

`QubitAI` is the recommended entry point. It wraps `QBNNFrontalEngine` with a Python-compatible API (mirrors `qubit_ai.py`) and tracks judgment history.

### Quick start

```ts
import { QubitAI } from "qubit_ai";

const qubit = new QubitAI();

const result = await qubit.judge(
  "ユーザーデータをログ出力",
  "デバッグモード"
);

console.log(result.decision);    // "Yes" | "No"
console.log(result.score);       // 0–100
console.log(result.confidence);  // "high" | "medium" | "low"
console.log(result.reasoning);   // human-readable explanation
console.log(result.factors);     // string[]
```

### Constructor options

```ts
// Default: Heuristic-based QBNN (fast, offline)
const qubit = new QubitAI({
  productName: "MyAI",          // default: "Qubit.ai"
  strictMode: true,             // default: false — score ≥ 70 = Yes
  maxJudgmentHistory: 500,      // default: 100
});

// With LLM backend (Claude, OpenAI, HuggingFace)
const qubit = new QubitAI({
  llmEnabled: true,
  llmProvider: 'claude',        // or 'openai' | 'hf'
  llmConfig: {
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: 'claude-3-5-sonnet-20241022',
    temperature: 0.7,
  },
});

// With NeuroQuantum backend (Python quantum-inspired)
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
    timeout: 30000,
  },
});

// Hybrid mode (NeuroQuantum + heuristic fallback, recommended for production)
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  fallbackToHeuristics: true,   // Use heuristics if API fails
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
  },
});
```

### `judge(action, context, judgmentType?, strict?)`

General-purpose judgment. `judgmentType` defaults to `"safety"`.

```ts
const result = await qubit.judge(
  "ユーザーのAPIキーを外部サーバーに送信",
  "本番環境",
  "safety",
  true  // per-call strict override
);
```

Available judgment types:

| Type | Description |
|---|---|
| `"safety"` | Is this action safe to perform? |
| `"ethics"` | Is this ethically acceptable? |
| `"quality"` | Does this content meet quality standards? |
| `"risk"` | What is the risk level? |
| `"decision"` | General decision-making |
| `"priority"` | Task prioritisation |

### `safetyCheck(action, context, opts?)`

Returns `[isSafe, result]`.

```ts
const [isSafe, result] = await qubit.safetyCheck(
  "APIキーをログに出力",
  "本番環境",
  { risks: ["情報漏洩", "セキュリティ違反"] }
);

if (!isSafe) {
  console.warn("Blocked:", result.reasoning);
}
```

### `evaluateQuality(content, opts?)`

```ts
const quality = await qubit.evaluateQuality(
  "生成されたテキスト",
  { criteria: ["正確性", "読みやすさ", "完全性"] }
);
console.log(`Quality: ${quality.score}/100`);
```

### `ethicsCheck(action, stakeholders?, potentialHarms?)`

```ts
const ethics = await qubit.ethicsCheck(
  "ユーザーデータを分析",
  ["ユーザー", "企業"],
  ["プライバシー侵害", "差別"]
);
```

### `prioritize(items, constraints?)`

Ranks items by priority score (0–1). Returns `[item, score][]` sorted descending.

```ts
const ranked = await qubit.prioritize(
  [
    { name: "バグ修正", description: "本番クラッシュ" },
    { name: "新機能", description: "検索機能" },
    { name: "ドキュメント", description: "API ドキュメント更新" },
  ],
  "チーム3人、納期2週間"
);

ranked.forEach(([item, score]) => {
  console.log(`${item.name}: ${(score * 100).toFixed(1)}%`);
});
```

### Status & history

```ts
qubit.getInfo();    // product, version, sessionId, status
qubit.getStatus();  // frontalEngineAvailable, judgmentHistorySize, …
qubit.getHistory(5);   // last 5 JudgmentRecord entries
qubit.clearHistory();
```

### `explain(result)`

Format a result as a readable Japanese string.

```ts
const result = await qubit.judge("...", "...");
console.log(qubit.explain(result));
// 【判断結果】
// 決定: Yes
// スコア: 75/100
// 信頼度: high
// ...
```

---

## Global convenience functions

Use these when you don't need instance-level config or history:

```ts
import { judge, safetyCheck, evaluateQuality, ethicsCheck } from "qubit_ai";

const result  = await judge("アクション", "コンテキスト");
const [ok, r] = await safetyCheck("アクション", "コンテキスト");
const quality = await evaluateQuality("コンテンツ");
const ethics  = await ethicsCheck("アクション", ["ユーザー"]);
```

All convenience functions share a global `QubitAI` singleton.

### Singleton management

```ts
import { getQubitAI, resetQubitAI } from "qubit_ai";

// Get or create the global instance (optionally configure on first call)
const qubit = getQubitAI({ strictMode: true });

// Reset — next call to getQubitAI() creates a fresh instance
resetQubitAI();
```

---

## `QBNNFrontalEngine` — low-level engine

A pure TypeScript quantum-inspired judgment engine. No Python, no model weights — runs entirely in Node.js or the browser.

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
  promptField: "input",
  completionField: "output",
  maxRows: 500,
  batchSize: 10,
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
  config: "default",
  split: "train",
  offset: 0,
  limit: 50,         // max 100
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
  // QubitAI types
  QubitAIConfig,
  QubitAIResult,
  QubitAIInfo,
  QubitAIStatus,
  JudgmentRecord,
  PriorityItem,
  PriorityItemResult,
  // Judgment engine types
  JudgmentResult,
  JudgmentType,
  JudgeOptions,
  GenerateOptions,
  GenerateResult,
  SafetyCheckOptions,
  QualityEvalOptions,
  RiskAssessmentOptions,
  // Dataset types
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

### Return types

**`QubitAIResult`** (returned by `judge`, `safetyCheck`, `evaluateQuality`, `ethicsCheck`):

```ts
interface QubitAIResult {
  decision:   "Yes" | "No";
  score:      number;                        // 0–100
  reasoning:  string;
  confidence: "high" | "medium" | "low";
  factors:    string[];
  timestamp:  string;                        // ISO 8601
}
```

**`PriorityItemResult`** (each element returned by `prioritize`):

```ts
type PriorityItemResult = [item: PriorityItem, score: number]; // score 0–1
```

### Scoring guide

```
Score 70–100  → Strong Yes (Recommended)
Score 50–69   → Weak Yes (Verify first)
Score 30–49   → Weak No (Concerns exist)
Score 0–29    → Strong No (Not recommended)

Confidence levels:
  high    Definitive decision
  medium  Some uncertainty
  low     Ambiguous — human review recommended
```

---

## 🧠 LLM Backend — Generative AI Reasoning

Use Claude, OpenAI, or HuggingFace for advanced generative reasoning:

```ts
const qubit = new QubitAI({
  llmEnabled: true,
  llmProvider: 'claude',
  llmConfig: {
    apiKey: process.env.ANTHROPIC_API_KEY,
  },
});

const result = await qubit.judge(
  "Delete user data",
  "Production environment",
  "safety"
);

console.log(result.reasoning);  // LLM-generated explanation
```

### LLM Providers

| Provider | Setup |
|----------|-------|
| **Claude** | `llmProvider: 'claude'`, set `ANTHROPIC_API_KEY` |
| **OpenAI** | `llmProvider: 'openai'`, set `OPENAI_API_KEY` |
| **HuggingFace** | `llmProvider: 'hf'`, set `HF_TOKEN` |

See [LLM Integration Guide](../docs/LLM_INTEGRATION.md) for detailed setup.

---

## 🌌 NeuroQuantum Backend — Python Quantum-Inspired Neural Networks

Connect to Python backend for advanced quantum-inspired reasoning:

```ts
// Start Python server first:
// python neuroquantum_api_server.py --host 127.0.0.1 --port 5000

const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
  },
});

const result = await qubit.judge(
  "データを削除",
  "本番環境",
  "safety"
);

console.log(result.reasoning);  // Quantum-inspired neural analysis
```

### Production Setup (Recommended)

Use **hybrid mode** for maximum reliability:

```ts
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  fallbackToHeuristics: true,  // Use QBNN if API fails
  neuroquantumConfig: {
    baseUrl: 'http://api.example.com:5000',
    timeout: 30000,
    maxRetries: 3,
  },
});
```

See [NeuroQuantum Integration Guide](../docs/NEUROQUANTUM_INTEGRATION.md) for complete setup and deployment.

---

## 📚 Training on HuggingFace Datasets

Fine-tune LLMs for specialized judgment tasks:

```ts
const qubit = new QubitAI({
  llmEnabled: true,
  llmProvider: 'hf',
});

// Train on safety dataset
const result = await qubit.trainOnHFDataset({
  dataset: 'llm-jp/oasst2-33k-ja',
  judgmentType: 'safety',
  maxExamples: 5000,
  onProgress: (prog) => {
    console.log(`${prog.processedExamples}/${prog.totalExamples}`);
  },
});

// Evaluate on test set
const metrics = await qubit.evaluateFineTunedModel({
  dataset: 'llm-jp/oasst2-33k-ja',
  split: 'test',
  sampleSize: 100,
});

console.log(`Accuracy: ${metrics.accuracy.toFixed(2)}`);
```

---

## Environment Variables

```bash
# Backend selection
QUBIT_NEUROQUANTUM_ENABLED=true          # Use Python backend
QUBIT_LLM_ENABLED=true                   # Use LLM backend
QUBIT_FALLBACK_TO_HEURISTICS=true        # Hybrid mode

# LLM configuration
QUBIT_LLM_PROVIDER=claude                # claude | openai | hf
ANTHROPIC_API_KEY=sk_...
OPENAI_API_KEY=sk_...
HF_TOKEN=hf_...

# NeuroQuantum configuration
QUBIT_NEUROQUANTUM_BASE_URL=http://localhost:5000
QUBIT_NEUROQUANTUM_TIMEOUT=30000

# Judgment settings
QUBIT_STRICT_MODE=true                   # Score ≥ 70 = Yes
```

---

## 📖 Documentation

- [Quick Start](../docs/NEUROQUANTUM_QUICKSTART.md) — Get running in 5 minutes
- [NeuroQuantum Integration](../docs/NEUROQUANTUM_INTEGRATION.md) — Full Python backend guide
- [LLM Integration](../docs/LLM_INTEGRATION.md) — Generative AI setup
- [Configuration](../docs/CONFIGURATION.md) — All config options
- [Examples](../examples/) — Complete usage examples
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md) — Architecture overview

---

## License

MIT © tapiocaTakeshi
