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
  JudgmentResult,
  JudgmentType,
  GenerateOptions,
} from "qubit_ai";
```

---

## License

MIT © tapiocaTakeshi
