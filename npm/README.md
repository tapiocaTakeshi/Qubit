# qubit_ai

**Generative AI & Model Training SDK for JavaScript / TypeScript**

`qubit_ai` is the official JavaScript/TypeScript SDK for [Qubit AI](https://github.com/tapiocaTakeshi/Qubit) — a **next-generation language model** that combines generative AI capabilities with quantum-inspired neural network architecture for text generation and model training.

## What is Qubit AI?

Qubit AI provides:

- 🧠 **Text Generation**: High-quality, contextual text generation with multiple sampling strategies
- 📚 **HuggingFace Integration**: Direct dataset access and fine-tuning on community models
- ⚛️ **Quantum-Inspired Backend**: Optional Python NeuroQuantum backend for advanced inference
- 🎯 **Fine-tuning**: Train models on HuggingFace datasets with streaming support
- ⚡ **Production-Ready**: Retry logic, timeout handling, and comprehensive error management

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

- ✅ **Text Generation**: Generate contextual text with configurable sampling (temperature, top-k, top-p)
- ✅ **HuggingFace Datasets**: Stream and load datasets directly from HuggingFace Hub
- ✅ **Fine-tuning**: Train models on custom datasets with batch processing
- ✅ **Few-shot Learning**: In-context learning with dataset examples
- ✅ **NeuroQuantum Support**: Optional Python backend for quantum-inspired inference
- ✅ **Full TypeScript Support**: Complete type safety and IDE support

---

## Modules

| Export | Description |
|---|---|
| `NeuroQuantumClient` | Text generation with HuggingFace inference endpoints |
| `HFDatasetLoader` | Load and stream HuggingFace datasets |
| `LLMTrainer` | Fine-tune models on HuggingFace datasets |

---

## `NeuroQuantumClient` — Text Generation

Generate text using HuggingFace inference endpoints.

### Quick start

```ts
import { NeuroQuantumClient } from "qubit_ai";

const client = new NeuroQuantumClient({
  hfToken: process.env.HF_TOKEN, // optional for public endpoints
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

### Generation options

```ts
interface GenerateOptions {
  maxNewTokens?: number;        // Max tokens to generate (default: 100)
  temperature?: number;          // Sampling temperature (default: 0.7)
  topK?: number;                // Top-k sampling (default: 40)
  topP?: number;                // Top-p (nucleus) sampling (default: 0.9)
  repetitionPenalty?: number;   // Penalize repeated tokens (default: 1.0)
}
```

---

## `HFDatasetLoader` — Dataset Access

Load and stream datasets from HuggingFace Hub.

```ts
import { HFDatasetLoader } from "qubit_ai";

const loader = new HFDatasetLoader({
  hfToken: process.env.HF_TOKEN, // required for private datasets
});
```

### Methods

**`fetchRows(opts)`** — Fetch a single page of rows.

```ts
const page = await loader.fetchRows({
  dataset: "llm-jp/oasst2-33k-ja",
  config: "default",
  split: "train",
  offset: 0,
  limit: 50,
});
// page.rows: HFDatasetRow[]
// page.numRowsTotal: number
```

**`streamRows(opts)`** — Async generator yielding rows page-by-page.

```ts
for await (const { rowIdx, row } of loader.streamRows({
  dataset: "llm-jp/oasst2-33k-ja",
  maxRows: 1000,
})) {
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

**`loadExamples(opts)`** — Load all examples into memory.

```ts
const examples = await loader.loadExamples({
  dataset: "llm-jp/oasst2-33k-ja",
  maxRows: 100,
});
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

## Few-shot Generation with Dataset Examples

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

### Few-shot options

| Option | Type | Default | Description |
|---|---|---|---|
| `numExamples` | `number` | `3` | Number of examples to include |
| `exampleSeparator` | `string` | `"\n\n"` | Separator between examples |
| `exampleTemplate` | `string` | `"Q: {prompt}\nA: {completion}"` | Format for each example |
| `queryTemplate` | `string` | `"Q: {prompt}\nA:"` | Format for the query |

---

## Fine-tuning on HuggingFace Datasets

Train models on custom datasets using streaming batch processing:

```ts
const client = new NeuroQuantumClient({
  hfToken: process.env.HF_TOKEN,
});

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

### Fine-tuning options

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

## Environment Variables

```bash
# HuggingFace configuration
HF_TOKEN=hf_...

# NeuroQuantum configuration
QUBIT_NEUROQUANTUM_BASE_URL=https://api.huggingface.co/models/
QUBIT_NEUROQUANTUM_TIMEOUT=600000
```

---

## TypeScript Support

**Fully compatible with TypeScript 5.0+, TypeScript 6.0+**. Full type definitions are included:

```ts
import type {
  GenerateOptions,
  GenerateResult,
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

**TypeScript Version Requirements:**
- TypeScript 5.0+: Fully supported
- TypeScript 6.0+: Fully supported with modern `moduleResolution: "node16"`

---

## 📚 Documentation

- [Quick Start](../docs/NEUROQUANTUM_QUICKSTART.md) — Get running in 5 minutes
- [Configuration](../docs/CONFIGURATION.md) — All config options
- [Examples](../examples/) — Complete usage examples

---

## License

MIT © tapiocaTakeshi
