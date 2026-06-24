# Qubit AI v3.0.0 — Pyodide Backend

**Quantum-inspired neural reasoning + HuggingFace dataset training, all in WebAssembly**

## What Changed

Qubit AI has been simplified and refactored to use **Pyodide** — a Python runtime compiled to WebAssembly. This enables:

- ✅ **No External Servers** - Python runs locally in your application
- ✅ **No REST API** - Direct Python execution with zero network overhead
- ✅ **Quantum-Inspired Reasoning** - APQB tensors for advanced judgment
- ✅ **HF Dataset Training** - Direct integration with HuggingFace datasets
- ✅ **Browser Compatible** - Works in Node.js and modern browsers
- ✅ **Simplified API** - Focused on quantum judgment + HF training

## What Was Removed

- ❌ REST API server (neuroquantum_api_server.py)
- ❌ LLM providers (Claude, OpenAI, HuggingFace LLM calls)
- ❌ Hybrid mode with multiple backends
- ❌ Complex configuration management

**Why?** The Pyodide version is purpose-built for quantum-inspired reasoning and HF dataset training. LLM providers add complexity without benefiting these core use cases.

## Installation

```bash
npm install qubit_ai
```

## Quick Start

```typescript
import { getQubitAIPyodide } from 'qubit_ai';

// Initialize
const qubit = getQubitAIPyodide();
await qubit.initialize();

// Make quantum-inspired judgment
const result = await qubit.judge(
  'Delete user data',
  'Production environment',
  'safety'
);

console.log(result.decision);  // "Yes" or "No"
console.log(result.score);     // 0-100
console.log(result.reasoning); // Quantum-inspired explanation

// Train on HuggingFace dataset
await qubit.trainOnHFDataset({
  dataset: 'llm-jp/oasst2-33k-ja',
  judgmentType: 'safety',
  maxExamples: 5000,
  onProgress: (p) => console.log(`${p.processedExamples}/${p.totalExamples}`),
});
```

## How It Works

```
TypeScript Code
    ↓
Pyodide Runtime (WebAssembly)
    ↓
Python (neuroquantum_layered.py)
    ↓
APQB Quantum-Inspired Tensors
    ↓
NeuroQuantumLayer (Entanglement-Inspired)
    ↓
Judgment Result (Decision, Score, Confidence)
    ↓
JSON Response
```

## Core Features

### 1. Quantum-Inspired Judgment

```typescript
// Binary decisions with confidence
const result = await qubit.judge(action, context, judgmentType);
// → { decision: "Yes"|"No", score: 0-100, confidence: "high"|"medium"|"low" }
```

### 2. Safety Checks

```typescript
const [isSafe, details] = await qubit.safetyCheck(action, context);
if (!isSafe) console.error(details.reasoning);
```

### 3. Ethics Evaluation

```typescript
const result = await qubit.ethicsCheck(
  action,
  ['stakeholders'],
  ['potential harms']
);
```

### 4. Quality Assessment

```typescript
const result = await qubit.evaluateQuality(content);
// Score: 0-100, Confidence: high/medium/low
```

### 5. Task Prioritization

```typescript
const ranked = await qubit.prioritize(
  [{ name: 'Task 1', description: '...' }, ...],
  'Constraints...'
);
// Returns: [[item, score], ...] sorted by priority
```

### 6. HuggingFace Dataset Training

```typescript
await qubit.trainOnHFDataset({
  dataset: 'dataset-name',
  judgmentType: 'safety',
  maxExamples: 5000,
  onProgress: (progress) => { ... },
});
```

## Judgment Types

| Type | Purpose |
|------|---------|
| `safety` | Is this action safe? |
| `ethics` | Is this ethically acceptable? |
| `quality` | Is content quality sufficient? |
| `risk` | What's the risk level? |
| `decision` | General decision support |
| `priority` | Task prioritization |

## Configuration

```typescript
const qubit = getQubitAIPyodide({
  productName: 'MyApp',
  strictMode: true,           // Require score >= 70 for "Yes"
  maxJudgmentHistory: 500,
});
```

## Global Functions

```typescript
import {
  judge,
  safetyCheck,
  ethicsCheck,
  evaluateQuality,
} from 'qubit_ai';

// Use global singleton
await judge(action, context, judgmentType);
await safetyCheck(action, context);
await ethicsCheck(action, stakeholders);
await evaluateQuality(content);
```

## Performance

| Operation | Time |
|-----------|------|
| Initialize Pyodide | 2-5s |
| Single judgment | 50-200ms |
| HF dataset batch | 1-2s |
| Training 1000 examples | 5-10s |

## Architecture Comparison

### v2.0.0 (LLM + NeuroQuantum with REST API)

```
┌─────────────────────┐
│   TypeScript        │
│   (Multiple LLMs)   │
└──────────┬──────────┘
           │ HTTP
           ↓
┌─────────────────────┐
│  REST API Server    │
│  (Python Flask)     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  neuroquantum_      │
│  layered.py         │
└─────────────────────┘
```

**Pros**: Multiple backends (LLM + NeuroQuantum)
**Cons**: External server, network latency, deployment complexity

### v3.0.0 (Pyodide-based)

```
┌─────────────────────┐
│   TypeScript        │
└──────────┬──────────┘
           │ (same process)
           ↓
┌─────────────────────┐
│   Pyodide WASM      │
│   Python Runtime    │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   APQB Tensors      │
│   Quantum Layers    │
│   HF Integration    │
└─────────────────────┘
```

**Pros**: Zero external servers, low latency, browser compatible, simplified
**Cons**: WASM payload size, no LLM providers

## Migration from v2.0.0

### Old Code (v2.0.0)
```typescript
import { QubitAI } from 'qubit_ai';

const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: { baseUrl: 'http://localhost:5000' }
});

const result = await qubit.judge(action, context);
```

### New Code (v3.0.0)
```typescript
import { getQubitAIPyodide } from 'qubit_ai';

const qubit = getQubitAIPyodide();
await qubit.initialize();

const result = await qubit.judge(action, context);
```

**Key Changes:**
1. Replace `new QubitAI()` with `getQubitAIPyodide()`
2. Call `await qubit.initialize()` once at startup
3. Remove `neuroquantumConfig` (no servers)
4. Remove LLM configuration options

## Features Removed in v3.0.0

These features from v2.0.0 are **not available** in v3.0.0:

- ❌ LLM providers (Claude, OpenAI, HuggingFace LLM inference)
- ❌ REST API client (NeuroQuantumAPIClient)
- ❌ REST API server setup required
- ❌ Hybrid mode with fallback
- ❌ Complex configuration management
- ❌ Multiple backend selection

**If you need these features**, continue using v2.0.0:

```bash
npm install qubit_ai@2.0.0
```

## Why Pyodide?

1. **Simplicity** - No servers to run, single npm package
2. **Performance** - Direct Python execution in WebAssembly, no network latency
3. **Reliability** - No external dependencies, fully contained
4. **Portability** - Works in Node.js, browsers, Electron, etc.
5. **Integration** - HuggingFace datasets load directly in Python
6. **Cost** - No server infrastructure needed

## Documentation

- [Quick Start](./docs/PYODIDE_QUICKSTART.md) - 5-minute setup
- [Examples](./examples/qubit-pyodide.ts) - 9 complete examples
- [API Reference](./docs/API.md) - Complete API

## Examples

```typescript
// Safety check with quantum reasoning
const [isSafe, details] = await qubit.safetyCheck(
  'Log API credentials',
  'Production'
);

// Ethics with stakeholder analysis
const result = await qubit.ethicsCheck(
  'Share user location',
  ['users', 'privacy advocates'],
  ['privacy violation', 'trust loss']
);

// Train on HF dataset
await qubit.trainOnHFDataset({
  dataset: 'llm-jp/oasst2-33k-ja',
  judgmentType: 'safety',
  maxExamples: 5000,
});

// Prioritize tasks with quantum-inspired reasoning
const ranked = await qubit.prioritize([
  { name: 'Security patch', description: 'Critical bug' },
  { name: 'Feature X', description: 'User request' },
]);
```

## Browser Usage

```html
<script src="https://cdn.jsdelivr.net/npm/qubit_ai@3.0.0/dist/esm/index.js"></script>
<script>
  const { getQubitAIPyodide } = window.qubitAI;
  
  async function runQubit() {
    const qubit = getQubitAIPyodide();
    await qubit.initialize();
    
    const result = await qubit.judge('action', 'context', 'safety');
    console.log(result);
  }
  
  runQubit();
</script>
```

## License

MIT © tapiocaTakeshi
