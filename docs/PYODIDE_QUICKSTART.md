# Qubit AI v3.0.0 — Pyodide Backend Quick Start

Quantum-inspired neural networks + HuggingFace dataset training, all in WebAssembly.

## Installation

```bash
npm install qubit_ai
```

## Basic Usage

```typescript
import { getQubitAIPyodide } from 'qubit_ai';

// Initialize
const qubit = getQubitAIPyodide();
await qubit.initialize();

// Make judgment
const result = await qubit.judge(
  'Delete user data',
  'Production environment',
  'safety'
);

console.log(result.decision);  // "Yes" or "No"
console.log(result.score);     // 0-100
console.log(result.reasoning); // Quantum-inspired explanation
```

## Safety Check

```typescript
const [isSafe, details] = await qubit.safetyCheck(
  'API key in logs',
  'Production'
);

if (!isSafe) {
  console.error('Blocked:', details.reasoning);
}
```

## Train on HuggingFace Dataset

```typescript
const result = await qubit.trainOnHFDataset({
  dataset: 'llm-jp/oasst2-33k-ja',
  judgmentType: 'safety',
  maxExamples: 5000,
  onProgress: (progress) => {
    console.log(
      `Progress: ${progress.processedExamples}/${progress.totalExamples}`
    );
  },
});

console.log(`Trained on ${result.totalExamples} examples`);
```

## Ethics & Quality Evaluation

```typescript
// Ethics check
const ethics = await qubit.ethicsCheck(
  'Share user data',
  ['users', 'regulators'],     // stakeholders
  ['privacy loss']              // potential harms
);

// Quality evaluation
const quality = await qubit.evaluateQuality(
  'AI-generated content'
);

console.log('Quality score:', quality.score);
```

## Task Prioritization

```typescript
const ranked = await qubit.prioritize([
  { name: 'Security patch', description: 'Critical fix' },
  { name: 'Documentation', description: 'Update docs' },
  { name: 'New feature', description: 'Feature X' },
]);

ranked.forEach(([item, score]) => {
  console.log(`${item.name}: ${(score * 100).toFixed(0)}%`);
});
```

## Judgment History

```typescript
// Get recent judgments
const history = qubit.getHistory(10);
history.forEach((record) => {
  console.log(`${record.judgmentType}: ${record.decision}`);
});

// Clear history
qubit.clearHistory();

// Get status
const status = await qubit.getStatus();
console.log(status.frontalEngineAvailable);
```

## System Information

```typescript
const info = qubit.getInfo();
console.log(info.product);    // "Qubit.ai"
console.log(info.version);    // "3.0.0"
console.log(info.sessionId);  // Unique session ID
```

## Configuration

```typescript
import { getQubitAIPyodide } from 'qubit_ai';

const qubit = getQubitAIPyodide({
  productName: 'MyApp',
  strictMode: true,           // Require score >= 70 for "Yes"
  maxJudgmentHistory: 500,
});
```

## Convenience Functions

```typescript
import {
  judge,
  safetyCheck,
  ethicsCheck,
  evaluateQuality,
} from 'qubit_ai';

// Use global singleton
const result = await judge('action', 'context', 'safety');
const [safe, details] = await safetyCheck('action', 'context');
const ethics = await ethicsCheck('action', ['stakeholders']);
const quality = await evaluateQuality('content');
```

## How It Works

```
TypeScript Code
    ↓
Pyodide Runtime (WebAssembly)
    ↓
Python (neuroquantum_layered.py)
    ↓
Quantum-Inspired APQB Tensors
    ↓
QBNN Scoring Result
    ↓
JSON Response
```

## Features

- ✅ **Zero External Servers**: All Python runs in WASM
- ✅ **Quantum-Inspired**: APQB tensors for advanced reasoning
- ✅ **HF Dataset Training**: Direct integration with HuggingFace
- ✅ **Type-Safe**: Full TypeScript support
- ✅ **Browser Compatible**: Works in Node.js and browsers
- ✅ **Offline**: No internet required after initialization

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Initialize | 2-5s | Load Pyodide + Python packages |
| Judgment | 50-200ms | Per-action quantum analysis |
| Train batch | 1-2s | Per-batch HF dataset processing |

## Troubleshooting

### Pyodide fails to load
```typescript
try {
  await qubit.initialize();
} catch (error) {
  console.error('Pyodide failed:', error);
  // Fallback to heuristics
}
```

### Memory issues
```typescript
// Reset instance to free memory
const { resetQubitAIPyodide } = require('qubit_ai');
resetQubitAIPyodide();
```

### Training slow
```typescript
// Reduce batch size
const result = await qubit.trainOnHFDataset({
  dataset: 'dataset-name',
  judgmentType: 'safety',
  maxExamples: 1000,  // Reduce examples
});
```

## See Also

- [Implementation Details](../IMPLEMENTATION_SUMMARY.md)
- [API Reference](./API.md)
- [Examples](../examples/)

## License

MIT © tapiocaTakeshi
