# Qubit AI v4.0.0 — Pyodide + NeuroQuantum Generative AI

**Quantum-inspired text generation without external APIs**

Pure in-browser or Node.js generative AI using Pyodide and quantum-inspired sampling strategies from neuroquantum_layered.py.

## Installation

```bash
npm install qubit_ai
```

## Quick Start

### Basic Text Generation

```typescript
import { getQubitAIGenerative } from 'qubit_ai';

const qubit = getQubitAIGenerative();

const result = await qubit.generate('こんにちは');

console.log(result.text);      // Generated text
console.log(result.tokensUsed); // Token count
```

### Custom Configuration

```typescript
const qubit = getQubitAIGenerative({
  vocabSize: 32000,
  seed: 42,
  sessionKey: 'my-session'
});

const result = await qubit.generate(
  'Write a haiku about AI',
  {
    temperature: 0.8,
    maxTokens: 100,
    topK: 50,
    topP: 0.95,
  }
);

console.log(result.text);
```

## API Reference

### QubitAIGenerative Class

#### Constructor

```typescript
const qubit = getQubitAIGenerative(config?: {
  vocabSize?: number;        // Vocabulary size (default: 32000)
  seed?: number;             // Random seed for reproducibility
  sessionKey?: string;       // Session identifier (default: 'default')
});
```

#### Methods

##### `generate(prompt, options)`

Generate text from a prompt.

```typescript
const result = await qubit.generate(
  'Translate to English: こんにちは',
  {
    temperature?: 0.7,
    maxTokens?: 500,
    topK?: 40,
    topP?: 0.9,
    repetitionPenalty?: 1.2,
  }
);

// Returns: { text, finishReason, tokensUsed, generatedAt }
```

**Options:**
- `temperature` (0-2): Randomness (default: 0.7)
  - Lower: More deterministic
  - Higher: More creative
- `maxTokens`: Max output length (default: 500)
- `topK`: Top-K sampling (default: 40)
  - Only sample from top K tokens
- `topP`: Nucleus sampling (default: 0.9)
  - Sample until cumulative probability reaches topP
- `repetitionPenalty`: Penalize repeated tokens (default: 1.2)
  - Higher: Stronger penalty

##### `generateWithExamples(prompt, examples, options)`

Few-shot generation with examples.

```typescript
const result = await qubit.generateWithExamples(
  'Translate to Japanese: Hello',
  [
    'Translate to Japanese: Good morning',
    'おはようございます',
    'Translate to Japanese: Good night',
    'おやすみなさい',
  ],
  { maxTokens: 100 }
);
```

##### `generateBatch(prompts, options)`

Generate multiple outputs efficiently.

```typescript
const results = await qubit.generateBatch(
  [
    'Write a haiku',
    'Write a limerick',
    'Write a sonnet',
  ],
  { temperature: 0.8 }
);

// Returns: GenerationResult[]
```

##### `train(texts)`

Train on custom text data.

```typescript
await qubit.train([
  'Custom training text 1',
  'Custom training text 2',
  'Custom training text 3',
]);
```

##### `trainOnHFDataset(options)`

Train on HuggingFace dataset.

```typescript
const result = await qubit.trainOnHFDataset({
  dataset: 'wikitext',
  split: 'train',
  maxExamples: 5000,
  batchSize: 32,
  onProgress: (progress) => {
    console.log(
      `Training: ${progress.processedExamples}/${progress.totalExamples}`
    );
  },
});

// Returns: { totalExamples, batches, durationMs, status, errors }
```

##### `getStatus()`

Get model status and availability.

```typescript
const status = await qubit.getStatus();
// {
//   available: boolean,
//   provider: "pyodide",
//   version: "4.0.0",
//   trained: boolean,
//   vocabSize: number
// }
```

##### `getConfig()`

Get current configuration.

```typescript
const config = qubit.getConfig();
// {
//   provider: "pyodide",
//   version: "4.0.0",
//   productName: "Qubit.ai Generative (Pyodide)",
//   sessionId: string,
//   vocabSize: number,
//   modelType: "neuroquantum-lightweight"
// }
```

### Global Convenience Functions

```typescript
import {
  generate,
  generateWithExamples,
  generateBatch,
  trainOnData,
  trainOnHFDataset,
} from 'qubit_ai';

// Single generation
const result = await generate('Write a story');

// Few-shot
const result2 = await generateWithExamples(
  'Q: What is AI?\nA:',
  ['Q: What is ML?\nA: Machine Learning...']
);

// Batch
const results = await generateBatch(['Prompt 1', 'Prompt 2']);

// Training
await trainOnData(['Text 1', 'Text 2', 'Text 3']);

// HF dataset
await trainOnHFDataset({
  dataset: 'wikitext',
  maxExamples: 1000,
});
```

## Complete Examples

### Example 1: Translation

```typescript
async function translateToJapanese() {
  const qubit = getQubitAIGenerative();

  const result = await qubit.generateWithExamples(
    'Translate to Japanese: The future is bright',
    [
      'Translate to Japanese: Hello world',
      'こんにちは世界',
      'Translate to Japanese: Good morning',
      'おはようございます',
    ],
    { temperature: 0.5 }
  );

  console.log(result.text);
}

translateToJapanese();
```

### Example 2: Content Generation

```typescript
async function generateBlogOutlines() {
  const qubit = getQubitAIGenerative();

  const topics = [
    'Quantum Computing and AI',
    'Future of Neural Networks',
    'Quantum-Inspired Algorithms',
  ];

  const results = await qubit.generateBatch(
    topics.map(topic => `Create a blog outline: ${topic}`),
    { maxTokens: 300, temperature: 0.7 }
  );

  results.forEach((result, i) => {
    console.log(`Topic: ${topics[i]}`);
    console.log(result.text);
    console.log('---');
  });
}

generateBlogOutlines();
```

### Example 3: Custom Training

```typescript
async function trainAndGenerate() {
  const qubit = getQubitAIGenerative();

  // Train on domain-specific data
  const trainingTexts = [
    'Quantum computing uses quantum bits (qubits).',
    'Qubits can exist in superposition.',
    'Quantum entanglement allows qubit correlation.',
    'Quantum gates manipulate qubit states.',
  ];

  await qubit.train(trainingTexts);

  // Now generate based on training
  const result = await qubit.generate(
    'Explain quantum superposition',
    { temperature: 0.6 }
  );

  console.log(result.text);
}

trainAndGenerate();
```

### Example 4: HuggingFace Dataset Training

```typescript
async function trainOnWikitext() {
  const qubit = getQubitAIGenerative();

  const result = await qubit.trainOnHFDataset({
    dataset: 'wikitext',
    split: 'train',
    maxExamples: 2000,
    batchSize: 64,
    onProgress: (progress) => {
      const percent = (
        (progress.processedExamples / progress.totalExamples) * 100
      ).toFixed(1);
      console.log(
        `Training: ${progress.processedExamples}/${progress.totalExamples} (${percent}%)`
      );
    },
  });

  console.log(`Training complete!`);
  console.log(`Total examples: ${result.totalExamples}`);
  console.log(`Duration: ${result.durationMs}ms`);
  console.log(`Status: ${result.status}`);

  if (result.errors.length > 0) {
    console.error('Errors:', result.errors);
  }
}

trainOnWikitext();
```

## Quantum-Inspired Sampling

Qubit AI uses quantum-inspired algorithms from neuroquantum_layered.py:

### Phase Evolution

```
θ(t) = quantum phase at step t
r = cos(2θ)   (correlation component)
T = |sin(2θ)| (entanglement component)
```

Temperature evolves dynamically:
```
T(t) = T_min + (T_max - T_min) × (0.5 + 0.5 × sin(θ(t)))
```

### Repetition Prevention

- **Recency-weighted penalty:** Recent tokens penalized more strongly
- **N-gram blocking:** Prevent repeating N-grams
- **Top-K/Top-P filtering:** Restrict sampling to high-probability tokens

### Sampling Strategy

1. Compute logits from model
2. Apply quantum influence (phase-dependent modulation)
3. Apply repetition penalty (recency-weighted)
4. Apply Top-K filtering
5. Apply Top-P (nucleus) filtering
6. Softmax normalization
7. Multinomial sampling

## Architecture

```
┌─────────────────────────────────┐
│   QubitAIGenerative (TypeScript) │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│   NeuroQuantumGenerator          │
│  (Quantum-inspired sampling)     │
└──────────────┬──────────────────┘
               ↓
┌──────────────────────────────────┐
│   LightweightLanguageModel       │
│  (Embedding + logit computation) │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│   SimpleTokenizer                │
│  (Japanese + English support)    │
└──────────────────────────────────┘
```

## Features

✅ **No External APIs**
- Pure in-browser or Node.js execution
- No internet required after initial load
- Privacy-preserving

✅ **Quantum-Inspired**
- Phase evolution (θ dynamics)
- Entanglement-inspired logit modulation
- Recency-weighted penalties

✅ **Flexible Training**
- Train on custom texts
- Support for HuggingFace datasets
- Progressive learning

✅ **Efficient Sampling**
- Top-K filtering
- Top-P (nucleus) sampling
- Temperature control

## Performance Considerations

1. **First Generation Slowdown**
   - Auto-trains on default dataset
   - Subsequent generations are faster

2. **Token Limits**
   - Default max: 500 tokens
   - Adjust via `maxTokens` option
   - Runtime scales linearly with output length

3. **Memory Usage**
   - Lightweight model (~10MB)
   - Vocabulary: 32K tokens
   - Embed dimension: 64

## Error Handling

```typescript
try {
  const result = await qubit.generate('prompt');
} catch (error) {
  if (error.message.includes('Training')) {
    console.error('Training failed');
  } else {
    console.error('Generation failed:', error);
  }
}
```

## Browser Support

✅ Chrome/Brave (recommended)
✅ Firefox
✅ Safari
✅ Edge
✅ Node.js 18+

## Limitations

- **Inference speed:** Slower than native Python
- **Model size:** Cannot easily load large models (>100M parameters)
- **Training:** Limited to in-memory datasets
- **Language:** Primarily optimized for Japanese and English

## Advanced Configuration

```typescript
// Multiple independent sessions
const session1 = getQubitAIGenerative({ sessionKey: 'chat' });
const session2 = getQubitAIGenerative({ sessionKey: 'translation' });

// Different seeds for variety
const creative = getQubitAIGenerative({ seed: 42 });
const consistent = getQubitAIGenerative({ seed: 0 });

// Reset sessions
resetQubitAIGenerativeSession('chat');
resetQubitAIGenerative(); // Reset all
```

## Migration from v3.0.0

v4.0.0 shifts from judgment/reasoning to pure generative AI:

```typescript
// v3.0.0 (Judgment) - No longer available
// qubit.judge(action, context, type)
// qubit.safetyCheck(action)

// v4.0.0 (Generation) - New approach
const result = await qubit.generate('Prompt text');
```

## License

MIT © tapiocaTakeshi
