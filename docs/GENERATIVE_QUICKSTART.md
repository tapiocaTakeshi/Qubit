# Qubit AI Generative — 5-Minute Quick Start

Generate content and fine-tune LLMs with Claude, OpenAI, and HuggingFace.

## Installation

```bash
npm install qubit_ai
```

## 1. Basic Generation (Claude)

```typescript
import { getQubitAIGenerative } from 'qubit_ai';

const qubit = getQubitAIGenerative('claude', {
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const result = await qubit.generate('Write a haiku about coding');

console.log(result.text);
```

## 2. OpenAI Alternative

```typescript
const qubit = getQubitAIGenerative('openai', {
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
});

const result = await qubit.generate('Explain quantum computing');
console.log(result.text);
```

## 3. Few-Shot Generation

```typescript
const result = await qubit.generateWithExamples(
  'Translate to French: How are you?',
  [
    'Translate to French: Hello',
    'Bonjour',
    'Translate to French: Goodbye',
    'Au revoir',
  ]
);

console.log(result.text);
```

## 4. Batch Generation

```typescript
const prompts = [
  'List 3 benefits of AI',
  'Explain machine learning',
  'What is neural networks?',
];

const results = await qubit.generateBatch(prompts, {
  temperature: 0.5,
  maxTokens: 200,
});

results.forEach((result, i) => {
  console.log(`Result ${i + 1}: ${result.text}\n`);
});
```

## 5. Fine-tune on HuggingFace Dataset

```typescript
await qubit.trainOnHFDataset({
  dataset: 'wikitext',
  split: 'train',
  maxExamples: 10000,
  onProgress: (progress) => {
    console.log(
      `Training: ${progress.processedExamples}/${progress.totalExamples}`
    );
  },
});

console.log('Fine-tuning complete!');
```

## 6. Specialized Generation

```typescript
// Generate using specialized prompt templates
const result = await qubit.generateForType(
  'User asked about data security',
  'safety',
  { maxTokens: 300 }
);

console.log(result.text);
```

## 7. Global Functions

```typescript
import {
  generate,
  generateBatch,
  trainOnHFDataset,
} from 'qubit_ai';

// No need to create instance
const result = await generate('Write a story', {
  provider: 'claude',
  temperature: 0.9,
});

// Batch processing
const results = await generateBatch(
  ['Prompt 1', 'Prompt 2'],
  'openai'
);

// Training
await trainOnHFDataset({
  dataset: 'wikitext',
  provider: 'hf',
  maxExamples: 5000,
});
```

## 8. Generation Options

```typescript
const result = await qubit.generate(
  'Write a creative story',
  {
    temperature: 0.9,      // More creative
    maxTokens: 1000,       // Longer output
    topK: 50,              // Top-K sampling
    topP: 0.95,            // Nucleus sampling
    repetitionPenalty: 1.2, // Avoid repetition
  }
);
```

## 9. Multi-Provider Management

```typescript
const claude = getQubitAIGenerative('claude');
const openai = getQubitAIGenerative('openai');
const hf = getQubitAIGenerative('hf');

// Use each independently
const r1 = await claude.generate('Claude prompt');
const r2 = await openai.generate('OpenAI prompt');
const r3 = await hf.generate('HF prompt');

// Reset specific provider
resetQubitAIGenerativeProvider('claude');

// Reset all
resetQubitAIGenerative();
```

## 10. Configuration

```typescript
const qubit = getQubitAIGenerative('claude', {
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-3-5-sonnet-20241022',
  temperature: 0.7,
  maxTokens: 500,
});

const config = qubit.getConfig();
console.log(config);
// { provider: 'claude', version: '4.0.0', ... }

const status = await qubit.getStatus();
console.log(status.available);
```

## Provider Selection

| Provider | Setup | Best For |
|----------|-------|----------|
| Claude | `ANTHROPIC_API_KEY` | Advanced reasoning |
| OpenAI | `OPENAI_API_KEY` | Cost-effective |
| HuggingFace | `HF_TOKEN` | Open-source models |

## Common Tasks

### Summarization
```typescript
const result = await qubit.generate(
  'Summarize this text in 100 words: ...',
  { maxTokens: 150 }
);
```

### Translation
```typescript
const result = await qubit.generateWithExamples(
  'Translate to Japanese: Hello',
  ['Translate to Japanese: Good morning', 'おはようございます']
);
```

### Code Generation
```typescript
const result = await qubit.generate(
  'Write Python function to calculate factorial',
  { maxTokens: 500 }
);
```

### Content Creation
```typescript
const results = await qubit.generateBatch([
  'Write a blog post about AI',
  'Write a product description',
  'Write marketing copy',
], { temperature: 0.8 });
```

## Error Handling

```typescript
try {
  const result = await qubit.generate('prompt');
} catch (error) {
  if (error.message.includes('rate_limit')) {
    console.error('Rate limited - retry later');
  } else if (error.message.includes('auth')) {
    console.error('Authentication failed');
  } else {
    console.error('Generation failed:', error);
  }
}
```

## Performance Tips

1. Use `generateBatch()` for multiple prompts
2. Set `maxTokens` to limit costs
3. Use lower `temperature` for consistency
4. Implement exponential backoff for retries
5. Cache results for repeated queries

## Environment Variables

```bash
export ANTHROPIC_API_KEY=sk_...
export OPENAI_API_KEY=sk_...
export HF_TOKEN=hf_...
```

## Next Steps

- [Full Documentation](./GENERATIVE_QUICKSTART.md)
- [Complete Examples](../examples/)
- [API Reference](./API.md)

---

That's it! You can now generate content and fine-tune LLMs. 🚀
