# Qubit AI v4.0.0 — Generative AI + HuggingFace Training

**LLM-based text generation and fine-tuning with Claude, OpenAI, and HuggingFace**

## Installation

```bash
npm install qubit_ai
```

## Quick Start

### Text Generation

```typescript
import { getQubitAIGenerative } from 'qubit_ai';

// Using Claude
const qubit = getQubitAIGenerative('claude', {
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// Generate text
const result = await qubit.generate(
  'Write a haiku about artificial intelligence'
);

console.log(result.text);      // Generated text
console.log(result.tokensUsed); // Token count
```

### Few-Shot Generation

```typescript
const result = await qubit.generateWithExamples(
  'Translate to Japanese: Hello world',
  [
    'Translate to Japanese: Good morning',
    'Japanese: おはようございます',
  ]
);

console.log(result.text);
```

### Batch Generation

```typescript
const results = await qubit.generateBatch([
  'Summarize: The quick brown fox...',
  'Explain: What is machine learning?',
  'Translate: Hola mundo',
]);

results.forEach((result) => {
  console.log(result.text);
});
```

### HuggingFace Dataset Fine-tuning

```typescript
await qubit.trainOnHFDataset({
  dataset: 'wikitext',
  split: 'train',
  maxExamples: 10000,
  batchSize: 32,
  onProgress: (progress) => {
    console.log(
      `Training: ${progress.processedExamples}/${progress.totalExamples}`
    );
  },
});
```

## Supported LLM Providers

### Claude (Recommended)

```typescript
const qubit = getQubitAIGenerative('claude', {
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-3-5-sonnet-20241022',
  temperature: 0.7,
  maxTokens: 1000,
});
```

### OpenAI

```typescript
const qubit = getQubitAIGenerative('openai', {
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  temperature: 0.7,
});
```

### HuggingFace

```typescript
const qubit = getQubitAIGenerative('hf', {
  hfToken: process.env.HF_TOKEN,
  model: 'meta-llama/Llama-2-7b-hf',
});
```

## API Reference

### Generation Methods

#### `generate(prompt, options)`

Generate text from a prompt.

```typescript
const result = await qubit.generate(
  'Write a poem about the moon',
  {
    temperature: 0.9,
    maxTokens: 500,
    topP: 0.9,
  }
);

// Returns: { text, finishReason, tokensUsed, generatedAt }
```

**Options:**
- `temperature` (0-2): Randomness (default: 0.7)
- `maxTokens`: Max output length (default: 500)
- `topK`: Top-K sampling
- `topP`: Nucleus sampling
- `repetitionPenalty`: Penalize repeated text

#### `generateWithExamples(prompt, examples, options)`

Few-shot generation with examples.

```typescript
const result = await qubit.generateWithExamples(
  'Q: What is AI?\nA:',
  [
    'Q: What is ML?\nA: Machine Learning is...',
    'Q: What is DL?\nA: Deep Learning is...',
  ],
  { maxTokens: 200 }
);
```

#### `generateBatch(prompts, options)`

Generate multiple outputs efficiently.

```typescript
const results = await qubit.generateBatch(
  [
    'Translate: Hello',
    'Translate: Goodbye',
    'Translate: Thank you',
  ],
  { temperature: 0.2 }
);

// Returns: GenerationResult[]
```

#### `generateForType(prompt, judgmentType, options)`

Generate using specialized prompts for specific tasks.

```typescript
const result = await qubit.generateForType(
  'User query about product safety',
  'safety',
  { maxTokens: 300 }
);
```

### Training Methods

#### `trainOnHFDataset(options)`

Fine-tune LLM on HuggingFace dataset.

```typescript
const result = await qubit.trainOnHFDataset({
  dataset: 'wikitext',
  split: 'train',
  maxExamples: 10000,
  batchSize: 32,
  onProgress: (progress) => {
    console.log(`${progress.processedExamples}/${progress.totalExamples}`);
  },
});

// Returns: { totalExamples, batches, durationMs, status, errors }
```

**Options:**
- `dataset` (required): HuggingFace dataset name
- `split`: Dataset split (default: 'train')
- `maxExamples`: Limit number of examples
- `batchSize`: Training batch size (default: 32)
- `onProgress`: Progress callback

### Status and Configuration

#### `getStatus()`

Get provider status.

```typescript
const status = await qubit.getStatus();
// { available: boolean, provider: string, version: string }
```

#### `getConfig()`

Get current configuration.

```typescript
const config = qubit.getConfig();
// { provider, version, productName, sessionId }
```

## Global Functions

```typescript
import {
  generate,
  generateWithExamples,
  generateBatch,
  trainOnHFDataset,
} from 'qubit_ai';

// Use without creating instance
const result = await generate('Write a story', {
  provider: 'claude',
  temperature: 0.8,
});

const batch = await generateBatch(['Prompt 1', 'Prompt 2'], 'openai');

await trainOnHFDataset({
  dataset: 'wikitext',
  provider: 'hf',
  maxExamples: 5000,
});
```

## Complete Examples

### Example 1: Content Generation

```typescript
import { getQubitAIGenerative } from 'qubit_ai';

async function generateBlogPost() {
  const qubit = getQubitAIGenerative('claude');

  const result = await qubit.generate(
    'Write a 500-word blog post about AI ethics',
    { maxTokens: 2000 }
  );

  console.log(result.text);
  console.log(`Tokens used: ${result.tokensUsed}`);
}

generateBlogPost();
```

### Example 2: Few-Shot Translation

```typescript
async function translateText() {
  const qubit = getQubitAIGenerative('claude');

  const result = await qubit.generateWithExamples(
    'Translate to Spanish: The future is bright',
    [
      'Translate to Spanish: Hello world',
      'Spanish: Hola mundo',
      'Translate to Spanish: Good morning',
      'Spanish: Buenos días',
    ]
  );

  console.log(result.text);
}

translateText();
```

### Example 3: Batch Processing

```typescript
async function processBatch() {
  const qubit = getQubitAIGenerative('openai');

  const prompts = [
    'Summarize: Machine learning is...',
    'Explain: What is deep learning?',
    'List: Top 5 AI frameworks',
  ];

  const results = await qubit.generateBatch(prompts, {
    temperature: 0.5,
    maxTokens: 300,
  });

  results.forEach((result, i) => {
    console.log(`Prompt ${i + 1}:`);
    console.log(result.text);
    console.log('---');
  });
}

processBatch();
```

### Example 4: Fine-tune on Custom Dataset

```typescript
async function trainModel() {
  const qubit = getQubitAIGenerative('hf');

  const result = await qubit.trainOnHFDataset({
    dataset: 'wikitext',
    split: 'train',
    maxExamples: 100000,
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

trainModel();
```

## Configuration via Environment Variables

```bash
# Claude provider
export ANTHROPIC_API_KEY=sk_...

# OpenAI provider
export OPENAI_API_KEY=sk_...

# HuggingFace provider
export HF_TOKEN=hf_...
```

## Managing Multiple Providers

```typescript
// Create instances for different providers
const claude = getQubitAIGenerative('claude');
const openai = getQubitAIGenerative('openai');
const hf = getQubitAIGenerative('hf');

// Use them independently
const claudeResult = await claude.generate('Prompt 1');
const openaiResult = await openai.generate('Prompt 2');
const hfResult = await hf.generate('Prompt 3');

// Reset specific provider
resetQubitAIGenerativeProvider('claude');

// Reset all
resetQubitAIGenerative();
```

## Performance Tips

1. **Batch Generation**: Use `generateBatch()` instead of multiple `generate()` calls
2. **Token Limits**: Set appropriate `maxTokens` to control costs
3. **Temperature**: Use lower values (0.3-0.5) for deterministic output
4. **Streaming**: For long outputs, consider provider-specific streaming
5. **Caching**: Implement prompt caching for repeated queries

## Error Handling

```typescript
try {
  const result = await qubit.generate('prompt');
} catch (error) {
  if (error.message.includes('rate_limit')) {
    console.error('API rate limit exceeded');
  } else if (error.message.includes('auth')) {
    console.error('Authentication failed');
  } else {
    console.error('Generation failed:', error);
  }
}
```

## License

MIT © tapiocaTakeshi
