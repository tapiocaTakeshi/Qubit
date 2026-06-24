# Qubit AI Inference Examples

Complete collection of inference examples demonstrating the capabilities of the qubit_ai library.

## Setup

Before running any examples, ensure you have:

1. **HuggingFace Token** (optional for public models):
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

2. **Dependencies installed**:
   ```bash
   npm install qubit_ai
   # or for development
   npm install
   ```

3. **TypeScript support** (if running `.ts` files directly):
   ```bash
   npm install -D ts-node typescript
   ```

## Examples Overview

### 1. **Basic Text Generation** (`qubit-inference-basics.ts`)

Learn the fundamentals of text generation with qubit_ai.

**What it demonstrates:**
- Simple text generation
- Temperature control (creativity vs. consistency)
- Custom generation parameters
- Multiple prompt examples

**Run:**
```bash
npx ts-node examples/qubit-inference-basics.ts
```

**Key concepts:**
- `NeuroQuantumClient` initialization
- `generateWithExamples()` with empty examples for zero-shot generation
- Parameter tuning: `temperature`, `topP`, `topK`, `repetitionPenalty`

**Example output:**
```
📝 Prompt: "人工知能は"
✅ Generated text: 人工知能は様々な分野で応用され、生産性向上を実現しています
```

---

### 2. **Few-shot Learning** (`qubit-few-shot-inference.ts`)

Master few-shot prompting techniques for specialized tasks.

**What it demonstrates:**
- Language translation with 3 examples
- Sentiment analysis from reviews
- Question answering from facts
- Loading examples from HuggingFace datasets

**Run:**
```bash
npx ts-node examples/qubit-few-shot-inference.ts
```

**Key concepts:**
- Few-shot examples shape model behavior
- Custom templates for different tasks
- `numExamples` parameter controls context
- Template variables: `{prompt}` and `{completion}`

**Example usage:**
```typescript
const examples = [
  { prompt: "Hello, how are you?", completion: "こんにちは、お元気ですか？" },
  { prompt: "Thank you", completion: "ありがとうございます" },
];

const result = await client.generateWithExamples(
  "Good morning",
  examples,
  {
    numExamples: 2,
    exampleTemplate: "English: {prompt}\nJapanese: {completion}",
    queryTemplate: "English: {prompt}\nJapanese:",
    maxNewTokens: 40,
    temperature: 0.3,
  }
);
```

---

### 3. **Batch Processing** (`qubit-batch-inference.ts`)

Efficiently process multiple texts with batch inference.

**What it demonstrates:**
- Processing multiple prompts sequentially
- Sentiment analysis on collections
- Topic classification
- Performance metrics and statistics

**Run:**
```bash
npx ts-node examples/qubit-batch-inference.ts
```

**Key concepts:**
- Process many items with consistent settings
- Track performance metrics
- Handle failures gracefully
- Calculate statistics (average time, success rate)

**Example output:**
```
📦 Batch Text Generation
Processing 5 prompts...

[1/5] Processing: "深層学習の基本は"
  ✅ Completed in 1245ms

📈 Statistics:
  Total items: 5
  Average time per item: 1180ms
```

---

### 4. **Advanced Prompting** (`qubit-advanced-prompting.ts`)

Explore sophisticated prompting techniques.

**What it demonstrates:**
- Chain-of-thought reasoning (step-by-step)
- Context-aware generation (domain-specific)
- Multi-turn conversation simulation
- Structured output generation (JSON)
- Temperature impact on creativity

**Run:**
```bash
npx ts-node examples/qubit-advanced-prompting.ts
```

**Key techniques:**

#### Chain-of-Thought
```typescript
const cotExample = {
  prompt: "If I have 3 apples and get 2 more, how many do I have?",
  completion: "Let me think step by step:\n1. I start with 3 apples\n2. I get 2 more...\n3. Total: 5 apples",
};
```

#### Context-Aware Generation
```typescript
// Medical domain examples
const medicalExamples = [
  { prompt: "What are symptoms of flu?", 
    completion: "Common symptoms include fever, cough, sore throat..." },
];
```

#### Structured Output
```typescript
const structuredExample = {
  prompt: "Extract: 'John Smith, age 28, works in engineering'",
  completion: JSON.stringify({ name: "John Smith", age: 28, job: "Engineer" }),
};
```

---

### 5. **Inference Pipeline** (`qubit-inference-pipeline.ts`)

Complete end-to-end production-ready pipeline.

**What it demonstrates:**
- Summarization pipeline
- Dataset-integrated inference
- Error handling and retries
- Output persistence (JSON)
- Pipeline statistics and logging

**Run:**
```bash
npx ts-node examples/qubit-inference-pipeline.ts
```

**Key features:**
- Robust error handling with retries
- Results saved to JSON files
- Detailed timing and metrics
- Dataset integration for examples
- Configurable parameters

**Output structure:**
```json
{
  "pipeline": "Code Explanation",
  "timestamp": "2026-06-24T...",
  "config": { "temperature": 0.7, "maxTokens": 100 },
  "summary": {
    "totalProcessed": 2,
    "totalErrors": 0,
    "successRate": "100%"
  },
  "results": [
    {
      "input": "...",
      "output": "...",
      "duration": 1200,
      "timestamp": "..."
    }
  ]
}
```

---

## Common Patterns

### Initialize Client
```typescript
import { NeuroQuantumClient } from "qubit_ai";

const client = new NeuroQuantumClient({
  hfToken: process.env.HF_TOKEN,
  timeoutMs: 30000,
  maxRetries: 3,
});
```

### Simple Generation
```typescript
const result = await client.generateWithExamples(
  "Your prompt here",
  [], // Empty for zero-shot
  {
    maxNewTokens: 50,
    temperature: 0.7,
  }
);
console.log(result.generatedText);
```

### Few-shot Generation
```typescript
const examples = [
  { prompt: "Example input 1", completion: "Example output 1" },
  { prompt: "Example input 2", completion: "Example output 2" },
];

const result = await client.generateWithExamples(
  "Test input",
  examples,
  {
    numExamples: 2,
    exampleTemplate: "Q: {prompt}\nA: {completion}",
    queryTemplate: "Q: {prompt}\nA:",
    maxNewTokens: 100,
    temperature: 0.5,
  }
);
```

### Load Dataset Examples
```typescript
import { HFDatasetLoader } from "qubit_ai";

const loader = new HFDatasetLoader({
  hfToken: process.env.HF_TOKEN,
});

// Quick preview
const examples = await loader.preview("dataset-name", 5);

// Or stream for large datasets
for await (const example of loader.streamExamples({
  dataset: "dataset-name",
  promptField: "input",
  completionField: "output",
  maxRows: 1000,
})) {
  console.log(example.prompt, "->", example.completion);
}
```

---

## Parameter Tuning Guide

### Temperature
- **0.0-0.3**: Deterministic, repetitive (good for facts)
- **0.4-0.7**: Balanced (general purpose)
- **0.8-1.5**: Creative, varied (good for storytelling)

### Top-K & Top-P
- **topK**: Limits to top K most probable tokens
- **topP**: Nucleus sampling (cumulative probability)
- Both help reduce nonsense while maintaining diversity

### Repetition Penalty
- **1.0**: No penalty (default)
- **1.2-1.5**: Discourages repetition
- Higher values = more variety

### Max Tokens
- Typical range: 50-200 tokens
- Longer: 200-500 for detailed responses
- Very long: 500+ for essays or code

---

## Troubleshooting

### "HF_TOKEN environment variable is not set"
```bash
export HF_TOKEN="your_hugging_face_token"
```

### "Service unavailable (503)"
The API is overloaded. The client automatically retries, but you can:
- Reduce `maxNewTokens`
- Reduce number of requests
- Try again later

### "Timeout exceeded"
Increase timeout:
```typescript
const client = new NeuroQuantumClient({
  timeoutMs: 60000, // 60 seconds
});
```

### "Module not found: qubit_ai"
Ensure the package is installed:
```bash
npm install qubit_ai
```

---

## Performance Tips

1. **Batch Processing**: Process multiple texts together
2. **Lower Temperature**: For factual tasks, use lower temperature
3. **Shorter Tokens**: Reduce `maxNewTokens` for faster inference
4. **Fewer Examples**: Using 2-3 examples is often sufficient
5. **Reuse Client**: Create one client and reuse it

---

## Next Steps

- Explore different **templates** for your specific use case
- Experiment with **few-shot examples** to improve quality
- Combine with your own data pipeline
- Integrate into a larger application

For more details, see the [main README](../npm/README.md).

---

## License

MIT © tapiocaTakeshi
