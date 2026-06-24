# NeuroQuantum Backend - Quick Start

Get quantum-inspired reasoning working in 5 minutes.

## 1. Start the Python Server

```bash
# Install Python dependencies (if not already installed)
pip install flask flask-cors torch

# Start the API server
python neuroquantum_api_server.py
```

Expected output:
```
NeuroQuantum REST API Server を起動します
  アドレス: 127.0.0.1:5000
  Debug: False
```

Verify it's running:
```bash
curl http://localhost:5000/api/v1/health
```

## 2. Create TypeScript Application

```typescript
// my-app.ts
import { QubitAI } from 'qubit-ai';

// Create QubitAI with NeuroQuantum backend
const qubit = new QubitAI({
  neuroquantumEnabled: true,
});

async function main() {
  // Perform quantum-inspired judgment
  const result = await qubit.judge(
    'Delete all user data',      // action
    'Production environment',     // context
    'safety'                      // judgment type
  );

  console.log('Decision:', result.decision);    // "Yes" or "No"
  console.log('Score:', result.score);          // 0-100
  console.log('Confidence:', result.confidence); // high/medium/low
  console.log('Reasoning:', result.reasoning);   // Quantum-inspired explanation
}

main().catch(console.error);
```

Run it:
```bash
npx tsx my-app.ts
```

## 3. Common Tasks

### Safety Check
```typescript
const [isSafe, details] = await qubit.safetyCheck(
  'Log API credentials',
  'Production',
  { risks: ['credential exposure'] }
);

if (isSafe) {
  console.log('✓ Safe to proceed');
} else {
  console.log('✗ Unsafe:', details.reasoning);
}
```

### Ethics Evaluation
```typescript
const result = await qubit.ethicsCheck(
  'Share user location',
  ['users', 'regulators'],        // stakeholders
  ['privacy violation']            // potential harms
);

console.log('Ethical?', result.decision);
```

### Content Quality
```typescript
const result = await qubit.evaluateQuality(
  'AI-generated content to check',
  { requirements: ['clarity', 'accuracy'] }
);

console.log('Quality score:', result.score);
```

### Task Prioritization
```typescript
const items = [
  { name: 'Security patch', description: 'Critical fix' },
  { name: 'Documentation', description: 'Update docs' },
  { name: 'Feature X', description: 'New feature' },
];

const ranked = await qubit.prioritize(items);

ranked.forEach(([item, score]) => {
  console.log(`${item.name}: ${(score * 100).toFixed(0)}%`);
});
```

## 4. Production Setup

### Hybrid Mode (Recommended)
Combine NeuroQuantum with keyword fallback for reliability:

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  fallbackToHeuristics: true,    // Use keywords if API fails
  neuroquantumConfig: {
    baseUrl: 'http://api.example.com:5000',
    timeout: 10000,
    maxRetries: 3,
  },
});
```

### Strict Mode
Enforce stricter judgment threshold:

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  strictMode: true,              // Requires score >= 70 for "Yes"
});
```

### Environment Variables
```bash
export QUBIT_NEUROQUANTUM_ENABLED=true
export QUBIT_NEUROQUANTUM_BASE_URL=http://localhost:5000
export QUBIT_FALLBACK_TO_HEURISTICS=true
```

## 5. Error Handling

```typescript
try {
  const result = await qubit.judge('action', 'context', 'safety');
} catch (error) {
  if (error.message.includes('timeout')) {
    console.error('API is slow - using fallback');
    // With fallbackToHeuristics, will use keyword engine
  } else if (error.message.includes('ECONNREFUSED')) {
    console.error('API server not running');
    console.error('Start with: python neuroquantum_api_server.py');
  } else {
    console.error('Error:', error.message);
  }
}
```

## 6. Batch Operations

Process multiple items efficiently:

```typescript
const { NeuroQuantumAPIClient } = await import('qubit-ai');

const client = new NeuroQuantumAPIClient({
  baseUrl: 'http://localhost:5000',
});

const results = await client.batchJudge([
  { action: 'Action 1', context: 'Context 1', judgment_type: 'safety' },
  { action: 'Action 2', context: 'Context 2', judgment_type: 'safety' },
  { action: 'Action 3', context: 'Context 3', judgment_type: 'safety' },
]);

console.log(`Processed ${results.count} judgments`);
results.results.forEach((r) => {
  console.log(`Decision: ${r.decision}, Score: ${r.score}`);
});
```

## 7. Monitor API

```typescript
// Check if API is available
const isAvailable = await client.isAvailable();
console.log(isAvailable ? '✓ API online' : '✗ API offline');

// Wait for API to start
try {
  await client.waitForAvailable(30000);  // Wait up to 30 seconds
  console.log('✓ API is ready');
} catch {
  console.error('API did not become available');
}

// Get judgment history
const history = qubit.getHistory(10);
history.forEach((h) => {
  console.log(`${h.judgmentType}: ${h.decision} (${h.score})`);
});
```

## 8. Docker Deployment

Containerize the Python server:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY neuroquantum*.py .

EXPOSE 5000

CMD ["python", "neuroquantum_api_server.py", "--host", "0.0.0.0", "--port", "5000"]
```

```bash
# Build image
docker build -t qubit-neuroquantum .

# Run server
docker run -p 5000:5000 qubit-neuroquantum

# Connect from TypeScript with Docker hostname
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://qubit-neuroquantum:5000',
  },
});
```

## 9. Multiple Backends

You can use different backends for different purposes:

```typescript
// Pure NeuroQuantum for important decisions
const strictQubit = new QubitAI({
  neuroquantumEnabled: true,
  strictMode: true,
});

// Hybrid mode for general purpose
const generalQubit = new QubitAI({
  neuroquantumEnabled: true,
  fallbackToHeuristics: true,
});

// Fallback to heuristics for speed
const fastQubit = new QubitAI({
  // neuroquantumEnabled: false (default)
  // Uses keyword-based heuristics
});

// Use appropriate backend for each case
const criticalResult = await strictQubit.judge(...);
const normalResult = await generalQubit.judge(...);
const quickResult = await fastQubit.judge(...);
```

## 10. Testing

```typescript
import { test, expect } from 'vitest';

test('quantum judgment works', async () => {
  const qubit = new QubitAI({
    neuroquantumEnabled: true,
  });

  const result = await qubit.judge('test', 'context', 'safety');

  expect(result.decision).toMatch(/^(Yes|No)$/);
  expect(result.score).toBeGreaterThanOrEqual(0);
  expect(result.score).toBeLessThanOrEqual(100);
  expect(result.confidence).toMatch(/^(high|medium|low)$/);
});
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Error: ECONNREFUSED` | Start Python server: `python neuroquantum_api_server.py` |
| `Timeout error` | Increase timeout: `timeout: 60000` |
| `503 Service Unavailable` | Server may be initializing, wait a moment |
| `GPU out of memory` | Use lower GPU tier: `--gpu-tier low` |
| `API slow` | Enable hybrid mode with fallback, run batch operations |

## Next Steps

- Read [Full NeuroQuantum Integration Guide](./NEUROQUANTUM_INTEGRATION.md)
- Check [API Examples](../examples/neuroquantum-backend.ts)
- Review [Configuration Options](./CONFIGURATION.md)
- Explore [Advanced Usage](./ADVANCED.md)

## Support

For issues or questions:
1. Check [Troubleshooting Guide](./NEUROQUANTUM_INTEGRATION.md#troubleshooting)
2. Review [API Endpoints](./NEUROQUANTUM_INTEGRATION.md#api-endpoints)
3. Examine example code in `examples/neuroquantum-backend.ts`
