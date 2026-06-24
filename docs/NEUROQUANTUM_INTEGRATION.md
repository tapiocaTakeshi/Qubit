# NeuroQuantum Backend Integration Guide

This guide explains how to use Qubit AI with the Python `neuroquantum_layered.py` quantum-inspired neural network backend via REST API.

## Overview

The integration combines:
- **TypeScript/Node.js Frontend**: Qubit AI client library with pluggable backends
- **Python Backend**: Quantum-inspired neural network reasoning via `neuroquantum_layered.py`
- **REST API**: Flask HTTP interface for communication between systems

This architecture enables true quantum-inspired reasoning in your TypeScript applications while leveraging sophisticated neural network implementations in Python.

## Architecture

```
TypeScript Application
    ↓
QubitAI (with neuroquantumEnabled: true)
    ↓
NeuroQuantumFrontalEngine
    ↓
NeuroQuantumAPIClient (HTTP)
    ↓
neuroquantum_api_server.py (Flask)
    ↓
neuroquantum_layered.py (QBNN Inference)
    ↓
Quantum-Inspired Neural Reasoning Result
    ↓
NeuroQuantumResponse (JSON)
    ↓
QubitAI Result
    ↓
TypeScript Application
```

## Prerequisites

### Python Dependencies
- Python 3.8+
- Flask 2.0+
- torch (for neuroquantum_layered.py)
- flask-cors (for cross-origin requests)

### TypeScript/Node.js
- Node.js 16+
- npm or yarn

## Setup

### 1. Start the Python API Server

The Python server provides REST API endpoints for quantum-inspired reasoning.

#### Option A: Default Configuration

```bash
# From the project root
python neuroquantum_api_server.py
```

This starts the server at `http://127.0.0.1:5000` with default GPU settings.

#### Option B: Custom Configuration

```bash
# Specify host, port, and GPU tier
python neuroquantum_api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-tier high \
  --debug
```

**Available GPU Tiers:**
- `ultra` - RTX 4090, A100 (>40GB VRAM)
- `high` - A6000, RTX 3090 (24GB VRAM)
- `mid` - RTX 3080, A5000 (10-24GB VRAM)
- `low` - RTX 2080, Tesla P100 (8-10GB VRAM)
- `cpu` - CPU-only inference

#### Verify Server is Running

```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "neuroquantum_available": true,
  "gpu_info": {...}
}
```

### 2. Configure TypeScript Client

```typescript
import { QubitAI } from 'qubit-ai';

const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',      // Server URL
    timeout: 30000,                        // Request timeout (ms)
    maxRetries: 3,                         // Retry attempts
    retryDelayMs: 1000,                    // Initial retry delay (ms)
  },
});
```

## API Endpoints

The Python server exposes the following REST endpoints:

### Health Check
```
GET /api/v1/health
```
Returns server status and configuration.

### Judge
```
POST /api/v1/judge
```
Perform quantum-inspired judgment.

**Request:**
```json
{
  "action": "description of action",
  "context": "situational context",
  "judgment_type": "safety|ethics|quality|risk|decision|priority",
  "strict_mode": false
}
```

**Response:**
```json
{
  "decision": "Yes" | "No",
  "score": 0-100,
  "reasoning": "explanation",
  "confidence": "high" | "medium" | "low",
  "factors": ["factor1", "factor2"],
  "timestamp": "2024-01-01T00:00:00Z",
  "system": "neuroquantum",
  "processing_time_ms": 150
}
```

### Batch Judge
```
POST /api/v1/batch_judge
```
Process multiple judgments efficiently.

**Request:**
```json
{
  "requests": [
    {
      "action": "action1",
      "context": "context1",
      "judgment_type": "safety",
      "strict_mode": false
    },
    ...
  ]
}
```

### Safety Check
```
POST /api/v1/safety_check
```
Evaluate safety with risk assessment.

**Request:**
```json
{
  "action": "action to evaluate",
  "context": "context",
  "risks": ["risk1", "risk2"]
}
```

### Ethics Check
```
POST /api/v1/ethics_check
```
Evaluate ethical implications.

**Request:**
```json
{
  "action": "action to evaluate",
  "stakeholders": ["group1", "group2"],
  "potential_harms": ["harm1", "harm2"]
}
```

### Quality Eval
```
POST /api/v1/quality_eval
```
Assess content quality.

**Request:**
```json
{
  "content": "content to evaluate",
  "requirements": ["clarity", "accuracy"],
  "user_intent": "evaluation purpose"
}
```

### Status
```
GET /api/v1/status
```
Get detailed server status and model configuration.

### Config
```
GET /api/v1/config
```
Retrieve current server configuration.

## Usage Examples

### Basic Judgment

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
});

const result = await qubit.judge(
  "Delete production database",
  "Maintenance operation",
  "safety"
);

console.log(result.decision);    // "Yes" or "No"
console.log(result.score);       // 0-100
console.log(result.reasoning);   // Quantum-inspired explanation
console.log(result.confidence);  // "high", "medium", or "low"
```

### Safety Check

```typescript
const [isSafe, details] = await qubit.safetyCheck(
  "Log API credentials",
  "Production environment",
  { risks: ["credential exposure", "security breach"] }
);

if (isSafe) {
  console.log("✓ Action is safe");
} else {
  console.log("✗ Action is unsafe");
  console.log("Details:", details.reasoning);
}
```

### Ethics Evaluation

```typescript
const result = await qubit.ethicsCheck(
  "Share user location data",
  ["users", "regulators"],
  ["privacy violation", "loss of trust"]
);

console.log(`Ethically sound? ${result.decision}`);
```

### Quality Assessment

```typescript
const result = await qubit.evaluateQuality(
  "Some content to evaluate",
  {
    requirements: ["clarity", "accuracy", "completeness"],
  }
);

console.log(`Quality score: ${result.score}/100`);
```

### Task Prioritization

```typescript
const tasks = [
  { name: "Security patch", description: "Critical vulnerability fix" },
  { name: "Documentation", description: "Update API docs" },
  { name: "Bug fix", description: "Handle edge case" },
];

const prioritized = await qubit.prioritize(tasks);

prioritized.forEach(([task, score], index) => {
  console.log(`${index + 1}. ${task.name} (priority: ${score})`);
});
```

### Batch Operations

```typescript
const { NeuroQuantumAPIClient } = await import('qubit-ai');

const client = new NeuroQuantumAPIClient({
  baseUrl: 'http://localhost:5000',
});

const results = await client.batchJudge([
  {
    action: "Delete data",
    context: "Production",
    judgment_type: "safety",
    strict_mode: false,
  },
  {
    action: "Share information",
    context: "External party",
    judgment_type: "ethics",
    strict_mode: false,
  },
]);

console.log(`Processed ${results.count} judgments`);
```

## Hybrid Mode

Combine NeuroQuantum reasoning with keyword-based fallback for robustness:

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  fallbackToHeuristics: true,  // Enable hybrid mode
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
    timeout: 5000,  // Faster timeout to trigger fallback
  },
});

const result = await qubit.judge("action", "context", "safety");

// System info shows which backend was used
console.log(result.system);  // "neuroquantum" | "hybrid" | "heuristic"
```

## Strict Mode

Enforce stricter judgment thresholds (requires score ≥ 70 for "Yes"):

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  strictMode: true,  // Stricter judgment threshold
});

const result = await qubit.judge("action", "context", "safety");

// With strictMode: decision="Yes" requires score >= 70
// Otherwise decision="No"
console.log(result.decision);  // Only "Yes" if score >= 70
```

## Monitoring and Debugging

### Check API Availability

```typescript
const client = new NeuroQuantumAPIClient({
  baseUrl: 'http://localhost:5000',
});

// Check if API is currently available
const isAvailable = await client.isAvailable();
console.log(isAvailable ? "✓ API available" : "✗ API unavailable");

// Wait for API to become available
try {
  await client.waitForAvailable(10000);  // Wait up to 10 seconds
  console.log("✓ API is now available");
} catch (error) {
  console.error("API did not become available within timeout");
}
```

### Get Server Status

```typescript
const status = await qubit.getStatus();
console.log(status);
/*
{
  status: "operational",
  frontalEngineAvailable: true,
  judgmentHistorySize: 42,
  maxHistory: 100,
  timestamp: "2024-01-01T00:00:00Z"
}
*/
```

### Judgment History

```typescript
// Get recent judgments
const history = qubit.getHistory(10);
history.forEach((record) => {
  console.log(
    `${record.judgmentType}: ${record.decision} (${record.score})`
  );
});

// Clear history
qubit.clearHistory();
```

## Configuration

### Environment Variables

```bash
# Enable NeuroQuantum backend
QUBIT_NEUROQUANTUM_ENABLED=true

# API endpoint
QUBIT_NEUROQUANTUM_BASE_URL=http://localhost:5000

# Request timeout (ms)
QUBIT_NEUROQUANTUM_TIMEOUT=30000

# Retry settings
QUBIT_NEUROQUANTUM_MAX_RETRIES=3
QUBIT_NEUROQUANTUM_RETRY_DELAY_MS=1000

# Hybrid mode
QUBIT_FALLBACK_TO_HEURISTICS=true

# Strict mode
QUBIT_STRICT_MODE=true
```

### Runtime Configuration

```typescript
const qubit = new QubitAI({
  // Basic settings
  strictMode: true,
  enableLogging: true,
  maxJudgmentHistory: 100,

  // NeuroQuantum backend
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
    timeout: 30000,
    maxRetries: 3,
    retryDelayMs: 1000,
  },

  // Hybrid mode
  fallbackToHeuristics: true,

  // Product info
  productName: "My Quantum App",
  description: "App using quantum-inspired reasoning",
});
```

## Error Handling

```typescript
try {
  const result = await qubit.judge("action", "context", "safety");
} catch (error) {
  if (error.message.includes("timeout")) {
    console.error("Request timed out - API may be slow");
  } else if (error.message.includes("ERR_NETWORK")) {
    console.error("Network error - API may be unreachable");
  } else if (error.message.includes("HTTP 500")) {
    console.error("Server error - check Python API logs");
  } else {
    console.error("Unknown error:", error);
  }
}
```

## Retry Logic

The client automatically retries on network errors with exponential backoff:

```
Attempt 1: Immediate
Attempt 2: Wait 1000ms (default retryDelayMs)
Attempt 3: Wait 2000ms (1000 * 2^1)
Attempt 4: Wait 4000ms (1000 * 2^2)
```

Configure retries:

```typescript
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    baseUrl: 'http://localhost:5000',
    maxRetries: 5,        // More aggressive retries
    retryDelayMs: 500,    // Shorter initial delay
  },
});
```

## Performance Optimization

### Batch Operations

Process multiple judgments in a single request:

```typescript
const client = new NeuroQuantumAPIClient();

// Instead of multiple individual requests
const results = await client.batchJudge([
  { action: "action1", context: "context1", judgment_type: "safety" },
  { action: "action2", context: "context2", judgment_type: "ethics" },
  { action: "action3", context: "context3", judgment_type: "quality" },
]);

console.log(`Processed ${results.count} judgments efficiently`);
```

### Connection Pooling

The API client handles connection reuse automatically. Create a single client instance and reuse it:

```typescript
// Good: Single instance reused
const qubit = new QubitAI({ neuroquantumEnabled: true });

async function processActions(actions) {
  for (const action of actions) {
    await qubit.judge(action, context);
  }
}

// Avoid: Creating new instances repeatedly
async function processActionsInefficient(actions) {
  for (const action of actions) {
    const qubit = new QubitAI({ neuroquantumEnabled: true });  // ✗ Inefficient
    await qubit.judge(action, context);
  }
}
```

### Timeout Configuration

Balance responsiveness and reliability:

```typescript
// Fast responses (risky for slow network)
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    timeout: 5000,  // 5 seconds
  },
});

// Patient waiting (better for slow network)
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    timeout: 60000,  // 60 seconds
  },
});
```

## Troubleshooting

### API Not Responding

```bash
# 1. Verify server is running
curl http://localhost:5000/api/v1/health

# 2. Check server logs
tail -f neuroquantum_api_server.log

# 3. Restart with debug output
python neuroquantum_api_server.py --debug

# 4. Check network connectivity
ping localhost
netstat -an | grep 5000
```

### Slow Responses

1. Check GPU availability: `python neuroquantum_api_server.py --gpu-tier high`
2. Monitor server CPU/memory usage
3. Increase request timeout
4. Use batch operations for multiple judgments
5. Consider running multiple server instances with load balancing

### Timeout Errors

```typescript
// Increase timeout for complex reasoning tasks
const qubit = new QubitAI({
  neuroquantumEnabled: true,
  neuroquantumConfig: {
    timeout: 60000,  // 60 seconds for complex analysis
    maxRetries: 2,   // Fewer retries with longer timeout
  },
});
```

### Connection Refused

1. Verify Python server is running: `ps aux | grep neuroquantum_api_server`
2. Check correct URL and port configuration
3. Verify firewall rules allow connections
4. Check if another process is using port 5000

## Integration Testing

```typescript
import { QubitAI } from 'qubit-ai';

describe('NeuroQuantum Integration', () => {
  it('should judge actions with quantum reasoning', async () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: 'http://localhost:5000',
      },
    });

    const result = await qubit.judge(
      "test action",
      "test context",
      "safety"
    );

    expect(result.decision).toMatch(/^(Yes|No)$/);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.confidence).toMatch(/^(high|medium|low)$/);
  });

  it('should handle API unavailability gracefully', async () => {
    const qubit = new QubitAI({
      neuroquantumEnabled: true,
      neuroquantumConfig: {
        baseUrl: 'http://localhost:9999',  // Non-existent server
        timeout: 1000,
        maxRetries: 1,
      },
    });

    // With fallback enabled, should use heuristics
    const qubit_fallback = new QubitAI({
      neuroquantumEnabled: true,
      fallbackToHeuristics: true,
      neuroquantumConfig: {
        baseUrl: 'http://localhost:9999',
        timeout: 1000,
        maxRetries: 1,
      },
    });

    const result = await qubit_fallback.judge(
      "action",
      "context"
    );

    // Should still get a result from heuristic fallback
    expect(result).toBeDefined();
  });
});
```

## See Also

- [QubitAI Core Documentation](./README.md)
- [LLM Backend Integration](./LLM_INTEGRATION.md)
- [Configuration Guide](./CONFIGURATION.md)
- [API Reference](./API.md)
