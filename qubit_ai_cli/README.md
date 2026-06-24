# Qubit AI CLI - Interactive Chat

🤖 **Interactive chat interface powered by Qubit AI and quantum-inspired neural networks**

A command-line chat application that uses the `qubit_ai` library to generate contextual, human-like responses in real-time.

## Features

✨ **Real-time Conversation**: Chat interactively with quantum-inspired AI  
🤖 **Model Selection**: Choose between QBNN, Gemma 2, and Gemma 7 models  
🧠 **Hybrid System**: Gemma + QBNN Frontal for intelligent reasoning + quality writing  
💾 **Conversation History**: Automatically saves and manages chat history  
⚙️ **Configurable Parameters**: Adjust temperature, token limits, and sampling strategies  
📊 **Performance Metrics**: See response generation time  
🎯 **Few-shot Learning**: Learns from examples for better responses  
💡 **Context Awareness**: Maintains conversation context for coherent multi-turn dialogue  
📥 **Export Conversations**: Save chats as JSON files  

## Installation

### From Source

```bash
cd qubit_ai_cli
npm install
npm run build
npm start
```

### From npm (when published)

```bash
npm install -g qubit_ai_cli
qubit-chat
```

### Development

```bash
npm run dev                    # Run with ts-node
```

## Models

### Available Models

The CLI supports three different models:

| Model | Endpoint | Best For | Speed |
|-------|----------|----------|-------|
| **QBNN** | `neuroq-ai/quantum-llm` | Logical reasoning, code generation | Medium (100-150ms) |
| **Gemma 2** | `google/gemma-2-9b-it` | General conversation, content creation | Good (150-200ms) |
| **Gemma 7B** | `google/gemma-7b-it` | Real-time chat, low latency | Fast (80-120ms) |

### Selecting a Model

**Start with a specific model:**

```bash
npm start -- --model qbnn      # Default
npm start -- --model gemma-2   # Google Gemma 2
npm start -- --model gemma-7   # Google Gemma 7B
```

**Switch models during chat:**

```
You: /model
# Shows available models

You: /model gemma-2
# Switches to Gemma 2 model
```

## Hybrid Chat System

### Gemma + QBNN Frontal

The hybrid system combines two AI models for superior reasoning and generation:

- **QBNN (Phase 1)**: Quantum-inspired reasoning engine analyzes your query, identifies key concepts, and creates a structured reasoning framework
- **Gemma (Phase 2)**: High-quality language model generates responses guided by QBNN's analysis

This combination excels at:
- **Complex technical questions**: Structured reasoning + clear explanations
- **Detailed analysis**: Logical decomposition + natural presentation
- **Problem-solving**: Step-by-step reasoning + actionable output
- **Professional writing**: Organized thinking + polished text

### Starting Hybrid Chat

```bash
npm run hybrid              # Interactive hybrid chat
npm run dev:hybrid         # Development mode
```

### Hybrid Chat Commands

| Command | Description |
|---------|-------------|
| `/reasoning` | Toggle QBNN reasoning display (see analysis) |
| `/config` | View hybrid configuration (temperatures, tokens) |
| `/temp <0-2>` | Set Gemma generation temperature |
| `/tokens <num>` | Set maximum output length |

### Hybrid Chat Example

```
🧠 Qubit AI Hybrid Chat - Gemma + QBNN

You: Explain quantum computing and its applications

🧠 Reasoning: [QBNN analyzes structure and key points]
🤖 Assistant: Quantum computing leverages principles of quantum mechanics...
   [Complete, well-structured response from Gemma]
```

### When to Use Each Mode

| Task | Recommended |
|------|-------------|
| Quick conversation | `npm start` (default model) |
| Specific model needed | `npm start -- --model gemma-2` |
| Complex analysis | `npm run hybrid` (Gemma + QBNN) |
| Fastest response | `npm start -- --model gemma-7` |

## Multi-Agent Collaborative System

### Four Specialized AI Agents

The multi-agent system brings together four powerful AI agents, each with distinct expertise and specialization:

#### 1. 🧠 Claude (Analyzer)
**Role:** Deep Analysis & Logical Reasoning

- **Specialty**: Complex problem decomposition and logical analysis
- **Temperature**: 0.3 (Analytical, deterministic)
- **Max Tokens**: 200
- **Key Strengths**:
  - Breaks down complex problems into manageable components
  - Identifies core concepts and their relationships
  - Provides deep logical analysis and structural frameworks
  - Maps critical dependencies between ideas
  - Excellent for technical architecture and system design
- **Best For**:
  - System design and architecture questions
  - Complex problem decomposition
  - Logical reasoning and algorithm analysis
  - Research methodology and framework creation
  
**Example Input**: "Design a scalable microservices architecture"
**Expected Output**: Detailed logical framework with components and relationships

---

#### 2. ✍️ ChatGPT (Writer)
**Role:** Natural Communication & Clear Explanation

- **Specialty**: Making complex ideas accessible and understandable
- **Temperature**: 0.6 (Balanced, conversational)
- **Max Tokens**: 250
- **Key Strengths**:
  - Converts technical concepts into simple, clear language
  - Creates engaging and natural explanations
  - Provides practical examples and analogies
  - Focuses on accessibility without losing accuracy
  - Excellent for communication and presentation
- **Best For**:
  - Explaining technical concepts to non-experts
  - Writing clear documentation
  - Creating educational content
  - Professional communication and messaging
  
**Example Input**: "Explain quantum computing to a 10-year-old"
**Expected Output**: Simple, engaging explanation with relatable analogies

---

#### 3. 🔄 Gemini (Synthesizer)
**Role:** Multi-Perspective Integration & Holistic Understanding

- **Specialty**: Connecting multiple viewpoints and seeing the bigger picture
- **Temperature**: 0.7 (Creative, integrative)
- **Max Tokens**: 300
- **Key Strengths**:
  - Integrates multiple perspectives into cohesive insights
  - Identifies cross-domain connections and patterns
  - Provides holistic understanding of complex systems
  - Highlights relationships between seemingly disparate concepts
  - Excellent for strategic thinking and comprehensive analysis
- **Best For**:
  - Strategic business planning
  - Cross-functional problem solving
  - System-wide impact analysis
  - Identifying hidden connections and opportunities
  
**Example Input**: "How does AI impact different business departments?"
**Expected Output**: Integrated analysis showing technical, business, organizational, and ecosystem dimensions

---

#### 4. 🔍 Perplexity (Researcher)
**Role:** Research & Evidence-Based Verification

- **Specialty**: Fact verification and evidence-based insights
- **Temperature**: 0.4 (Analytical, precision-focused)
- **Max Tokens**: 200
- **Key Strengths**:
  - Identifies and verifies key facts with evidence
  - Provides current information and research findings
  - Notes gaps and areas requiring verification
  - Grounds recommendations in factual evidence
  - Excellent for research and validation
- **Best For**:
  - Fact-checking and verification
  - Literature review and research synthesis
  - Evidence-based recommendations
  - Identifying knowledge gaps and research needs
  
**Example Input**: "What are the latest advances in AI?"
**Expected Output**: Verified facts with sources, emerging trends, and items needing further verification

---

### Agent Coordination & Synthesis

When you ask the multi-agent system a question, here's what happens:

```
1. Query Distribution
   └─> All 4 agents receive your question simultaneously

2. Parallel Processing
   ├─> Claude analyzes logical structure (0.3°C)
   ├─> ChatGPT crafts clear explanations (0.6°C)
   ├─> Gemini integrates perspectives (0.7°C)
   └─> Perplexity verifies facts (0.4°C)

3. Response Aggregation
   └─> All four responses collected (~1200ms total)

4. Intelligent Synthesis
   └─> Unified answer combining all perspectives
```

### Multi-Agent System Features

- **⚡ Parallel Processing**: All agents work simultaneously (~2 seconds total)
- **📊 Comprehensive Coverage**: Four specialized viewpoints in one response
- **🎯 Role Specialization**: Each agent optimized for their domain
- **🔗 Integrated Intelligence**: Final synthesis combines all insights
- **💾 Conversation Management**: Full history with agent attribution
- **📤 Export Capability**: Save complete multi-agent discussions

### Starting Multi-Agent Chat

```bash
npm run multi-agent              # Interactive multi-agent chat
npm run dev:multi-agent         # Development mode with hot reload
node demo-multi-agent.js        # View demonstration
```

### Multi-Agent Chat Commands

| Command | Description |
|---------|-------------|
| `/agents` | List all agents with their roles and specialties |
| `/details` | Toggle display of individual agent responses |
| `/config` | Show agent configuration and temperatures |
| `/history` | View conversation history |
| `/export` | Save multi-agent conversation to JSON |
| `/clear` | Clear conversation history |
| `/help` | Show all available commands |
| `/exit` or `/quit` | Exit the chat |

### Multi-Agent Usage Example

```bash
npm run multi-agent

You: Analyze the impact of AI on the software industry

🤖 Claude (Analyzer): [Logical framework with architecture]
✍️  ChatGPT (Writer): [Clear, engaging explanation]
🔄 Gemini (Synthesizer): [Multi-dimensional analysis]
🔍 Perplexity (Researcher): [Evidence and verification]

📊 Final Synthesis: [Integrated comprehensive answer]
```

### When to Use Multi-Agent System

Use the multi-agent system for:
- ✅ Complex questions requiring multiple perspectives
- ✅ Strategic decision-making and planning
- ✅ Comprehensive analysis and research
- ✅ Questions requiring verification and evidence
- ✅ Technical architecture and design decisions
- ✅ Business impact analysis
- ✅ Learning and knowledge synthesis

### Agent Characteristics Comparison

| Characteristic | Claude | ChatGPT | Gemini | Perplexity |
|---|---|---|---|---|
| **Focus** | Logic & Analysis | Clarity & Communication | Integration & Strategy | Verification & Evidence |
| **Temperature** | 0.3 (Analytical) | 0.6 (Balanced) | 0.7 (Creative) | 0.4 (Precise) |
| **Output Length** | 200 tokens | 250 tokens | 300 tokens | 200 tokens |
| **Speed** | Fast | Medium | Medium | Fast |
| **Best For** | Architecture, Code | Documentation, Guides | Planning, Strategy | Research, Validation |
| **Creativity Level** | Low | Medium | High | Low |
| **Detail Level** | Structural | Narrative | Integrated | Factual |

---

## Usage

### Interactive Mode

Start the chat interface:

```bash
npm start
# or
node dist/bin/cli.js

# Or with specific model
npm start -- --model gemma-2
```

Then type your messages:
```
You: こんにちは、今日はどんな日ですか？
🤖 Assistant: 今日は素晴らしい一日ですね...
```

### Single Query Mode

Ask a single question:

```bash
npm start "Tell me about artificial intelligence"
```

Or programmatically:

```bash
node dist/bin/cli.js "What is quantum computing?"
```

## Commands

Type these commands during interactive mode:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Show available models and current selection |
| `/model <name>` | Switch to a different model (qbnn, gemma-2, gemma-7) |
| `/clear` | Clear conversation history |
| `/history` | View conversation history |
| `/export` | Save conversation to JSON |
| `/config` | Show current configuration (including model) |
| `/temp <0-2>` | Set temperature (creativity) |
| `/tokens <10-500>` | Set max tokens to generate |
| `/exit` or `/quit` | Exit the chat |

### Examples

**Start with a specific model:**
```bash
npm start -- --model gemma-2     # Start with Gemma 2
npm start -- --model gemma-7     # Start with Gemma 7B
npm start -- --model qbnn        # Start with QBNN (default)
```

**Get help:**
```
You: /help
```

**View and switch models:**
```
You: /model              # Show all available models
You: /model gemma-2      # Switch to Gemma 2
You: /config             # Show current configuration with model
```

**Adjust creativity:**
```
You: /temp 0.5        # More deterministic
You: /temp 1.5        # More creative
```

**Export conversation:**
```
You: /export
```

## Environment Variables

### Required

`HF_TOKEN` - Your HuggingFace API token (optional for public models)

```bash
export HF_TOKEN="hf_..."
npm start
```

### Optional

`QUBIT_HISTORY_DIR` - Custom directory for saving chat history (default: `.qubit-history`)

```bash
export QUBIT_HISTORY_DIR="/path/to/history"
npm start
```

## Configuration

### Default Settings

```typescript
{
  generation: {
    maxTokens: 150,        // Maximum tokens to generate
    temperature: 0.7,      // Sampling temperature
    topK: 40,             // Top-K sampling
    topP: 0.9,            // Top-P (nucleus) sampling
    repetitionPenalty: 1.2, // Penalize repetition
  },
  contextWindowSize: 5,    // Recent messages to consider
  enableHistory: true,     // Save conversation history
}
```

### Customizing Configuration

Edit `src/bin/cli.ts` to modify defaults:

```typescript
const chat = await createChat({
  generation: {
    maxTokens: 200,
    temperature: 0.8,
  },
});
```

## Temperature Guide

- **0.0-0.3**: Deterministic, factual (use for information retrieval)
- **0.4-0.7**: Balanced (default, good for general conversation)
- **0.8-1.2**: Creative, varied (use for creative writing)
- **1.3-2.0**: Very creative, unpredictable (experimental)

## Output Examples

### Interactive Mode

```
╔════════════════════════════════════════════════════════════╗
║         🤖 Qubit AI Interactive Chat CLI 🤖              ║
║  Powered by quantum-inspired neural networks              ║
╚════════════════════════════════════════════════════════════╝

ℹ️  Type 'help' for commands | 'exit' to quit

You: What is machine learning?
🤖 Assistant: Machine learning is a subset of artificial intelligence...
ℹ️  Generated in 1245ms

You: /help
📖 Available Commands:

  help          - Show this help message
  clear         - Clear conversation history
  history       - Show conversation history
  ...
```

### Single Query Mode

```
👤 You: Explain quantum computing
🤖 Assistant: Quantum computing uses quantum mechanics principles...
ℹ️  Generated in 1180ms

✅ Chat session ended. (1 messages)
```

## Conversation History

Conversations are automatically saved to `.qubit-history/` directory:

```
.qubit-history/
├── history-1687892345123.json
├── history-1687892456789.json
└── ...
```

Each file contains:
- Session metadata (ID, timestamps, message count)
- Configuration used
- All messages with timestamps

### Export Format

```json
{
  "session": {
    "id": "session_1687892345_abc123def",
    "createdAt": "2024-06-24T12:34:56Z",
    "updatedAt": "2024-06-24T12:45:30Z",
    "messageCount": 5
  },
  "config": {
    "generation": {
      "maxTokens": 150,
      "temperature": 0.7,
      ...
    }
  },
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?",
      "timestamp": "2024-06-24T12:34:56Z"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking...",
      "timestamp": "2024-06-24T12:34:58Z"
    }
  ]
}
```

## Use Cases

### 1. Learning & Research

```bash
npm start "Explain the theory of relativity in simple terms"
npm start "What are the main causes of climate change?"
npm start "Summarize the benefits of machine learning"
```

### 2. Brainstorming

```bash
npm start "Give me 5 creative business ideas"
npm start "What are innovative solutions for traffic congestion?"
```

### 3. Writing Assistance

```bash
npm start "Help me write an introduction to my blog post"
npm start "Improve this sentence: [your sentence]"
```

### 4. Q&A Sessions

```
npm start
You: What is the capital of France?
You: Tell me more about its history
You: What are popular tourist attractions?
```

## Architecture

```
qubit_ai_cli/
├── src/
│   ├── chat.ts          # Core chat functionality
│   ├── types.ts         # Type definitions
│   ├── index.ts         # Public API
│   └── bin/
│       └── cli.ts       # Interactive CLI interface
├── dist/                # Compiled JavaScript
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
└── README.md            # This file
```

### Key Classes

**QubitAIChat**: Main chat class

```typescript
const chat = await createChat();

// Send a message
const response = await chat.sendMessage("Hello!");

// Get history
const messages = chat.getHistory();

// Clear history
chat.clearHistory();

// Update config
chat.updateConfig({ temperature: 0.8 });

// Export conversation
const json = chat.exportConversation();
```

## Troubleshooting

### "Cannot find module 'qubit_ai'"

Install dependencies:
```bash
npm install
```

### "HF_TOKEN not set"

The CLI works without a token for public models, but performance is limited. Set your token:

```bash
export HF_TOKEN="hf_your_token_here"
```

Get a free token at: https://huggingface.co/settings/tokens

### "Connection timeout"

The API might be overloaded. The CLI retries automatically. Try:
1. Wait a few seconds and try again
2. Use a shorter prompt
3. Reduce `maxTokens`

### "Out of memory"

If the process crashes with memory errors:
1. Clear history: `/clear`
2. Reduce `contextWindowSize` in the config
3. Exit and restart: `/exit`

## Demo Scripts

### Model Selection Demo

Learn about available models and how to use them:

```bash
node demo-model-selection.js
```

This demonstrates:
- Available models (QBNN, Gemma 2, Gemma 7B)
- How to select models at startup
- Runtime model switching
- Model comparison and selection guide
- Practical usage examples

### Advanced Features Demo

See long-form reasoning, code generation, and deep analysis:

```bash
node demo-advanced-features.js
```

### Chat Simulation Demo

Watch an interactive chat simulation:

```bash
node demo-chat-simulation.js
```

### Hybrid Chat Demo

Learn about the Gemma + QBNN Frontal system:

```bash
node demo-hybrid-chat.js
```

This demonstrates:
- Hybrid system architecture (QBNN reasoning + Gemma generation)
- Comparison with single-model approaches
- When to use hybrid chat vs. standard models
- Configuration options
- Use cases and best practices

### Multi-Agent Collaborative System Demo

Experience the power of four specialized AI agents working together:

```bash
node demo-multi-agent.js
```

This demonstrates:
- Four specialized AI agents (Claude, ChatGPT, Gemini, Perplexity)
- Parallel agent processing and coordination
- Role-based specialization and expertise
- Real-world use cases:
  - Technology analysis with cloud computing example
  - Strategic business planning with AI startup example
- System architecture diagram
- Performance characteristics (~2 seconds total)
- Command reference and features
- When to use multi-agent system

## Development

### Build

```bash
npm run build
```

### Run in Development

```bash
npm run dev
```

### Type Checking

```bash
npx tsc --noEmit
```

## Performance

Typical response times:
- **Simple responses**: 500-1000ms
- **Complex responses**: 1000-3000ms
- **Dataset loading**: 2000-5000ms

Times vary based on:
- Network latency
- API server load
- Token count
- System resources

## Limitations

- Maximum input length depends on model
- History is stored locally in JSON (not encrypted)
- Requires active internet connection for generation
- Single-turn chat (no persistent server)

## Future Enhancements

- [ ] Streaming responses
- [ ] Voice input/output
- [ ] Database for persistent history
- [ ] Cloud sync for conversations
- [ ] Custom model fine-tuning
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] Multi-language support
- [ ] Plugin system

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT © tapiocaTakeshi

## Support

- 📖 [Main Qubit AI Documentation](../npm/README.md)
- 🐛 [Report Issues](https://github.com/tapiocaTakeshi/Qubit/issues)
- 💬 [Discussions](https://github.com/tapiocaTakeshi/Qubit/discussions)

---

**Made with ❤️ using Qubit AI and quantum-inspired neural networks**
