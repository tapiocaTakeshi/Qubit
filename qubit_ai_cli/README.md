# Qubit AI CLI - Interactive Chat

🤖 **Interactive chat interface powered by Qubit AI and quantum-inspired neural networks**

A command-line chat application that uses the `qubit_ai` library to generate contextual, human-like responses in real-time.

## Features

✨ **Real-time Conversation**: Chat interactively with quantum-inspired AI  
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

## Usage

### Interactive Mode

Start the chat interface:

```bash
npm start
# or
node dist/bin/cli.js
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
| `/clear` | Clear conversation history |
| `/history` | View conversation history |
| `/export` | Save conversation to JSON |
| `/config` | Show current configuration |
| `/temp <0-2>` | Set temperature (creativity) |
| `/tokens <10-500>` | Set max tokens to generate |
| `/exit` or `/quit` | Exit the chat |

### Examples

**Get help:**
```
You: /help
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
