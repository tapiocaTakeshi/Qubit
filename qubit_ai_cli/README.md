# Qubit AI CLI - Interactive Chat

🤖 **A full-screen terminal chat interface — like Claude Code / Codex CLI — powered by the QBNN (quantum-inspired) engine**

Running the CLI drops you straight into an interactive chat screen built with
[Ink](https://github.com/vadimdemedes/ink): a header banner, a scrolling
conversation area, a live status line, and a bordered input box pinned to the
bottom of the terminal. Just start typing.

## Features

🖥️ **Full-screen TUI**: A polished chat screen (Ink-based) like Claude Code / Codex CLI  
✨ **Real-time Conversation**: Chat interactively with the quantum-inspired QBNN engine  
⏳ **Live status**: Animated spinner while the model is thinking + per-response timing  
💾 **Conversation History**: In-session history with one-command JSON export  
⚙️ **Configurable Parameters**: Adjust temperature and token limits on the fly via slash commands  
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
qbnn          # launches the chat screen
# qubit-chat  # alias, same thing
```

### Development

```bash
npm run dev                    # Run the TUI directly with tsx
```

## Usage

### Interactive Mode (default)

Just run the command with no arguments and the full-screen chat interface opens:

```bash
qbnn
# or, from source:
npm start
# or
node dist/bin/cli.js
```

You'll see a chat screen like this:

```
╭────────────────────────────────────────────────────────────╮
│ ✶ Qubit AI  quantum-inspired chat · QBNN engine             │
│ temp 0.7 · max 150 tokens · top-k 40 · top-p 0.9            │
╰────────────────────────────────────────────────────────────╯
 Type a message and press Enter. /help for commands, /exit to quit.

› You
  こんにちは、今日はどんな日ですか？
⏺ Qubit   1245ms
  今日は素晴らしい一日ですね...

╭────────────────────────────────────────────────────────────╮
│ › Send a message…                                           │
╰────────────────────────────────────────────────────────────╯
 temp 0.7 · 150 tokens · 2 msgs · Ctrl+C to quit
```

Type a message and press Enter to chat. Press `Ctrl+C` or type `/exit` to quit.

### Single Query Mode

Pass a question as an argument to get a one-shot answer (no TUI):

```bash
qbnn "Tell me about artificial intelligence"
# or
node dist/bin/cli.js "What is quantum computing?"
```

### Help

```bash
qbnn --help
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

Edit `src/bin/cli.tsx` to modify defaults:

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

### Single Query Mode

```
› You: Explain quantum computing
⏺ Qubit: Quantum computing uses quantum mechanics principles...
  generated in 1180ms
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
│   ├── chat.ts          # Core chat engine (QubitAIChat)
│   ├── types.ts         # Type definitions
│   ├── index.ts         # Public API
│   ├── ui/
│   │   └── app.tsx      # Full-screen Ink chat UI
│   └── bin/
│       └── cli.tsx      # CLI entry point (TUI + single-query mode)
├── dist/                # Compiled JavaScript
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
└── README.md            # This file
```

The interactive UI is built with [Ink](https://github.com/vadimdemedes/ink)
(React for the terminal). `cli.tsx` decides between launching the full-screen
`<App>` (no arguments) and one-shot query mode (a question passed as an
argument); `app.tsx` owns the chat screen, slash commands, and rendering.

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
