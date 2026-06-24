# Qubit AI CLI - Usage Examples

Complete guide with real-world usage examples for the interactive chat interface.

## Quick Start

### Setup

```bash
cd qubit_ai_cli
npm install
npm run build
```

### Get HuggingFace Token

1. Visit https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Set environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Usage Modes

### 1. Interactive Mode (full-screen chat)

Start the full-screen chat interface (like Claude Code / Codex CLI):

```bash
qbnn
# or, from source:
npm start
```

This opens a chat screen with a header banner, scrolling conversation area,
and an input box pinned to the bottom:

```
╭────────────────────────────────────────────────────────────╮
│ ✶ Qubit AI  quantum-inspired chat · QBNN engine             │
│ temp 0.7 · max 150 tokens · top-k 40 · top-p 0.9            │
╰────────────────────────────────────────────────────────────╯
 Type a message and press Enter. /help for commands, /exit to quit.

› You
  What is artificial intelligence?
⏺ Qubit   1243ms
  Artificial intelligence is the field of computer science...

› You
  Can you explain it more simply?
⏺ Qubit   987ms
  Sure! AI is when computers can learn and make decisions...

╭────────────────────────────────────────────────────────────╮
│ › Send a message…                                           │
╰────────────────────────────────────────────────────────────╯
 temp 0.7 · 150 tokens · 4 msgs · Ctrl+C to quit
```

Type a message and press Enter to chat. Slash commands (`/help`, `/config`,
`/temp`, …) work inline. Press `Ctrl+C` or type `/exit` to quit.

### 2. Single Query Mode

Ask a single question without opening interactive mode:

```bash
npm start "What is machine learning?"
```

Output:

```
👤 You: What is machine learning?
🤖 Assistant: Machine learning is a branch of artificial intelligence...
ℹ️  Generated in 1156ms

✅ Chat session ended. (1 messages)
```

## Commands Reference

All commands are prefixed with `/` in interactive mode.

### Help

```
You: /help
```

Shows all available commands and tips.

### Conversation Management

**Clear history:**
```
You: /clear
✅ Conversation history cleared
```

**View conversation:**
```
You: /history

📜 Conversation History:

[1] 👤 You: Hello
[2] 🤖 Assistant: Hi there! How can I help you today?
```

**Export conversation:**
```
You: /export
✅ Conversation exported to qubit-chat-2024-06-24.json
```

### Configuration

**View current settings:**
```
You: /config

⚙️  Current Configuration:

  Temperature: 0.7
  Max Tokens: 150
  Top K: 40
  Top P: 0.9
  Repetition Penalty: 1.2
```

**Adjust temperature:**
```
You: /temp 0.3
✅ Temperature set to 0.3

You: /temp 1.5
✅ Temperature set to 1.5
```

**Adjust max tokens:**
```
You: /tokens 200
✅ Max tokens set to 200

You: /tokens 100
✅ Max tokens set to 100
```

### Exit

```
You: /exit
# or
You: /quit
```

## Practical Examples

### Example 1: Learning & Research

```bash
$ export HF_TOKEN="hf_..."
$ npm start
```

```
ℹ️  Type 'help' for commands | 'exit' to quit

You: What are the main types of machine learning?
🤖 Assistant: The main types of machine learning are:
1. Supervised Learning - learning from labeled data
2. Unsupervised Learning - finding patterns in unlabeled data
3. Reinforcement Learning - learning through trial and error

You: Can you explain supervised learning in more detail?
🤖 Assistant: Supervised learning involves training a model on labeled data...

You: Give me some practical examples
🤖 Assistant: Some practical examples include:
- Email spam detection
- House price prediction
- Image classification
- Disease diagnosis

You: /export
✅ Conversation exported to qubit-chat-2024-06-24.json

You: /exit
✅ Chat session ended. (3 messages)
```

### Example 2: Creative Writing

```bash
npm start
```

```
You: /temp 1.2
✅ Temperature set to 1.2

You: Write me a short story opening about a mysterious library
🤖 Assistant: The ancient library materialized before her eyes as if 
emerging from the fog itself. Shelves of impossible architecture spiraled 
upward into darkness, books bound in leather that glowed with inner light...

You: Continue the story, add more mystery
🤖 Assistant: She reached for a tome bound in silver, but her hand 
trembled. The book seemed to whisper, calling to her in a language she 
somehow understood but couldn't quite remember learning...

You: /history
📜 Conversation History:
[1] 👤 You: Write me a short story opening about a mysterious library
[2] 🤖 Assistant: The ancient library materialized before her eyes...
[3] 👤 You: Continue the story, add more mystery
[4] 🤖 Assistant: She reached for a tome bound in silver...

You: /exit
```

### Example 3: Q&A Sessions

```bash
npm start "List 5 benefits of renewable energy"
```

```
👤 You: List 5 benefits of renewable energy
🤖 Assistant: Here are 5 key benefits of renewable energy:

1. Environmental: Reduces greenhouse gas emissions and air pollution
2. Economic: Creates jobs and reduces long-term energy costs
3. Sustainable: Energy sources like solar and wind are infinite
4. Health: Cleaner air leads to better public health outcomes
5. Energy Independence: Reduces reliance on fossil fuel imports

ℹ️  Generated in 1432ms

✅ Chat session ended. (1 messages)
```

### Example 4: Problem Solving

```bash
npm start
```

```
You: I need to optimize my JavaScript code for performance
🤖 Assistant: Here are key strategies for optimizing JavaScript:

1. Minimize DOM manipulation
2. Use efficient data structures
3. Cache frequently accessed values
4. Use async/await for I/O operations
5. Profile with Chrome DevTools

You: Can you show me an example?
🤖 Assistant: Sure! Here's a common optimization pattern...

You: /config
⚙️  Current Configuration:
  Temperature: 0.7
  Max Tokens: 150
  ...

You: /tokens 200
✅ Max tokens set to 200

You: Now give me a longer example with explanation
🤖 Assistant: [Longer, more detailed response with 200 tokens available]

You: /exit
```

## Temperature Settings Guide

Use appropriate temperature for different tasks:

### Factual Tasks (Low Temperature)

```bash
npm start
```

```
You: /temp 0.2
✅ Temperature set to 0.2

You: What is the capital of France?
🤖 Assistant: Paris is the capital of France.
```

**Good for:** Facts, math, definitions, consistent answers

### General Conversation (Medium Temperature)

```
You: /temp 0.7
✅ Temperature set to 0.7

You: How should I spend my weekend?
🤖 Assistant: Here are some ideas for your weekend:
1. Explore a new restaurant
2. Outdoor activities...
```

**Good for:** General conversation, Q&A, recommendations

### Creative Tasks (High Temperature)

```
You: /temp 1.3
✅ Temperature set to 1.3

You: Write a creative poem about technology
🤖 Assistant: Silicon dreams cascade through fiber-optic veins,
Dancing between zeros and ones...
```

**Good for:** Creative writing, brainstorming, storytelling

## Token Management

Adjust max tokens based on response length needs:

```bash
npm start
```

```
# Short responses (faster, cheaper)
You: /tokens 50
✅ Max tokens set to 50

You: What's 2+2?
🤖 Assistant: 4
ℹ️  Generated in 234ms

# Long responses (slower, more detailed)
You: /tokens 300
✅ Max tokens set to 300

You: Explain quantum mechanics
🤖 Assistant: [300 tokens of detailed explanation]
ℹ️  Generated in 2341ms
```

## Export & Sharing

Save conversations for documentation or sharing:

```bash
npm start
```

```
You: [Your conversation...]

You: /export
✅ Conversation exported to qubit-chat-2024-06-24.json

You: /exit
```

The exported file contains:

```json
{
  "session": {
    "id": "session_1687892345_abc123",
    "createdAt": "2024-06-24T12:34:56Z",
    "updatedAt": "2024-06-24T12:45:30Z",
    "messageCount": 5
  },
  "config": {
    "generation": {
      "maxTokens": 150,
      "temperature": 0.7,
      "topK": 40,
      "topP": 0.9,
      "repetitionPenalty": 1.2
    }
  },
  "messages": [
    {
      "role": "user",
      "content": "Your question",
      "timestamp": "2024-06-24T12:34:56Z"
    },
    {
      "role": "assistant",
      "content": "AI response",
      "timestamp": "2024-06-24T12:34:58Z"
    }
  ]
}
```

## Performance Tips

1. **Shorter context**: Faster responses with fewer previous messages
2. **Lower tokens**: Reduce max tokens for faster generation
3. **Batch queries**: Process multiple similar queries in one session
4. **Cache results**: Export and reuse good responses

Example:

```
You: /tokens 100
✅ Max tokens set to 100

You: Quick fact: What's the population of Tokyo?
🤖 Assistant: Tokyo has approximately 13.9 million people
ℹ️  Generated in 487ms
```

## Troubleshooting

### "fetch failed"

**Cause**: No HF_TOKEN or network issue

**Solution**:
```bash
export HF_TOKEN="hf_your_token"
npm start
```

### Response takes too long

**Solution**:
```
You: /tokens 100
✅ Max tokens set to 100
```

### Want more creative responses

**Solution**:
```
You: /temp 1.2
✅ Temperature set to 1.2
```

### Want more factual responses

**Solution**:
```
You: /temp 0.3
✅ Temperature set to 0.3
```

## Keyboard Shortcuts

- `Ctrl+C` - Exit immediately
- `Ctrl+D` - Exit (on some systems)
- Up Arrow - Recall previous input (in some terminals)
- Tab - Autocomplete (if supported)

## Advanced Usage

### Batch Processing

Process multiple queries:

```bash
npm start "Question 1"
npm start "Question 2"
npm start "Question 3"
```

### Scripting

Save commands to a file:

```bash
# Save as script.txt
echo "What is AI?" > questions.txt
echo "What is ML?" >> questions.txt
```

Then process:

```bash
while read question; do
  npm start "$question"
done < questions.txt
```

### Integration with Other Tools

Export and analyze conversations:

```bash
npm start
# ... have conversation ...
# /export creates qubit-chat-YYYY-MM-DD.json

# Parse with jq or other JSON tools
jq '.messages' qubit-chat-2024-06-24.json
```

## Best Practices

1. **Use clear prompts**: "Explain X in simple terms" works better
2. **Provide context**: "As a beginner, help me understand..."
3. **Ask follow-ups**: Build on previous responses
4. **Adjust parameters**: Temperature and tokens for your needs
5. **Save important conversations**: Use `/export`

## Limitations

- Single session per chat (no persistent server)
- Local history only (not synced to cloud)
- Requires internet for API calls
- Token limit based on model (typically 2000-4000)

---

**For more information, see [README.md](README.md)**
