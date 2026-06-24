/**
 * Qubit AI — Full-screen interactive chat TUI
 *
 * A Claude Code / Codex CLI style terminal chat interface built with Ink.
 * Renders a header banner, a scrolling conversation area, a live status
 * line and a bordered input box at the bottom of the screen.
 */

import React, { useState, useCallback } from "react";
import { Box, Text, useApp, useStdout } from "ink";
import TextInput from "ink-text-input";
import Spinner from "ink-spinner";
import * as fs from "fs";
import type { QubitAIChat } from "../chat.js";

export interface UIMessage {
  role: "user" | "assistant" | "system";
  content: string;
  meta?: string;
}

interface AppProps {
  chat: QubitAIChat;
  onExit?: (messageCount: number) => void;
}

const TITLE = "Qubit AI";
const SUBTITLE = "quantum-inspired chat · QBNN engine";

/**
 * Header banner shown once at the top of the session.
 */
function Banner({ config }: { config: ReturnType<QubitAIChat["getConfig"]> }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box
        borderStyle="round"
        borderColor="cyan"
        paddingX={1}
        flexDirection="column"
      >
        <Text>
          <Text color="cyan" bold>
            ✶ {TITLE}
          </Text>{" "}
          <Text dimColor>{SUBTITLE}</Text>
        </Text>
        <Text dimColor>
          temp {config.temperature} · max {config.maxTokens} tokens · top-k{" "}
          {config.topK} · top-p {config.topP}
        </Text>
      </Box>
      <Box paddingX={1}>
        <Text dimColor>
          Type a message and press Enter. <Text color="cyan">/help</Text> for
          commands, <Text color="cyan">/exit</Text> to quit.
        </Text>
      </Box>
    </Box>
  );
}

/**
 * A single rendered message row.
 */
function MessageRow({ message }: { message: UIMessage }) {
  if (message.role === "system") {
    return (
      <Box flexDirection="column" marginBottom={1}>
        {message.content.split("\n").map((line, i) => (
          <Text key={i} color="yellow" dimColor>
            {line}
          </Text>
        ))}
      </Box>
    );
  }

  const isUser = message.role === "user";
  const marker = isUser ? "›" : "⏺";
  const markerColor = isUser ? "magenta" : "green";
  const label = isUser ? "You" : "Qubit";

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text color={markerColor} bold>
          {marker}{" "}
        </Text>
        <Text bold color={markerColor}>
          {label}
        </Text>
        {message.meta ? <Text dimColor>  {message.meta}</Text> : null}
      </Box>
      <Box paddingLeft={2}>
        <Text wrap="wrap">{message.content}</Text>
      </Box>
    </Box>
  );
}

export function App({ chat, onExit }: AppProps) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [config, setConfig] = useState(chat.getConfig());

  const pushMessage = useCallback((msg: UIMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const pushSystem = useCallback(
    (content: string) => pushMessage({ role: "system", content }),
    [pushMessage]
  );

  const quit = useCallback(() => {
    onExit?.(chat.getHistory().length / 2);
    exit();
  }, [chat, exit, onExit]);

  /**
   * Handle a slash command. Returns true if the input was a command.
   */
  const handleCommand = useCallback(
    (raw: string): boolean => {
      const parts = raw.trim().slice(1).split(/\s+/);
      const cmd = (parts[0] || "").toLowerCase();

      switch (cmd) {
        case "help":
          pushSystem(
            [
              "Commands:",
              "  /help          Show this help",
              "  /clear         Clear the conversation",
              "  /history       Show full history",
              "  /export        Save conversation to JSON",
              "  /config        Show generation settings",
              "  /temp <0-2>    Set temperature",
              "  /tokens <n>    Set max tokens (10-500)",
              "  /exit, /quit   Leave the chat",
            ].join("\n")
          );
          return true;

        case "clear":
          chat.clearHistory();
          setMessages([]);
          return true;

        case "history": {
          const history = chat.getHistory();
          if (history.length === 0) {
            pushSystem("No messages yet.");
            return true;
          }
          pushSystem(
            history
              .map(
                (m, i) =>
                  `[${i + 1}] ${m.role === "user" ? "You" : "Qubit"}: ${m.content}`
              )
              .join("\n")
          );
          return true;
        }

        case "export": {
          const timestamp = new Date().toISOString().split("T")[0];
          const filename = `qubit-chat-${timestamp}.json`;
          fs.writeFileSync(filename, chat.exportConversation());
          pushSystem(`Conversation exported to ${filename}`);
          return true;
        }

        case "config": {
          const c = chat.getConfig();
          pushSystem(
            [
              "Generation settings:",
              `  temperature        ${c.temperature}`,
              `  max tokens         ${c.maxTokens}`,
              `  top-k              ${c.topK}`,
              `  top-p              ${c.topP}`,
              `  repetition penalty ${c.repetitionPenalty}`,
            ].join("\n")
          );
          return true;
        }

        case "temp": {
          const value = parseFloat(parts[1]);
          if (isNaN(value) || value < 0 || value > 2) {
            pushSystem("Usage: /temp <0.0-2.0>");
            return true;
          }
          chat.updateConfig({ temperature: value });
          setConfig(chat.getConfig());
          pushSystem(`Temperature set to ${value}`);
          return true;
        }

        case "tokens": {
          const value = parseInt(parts[1], 10);
          if (isNaN(value) || value < 10 || value > 500) {
            pushSystem("Usage: /tokens <10-500>");
            return true;
          }
          chat.updateConfig({ maxTokens: value });
          setConfig(chat.getConfig());
          pushSystem(`Max tokens set to ${value}`);
          return true;
        }

        case "exit":
        case "quit":
        case "q":
          quit();
          return true;

        default:
          pushSystem(`Unknown command: /${cmd} — try /help`);
          return true;
      }
    },
    [chat, pushSystem, quit]
  );

  const handleSubmit = useCallback(
    async (value: string) => {
      const text = value.trim();
      if (!text || isGenerating) {
        return;
      }
      setInput("");

      if (text.startsWith("/")) {
        handleCommand(text);
        return;
      }

      pushMessage({ role: "user", content: text });
      setIsGenerating(true);

      try {
        const start = Date.now();
        const response = await chat.sendMessage(text);
        const duration = Date.now() - start;
        pushMessage({
          role: "assistant",
          content: response,
          meta: `${duration}ms`,
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : String(error);
        let hint = "";
        if (message.includes("HF_TOKEN")) {
          hint = "\nSet your token: export HF_TOKEN='hf_...'";
        }
        pushSystem(`⚠ Failed to generate response: ${message}${hint}`);
      } finally {
        setIsGenerating(false);
      }
    },
    [chat, handleCommand, isGenerating, pushMessage, pushSystem]
  );

  return (
    <Box flexDirection="column" width={stdout?.columns}>
      <Banner config={config} />

      <Box flexDirection="column">
        {messages.map((message, i) => (
          <MessageRow key={i} message={message} />
        ))}
      </Box>

      {isGenerating ? (
        <Box marginBottom={1}>
          <Text color="green">
            <Spinner type="dots" />
          </Text>
          <Text dimColor> Qubit is thinking…</Text>
        </Box>
      ) : null}

      <Box
        borderStyle="round"
        borderColor={isGenerating ? "gray" : "cyan"}
        paddingX={1}
      >
        <Text color="magenta" bold>
          ›{" "}
        </Text>
        {isGenerating ? (
          <Text dimColor>waiting for response…</Text>
        ) : (
          <TextInput
            value={input}
            onChange={setInput}
            onSubmit={handleSubmit}
            placeholder="Send a message…"
          />
        )}
      </Box>

      <Box paddingX={1}>
        <Text dimColor>
          temp {config.temperature} · {config.maxTokens} tokens ·{" "}
          {messages.filter((m) => m.role !== "system").length} msgs · Ctrl+C to
          quit
        </Text>
      </Box>
    </Box>
  );
}

export default App;
