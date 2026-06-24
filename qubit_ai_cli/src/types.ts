/**
 * Type definitions for Qubit AI Chat CLI
 */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

export interface GenerationConfig {
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
}

export interface ChatConfig {
  generation: GenerationConfig;
  systemPrompt?: string;
  contextWindowSize: number;
  enableHistory: boolean;
}

export interface CLIOptions {
  token?: string;
  temperature?: number;
  maxTokens?: number;
  quiet?: boolean;
  saveHistory?: boolean;
  historyFile?: string;
}
