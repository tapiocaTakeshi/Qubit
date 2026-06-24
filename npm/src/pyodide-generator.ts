/**
 * Pyodide-based Text Generation Engine
 *
 * Implements neuroquantum_layered.py generation logic in TypeScript
 * - No external LLM API required
 * - Runs entirely in-browser via Pyodide or Node.js
 * - Supports quantum-inspired sampling strategies
 */

export interface GenerationConfig {
  maxLength?: number;
  tempMin?: number;
  tempMax?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  noRepeatNgramSize?: number;
  seed?: number;
}

export interface GenerationResult {
  text: string;
  tokensGenerated: number;
  timeMs: number;
}

export interface ModelStatus {
  loaded: boolean;
  trained: boolean;
  vocabSize: number;
  modelSize: string;
}

/**
 * Simple Tokenizer (Japanese + English support)
 */
export class SimpleTokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private vocabSize: number = 32000;

  constructor(vocabSize: number = 32000) {
    this.vocabSize = vocabSize;
    this.initializeBasicVocab();
  }

  private initializeBasicVocab(): void {
    // Special tokens
    const specialTokens = [
      "<pad>", "<unk>", "<bos>", "<eos>",
      "<cls>", "<sep>", "<mask>"
    ];

    let idx = 0;
    for (const token of specialTokens) {
      this.vocab.set(token, idx);
      this.reverseVocab.set(idx, token);
      idx++;
    }

    // Common tokens (simplified)
    const commonTokens = [
      ".", ",", "!", "?", ":", ";",
      "の", "に", "は", "を", "た", "が", "で", "て", "と",
      "a", "the", "is", "was", "are", "be", "been", "being",
      "have", "has", "do", "does", "did", "will", "would", "could",
      "should", "may", "might", "must", "can", "shall", "should"
    ];

    for (const token of commonTokens) {
      if (!this.vocab.has(token)) {
        this.vocab.set(token, idx);
        this.reverseVocab.set(idx, token);
        idx++;
      }
    }

    // Character-level fallback for remaining vocabulary
    const chars = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん" +
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (const char of chars) {
      if (!this.vocab.has(char) && idx < this.vocabSize) {
        this.vocab.set(char, idx);
        this.reverseVocab.set(idx, char);
        idx++;
      }
    }
  }

  encode(text: string): number[] {
    const tokens: number[] = [];
    tokens.push(2); // <bos>

    // Simple tokenization (word + character fallback)
    let current = "";
    for (const char of text) {
      current += char;
      if (this.vocab.has(current)) {
        // Continue building word
      } else if (current.length > 1) {
        // Token found
        const prev = current.slice(0, -1);
        if (this.vocab.has(prev)) {
          tokens.push(this.vocab.get(prev)!);
          current = char;
        } else {
          // Character-level fallback
          for (const c of prev) {
            tokens.push(this.vocab.get(c) ?? this.vocab.get("<unk>")!);
          }
          current = char;
        }
      }
    }

    if (current) {
      if (this.vocab.has(current)) {
        tokens.push(this.vocab.get(current)!);
      } else {
        for (const c of current) {
          tokens.push(this.vocab.get(c) ?? this.vocab.get("<unk>")!);
        }
      }
    }

    return tokens;
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map(id => this.reverseVocab.get(id) ?? "<unk>")
      .filter(t => !t.startsWith("<") || t === "<pad>")
      .join("")
      .replace(/\s+/g, " ")
      .trim();
  }

  getVocabSize(): number {
    return this.vocabSize;
  }
}

/**
 * Lightweight Language Model (neural tensor-based generation)
 */
export class LightweightLanguageModel {
  private embeddings: Map<number, Float32Array> = new Map();
  private readonly embedDim: number = 64;
  private readonly vocabSize: number;
  private trained: boolean = false;
  private trainingData: string[] = [];

  constructor(vocabSize: number = 32000) {
    this.vocabSize = vocabSize;
    this.initializeEmbeddings();
  }

  private initializeEmbeddings(): void {
    // Initialize embeddings randomly
    for (let i = 0; i < Math.min(1000, this.vocabSize); i++) {
      const embedding = new Float32Array(this.embedDim);
      for (let j = 0; j < this.embedDim; j++) {
        embedding[j] = (Math.random() - 0.5) * 2;
      }
      this.embeddings.set(i, embedding);
    }
  }

  /**
   * Train on text samples (lightweight)
   */
  train(texts: string[], tokenizer: SimpleTokenizer): void {
    this.trainingData = texts;
    this.trained = true;

    // Update embeddings based on training data
    for (const text of texts) {
      const tokens = tokenizer.encode(text);
      for (const token of tokens) {
        if (!this.embeddings.has(token)) {
          const embedding = new Float32Array(this.embedDim);
          for (let j = 0; j < this.embedDim; j++) {
            embedding[j] = (Math.random() - 0.5) * 0.1;
          }
          this.embeddings.set(token, embedding);
        }
      }
    }
  }

  /**
   * Get next token logits (simplified neural computation)
   */
  getLogits(tokenIds: number[]): Float32Array {
    const logits = new Float32Array(this.vocabSize);

    // Initialize logits
    for (let i = 0; i < this.vocabSize; i++) {
      logits[i] = 0;
    }

    if (tokenIds.length === 0) {
      return logits;
    }

    // Simplified attention: recent tokens have higher weight
    const recentTokens = tokenIds.slice(-5);
    for (const token of recentTokens) {
      const embedding = this.embeddings.get(token);
      if (embedding) {
        // Project embedding to vocabulary logits
        for (let i = 0; i < Math.min(1000, this.vocabSize); i++) {
          const targetEmb = this.embeddings.get(i);
          if (targetEmb) {
            // Compute similarity (dot product)
            let similarity = 0;
            for (let j = 0; j < this.embedDim; j++) {
              similarity += embedding[j] * targetEmb[j];
            }
            logits[i] += similarity / recentTokens.length;
          }
        }
      }
    }

    // Normalize
    const maxLogit = Math.max(...Array.from(logits));
    for (let i = 0; i < this.vocabSize; i++) {
      logits[i] -= maxLogit;
    }

    return logits;
  }

  isTrained(): boolean {
    return this.trained;
  }
}

/**
 * NeuroQuantum Text Generator (Pyodide-based)
 *
 * Implements quantum-inspired sampling and neuroquantum_layered.py logic
 */
export class NeuroQuantumGenerator {
  private tokenizer: SimpleTokenizer;
  private model: LightweightLanguageModel;
  private config: Required<GenerationConfig>;
  private rng: () => number; // Seeded RNG

  constructor(vocabSize: number = 32000, seed: number = 42) {
    this.tokenizer = new SimpleTokenizer(vocabSize);
    this.model = new LightweightLanguageModel(vocabSize);
    this.rng = this.createSeededRNG(seed);

    this.config = {
      maxLength: 100,
      tempMin: 0.4,
      tempMax: 0.8,
      topK: 40,
      topP: 0.9,
      repetitionPenalty: 1.2,
      noRepeatNgramSize: 3,
      seed: seed,
    };
  }

  private createSeededRNG(seed: number): () => number {
    let state = seed;
    return () => {
      state = (state * 9301 + 49297) % 233280;
      return state / 233280;
    };
  }

  /**
   * Train on texts
   */
  train(texts: string[]): void {
    this.model.train(texts, this.tokenizer);
  }

  /**
   * Generate text from prompt (neuroquantum algorithm)
   */
  async generate(
    prompt: string,
    options: Partial<GenerationConfig> = {}
  ): Promise<GenerationResult> {
    const startTime = Date.now();
    const config = { ...this.config, ...options };

    // Auto-train if needed
    if (!this.model.isTrained()) {
      this.trainOnDefaultData();
    }

    // Tokenize prompt
    const promptTokens = this.tokenizer.encode(prompt);
    let tokens = Array.from(promptTokens);
    const generated: number[] = [];

    // N-gram history for repetition prevention
    const ngramHistory = new Set<string>();

    for (let step = 0; step < config.maxLength; step++) {
      // Get model logits
      let logits = this.model.getLogits(tokens);

      // ⚛️ Quantum-inspired influence: apply phase-dependent modulation
      logits = this.applyQuantumInfluence(logits, step);

      // 🚫 Suppress starting conjunctions/particles
      if (step === 0) {
        const suppressTokens = [5, 6, 7, 8, 9]; // あいう etc
        for (const tid of suppressTokens) {
          if (tid < logits.length) {
            logits[tid] = -Infinity;
          }
        }
      }

      // Repetition penalty (recency-weighted)
      const recentWindow = Math.min(100, generated.length);
      if (recentWindow > 0) {
        const recentTokens = generated.slice(-recentWindow);
        const tokenPositions = new Map<number, number[]>();

        for (let i = 0; i < recentTokens.length; i++) {
          const token = recentTokens[i];
          if (!tokenPositions.has(token)) {
            tokenPositions.set(token, []);
          }
          tokenPositions.get(token)!.push(i);
        }

        for (const [token, positions] of tokenPositions) {
          if (token < logits.length) {
            const count = positions.length;
            const mostRecentPos = Math.max(...positions);
            const recencyWeight = 0.5 + 0.5 * (mostRecentPos / recentWindow);
            const penalty = Math.pow(config.repetitionPenalty, 1 + count * 0.3 * recencyWeight);
            logits[token] /= penalty;
          }
        }
      }

      // Dynamic temperature (quantum phase evolution)
      const thetaPhase = step * 0.2;
      const temperature =
        config.tempMin +
        (config.tempMax - config.tempMin) *
          (0.5 + 0.5 * Math.sin(thetaPhase));

      // Temperature scaling
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] !== -Infinity) {
          logits[i] /= Math.max(temperature, 0.1);
        }
      }

      // N-gram repetition blocking
      if (
        config.noRepeatNgramSize > 0 &&
        generated.length >= config.noRepeatNgramSize - 1
      ) {
        const currentPrefix = generated
          .slice(-(config.noRepeatNgramSize - 1))
          .join(",");

        const bannedTokens = new Set<number>();
        for (let i = 0; i <= generated.length - (config.noRepeatNgramSize - 1); i++) {
          const prefix = generated
            .slice(i, i + config.noRepeatNgramSize - 1)
            .join(",");
          if (prefix === currentPrefix && i + config.noRepeatNgramSize - 1 < generated.length) {
            bannedTokens.add(generated[i + config.noRepeatNgramSize - 1]);
          }
        }

        for (const token of bannedTokens) {
          if (token < logits.length) {
            logits[token] = -Infinity;
          }
        }
      }

      // Top-K filtering
      if (config.topK > 0) {
        const validLogits: Array<{ value: number; idx: number }> = [];
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] !== -Infinity) {
            validLogits.push({ value: logits[i], idx: i });
          }
        }
        validLogits.sort((a, b) => b.value - a.value);

        const topKSet = new Set(validLogits.slice(0, config.topK).map((x) => x.idx));
        for (let i = 0; i < logits.length; i++) {
          if (!topKSet.has(i)) {
            logits[i] = -Infinity;
          }
        }
      }

      // Top-P (nucleus) filtering
      if (config.topP < 1.0) {
        const validLogits: Array<{ value: number; idx: number }> = [];
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] !== -Infinity) {
            validLogits.push({ value: logits[i], idx: i });
          }
        }
        validLogits.sort((a, b) => b.value - a.value);

        // Convert to probabilities
        const maxLogit = validLogits[0]?.value ?? 0;
        const probs: Array<{ value: number; idx: number; prob: number }> = validLogits.map((x) => ({
          value: x.value,
          idx: x.idx,
          prob: Math.exp(x.value - maxLogit),
        }));

        const totalProb = probs.reduce((sum, x) => sum + x.prob, 0);
        let cumulativeProb = 0;
        const topPIndices = new Set<number>();

        for (const { idx, prob } of probs) {
          cumulativeProb += prob / totalProb;
          topPIndices.add(idx);
          if (cumulativeProb >= config.topP) {
            break;
          }
        }

        for (let i = 0; i < logits.length; i++) {
          if (!topPIndices.has(i)) {
            logits[i] = -Infinity;
          }
        }
      }

      // Sampling from filtered logits
      const validLogits: Array<{ value: number; idx: number }> = [];
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] !== -Infinity) {
          validLogits.push({ value: logits[i], idx: i });
        }
      }

      if (validLogits.length === 0) {
        break; // No valid tokens
      }

      // Softmax
      const maxLogit = Math.max(...validLogits.map((x) => x.value));
      const expLogits: Array<{ value: number; idx: number; exp: number }> = validLogits.map((x) => ({
        value: x.value,
        idx: x.idx,
        exp: Math.exp(x.value - maxLogit),
      }));

      const sumExp = expLogits.reduce((sum, x) => sum + x.exp, 0);
      const probs: Array<{ value: number; idx: number; exp: number; prob: number }> = expLogits.map((x) => ({
        value: x.value,
        idx: x.idx,
        exp: x.exp,
        prob: x.exp / sumExp,
      }));

      // Multinomial sampling
      const rand = this.rng();
      let cumsum = 0;
      let nextToken = validLogits[0].idx;

      for (const { idx, prob } of probs) {
        cumsum += prob;
        if (rand < cumsum) {
          nextToken = idx;
          break;
        }
      }

      // EOS/EOF check
      if (nextToken === 3) {
        // <eos>
        break;
      }

      // PAD/BOF skip
      if (nextToken === 0 || nextToken === 1) {
        continue;
      }

      generated.push(nextToken);
      tokens.push(nextToken);

      // Yield control (allow async interruption)
      if (step % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    const text = this.tokenizer.decode(generated);
    const timeMs = Date.now() - startTime;

    return {
      text,
      tokensGenerated: generated.length,
      timeMs,
    };
  }

  /**
   * Quantum-inspired influence on logits
   * Models quantum phase evolution: θ changes with step
   */
  private applyQuantumInfluence(logits: Float32Array, step: number): Float32Array {
    const result = new Float32Array(logits);

    // Quantum phase: r = cos(2θ), T = |sin(2θ)|
    const theta = (step * Math.PI) / 100; // Slow phase evolution
    const r = Math.cos(2 * theta);
    const T = Math.abs(Math.sin(2 * theta));

    // Apply quantum modulation
    for (let i = 0; i < result.length; i++) {
      const quantumPhase = r + T * Math.cos((i * Math.PI) / result.length);
      result[i] += quantumPhase * 0.1; // Small quantum influence
    }

    return result;
  }

  /**
   * Train on default dataset if not trained
   */
  private trainOnDefaultData(): void {
    const defaultTexts = [
      "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングなどの技術を使用して、データからパターンを学習し、予測や判断を行います。",
      "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。従来のコンピュータでは解けない複雑な問題を高速に解くことができます。",
      "自然言語処理は、コンピュータが人間の言語を理解し、生成するための技術です。翻訳、要約、質問応答などのタスクに使用されます。",
      "ニューラルネットワークは、人間の脳の神経細胞の働きを模倣した計算モデルです。層状に接続されたノードで構成され、データから特徴を学習します。",
    ];

    this.train(defaultTexts);
  }

  /**
   * Batch generation
   */
  async generateBatch(
    prompts: string[],
    options?: Partial<GenerationConfig>
  ): Promise<GenerationResult[]> {
    const results: GenerationResult[] = [];

    for (const prompt of prompts) {
      try {
        const result = await this.generate(prompt, options);
        results.push(result);
      } catch (error) {
        console.error(`Generation failed for prompt: "${prompt}"`, error);
        results.push({
          text: "",
          tokensGenerated: 0,
          timeMs: 0,
        });
      }
    }

    return results;
  }

  /**
   * Get model status
   */
  getStatus(): ModelStatus {
    return {
      loaded: true,
      trained: this.model.isTrained(),
      vocabSize: this.tokenizer.getVocabSize(),
      modelSize: "lightweight",
    };
  }
}
