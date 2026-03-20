#!/usr/bin/env python3
"""ChatGPT・AI サービスに関するQAデータで追加学習するスクリプト。"""
import os
import sys
import torch
import torch.nn.functional as F
import math
import random

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

# ChatGPT・AIサービス関連のQAデータ
CHATGPT_QA = [
    "質問: ChatGPTについて教えて\n回答: ChatGPTはOpenAIが開発した大規模言語モデルです。GPT（Generative Pre-trained Transformer）アーキテクチャを基にしており、人間のような自然な文章を生成できます。2022年11月に公開され、質問応答、文章作成、翻訳、プログラミング支援など幅広いタスクに対応します。",
    "質問: ChatGPTとは何ですか？\n回答: ChatGPTはOpenAIが開発した対話型AIです。大量のテキストデータで事前学習したGPTモデルをベースに、人間のフィードバックによる強化学習（RLHF）で調整されています。自然な会話ができ、文章生成、要約、翻訳、コード生成など多くのタスクをこなします。",
    "質問: ChatGPTはどのように動作しますか？\n回答: ChatGPTはTransformerアーキテクチャに基づく大規模言語モデルです。膨大なテキストデータで事前学習し、次の単語を予測する能力を獲得します。さらにRLHF（人間のフィードバックによる強化学習）で微調整され、有用で安全な回答を生成できるようになっています。",
    "質問: GPTとは何の略ですか？\n回答: GPTはGenerative Pre-trained Transformerの略です。Generative（生成的）、Pre-trained（事前学習済み）、Transformer（トランスフォーマー）の頭文字を取ったものです。テキスト生成に特化した事前学習済みのTransformerモデルを意味します。",
    "質問: OpenAIとは何ですか？\n回答: OpenAIはアメリカの人工知能研究企業です。2015年にサム・アルトマンやイーロン・マスクらによって設立されました。ChatGPT、GPT-4、DALL-Eなどの先進的なAIモデルを開発しています。安全で有益なAIの開発を使命としています。",
    "質問: GPT-4について教えてください\n回答: GPT-4はOpenAIが2023年3月に公開した大規模言語モデルです。GPT-3.5の後継モデルで、より高い推論能力と正確性を持ちます。テキストだけでなく画像の入力にも対応するマルチモーダルモデルです。複雑な問題解決や創造的なタスクに優れています。",
    "質問: 大規模言語モデルとは何ですか？\n回答: 大規模言語モデル（LLM）は、膨大なテキストデータで訓練された巨大なニューラルネットワークです。数十億から数兆のパラメータを持ち、文章の理解と生成ができます。ChatGPT、Claude、Geminiなどが代表的なLLMです。自然言語処理の多くのタスクを高い精度でこなします。",
    "質問: Transformerとは何ですか？\n回答: TransformerはGoogleが2017年に発表した深層学習アーキテクチャです。自己注意機構（Self-Attention）により、入力テキスト全体の関係性を効率的に捉えます。GPT、BERT、T5など現代の言語モデルの基盤技術であり、自然言語処理に革命をもたらしました。",
    "質問: 生成AIとは何ですか？\n回答: 生成AI（ジェネレーティブAI）は、新しいコンテンツを生成できる人工知能です。テキスト生成のChatGPT、画像生成のDALL-EやStable Diffusion、音楽生成のSunoなどがあります。大量のデータからパターンを学習し、創造的な出力を行います。",
    "質問: Claudeとは何ですか？\n回答: ClaudeはAnthropic社が開発した大規模言語モデルです。安全性と有用性を重視して設計されており、長文の理解や分析に優れています。Constitutional AI（憲法AI）という独自の安全技術を用いて訓練されています。",
    "質問: Geminiとは何ですか？\n回答: GeminiはGoogleが開発した大規模言語モデルです。テキスト、画像、音声、動画など複数のモダリティに対応するマルチモーダルAIです。Google検索やGmailなどのGoogleサービスにも統合されています。",
    "質問: RLHFとは何ですか？\n回答: RLHF（Reinforcement Learning from Human Feedback）は人間のフィードバックによる強化学習です。AIモデルの出力を人間が評価し、その評価を基にモデルを改善する手法です。ChatGPTなどの対話AIを安全で有用にするために使われています。",
    "質問: プロンプトエンジニアリングとは何ですか？\n回答: プロンプトエンジニアリングは、AIに効果的な指示（プロンプト）を与える技術です。適切なプロンプトを設計することで、AIからより正確で有用な回答を引き出せます。具体的な指示、文脈の提供、出力形式の指定などのテクニックがあります。",
    "質問: AIのハルシネーションとは何ですか？\n回答: ハルシネーション（幻覚）は、AIが事実に基づかない情報を自信を持って生成する現象です。大規模言語モデルは統計的なパターンで文章を生成するため、実在しない情報をもっともらしく出力することがあります。AIの重要な課題の一つです。",
    "質問: ファインチューニングとは何ですか？\n回答: ファインチューニングは、事前学習済みモデルを特定のタスクやドメインに適応させる追加学習のことです。少量の専門データで微調整することで、特定の用途に特化した性能を得られます。ChatGPTもGPTモデルをファインチューニングして作られています。",
    "質問: 自然言語処理とは何ですか？\n回答: 自然言語処理（NLP）は、コンピュータが人間の言語を理解・生成する技術分野です。機械翻訳、感情分析、質問応答、文書要約などのタスクがあります。近年はTransformerベースの大規模言語モデルにより飛躍的に進歩しました。",
    "質問: DALL-Eとは何ですか？\n回答: DALL-EはOpenAIが開発した画像生成AIです。テキストの説明から画像を生成できます。DALL-E 2、DALL-E 3と進化し、より高品質で詳細な画像生成が可能になりました。ChatGPTと統合され、会話の中で画像を生成できます。",
    "質問: Stable Diffusionとは何ですか？\n回答: Stable Diffusionはオープンソースの画像生成AIモデルです。テキストから画像を生成する拡散モデルで、Stability AI社が中心となって開発されました。無料で利用でき、ローカル環境でも動作するため、多くのクリエイターに使われています。",
    "質問: トークンとは何ですか？\n回答: トークンは、AIモデルがテキストを処理する際の最小単位です。単語や文字の一部に分割されます。例えばChatGPTでは、英語の1単語が約1トークン、日本語の1文字が1〜3トークンに相当します。モデルの入出力量はトークン数で計測されます。",
    "質問: AIの安全性について教えてください\n回答: AIの安全性は、AIシステムが人間に害を与えないようにする研究分野です。有害なコンテンツの生成防止、偏見の排除、プライバシー保護などが重要な課題です。OpenAIやAnthropicなどの企業が、安全なAI開発のための研究を進めています。",
]

# 追加の強化QA（バリエーション）
REINFORCEMENT_QA = [
    "質問: ChatGPTの特徴は？\n回答: ChatGPTの主な特徴は、自然な対話能力、幅広い知識、多言語対応です。質問応答、文章作成、コード生成、翻訳など多様なタスクに対応します。OpenAIが開発し、GPTアーキテクチャを基盤としています。",
    "質問: ChatGPTは何ができますか？\n回答: ChatGPTは文章の生成、質問への回答、文書の要約、翻訳、プログラミングコードの作成、ブレインストーミング、文章の校正など多くのことができます。対話形式で様々なタスクを支援する汎用的なAIアシスタントです。",
    "質問: ChatGPTの仕組みを教えて\n回答: ChatGPTはTransformerアーキテクチャの大規模言語モデルです。大量のテキストで事前学習した後、RLHF（人間のフィードバックによる強化学習）で対話用に微調整されています。入力されたテキストの次に続く単語を確率的に予測して文章を生成します。",
    "質問: ChatGPTはいつ公開されましたか？\n回答: ChatGPTは2022年11月30日にOpenAIから公開されました。公開後わずか5日で100万ユーザーを獲得し、史上最速で成長したアプリケーションの一つとなりました。",
    "質問: AIチャットボットとは何ですか？\n回答: AIチャットボットは、人工知能を使って人間と対話するプログラムです。ChatGPT、Claude、Geminiなどが代表的です。自然言語を理解し、適切な応答を生成します。カスタマーサポート、教育、エンターテインメントなど幅広い分野で活用されています。",
    "質問: GPT-3.5とGPT-4の違いは？\n回答: GPT-4はGPT-3.5より大幅に進化しています。より高い推論能力、正確性、創造性を持ちます。GPT-4は画像入力にも対応するマルチモーダルモデルです。複雑な問題解決能力や指示追従能力が向上しています。",
    "質問: LLMの学習方法を教えてください\n回答: LLM（大規模言語モデル）の学習は主に2段階です。まず大量のテキストデータで事前学習し、次の単語を予測する能力を獲得します。次にファインチューニングで特定のタスクに適応させます。ChatGPTではさらにRLHFで人間の好みに合わせて調整されます。",
    "質問: AIと機械学習の違いは？\n回答: AI（人工知能）は人間の知能を模倣する技術全体を指します。機械学習はAIの一分野で、データからパターンを学習する手法です。深層学習は機械学習の一種で、ニューラルネットワークを多層にしたものです。ChatGPTは深層学習を使ったAIです。",
]


def tokenize_texts(texts, tok, max_seq_len):
    sequences = []
    for t in texts:
        ids = tok.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    cfg = checkpoint["config"]

    tokenizer = NeuroQuantumTokenizer(vocab_size=cfg["vocab_size"], model_file=TOKENIZER_PATH)
    nq_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg.get("hidden_dim", cfg["embed_dim"] * 2),
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.1),
        lambda_entangle=cfg.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params")

    # Build training data - repeat many times for reinforcement
    all_qa = []
    for _ in range(100):
        all_qa.extend(CHATGPT_QA)
    for _ in range(80):
        all_qa.extend(REINFORCEMENT_QA)
    print(f"Total QA samples: {len(all_qa)}")

    # Tokenize
    max_seq_len = cfg["max_seq_len"]
    sequences = tokenize_texts(all_qa, tokenizer, max_seq_len)
    print(f"Training sequences: {len(sequences)}")

    # Training params
    epochs = 30
    batch_size = 4
    grad_accum_steps = 4
    lr = 3e-5
    warmup_steps = 20
    min_lr_ratio = 0.1

    steps_per_epoch = len(sequences) // batch_size
    total_steps = (steps_per_epoch * epochs) // grad_accum_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
            input_ids = []
            labels = []
            for s in batch_seqs:
                ids = s[:max_len]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                labels.append(ids + [-100] * pad_len)

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids_t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / grad_accum_steps
            loss.backward()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

            if n_batches % grad_accum_steps == 0:
                if global_step < warmup_steps:
                    cur_lr = lr * global_step / max(warmup_steps, 1)
                else:
                    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    cur_lr = lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if n_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    model.eval()
    from datetime import datetime, timezone
    prev_log = checkpoint.get("training_log", [])
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": prev_log + [{"epoch": len(prev_log) + 1, "loss": best_loss, "type": "chatgpt_qa"}],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(checkpoint.get("datasets", []) + ["chatgpt_qa_custom"])),
        "qa_training": True,
    }
    torch.save(new_checkpoint, CKPT_PATH)
    print(f"\nCheckpoint saved! Best loss: {best_loss:.4f}")

    # Test inference
    print("\n=== テスト推論 ===")
    from api import generate_text
    # Reload for inference
    model.eval()
    import api
    api.model = model
    api.tokenizer = tokenizer
    api.config = cfg
    api.device = device

    test_prompts = [
        "質問: ChatGPTについて教えて\n回答:",
        "質問: ChatGPTとは何ですか？\n回答:",
        "質問: 大規模言語モデルとは何ですか？\n回答:",
        "質問: 生成AIとは何ですか？\n回答:",
    ]

    for prompt in test_prompts:
        result = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)
        q = prompt.split("\n")[0]
        print(f"\n{q}")
        print(f"回答: {result}")


if __name__ == "__main__":
    main()
