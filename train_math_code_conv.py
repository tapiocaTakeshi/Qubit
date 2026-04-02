#!/usr/bin/env python3
"""
数学・コーディング・会話の各ドメインのHuggingFaceデータセットでファインチューニング。
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from datetime import datetime, timezone
import json
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# Training hyperparameters
EPOCHS = 10
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

# Math, Coding, Conversation datasets
MATH_CODE_CONV_DATASETS = [
    # Math
    {
        "id": "openai/gsm8k",
        "max_samples": 3000,
        "format": "gsm8k",  # question, answer
    },
    # Coding
    {
        "id": "sahil2801/CodeAlpaca-20k",
        "max_samples": 5000,
        "format": "alpaca",  # instruction, input, output
    },
    {
        "id": "iamtarun/code_instructions_120k_alpaca",
        "max_samples": 3000,
        "format": "alpaca",
    },
    # Conversation
    {
        "id": "kunishou/databricks-dolly-15k-ja",
        "max_samples": 5000,
        "format": "dolly",  # instruction, context, response
    },
    {
        "id": "kunishou/oasst1-chat-44k-ja",
        "max_samples": 5000,
        "format": "conversations",
    },
    {
        "id": "shi3z/Japanese_wikipedia_conversation_100K",
        "max_samples": 3000,
        "format": "conversations",
    },
]


def format_qa_gsm8k(row):
    """Format GSM8K math data as QA."""
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()
    if not question or not answer:
        return None
    return f"質問: {question}\n回答: {answer}"


def format_qa_alpaca(row):
    """Format alpaca-style data as QA."""
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    if inp:
        q = f"{instruction}\n{inp}"
    else:
        q = instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_dolly(row):
    """Format dolly-style data as QA."""
    instruction = row.get("instruction", "").strip()
    context = row.get("context", "").strip()
    response = row.get("response", "").strip()
    if not instruction or not response:
        return None
    if context:
        q = f"{instruction}\n{context}"
    else:
        q = instruction
    return f"質問: {q}\n回答: {response}"


def format_qa_conversations(row):
    """Format conversation-style data as QA pairs."""
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn = convs[i]
        next_turn = convs[i + 1]
        # Extract text from turn
        q_text = ""
        a_text = ""
        if isinstance(turn, dict):
            q_text = turn.get("value", turn.get("content", "")).strip()
        elif isinstance(turn, str):
            q_text = turn.strip()
        if isinstance(next_turn, dict):
            a_text = next_turn.get("value", next_turn.get("content", "")).strip()
        elif isinstance(next_turn, str):
            a_text = next_turn.strip()
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    if pairs:
        return "\n\n".join(pairs)
    return None


def load_qa_data():
    """Load and format all math, coding, and conversation datasets."""
    all_qa = []

    for ds_info in MATH_CODE_CONV_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        max_samples = ds_info["max_samples"]
        print(f"  Loading {ds_id}...")

        try:
            ds = safe_load_dataset(ds_id, split="train")
            n = min(max_samples, len(ds))
            count = 0

            for row in ds.select(range(n)):
                if fmt == "gsm8k":
                    text = format_qa_gsm8k(row)
                elif fmt == "alpaca":
                    text = format_qa_alpaca(row)
                elif fmt == "dolly":
                    text = format_qa_dolly(row)
                elif fmt == "conversations":
                    text = format_qa_conversations(row)
                else:
                    continue

                if text and len(text) > 10:
                    all_qa.append(text)
                    count += 1

            print(f"    -> {count} QA samples")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    # Add hand-crafted QA examples for pattern reinforcement
    crafted_qa = [
        # Math
        "質問: 123 + 456 はいくつですか？\n回答: 123 + 456 = 579 です。",
        "質問: 15 × 24 はいくつですか？\n回答: 15 × 24 = 360 です。15 × 20 = 300、15 × 4 = 60、合わせて360です。",
        "質問: 1/3 + 1/4 はいくつですか？\n回答: 1/3 + 1/4 = 4/12 + 3/12 = 7/12 です。",
        "質問: 200の30%はいくつですか？\n回答: 200 × 0.3 = 60 です。",
        "質問: 2x + 5 = 13 のとき、xはいくつですか？\n回答: 2x + 5 = 13 より、2x = 8、x = 4 です。",
        "質問: 半径5cmの円の面積は？\n回答: 円の面積 = π × r² = π × 25 ≈ 78.54 cm²です。",
        "質問: データ {2, 4, 6, 8, 10} の平均値は？\n回答: 平均値 = (2+4+6+8+10) / 5 = 30 / 5 = 6 です。",
        "質問: x² - 5x + 6 = 0 の解は？\n回答: 因数分解すると (x-2)(x-3) = 0 なので、x = 2 または x = 3 です。",
        # Coding
        "質問: Pythonでリストの要素を逆順にするには？\n回答: list.reverse() メソッドまたは list[::-1] スライスを使います。例: my_list = [1, 2, 3]; my_list.reverse() で [3, 2, 1] になります。",
        "質問: Pythonで辞書のキーと値を入れ替えるには？\n回答: 辞書内包表記を使います。{v: k for k, v in original_dict.items()} で入れ替えられます。",
        "質問: JavaScriptで配列の重複を除去するには？\n回答: Set を使います。const unique = [...new Set(array)]; で重複を除去できます。",
        "質問: Pythonでファイルを読み込むには？\n回答: with open('filename.txt', 'r', encoding='utf-8') as f: content = f.read() のように書きます。with文を使うとファイルが自動的に閉じられます。",
        "質問: SQLでテーブルからデータを取得するには？\n回答: SELECT文を使います。SELECT * FROM table_name WHERE condition; で条件に合うデータを取得できます。",
        "質問: Gitでブランチを作成するには？\n回答: git branch ブランチ名 で作成し、git checkout ブランチ名 で切り替えます。git checkout -b ブランチ名 で作成と切り替えを同時にできます。",
        "質問: Pythonのリスト内包表記とは？\n回答: リスト内包表記は、リストを簡潔に作成する構文です。例: squares = [x**2 for x in range(10)] で0から9の二乗のリストを作成します。条件付き: evens = [x for x in range(20) if x % 2 == 0]",
        "質問: HTTPステータスコード200, 404, 500の意味は？\n回答: 200はリクエスト成功（OK）、404はリソースが見つからない（Not Found）、500はサーバー内部エラー（Internal Server Error）です。",
        # Conversation
        "質問: 今日の天気はどうですか？\n回答: 申し訳ありませんが、リアルタイムの天気情報にはアクセスできません。天気予報サイトやアプリをご確認ください。",
        "質問: おすすめの本はありますか？\n回答: ジャンルによりますが、日本文学なら夏目漱石の「坊っちゃん」、SFなら星新一のショートショート集がおすすめです。何か特定のジャンルはありますか？",
        "質問: 料理が上手になるコツは？\n回答: まずはレシピ通りに作ることを心がけましょう。基本的な調理技術（切る、焼く、煮る）を練習し、味見を頻繁にすることが大切です。",
        "質問: ストレス解消法を教えてください。\n回答: 効果的なストレス解消法には、適度な運動、十分な睡眠、深呼吸やメディテーション、趣味の時間を持つことなどがあります。自分に合った方法を見つけることが大切です。",
        "質問: 英語を効率的に学ぶ方法は？\n回答: 毎日少しずつ継続することが重要です。リスニング、スピーキング、リーディング、ライティングの4技能をバランスよく練習しましょう。映画や音楽など楽しめる教材を活用するのも効果的です。",
        "質問: 自己紹介をしてください。\n回答: 私はneuroQという日本語AIアシスタントです。量子ビットニューラルネットワーク（QBNN）という独自のアーキテクチャで構築されています。質問への回答や会話のお手伝いができます。",
        "質問: こんにちは\n回答: こんにちは！何かお手伝いできることはありますか？質問があればお気軽にどうぞ。",
        "質問: ありがとう\n回答: どういたしまして！他にも質問があればいつでもお聞きください。",
    ]
    # Repeat crafted examples to reinforce QA pattern
    for _ in range(30):
        all_qa.extend(crafted_qa)
    print(f"  + {len(crafted_qa) * 30} crafted QA samples")

    return all_qa


def tokenize_texts(texts, tokenizer, max_seq_len):
    """Tokenize texts into training sequences with BOF/EOF boundary markers."""
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True, add_boundary=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            chunks = list(range(0, len(ids) - max_seq_len + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = ids[start:start + max_seq_len]
                # 先頭チャンク以外はBOFを除去、末尾チャンク以外はEOFを除去
                if idx > 0 and chunk[0] == tokenizer.bof_id:
                    chunk = chunk[1:]
                if idx < len(chunks) - 1 and chunk[-1] == tokenizer.eof_id:
                    chunk = chunk[:-1]
                sequences.append(chunk)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, "
          f"vocab={config['vocab_size']}")
    if prev_log:
        print(f"Previous training: {len(prev_log)} epochs, last loss: {prev_log[-1]['loss']:.4f}")

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

    # Build model
    max_seq_len = config["max_seq_len"]
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Load QA data
    print("\n=== Loading Math/Code/Conv datasets ===")
    qa_texts = load_qa_data()
    print(f"\nTotal QA texts: {len(qa_texts)}")

    # Tokenize
    print("\n=== Tokenizing ===")
    sequences = tokenize_texts(qa_texts, tokenizer, max_seq_len)
    print(f"Training sequences: {len(sequences)}")

    # Calculate steps
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\n=== Training for {EPOCHS} epochs (Math/Code/Conv format) ===")
    model.train()
    training_log = []
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
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

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, nq_config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

            if n_batches % GRAD_ACCUM_STEPS == 0:
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 200 == 0:
                avg = total_loss / n_batches
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {n_batches} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e} | Step: {global_step}")

        # Handle remaining gradients
        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": len(prev_log) + epoch + 1, "loss": avg_loss})

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
            save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # Final save
    save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # Inference test
    print("\n=== Math/Code/Conv Inference test ===")
    model.eval()
    test_qa = [
        "質問: 25 × 16 はいくつですか？\n回答:",
        "質問: x² + 3x - 10 = 0 の解は？\n回答:",
        "質問: Pythonでリストをソートするには？\n回答:",
        "質問: HTMLとCSSの違いは？\n回答:",
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: おすすめの趣味はありますか？\n回答:",
        "質問: 機械学習とは何ですか？\n回答:",
    ]
    for prompt in test_qa:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(100):
                seq = input_tensor[:, -max_seq_len:]
                logits = model(seq)[:, -1, :] / 0.7
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                    break
                if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                    continue
                # Stop if model starts new question
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                # Check for new 質問: marker
                partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                if "質問:" in partial:
                    break

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        # Trim if it started a new question
        if "質問:" in generated_text:
            generated_text = generated_text[:generated_text.index("質問:")].strip()
        print(f'  {prompt.strip()} {generated_text}')

    print("\nDone!")


def save_checkpoint(model, config, training_log, original_ckpt):
    """Save checkpoint."""
    math_code_conv_datasets = [d["id"] for d in MATH_CODE_CONV_DATASETS]
    all_datasets = list(set(
        original_ckpt.get("datasets", []) + math_code_conv_datasets
    ))
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_datasets,
        "math_code_conv_training": True,
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")
    sync_checkpoint_to_network_volume(CKPT_PATH)


if __name__ == "__main__":
    main()
