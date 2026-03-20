#!/usr/bin/env python3
"""
超軽量QA学習: 手作りQAのみで弱点を集中強化。CPU環境で数分で完了。
"""
import os
import sys
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

EPOCHS = 15
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
WARMUP_STEPS = 10
GRAD_CLIP = 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])

    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

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
    model.load_state_dict(checkpoint["model_state"])
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Crafted QA data - focused on weak areas
    crafted = [
        # === 安定している問題（維持） ===
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さは？\n回答: 富士山の高さは3,776メートルです。日本一高い山です。",
        "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。水素原子2つと酸素原子1つからなります。",
        "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートルです。宇宙で最速です。",
        "質問: 三角形の内角の和は？\n回答: 三角形の内角の和は180度です。",
        "質問: 量子コンピュータとは？\n回答: 量子コンピュータは量子力学の原理で計算を行うコンピュータです。量子ビットを使い従来困難な問題を解けます。",
        "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は人間の知能を模倣するコンピュータシステムです。機械学習や深層学習などの技術を含みます。",
        # === 弱点強化: 地球の年齢 ===
        "質問: 地球の年齢は？\n回答: 地球の年齢は約46億年です。",
        "質問: 地球の年齢はどれくらい？\n回答: 地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。",
        "質問: 地球は何歳ですか？\n回答: 地球は約46億歳です。",
        "質問: 地球はいつ誕生しましたか？\n回答: 地球は約46億年前に誕生しました。",
        # === 弱点強化: DNA ===
        "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、生物の遺伝情報を保持する分子です。二重らせん構造を持ちます。",
        "質問: DNAとは？\n回答: DNAは生物の遺伝情報を担う分子です。二重らせん構造を持ち、4つの塩基で構成されています。",
        "質問: DNAの構造は？\n回答: DNAは二重らせん構造を持つ分子です。アデニン、チミン、グアニン、シトシンの4つの塩基から成ります。",
        "質問: 遺伝子とは？\n回答: 遺伝子はDNA上にある遺伝情報の単位で、タンパク質の設計図となります。",
        # === 弱点強化: 人間の臓器 ===
        "質問: 人間の体で一番大きい臓器は？\n回答: 人間の体で一番大きい臓器は皮膚です。体全体を覆い、外部から身体を保護しています。",
        "質問: 人間の体で最も大きい臓器は何ですか？\n回答: 皮膚が人間の体で最も大きい臓器です。面積は約1.6〜1.8平方メートルあります。",
        "質問: 人体で一番大きな臓器は？\n回答: 皮膚です。人体で最も大きい臓器で、体重の約16%を占めます。",
        # === その他の知識 ===
        "質問: 円周率とは？\n回答: 円周率（π）は円の直径に対する円周の比率で、約3.14159です。無限に続く無理数です。",
        "質問: 東京タワーの高さは？\n回答: 東京タワーの高さは333メートルです。1958年に完成しました。",
        "質問: 1+1は？\n回答: 1+1は2です。",
        "質問: 日本で一番長い川は？\n回答: 日本で一番長い川は信濃川で、全長367キロメートルです。",
        "質問: プログラミングとは？\n回答: プログラミングとはコンピュータに命令を書くことです。Python、Java、C++などの言語を使います。",
        "質問: インターネットとは？\n回答: インターネットは世界中のコンピュータを接続した通信網です。",
        "質問: 太陽系の惑星は？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
        "質問: 相対性理論とは？\n回答: アインシュタインが提唱した物理学の理論で、時間と空間が相対的であることを示しました。",
        "質問: 織田信長はどんな人物ですか？\n回答: 織田信長は戦国時代の武将で、天下統一を目指しました。1582年に本能寺の変で討たれました。",
        "質問: Pythonとは？\n回答: Pythonは読みやすく書きやすい汎用プログラミング言語です。AI開発やWeb開発で広く使われています。",
        "質問: 機械学習とは？\n回答: 機械学習はデータからパターンを学習し予測を行うAIの一分野です。",
        "質問: 北海道の特徴は？\n回答: 北海道は日本最北の島で、面積は日本最大です。農業、酪農、漁業が盛んです。",
        "質問: 水は何度で沸騰しますか？\n回答: 水は1気圧のもとで100度で沸騰します。",
        "質問: こんにちは\n回答: こんにちは！何かお手伝いできることはありますか？",
    ]

    # Build training data with emphasis on weak areas
    qa_texts = []
    for _ in range(20):
        qa_texts.extend(crafted)
    print(f"Training QA pairs: {len(qa_texts)}")

    # Tokenize
    sequences = []
    for t in qa_texts:
        ids = tokenizer.encode(t, add_special=True)
        if 4 <= len(ids) <= max_seq_len:
            sequences.append(ids)
    print(f"Training sequences: {len(sequences)}")

    # Training
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps/epoch: {steps_per_epoch}, Total opt steps: {total_steps}, Epochs: {EPOCHS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
                if global_step < WARMUP_STEPS:
                    lr_now = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    lr_now = LR * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_now
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f} | Step: {global_step}")
        training_log.append({"epoch": len(prev_log) + epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            if (epoch + 1) % 10 == 0:
                print(f"  Best loss: {best_loss:.4f}, saving...")
                save_ckpt(model, config, prev_log + training_log, checkpoint)

    save_ckpt(model, config, prev_log + training_log, checkpoint)
    print(f"Final best loss: {best_loss:.4f}")

    # Inference test
    print("\n=== QA Inference test ===")
    model.eval()
    test_prompts = [
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: 地球の年齢はどれくらい？\n回答:",
        "質問: DNAとは何ですか？\n回答:",
        "質問: 人間の体で一番大きい臓器は？\n回答:",
        "質問: 量子コンピュータとは？\n回答:",
        "質問: 三角形の内角の和は？\n回答:",
        "質問: 富士山の高さは？\n回答:",
        "質問: 水の化学式は？\n回答:",
        "質問: 円周率とは？\n回答:",
        "質問: 光の速さは？\n回答:",
    ]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(80):
                seq = input_tensor[:, -max_seq_len:]
                logits = model(seq)[:, -1, :] / 0.5
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id == tokenizer.eos_id:
                    break
                if nxt_id == tokenizer.pad_id:
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                if "質問:" in partial:
                    break
        text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        if "質問:" in text:
            text = text[:text.index("質問:")].strip()
        print(f'  {prompt.strip()} {text}')

    print("\nDone!")


def save_ckpt(model, config, training_log, original_ckpt):
    all_ds = list(set(original_ckpt.get("datasets", [])))
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_ds,
        "qa_training": True,
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"  Saved: {os.path.getsize(CKPT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
