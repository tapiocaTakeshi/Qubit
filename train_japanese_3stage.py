#!/usr/bin/env python3
"""
megabyte_100mb 日本語3段階学習スクリプト（チャンク・ストリーミング方式）
1. Pre-training (事前学習)
2. SFT (Supervised Fine-Tuning)
3. Evaluation (評価)

データセット全体を一度にメモリに載せず、
「チャンク単位でダウンロード → トークン化 → 学習 → 破棄」を繰り返すことで
メモリ使用量を抑え、最初のチャンクの学習をすぐに開始できるようにしている。
"""
import sys
import os
import gc
import time
import threading
import queue
import torch
import torch.nn.functional as F
from typing import List, Optional
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def safe_save(state_dict, path: str, retries: int = 3, retry_delay: float = 5.0) -> bool:
    """
    ネットワークボリューム上での一時的なI/O障害でクラッシュしないよう、
    一時ファイルへ書き込んでからアトミックにrenameし、失敗時はリトライする。
    保存に失敗してもプロセスは継続する（例外を投げない）。
    """
    tmp_path = f"{path}.tmp{os.getpid()}"
    for attempt in range(1, retries + 1):
        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            print(f"         ⚠️  チェックポイント保存失敗 (試行 {attempt}/{retries}): {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(retry_delay)
    print(f"         ✗ チェックポイント保存を諦めました: {path}")
    return False

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

# ========================================
# Stage 1: Pre-training Datasets
# ========================================
PRETRAINING_DATASETS = [
    ("wikimedia/wikipedia", "20231101.ja", "Wikipedia 日本語 (2023/11/01)"),
    # uonlp/CulturaX はgatedデータセットでアクセス権限がなく、かつストリーム読み込みが
    # ハングを起こすことがあったため除外
    # ("uonlp/CulturaX", "ja", "CulturaX 日本語"),
    # 青空文庫クリーン版は globis-university/aozorabunko-clean のgzipストリーム読み込みが
    # ハングを起こすことがあったため除外
    # ("globis-university/aozorabunko-clean", None, "青空文庫クリーン版"),
]

# ========================================
# Stage 2: SFT Dataset
# ========================================
SFT_DATASET = ("kunishou/databricks-dolly-15k-ja", None, "Databricks Dolly 日本語 (15K)")

# ========================================
# Stage 3: Evaluation Datasets
# ========================================
EVALUATION_DATASETS = [
    ("shunk031/JGLUE", "JNLI", "JGLUE - JNLI (自然言語推論)"),
    ("shunk031/JGLUE", "JCommonsenseQA", "JGLUE - JCommonsenseQA (常識推論)"),
]

CHUNK_SIZE = 20000          # 一度に処理するテキスト件数
CHECKPOINT_EVERY_N_CHUNKS = 5  # 何チャンクごとにチェックポイントを保存するか
CHECKPOINT_DIR = "/workspace/checkpoints"  # ネットワークボリューム（ポッド再起動でも消えない）

EARLY_STOP_PATIENCE = 8     # 何チャンク連続で改善がなければ打ち切るか
EARLY_STOP_MIN_DELTA = 0.01  # 「改善」とみなす最小のLoss減少幅


def extract_text(sample: dict) -> Optional[str]:
    """サンプルから本文フィールドを抽出"""
    for key in ['text', 'content', 'passage', 'summary', 'document',
                'instruction', 'response', 'premise', 'hypothesis']:
        if key in sample and isinstance(sample[key], str):
            candidate = sample[key].strip()
            if len(candidate) > 20:
                return candidate[:1500]
    return None


def open_stream(dataset_id: str, split_name: str = None):
    """ストリーミングDatasetIteratorを開く"""
    from datasets import load_dataset

    try:
        if split_name:
            return load_dataset(dataset_id, split_name, streaming=True, split="train")
        else:
            return load_dataset(dataset_id, streaming=True, split="train")
    except Exception:
        return load_dataset(dataset_id, streaming=True, split="train")


def stream_text_chunks(dataset_id: str, split_name: str, chunk_size: int, max_samples: int = None):
    """
    データセットをストリーミングしながら chunk_size 件ごとにテキストのリストをyieldする
    ジェネレータ。全件を一度にメモリへ載せない。
    """
    ds = open_stream(dataset_id, split_name)

    chunk = []
    total = 0
    for sample in ds:
        text = extract_text(sample)
        if text:
            chunk.append(text)
            total += 1

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

        if max_samples is not None and total >= max_samples:
            break

    if chunk:
        yield chunk


def tokenize_texts(texts: List[str], tokenizer, max_seq_len: int) -> List[List[int]]:
    """テキストをトークン化"""
    sequences = []

    for text in texts:
        try:
            token_ids = tokenizer.encode(text, add_special=False)

            if len(token_ids) < 2:
                continue

            max_content = max_seq_len - 4
            if len(token_ids) > max_content:
                token_ids = token_ids[:max_content]

            seq = [tokenizer.bof_id, tokenizer.bos_id] + token_ids + \
                  [tokenizer.eos_id, tokenizer.eof_id]

            pad_len = max_seq_len - len(seq)
            if pad_len > 0:
                seq = seq + [tokenizer.pad_id] * pad_len
            else:
                seq = seq[:max_seq_len]

            sequences.append(seq)
        except Exception:
            continue

    return sequences


class _StopSentinel:
    """キューの終端を表すセンチネル"""
    pass


class _ProducerError:
    """プロデューサースレッドで発生した例外を運ぶラッパー"""
    def __init__(self, exc):
        self.exc = exc


def _produce_chunks(datasets: list, tokenizer, max_seq_len: int, chunk_size: int,
                     out_queue: "queue.Queue", max_samples_per_dataset: int = None,
                     stop_event: Optional[threading.Event] = None):
    """
    バックグラウンドスレッドで実行される関数。
    HTTPストリーミング接続を学習側の都合で止めないよう、
    ダウンロード→トークン化を絶え間なく行い、結果をキューに積み続ける。
    stop_event がセットされたら、次のチャンクに進む前に早期終了する
    （アーリーストッピング発動時に呼び出し側からセットされる）。
    """
    try:
        for dataset_id, split, desc in datasets:
            if stop_event is not None and stop_event.is_set():
                break
            out_queue.put(("desc", desc))
            try:
                for texts in stream_text_chunks(dataset_id, split, chunk_size, max_samples_per_dataset):
                    if stop_event is not None and stop_event.is_set():
                        break
                    sequences = tokenize_texts(texts, tokenizer, max_seq_len)
                    out_queue.put(("chunk", len(texts), sequences))
            except Exception as e:
                out_queue.put(("error", desc, str(e)))
                continue
    except Exception as e:
        out_queue.put(("fatal", _ProducerError(e)))
    finally:
        out_queue.put(("stop", None))


def merge_checkpoints_final(pattern: str, output_name: str):
    """チェックポイントをマージ（クラッシュ復旧用の_latest.ptと_best.ptは対象外、エポック単位のみ）"""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    # pattern末尾の"*"と後続の"*.pt"が連続すると pathlib が
    # "Invalid pattern: '**' can only be an entire path component" で例外を出すため、
    # 末尾の"*"を除去してから"*.pt"を付与する
    glob_pattern = pattern.rstrip("*") + "*.pt"
    checkpoint_files = sorted(
        f for f in checkpoint_dir.glob(glob_pattern)
        if "_latest" not in f.name and "_best" not in f.name
    )

    if len(checkpoint_files) < 2:
        print(f"      ℹ️  マージ対象が1個以下のため、スキップします")
        return

    print(f"\n      📂 {len(checkpoint_files)} 個のチェックポイントをマージ中...")

    averaged_state = defaultdict(lambda: None)

    for ckpt_path in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    if averaged_state[key] is None:
                        averaged_state[key] = value.float().clone()
                    else:
                        averaged_state[key] += value.float()
        except Exception as e:
            print(f"      ✗ Error: {e}")
            return

    for key in averaged_state:
        if averaged_state[key] is not None:
            averaged_state[key] = averaged_state[key] / len(checkpoint_files)

    output_path = checkpoint_dir / f"{output_name}_merged.pt"
    if safe_save(dict(averaged_state), str(output_path)):
        output_size_mb = output_path.stat().st_size / (1024**2)
        print(f"      ✅ マージ完了: {output_path.name} ({output_size_mb:.1f}MB)")


def train_on_sequences(model, sequences: List[List[int]], tokenizer, optimizer, device: str,
                        batch_size: int = 4, progress_every: int = 200) -> tuple:
    """1チャンク分のシーケンスで1パス学習する。(total_loss, batch_count) を返す"""
    if len(sequences) < 2:
        return 0.0, 0

    model.train()
    random.shuffle(sequences)

    total_loss = 0.0
    batch_count = 0
    skipped_batches = 0
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    t_start = time.time()

    for batch_idx in range(0, len(sequences), batch_size):
        batch = sequences[batch_idx:batch_idx + batch_size]

        if len(batch) < 2:
            continue

        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        labels = torch.tensor([s[1:] + [tokenizer.pad_id] for s in batch],
                               dtype=torch.long, device=device)

        optimizer.zero_grad()

        with torch.autocast(device, enabled=device == "cuda"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1))

        # NaN/Infガード: 異常なlossが出たバッチはモデルを汚染する前にスキップする
        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if batch_count % progress_every == 0:
            elapsed = time.time() - t_start
            sec_per_batch = elapsed / batch_count
            skip_info = f" | skip {skipped_batches}" if skipped_batches else ""
            print(f"           …batch {batch_count}/{total_batches} "
                  f"| Loss = {loss.item():.4f} | {sec_per_batch*1000:.0f}ms/batch "
                  f"| 経過 {elapsed:.1f}s{skip_info}", flush=True)

    if skipped_batches:
        print(f"           ⚠️  NaN/Inf検出により {skipped_batches} バッチをスキップしました", flush=True)

    return total_loss, batch_count


def stream_train_stage(model, tokenizer, optimizer, scheduler, datasets: list,
                        stage_name: str, epochs: int, device: str, config,
                        checkpoint_prefix: str = None, chunk_size: int = CHUNK_SIZE,
                        max_samples_per_dataset: int = None):
    """
    バックグラウンドの「プロデューサー」スレッドが絶え間なくダウンロード→トークン化を行い
    キューに結果を積み続ける。メインスレッド（GPU学習）はキューから取り出して学習するだけなので、
    学習中にHTTPストリーミング接続がアイドル状態になって固まる問題を回避できる。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # アーリーストッピング用の状態（エポックをまたいでステージ全体で追跡）
    best_loss = float('inf')
    chunks_without_improvement = 0
    early_stopped = False

    for epoch in range(epochs):
        if early_stopped:
            break

        print(f"\n       📍 {stage_name} Epoch {epoch + 1}/{epochs}")

        epoch_total_loss = 0.0
        epoch_batch_count = 0
        chunk_idx = 0
        total_seqs_trained = 0
        t_epoch_start = time.time()

        # プロデューサースレッドを起動（キューサイズ2 = 最大2チャンク先読み）
        chunk_queue: "queue.Queue" = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        producer = threading.Thread(
            target=_produce_chunks,
            args=(datasets, tokenizer, config.max_seq_len, chunk_size, chunk_queue,
                  max_samples_per_dataset, stop_event),
            daemon=True,
        )
        producer.start()

        while True:
            item = chunk_queue.get()
            tag = item[0]

            if tag == "stop":
                break
            elif tag == "desc":
                print(f"\n         📥 {item[1]} をストリーミング中...")
                continue
            elif tag == "error":
                print(f"         ✗ {item[1]} の読み込み中にエラー: {item[2][:80]}")
                continue
            elif tag == "fatal":
                print(f"         ✗ プロデューサースレッドで致命的エラー: {item[1].exc}")
                break

            _, n_texts, sequences = item

            if not sequences:
                continue

            t1 = time.time()
            loss_sum, batch_count = train_on_sequences(
                model, sequences, tokenizer, optimizer, device
            )
            t_train = time.time() - t1

            epoch_total_loss += loss_sum
            epoch_batch_count += batch_count
            total_seqs_trained += len(sequences)
            chunk_idx += 1

            avg_loss = loss_sum / max(batch_count, 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_epoch_start

            # アーリーストッピング判定
            if avg_loss < best_loss - EARLY_STOP_MIN_DELTA:
                best_loss = avg_loss
                chunks_without_improvement = 0
                if checkpoint_prefix:
                    best_ckpt_path = f"{CHECKPOINT_DIR}/{checkpoint_prefix}_best.pt"
                    safe_save(model.state_dict(), best_ckpt_path)
                improve_tag = " 🌟best更新"
            else:
                chunks_without_improvement += 1
                improve_tag = f" (改善なし {chunks_without_improvement}/{EARLY_STOP_PATIENCE})"

            print(f"         chunk {chunk_idx:4d}: {n_texts:6d} 件 → {len(sequences):6d} seqs "
                  f"| Loss = {avg_loss:.4f} | LR = {lr:.2e} "
                  f"| train {t_train:.1f}s "
                  f"| 累計 {total_seqs_trained} seqs / {elapsed:.0f}s経過{improve_tag}")

            # チャンクのデータを破棄してメモリを解放
            del sequences
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 定期チェックポイント保存（ファイルを増やさず、1つの「最新」ファイルに上書き）
            if checkpoint_prefix and chunk_idx % CHECKPOINT_EVERY_N_CHUNKS == 0:
                ckpt_path = f"{CHECKPOINT_DIR}/{checkpoint_prefix}_latest.pt"
                if safe_save(model.state_dict(), ckpt_path):
                    print(f"         💾 {ckpt_path} (chunk {chunk_idx} 時点で上書き保存)")

            if chunks_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n       🛑 アーリーストッピング発動: "
                      f"{EARLY_STOP_PATIENCE}チャンク連続でLoss改善なし（best={best_loss:.4f}）")
                stop_event.set()
                # プロデューサーがput()でブロックしていれば解除するためキューを空にする
                while True:
                    try:
                        drained = chunk_queue.get_nowait()
                        if drained[0] == "stop":
                            break
                    except queue.Empty:
                        break
                early_stopped = True
                break

        producer.join(timeout=10)

        if scheduler:
            scheduler.step()

        avg_epoch_loss = epoch_total_loss / max(epoch_batch_count, 1)
        print(f"\n       ✅ {stage_name} Epoch {epoch + 1} "
              f"{'（アーリーストップ）' if early_stopped else '完了'}: "
              f"Average Loss = {avg_epoch_loss:.4f} | 総シーケンス数 = {total_seqs_trained}")

        # エポック終了時チェックポイント保存
        if checkpoint_prefix:
            ckpt_path = f"{CHECKPOINT_DIR}/{checkpoint_prefix}_epoch{epoch+1}.pt"
            if safe_save(model.state_dict(), ckpt_path):
                print(f"       💾 {ckpt_path}")

    if early_stopped:
        print(f"\n       ℹ️  {stage_name}: アーリーストッピングによりステージを終了しました "
              f"(best_loss={best_loss:.4f})")

        if total_seqs_trained == 0:
            print(f"       ⚠️  このエポックで学習できたシーケンスがありません")


def main():
    print("\n" + "=" * 90)
    print("🚀 NeuroQuantum megabyte_100mb - 日本語3段階学習（チャンク・ストリーミング方式）")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # トークナイザー（本物のSentencePieceを学習・使用）
    print("\n🔤 Building SentencePiece tokenizer...")
    tokenizer_model_path = "neuroq_tokenizer.model"

    if os.path.exists(tokenizer_model_path):
        print(f"  既存のSentencePieceモデルを読み込み中: {tokenizer_model_path}")
        tokenizer = NeuroQuantumTokenizer(vocab_size=32000, model_file=tokenizer_model_path)
    else:
        print("  SentencePiece学習用コーパスを収集中（Wikipediaから最大50,000サンプル）...")
        vocab_texts = []
        for chunk in stream_text_chunks(
            PRETRAINING_DATASETS[0][0], PRETRAINING_DATASETS[0][1], 50000, max_samples=50000
        ):
            vocab_texts.extend(chunk)
        print(f"  ✓ {len(vocab_texts)} 件のテキストを収集")

        tokenizer = NeuroQuantumTokenizer(vocab_size=32000)
        tokenizer.build_vocab(vocab_texts, model_prefix="neuroq_tokenizer")
        del vocab_texts
        gc.collect()

    actual_vocab_size = len(tokenizer)
    print(f"  ✓ 実際の語彙サイズ: {actual_vocab_size}")

    # モデル設定（トークナイザーの実際の語彙サイズに合わせる）
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=actual_vocab_size)
    config = NeuroQuantumConfig(
        vocab_size=actual_vocab_size,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=6,
        max_seq_len=512,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    # モデル（embedding-gemma-300mを常時使用）
    print("\n🧠 Initializing model (embedding-gemma-300m 使用)...")
    model = NeuroQuantum(
        config=config,
        tokenizer=tokenizer,
        use_sentence_transformers=True,
        sentence_transformers_model="google/embeddinggemma-300m",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {n_params:,}")

    # ========================================
    # STAGE 1: Pre-training
    # ========================================
    print("\n" + "▓" * 90)
    print("📚 STAGE 1: Pre-training (事前学習) - チャンク・ストリーミング方式")
    print("▓" * 90)

    # 既にStage 1が完了・マージ済みならスキップして再開する
    # （ネットワークI/O障害などでクラッシュした場合に、長時間かかるStage 1を
    #   毎回やり直さずに済むようにするため）
    pretraining_merged_path = f"{CHECKPOINT_DIR}/megabyte_100mb_pretraining_merged.pt"
    if os.path.exists(pretraining_merged_path):
        print(f"\n  ℹ️  既存のStage 1成果物を検出、スキップして再開します: {pretraining_merged_path}")
        model.load_state_dict(torch.load(pretraining_merged_path, map_location=device))
    else:
        # embedding-gemmaは語彙埋め込みの事前計算・初期化のみに使用し、実行時は通常の
        # 学習可能Embeddingとして扱われるため、純粋なQBNN版と同じ学習率で問題ない
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        stream_train_stage(model, tokenizer, optimizer, scheduler, PRETRAINING_DATASETS,
                            "Pre-training", epochs=3, device=device, config=config,
                            checkpoint_prefix="megabyte_100mb_pretraining")

        merge_checkpoints_final("*pretraining*", "megabyte_100mb_pretraining")

    # ========================================
    # STAGE 2: SFT (Supervised Fine-Tuning)
    # ========================================
    print("\n" + "▓" * 90)
    print("📝 STAGE 2: SFT - Supervised Fine-Tuning (指示追従学習) - チャンク・ストリーミング方式")
    print("▓" * 90)

    sft_merged_path = f"{CHECKPOINT_DIR}/megabyte_100mb_sft_merged.pt"
    if os.path.exists(sft_merged_path):
        print(f"\n  ℹ️  既存のStage 2成果物を検出、スキップして再開します: {sft_merged_path}")
        model.load_state_dict(torch.load(sft_merged_path, map_location=device))
    else:
        # 学習率を下げる
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        stream_train_stage(model, tokenizer, optimizer, scheduler, [SFT_DATASET],
                            "SFT", epochs=2, device=device, config=config,
                            checkpoint_prefix="megabyte_100mb_sft", chunk_size=5000)

        merge_checkpoints_final("*sft*", "megabyte_100mb_sft")

    # ========================================
    # STAGE 3: Evaluation
    # ========================================
    print("\n" + "▓" * 90)
    print("📊 STAGE 3: Evaluation (評価)")
    print("▓" * 90)

    model.eval()

    print("\n  Loading evaluation datasets (少量なので一括読み込み)...")
    eval_sequences = []

    for dataset_id, split, desc in EVALUATION_DATASETS:
        print(f"    {desc}... ", end="", flush=True)
        try:
            texts = []
            for chunk in stream_text_chunks(dataset_id, split, 5000, max_samples=5000):
                texts.extend(chunk)

            if texts:
                sequences = tokenize_texts(texts, tokenizer, config.max_seq_len)
                eval_sequences.extend(sequences)
                print(f"✓ {len(sequences)} seqs")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    print(f"\n  ✓ Total evaluation samples: {len(eval_sequences)}")

    batch_count = 0
    if eval_sequences:
        print("\n  🔍 Running evaluation...")

        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx in range(0, min(100, len(eval_sequences)), 4):
                batch = eval_sequences[batch_idx:batch_idx + 4]

                if len(batch) < 2:
                    continue

                input_ids = torch.tensor(batch, dtype=torch.long, device=device)
                labels = torch.tensor([s[1:] + [tokenizer.pad_id] for s in batch],
                                      dtype=torch.long, device=device)

                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

                total_loss += loss.item()
                batch_count += 1

                if batch_idx % 20 == 0:
                    print(f"    Batch {batch_idx // 4:3d}: Loss = {loss.item():.4f}")

        avg_eval_loss = total_loss / max(batch_count, 1)
        print(f"\n  ✅ Evaluation Loss: {avg_eval_loss:.4f}")

    # ========================================
    # Final Merge
    # ========================================
    print("\n" + "=" * 90)
    print("🔀 Final Checkpoint Merge")
    print("=" * 90)

    merge_checkpoints_final("*pretraining*", "megabyte_100mb_final_pretraining")
    merge_checkpoints_final("*sft*", "megabyte_100mb_final_sft")

    print("\n" + "=" * 90)
    print("✅ 3段階学習完了！")
    print("=" * 90)
    print(f"\n📊 Summary:")
    print(f"  Model: megabyte_100mb")
    print(f"  Parameters: {n_params:,}")
    print(f"\n  Stage 1 - Pre-training: 3 epochs ✓")
    print(f"  Stage 2 - SFT: 2 epochs ✓")
    print(f"  Stage 3 - Evaluation: {batch_count} batches ✓")

    print(f"\n💾 Checkpoints:")
    ckpt_dir = Path(CHECKPOINT_DIR)
    merged_files = sorted(ckpt_dir.glob("*merged.pt"))
    for ckpt in merged_files:
        size_mb = ckpt.stat().st_size / (1024**2)
        print(f"  ✓ {ckpt.name} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
