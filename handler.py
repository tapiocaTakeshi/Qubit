"""
Custom Handler for HuggingFace Inference Endpoints & RunPod Serverless
neuroQ - NeuroQuantum Transformer

Supports both inference and training via the "action" field:
  - action: "inference" (default) — text generation
  - action: "train"               — fine-tuning on HF datasets
  - action: "train_qa"            — QA pairs training (qa_pairs format)
  - action: "train_qa_dataset"    — QA-format fine-tuning on HF datasets
  - action: "train_split"         — split dataset training (chunked)
  - action: "train_split_next"    — train next chunk only (timeout-safe)
  - action: "split_status"        — check split training progress
  - action: "split_reset"         — reset split training state
  - action: "status"              — model status & training info

Action resolution priority:
  1. data["action"]
  2. parameters["action"]
  3. inputs の __xxx__ 形式 (e.g. "__train__")
  4. デフォルト "inference"

RunPod Serverless:
  このファイルを直接実行すると RunPod サーバーレスハンドラーとして起動します。
    python handler.py
  RunPod の入力形式:
    {"input": {"prompt": "...", "action": "...", "parameters": {...}}}

HuggingFace Inference Endpoints:
  Reference: https://huggingface.co/docs/inference-endpoints/main/en/engines/toolkit#create-a-custom-inference-handler
"""

import os
import sys
import json
import math
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# Import NeuroQuantum architecture
# ============================================================
try:
    from neuroquantum_layered import (
        NeuroQuantum,
        NeuroQuantumConfig,
        NeuroQuantumTokenizer,
        migrate_legacy_state_dict,
    )
    NEUROQUANTUM_AVAILABLE = True
except ImportError:
    NEUROQUANTUM_AVAILABLE = False


# ============================================================
# Default model config
# ============================================================

DEFAULT_CONFIG = {
    "vocab_size": 8000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 10000,
    "entangle_strength": 0.5,
    "dropout": 0.1,
    "architecture": "neuroquantum",
}


# ============================================================
# Default datasets for training
# ============================================================

DEFAULT_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]

QA_DATASETS_INFO = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]

CRAFTED_QA = [
    # 一般知識
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さはどのくらいですか？\n回答: 富士山の高さは3,776メートルです。日本で最も高い山です。",
    "質問: 太陽系にはいくつの惑星がありますか？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
    "質問: 光の速さはどのくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。これは宇宙で最も速い速度です。",
    "質問: インターネットとは何ですか？\n回答: インターネットは、世界中のコンピュータネットワークを相互に接続した通信網です。Webサイトの閲覧、メール、動画配信などのサービスを支えています。",
    # AI・コンピュータサイエンス
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。",
    "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。量子ビット（キュービット）を使い、従来のコンピュータでは困難な問題を解くことができます。",
    "質問: 機械学習とは何ですか？\n回答: 機械学習は、データからパターンを学習し、予測や判断を行う人工知能の一分野です。教師あり学習、教師なし学習、強化学習などの手法があります。",
    "質問: ニューラルネットワークとは何ですか？\n回答: ニューラルネットワークは、人間の脳の神経回路を模倣した計算モデルです。入力層、隠れ層、出力層から構成され、重みを調整することで学習します。深層学習では多数の隠れ層を持つディープニューラルネットワークが使われます。",
    "質問: トランスフォーマーとは何ですか？\n回答: トランスフォーマーは、自己注意機構（Self-Attention）を使ったニューラルネットワークアーキテクチャです。2017年にGoogleが提案し、GPTやBERTなどの大規模言語モデルの基盤となっています。並列処理が可能で、長い文脈を効率的に扱えます。",
    # 数学
    "質問: 素数とは何ですか？\n回答: 素数とは、1とその数自身以外に正の約数を持たない、1より大きい自然数です。例えば2, 3, 5, 7, 11, 13, 17, 19, 23, 29などが素数です。2は唯一の偶数の素数です。素数は無限に存在することがユークリッドによって証明されています。",
    "質問: 微分とは何ですか？\n回答: 微分とは、関数の変化率を求める数学的操作です。関数f(x)の微分f'(x)は、xの微小な変化に対するf(x)の変化の割合を表します。例えば、f(x) = x²の微分はf'(x) = 2xです。物理学では速度（位置の微分）や加速度（速度の微分）の計算に使われます。",
    "質問: 積分とは何ですか？\n回答: 積分とは、微分の逆の操作で、関数の面積や累積量を求める数学的手法です。定積分∫[a,b]f(x)dxは、区間[a,b]でのf(x)とx軸の間の面積を表します。不定積分は微分の逆操作（原始関数）を求めます。例えば、∫2x dx = x² + Cです。",
    "質問: 行列とは何ですか？\n回答: 行列は、数や式を長方形に並べたものです。m行n列の行列をm×n行列と呼びます。行列の演算には加算、スカラー倍、行列積などがあります。機械学習ではデータの表現や線形変換に広く使われます。例えば、2×2行列 [[1,2],[3,4]] は2行2列の行列です。",
    "質問: ピタゴラスの定理とは何ですか？\n回答: ピタゴラスの定理は、直角三角形の3辺の関係を表す定理です。直角三角形の斜辺の長さをc、他の2辺の長さをa, bとすると、a² + b² = c²が成り立ちます。例えば、a=3, b=4のとき、c=√(9+16)=√25=5となります。",
    "質問: 確率とは何ですか？\n回答: 確率は、ある事象が起こる可能性の度合いを0から1の間の数値で表したものです。確率0は「絶対に起こらない」、確率1は「必ず起こる」を意味します。例えば、公正なコインを投げて表が出る確率は1/2=0.5です。サイコロで6が出る確率は1/6≈0.167です。",
    "質問: 対数とは何ですか？\n回答: 対数（logarithm）は、指数の逆演算です。a^x = bのとき、x = log_a(b)と書きます。例えば、2³=8なので、log₂(8)=3です。自然対数ln(x)は底がeの対数で、常用対数log₁₀(x)は底が10の対数です。情報理論やアルゴリズムの計算量の分析に使われます。",
    "質問: 1から100までの整数の合計はいくつですか？\n回答: 1から100までの整数の合計は5050です。ガウスの公式 n(n+1)/2 を使って計算できます。n=100の場合、100×101/2 = 5050となります。",
    "質問: フィボナッチ数列とは何ですか？\n回答: フィボナッチ数列は、前の2つの数を足して次の数を作る数列です。0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...と続きます。自然界の多くのパターン（花びらの数、貝殻の螺旋など）にフィボナッチ数が現れます。隣接する2項の比は黄金比φ≈1.618に収束します。",
    "質問: 線形代数とは何ですか？\n回答: 線形代数は、ベクトル空間と線形写像を扱う数学の分野です。ベクトル、行列、固有値、固有ベクトルなどの概念を含みます。機械学習、コンピュータグラフィックス、物理学の基礎となる重要な分野です。",
    # コーディング・プログラミング
    "質問: Pythonとは何ですか？\n回答: Pythonは、読みやすく書きやすい汎用プログラミング言語です。データ分析、AI開発、Web開発など幅広い分野で使われています。インデントでブロックを区切る構文が特徴的です。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させる命令を書くことです。Python、Java、C++などのプログラミング言語を使って、ソフトウェアやアプリケーションを作成します。",
    "質問: Pythonでリストをソートする方法を教えてください。\n回答: Pythonでリストをソートするには、sort()メソッドまたはsorted()関数を使います。sort()は元のリストを変更し、sorted()は新しいリストを返します。例: nums = [3,1,2]; nums.sort() で [1,2,3] になります。逆順にするには reverse=True を指定します。",
    "質問: Pythonで関数を定義する方法を教えてください。\n回答: Pythonでは def キーワードを使って関数を定義します。例:\ndef greet(name):\n    return f'こんにちは、{name}さん！'\nresult = greet('太郎')  # 'こんにちは、太郎さん！'\n引数にはデフォルト値を設定でき、*argsや**kwargsで可変長引数も扱えます。",
    "質問: Pythonのリスト内包表記とは何ですか？\n回答: リスト内包表記は、リストを簡潔に生成するPythonの構文です。例: squares = [x**2 for x in range(10)] は0から9の2乗のリスト [0,1,4,9,16,25,36,49,64,81] を生成します。条件も付けられます: evens = [x for x in range(20) if x % 2 == 0]",
    "質問: Pythonの辞書（dict）とは何ですか？\n回答: 辞書は、キーと値のペアを格納するデータ構造です。例: person = {'name': '太郎', 'age': 25, 'city': '東京'} のように作成します。person['name']で値を取得し、person['email'] = 'taro@example.com'で追加できます。keys(), values(), items()メソッドで要素を列挙できます。",
    "質問: Pythonのクラスとは何ですか？\n回答: クラスは、データ（属性）と操作（メソッド）をまとめたオブジェクトの設計図です。例:\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return f'{self.name}がワンと鳴いた'\ndog = Dog('ポチ')\nprint(dog.bark())  # ポチがワンと鳴いた\n継承やポリモーフィズムなどの概念をサポートします。",
    "質問: 再帰関数とは何ですか？\n回答: 再帰関数は、自分自身を呼び出す関数です。例えば、階乗を計算する再帰関数:\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\nfactorial(5) は 5×4×3×2×1 = 120 を返します。再帰には必ず終了条件（ベースケース）が必要です。",
    "質問: ソートアルゴリズムにはどのようなものがありますか？\n回答: 代表的なソートアルゴリズムには以下があります。バブルソート（O(n²)）: 隣接要素を比較・交換。クイックソート（平均O(n log n)）: ピボットで分割して再帰的にソート。マージソート（O(n log n)）: 分割統治法で安定ソート。ヒープソート（O(n log n)）: ヒープ構造を利用。",
    "質問: 二分探索とは何ですか？\n回答: 二分探索は、ソート済みの配列から効率的に値を探すアルゴリズムです。中央の要素と比較し、探す値が大きければ右半分、小さければ左半分を探索します。計算量はO(log n)で、線形探索O(n)より高速です。例: 1000要素の配列でも最大10回の比較で見つかります。",
    "質問: APIとは何ですか？\n回答: API（Application Programming Interface）は、ソフトウェア間の通信インターフェースです。RESTful APIではHTTPメソッド（GET, POST, PUT, DELETE）を使ってリソースを操作します。JSONフォーマットでデータをやり取りすることが一般的です。",
    "質問: Gitとは何ですか？\n回答: Gitは、ソースコードのバージョン管理システムです。変更履歴の追跡、ブランチによる並行開発、マージによる統合ができます。git init, git add, git commit, git push, git pullなどのコマンドで操作します。GitHubやGitLabなどのホスティングサービスと組み合わせて使います。",
    "質問: データベースとは何ですか？\n回答: データベースは、構造化されたデータを効率的に保存・管理するシステムです。リレーショナルDB（MySQL, PostgreSQL）ではSQLを使ってデータを操作します。NoSQL（MongoDB, Redis）は柔軟なスキーマで大量データを扱えます。CRUD操作（Create, Read, Update, Delete）が基本です。",
    "質問: 計算量のO記法とは何ですか？\n回答: O記法（ビッグオー記法）は、アルゴリズムの効率を表す表記法です。O(1)は定数時間、O(log n)は対数時間、O(n)は線形時間、O(n log n)は線形対数時間、O(n²)は二乗時間です。nはデータの大きさを表し、nが大きくなったときの最悪計算量を示します。",
]


# ============================================================
# Network Volume path (RunPod network volume for persistent storage)
# ============================================================
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")


# ============================================================
# Utility functions
# ============================================================

def find_checkpoint(path: str):
    """Search for a checkpoint file in the network volume (priority) and model directory."""
    # Prioritize network volume for persistent checkpoints across pod restarts
    search_dirs = []
    if os.path.isdir(NETWORK_VOLUME_PATH):
        search_dirs.append(NETWORK_VOLUME_PATH)
    search_dirs.append(path)

    for search_dir in search_dirs:
        candidates = [
            os.path.join(search_dir, "qbnn_checkpoint.pt"),
            os.path.join(search_dir, "neuroq_checkpoint.pt"),
            os.path.join(search_dir, "checkpoint.pt"),
            os.path.join(search_dir, "model.pt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        if os.path.isdir(search_dir):
            for fname in os.listdir(search_dir):
                if fname.endswith(".pt"):
                    return os.path.join(search_dir, fname)
    return None


def extract_texts(ds, text_column, max_samples):
    """Extract text from dataset."""
    texts = []
    n = min(max_samples, len(ds))
    for row in ds.select(range(n)):
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
        elif isinstance(col_data, list):
            parts = []
            for turn in col_data:
                if isinstance(turn, dict) and "value" in turn:
                    parts.append(turn["value"])
                elif isinstance(turn, dict) and "content" in turn:
                    parts.append(turn["content"])
                elif isinstance(turn, str):
                    parts.append(turn)
            combined = "\n".join(parts)
            if len(combined.strip()) > 4:
                texts.append(combined.strip())
    return texts


def tokenize_texts(texts, tok, max_seq_len):
    """Tokenize texts into training sequences with BOS/EOS and BOF/EOF markers."""
    sequences = []
    for t in texts:
        content_ids = tok.encode(t, add_special=False)
        max_content = max_seq_len - 2
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tok.bof_id, tok.bos_id] + content_ids + [tok.eos_id, tok.eof_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            chunks = list(range(0, len(content_ids) - max_content + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = content_ids[start:start + max_content]
                prefix = [tok.bof_id, tok.bos_id] if idx == 0 else [tok.bos_id]
                suffix = [tok.eos_id, tok.eof_id] if idx == len(chunks) - 1 else [tok.eos_id]
                seq = prefix + chunk + suffix
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tok.bos_id] + remaining + [tok.eos_id, tok.eof_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def format_qa_alpaca(row):
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    q = f"{instruction}\n{inp}" if inp else instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn, next_turn = convs[i], convs[i + 1]
        q_text = turn.get("value", turn.get("content", "")).strip() if isinstance(turn, dict) else (turn.strip() if isinstance(turn, str) else "")
        a_text = next_turn.get("value", next_turn.get("content", "")).strip() if isinstance(next_turn, dict) else (next_turn.strip() if isinstance(next_turn, str) else "")
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    return "\n\n".join(pairs) if pairs else None


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    return f"質問: {instruction}\n回答: {output}" if instruction else f"回答: {output}"


# ============================================================
# HuggingFace Inference Endpoints Handler
# ============================================================

class EndpointHandler:
    """
    Custom handler for HuggingFace Inference Endpoints.

    Supports:
      - action: "inference" (default) — text generation
      - action: "train"               — general dataset training
      - action: "train_qa"            — QA pairs training (qa_pairs format)
      - action: "train_qa_dataset"    — QA-format training on HF datasets
      - action: "train_split"         — split dataset training (chunked)
      - action: "train_split_next"    — train next chunk only (timeout-safe)
      - action: "split_status"        — check split training progress
      - action: "split_reset"         — reset split training state
      - action: "status"              — model status & training info
    """

    def __init__(self, path: str = ""):
        """Load the model, tokenizer, and config from the model directory."""
        self.model_path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_status = {"running": False, "log": [], "message": "idle"}
        self.split_state_path = os.path.join(path or ".", "split_training_state.json")

        # Find checkpoint
        self.ckpt_path = find_checkpoint(path) if path else None

        if self.ckpt_path and NEUROQUANTUM_AVAILABLE:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            self.config = checkpoint.get("config", dict(DEFAULT_CONFIG))

            # Load tokenizer (prioritize network volume for consistency with checkpoint)
            tokenizer_path = None
            if os.path.isdir(NETWORK_VOLUME_PATH):
                nv_tok = os.path.join(NETWORK_VOLUME_PATH, "neuroq_tokenizer.model")
                if os.path.isfile(nv_tok):
                    tokenizer_path = nv_tok
            if not tokenizer_path:
                tokenizer_path = os.path.join(path, "neuroq_tokenizer.model")
            self.tokenizer = NeuroQuantumTokenizer(
                vocab_size=self.config["vocab_size"],
                model_file=tokenizer_path if os.path.isfile(tokenizer_path) else None,
            )

            # Sync vocab_size: tokenizer's actual size takes priority
            tok_vocab = self.tokenizer.actual_vocab_size or self.tokenizer.vocab_size
            if tok_vocab and tok_vocab != self.config["vocab_size"]:
                print(f"[handler] vocab_size mismatch: config={self.config['vocab_size']}, tokenizer={tok_vocab}. Resizing to {tok_vocab}.")
                self.config["vocab_size"] = tok_vocab

            # Build model
            nq_config = NeuroQuantumConfig(
                vocab_size=self.config["vocab_size"],
                embed_dim=self.config["embed_dim"],
                hidden_dim=self.config.get("hidden_dim", self.config["embed_dim"] * 2),
                num_heads=self.config["num_heads"],
                num_layers=self.config["num_layers"],
                max_seq_len=self.config["max_seq_len"],
                dropout=self.config.get("dropout", 0.1),
                lambda_entangle=self.config.get("entangle_strength", 0.5),
            )
            self.model = NeuroQuantum(config=nq_config).to(self.device)
            migrated = migrate_legacy_state_dict(checkpoint["model_state"], self.model)

            # Handle vocab_size mismatch: partially load weights for resized layers
            model_state = self.model.state_dict()
            resized_keys = []
            for key in list(migrated.keys()):
                if key in model_state and migrated[key].shape != model_state[key].shape:
                    old_shape = migrated[key].shape
                    new_shape = model_state[key].shape
                    # Copy overlapping portion
                    new_tensor = model_state[key].clone()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                    src_slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                    new_tensor[slices] = migrated[key][src_slices]
                    migrated[key] = new_tensor
                    resized_keys.append(f"{key}: {list(old_shape)}->{list(new_shape)}")
            if resized_keys:
                print(f"[handler] Resized layers: {', '.join(resized_keys)}")

            self.model.load_state_dict(migrated)
            self.model.eval()

            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[handler] NeuroQuantum model loaded: {n_params:,} params on {self.device}")
        else:
            # Fallback: initialize fresh model
            self.config = dict(DEFAULT_CONFIG)
            if NEUROQUANTUM_AVAILABLE:
                # Try to find tokenizer model file
                tokenizer_path = None
                for candidate in [
                    os.path.join(path or ".", "neuroq_tokenizer.model"),
                    os.path.join(path or ".", "neuroq_tokenizer_8k.model"),
                    "/app/neuroq_tokenizer.model",
                    "/app/neuroq_tokenizer_8k.model",
                    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model"),
                    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_8k.model"),
                ]:
                    if os.path.isfile(candidate):
                        tokenizer_path = candidate
                        break
                self.tokenizer = NeuroQuantumTokenizer(
                    vocab_size=self.config["vocab_size"],
                    model_file=tokenizer_path,
                )
                print(f"[handler] Tokenizer: sp={self.tokenizer.sp is not None} path={tokenizer_path}")
                nq_config = NeuroQuantumConfig(
                    vocab_size=self.config["vocab_size"],
                    embed_dim=self.config["embed_dim"],
                    hidden_dim=self.config.get("hidden_dim", self.config["embed_dim"] * 2),
                    num_heads=self.config["num_heads"],
                    num_layers=self.config["num_layers"],
                    max_seq_len=self.config["max_seq_len"],
                    dropout=self.config.get("dropout", 0.1),
                    lambda_entangle=self.config.get("entangle_strength", 0.5),
                )
                self.model = NeuroQuantum(config=nq_config).to(self.device)
                self.model.eval()
                print("[handler] NeuroQuantum model initialized (no checkpoint)")
            else:
                raise RuntimeError("neuroquantum_layered.py not available")

    def _resolve_action(self, data: Dict[str, Any]) -> str:
        """Resolve action with priority: data["action"] > parameters["action"] > inputs __xxx__ > "inference"."""
        # 1. Top-level action field
        action = data.get("action")
        if action:
            return action

        # 2. parameters.action
        params = data.get("parameters", {})
        action = params.get("action") if isinstance(params, dict) else None
        if action:
            return action

        # 3. inputs の __xxx__ 形式 (e.g. "__train__", "__status__")
        inputs = data.get("inputs", "")
        if isinstance(inputs, str):
            stripped = inputs.strip()
            if stripped.startswith("__") and stripped.endswith("__") and len(stripped) > 4:
                return stripped[2:-2]

        # 4. デフォルト
        return "inference"

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Route request based on "action" field.

        Action resolution priority:
            1. data["action"]
            2. parameters["action"]
            3. inputs の __xxx__ 形式 (e.g. "__train__")
            4. デフォルト "inference"

        Supported actions:
            inference, train, train_qa, train_split, train_split_next,
            split_status, split_reset, status

        Returns:
            List of dicts with results.
        """
        action = self._resolve_action(data)

        # Action routing table
        _routes = {
            "train":            self._handle_train,
            "train_qa":         self._train_qa,
            "train_qa_dataset": self._handle_train_qa,
            "train_split":      self._handle_train_split,
            "train_split_next": self._handle_train_split_next,
        }
        _routes_no_data = {
            "split_status": self._handle_split_status,
            "split_reset":  self._handle_split_reset,
            "status":       self._handle_status,
        }

        if action in _routes:
            return _routes[action](data)
        elif action in _routes_no_data:
            return _routes_no_data[action]()
        else:
            return self._handle_inference(data)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    def _handle_inference(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text from prompt."""
        inputs = data.get("inputs", data.get("prompt", ""))
        if isinstance(inputs, list):
            inputs = inputs[0] if inputs else ""
        raw_input = str(inputs).strip()
        # Format as QA prompt matching training data format
        if not raw_input.startswith("質問:") and not raw_input.startswith("回答:"):
            raw_input = f"質問: {raw_input}\n回答:"

        params = data.get("parameters", {})
        temperature = float(params.get("temperature", 0.7))
        max_new_tokens = int(params.get("max_new_tokens", params.get("max_tokens", 100)))
        top_k = int(params.get("top_k", 40))
        top_p = float(params.get("top_p", 0.9))
        repetition_penalty = float(params.get("repetition_penalty", 1.3))
        ignore_eos = bool(params.get("ignore_eos", False))

        # Match training format: [BOF, BOS] + content + [EOS, EOF]
        # For inference, use [BOF, BOS] + content (no EOS, so model generates)
        content_ids = self.tokenizer.encode(raw_input, add_special=False)
        tokens = [self.tokenizer.bof_id, self.tokenizer.bos_id] + content_ids
        if len(content_ids) == 0:
            return [{"generated_text": ""}]

        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated = list(tokens)
        max_seq_len = self.config["max_seq_len"]

        # Debug: track first generated token IDs
        debug_tokens = []

        self.model.eval()
        with torch.no_grad():
            for step in range(max_new_tokens):
                seq = input_tensor[:, -max_seq_len:]
                logits = self.model(seq)[:, -1, :] / max(temperature, 1e-5)

                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    topk_vals = torch.topk(logits, k)[0]
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    to_remove = cumulative_probs > top_p
                    to_remove[:, 1:] = to_remove[:, :-1].clone()
                    to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[to_remove]
                    logits[0, indices_to_remove] = float('-inf')

                if repetition_penalty > 1.0 and len(generated) > 1:
                    for prev in set(generated[-50:]):
                        if prev < logits.size(-1):
                            if logits[0, prev] > 0:
                                logits[0, prev] /= repetition_penalty
                            else:
                                logits[0, prev] *= repetition_penalty

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()

                # Debug: record first 10 generated tokens
                if step < 10:
                    top5 = torch.topk(probs, 5)
                    top5_ids = top5.indices[0].tolist()
                    top5_probs = top5.values[0].tolist()
                    debug_tokens.append({
                        "step": step, "generated_id": nxt_id,
                        "top5_ids": top5_ids,
                        "top5_probs": [round(p, 4) for p in top5_probs],
                        "is_eos": nxt_id == self.tokenizer.eos_id,
                        "is_eof": nxt_id == self.tokenizer.eof_id,
                    })

                if not ignore_eos and nxt_id in (self.tokenizer.eos_id, self.tokenizer.eof_id):
                    break
                if nxt_id in (self.tokenizer.pad_id, self.tokenizer.bof_id):
                    # Still update input_tensor to avoid regenerating the same token
                    input_tensor = torch.cat([input_tensor, nxt], dim=1)
                    continue

                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)

        generated_text = self.tokenizer.decode(generated[len(tokens):], skip_special=True)
        return [{"generated_text": generated_text,
                 "debug": {
                     "input_len": len(tokens),
                     "content_ids_len": len(content_ids),
                     "first_5_content_ids": content_ids[:5],
                     "last_5_content_ids": content_ids[-5:],
                     "bof_id": self.tokenizer.bof_id,
                     "bos_id": self.tokenizer.bos_id,
                     "eos_id": self.tokenizer.eos_id,
                     "eof_id": self.tokenizer.eof_id,
                     "vocab_size": self.config.get("vocab_size"),
                     "has_sp": self.tokenizer.sp is not None,
                     "generated_token_count": len(generated) - len(tokens),
                     "debug_tokens": debug_tokens,
                 }}]

    # --------------------------------------------------------
    # Training (general)
    # --------------------------------------------------------

    def _handle_train(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run general dataset training."""
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        from dataset_utils import safe_load_dataset

        params = data.get("parameters", {})
        epochs = int(params.get("epochs", 10))
        lr = float(params.get("lr", 1e-4))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 8))
        warmup_steps = int(params.get("warmup_steps", 100))
        max_samples = int(params.get("max_samples_per_dataset", 5000))
        dataset_ids = params.get("dataset_ids", None)

        self.training_status = {"running": True, "log": [], "message": "Loading datasets..."}

        mode = params.get("mode", "general")
        crafted_repeat = int(params.get("crafted_repeat", 20))

        try:
            all_texts = []
            if dataset_ids:
                # Use _load_custom_datasets for proper format detection & colon parsing
                all_texts = self._load_custom_datasets(dataset_ids, max_samples, mode)
                if mode == "qa" and crafted_repeat > 0:
                    for _ in range(crafted_repeat):
                        all_texts.extend(CRAFTED_QA)
                    self.training_status["log"].append(
                        f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
                    )
            else:
                if mode == "qa":
                    all_texts = self._load_all_qa_texts(max_samples)
                    for _ in range(crafted_repeat):
                        all_texts.extend(CRAFTED_QA)
                    self.training_status["log"].append(
                        f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
                    )
                else:
                    datasets_to_use = DEFAULT_DATASETS
                    for ds_info in datasets_to_use:
                        try:
                            self.training_status["message"] = f"Loading {ds_info['id']}..."
                            ds = safe_load_dataset(ds_info["id"], split="train")
                            texts = extract_texts(ds, ds_info["col"], max_samples)
                            all_texts.extend(texts)
                            self.training_status["log"].append(f"Loaded {ds_info['id']}: {len(texts)} texts")
                        except Exception as e:
                            self.training_status["log"].append(f"Error loading {ds_info['id']}: {e}")

            # Also load cc100-ja for general mode without custom datasets
            if not dataset_ids and mode != "qa":
                try:
                    self.training_status["message"] = "Loading cc100-ja..."
                    ds_cc = safe_load_dataset("range3/cc100-ja", split="train", streaming=True)
                    cc_texts = []
                    for i, row in enumerate(ds_cc):
                        if i >= max_samples:
                            break
                        text = row.get("text", "").strip()
                        if len(text) > 10:
                            cc_texts.append(text)
                    all_texts.extend(cc_texts)
                    self.training_status["log"].append(f"Loaded cc100-ja: {len(cc_texts)} texts")
                except Exception as e:
                    self.training_status["log"].append(f"Error loading cc100-ja: {e}")

            if not all_texts:
                self.training_status["running"] = False
                self.training_status["message"] = "No training data loaded"
                return [{"status": "error", "message": "No training data loaded",
                         "log": self.training_status["log"]}]

            self.training_status["message"] = "Tokenizing..."
            max_seq_len = self.config["max_seq_len"]
            sequences = tokenize_texts(all_texts, self.tokenizer, max_seq_len)
            self.training_status["log"].append(
                f"Total: {len(all_texts)} texts -> {len(sequences)} sequences"
            )

            self._run_training_loop(
                sequences, epochs, lr, batch_size, grad_accum_steps,
                warmup_steps, max_seq_len,
            )

            # Save checkpoint
            ds_names = dataset_ids if dataset_ids else [d["id"] for d in DEFAULT_DATASETS] + ["range3/cc100-ja"]
            self._save_checkpoint(datasets=ds_names)

            return [{"status": "success", "message": self.training_status["message"],
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Training (QA format)
    # --------------------------------------------------------

    def _handle_train_qa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run QA-format training."""
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        from dataset_utils import safe_load_dataset

        params = data.get("parameters", {})
        epochs = int(params.get("epochs", 20))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 30))
        max_samples = int(params.get("max_samples_per_dataset", 1500))
        dataset_id = params.get("dataset_id", None)
        crafted_repeat = int(params.get("crafted_repeat", 40))

        self.training_status = {"running": True, "log": [], "message": "Loading QA datasets..."}

        try:
            all_qa = []

            if dataset_id:
                # Custom single dataset
                try:
                    self.training_status["message"] = f"Loading {dataset_id}..."
                    ds = safe_load_dataset(dataset_id, split="train", streaming=True)
                    count = 0
                    for row in ds:
                        if count >= max_samples:
                            break
                        q = row.get("question", row.get("instruction", "")).strip()
                        a = row.get("answer", row.get("output", row.get("response", ""))).strip()
                        if q and a and len(q) > 2 and len(a) > 2:
                            all_qa.append(f"質問: {q}\n回答: {a}")
                            count += 1
                    self.training_status["log"].append(f"Loaded {dataset_id}: {count} QA samples")
                except Exception as e:
                    self.training_status["log"].append(f"Error loading {dataset_id}: {e}")
            else:
                # Default QA datasets
                for ds_info in QA_DATASETS_INFO:
                    ds_id = ds_info["id"]
                    fmt = ds_info["format"]
                    ms = min(1000, max_samples) if fmt == "izumi" else max_samples
                    try:
                        self.training_status["message"] = f"Loading {ds_id}..."
                        ds = safe_load_dataset(ds_id, split="train")
                        n = min(ms, len(ds))
                        count = 0
                        for row in ds.select(range(n)):
                            if fmt == "alpaca":
                                text = format_qa_alpaca(row)
                            elif fmt == "conversations":
                                text = format_qa_conversations(row)
                            elif fmt == "izumi":
                                text = format_qa_izumi(row)
                            else:
                                continue
                            if text and len(text) > 10:
                                all_qa.append(text)
                                count += 1
                        self.training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
                    except Exception as e:
                        self.training_status["log"].append(f"Error loading {ds_id}: {e}")

            # Add crafted QA
            for _ in range(crafted_repeat):
                all_qa.extend(CRAFTED_QA)
            self.training_status["log"].append(
                f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
            )
            self.training_status["log"].append(f"Total QA texts: {len(all_qa)}")

            if not all_qa:
                self.training_status["running"] = False
                self.training_status["message"] = "No QA data loaded"
                return [{"status": "error", "message": "No QA data loaded",
                         "log": self.training_status["log"]}]

            self.training_status["message"] = "Tokenizing..."
            max_seq_len = self.config["max_seq_len"]
            sequences = tokenize_texts(all_qa, self.tokenizer, max_seq_len)
            self.training_status["log"].append(f"Training sequences: {len(sequences)}")

            self._run_training_loop(
                sequences, epochs, lr, batch_size, grad_accum_steps,
                warmup_steps, max_seq_len,
            )

            # Save checkpoint
            extra_ds = [dataset_id] if dataset_id else []
            self._save_checkpoint(
                datasets=[d["id"] for d in QA_DATASETS_INFO] + extra_ds,
                extra_meta={"qa_training": True},
            )

            return [{"status": "success", "message": self.training_status["message"],
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Training (QA pairs — lightweight API)
    # --------------------------------------------------------

    def _train_qa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Train on user-supplied QA pairs.

        Expects parameters.qa_pairs: [{"question": "...", "answer": "..."}, ...]
        Optional parameters.repeat (default 3): duplicate data for better learning on small sets.
        """
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        params = data.get("parameters", {})
        qa_pairs = params.get("qa_pairs", [])
        if not qa_pairs:
            return [{"status": "error", "message": "parameters.qa_pairs is required (list of {question, answer})"}]

        repeat = int(params.get("repeat", 3))
        epochs = int(params.get("epochs", 20))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 10))

        self.training_status = {"running": True, "log": [], "message": "Preparing QA pairs..."}

        try:
            # Format QA pairs into training texts
            texts = []
            for pair in qa_pairs:
                q = pair.get("question", "").strip()
                a = pair.get("answer", "").strip()
                if q and a:
                    texts.append(f"質問: {q}\n回答: {a}")

            if not texts:
                self.training_status["running"] = False
                return [{"status": "error", "message": "No valid QA pairs found"}]

            # Repeat for better learning on small datasets
            texts = texts * repeat
            self.training_status["log"].append(
                f"QA pairs: {len(qa_pairs)} original x {repeat} repeat = {len(texts)} texts"
            )

            self._train_from_texts(
                texts, epochs=epochs, lr=lr, batch_size=batch_size,
                grad_accum_steps=grad_accum_steps, warmup_steps=warmup_steps,
            )

            self._save_checkpoint(
                datasets=["user_qa_pairs"],
                extra_meta={"qa_training": True, "qa_pairs_count": len(qa_pairs)},
            )

            return [{"status": "success", "message": self.training_status["message"],
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    def _train_from_texts(self, texts, epochs=20, lr=3e-5, batch_size=4,
                          grad_accum_steps=4, warmup_steps=10):
        """Tokenize texts and run the shared training loop."""
        self.training_status["message"] = "Tokenizing..."
        max_seq_len = self.config["max_seq_len"]
        sequences = tokenize_texts(texts, self.tokenizer, max_seq_len)
        self.training_status["log"].append(f"Training sequences: {len(sequences)}")

        self._run_training_loop(
            sequences, epochs, lr, batch_size, grad_accum_steps,
            warmup_steps, max_seq_len,
        )

    # --------------------------------------------------------
    # Training loop (shared)
    # --------------------------------------------------------

    def _run_training_loop(self, sequences, epochs, lr, batch_size,
                           grad_accum_steps, warmup_steps, max_seq_len):
        """Core training loop used by both general and QA training."""
        steps_per_epoch = len(sequences) // batch_size
        total_steps = (steps_per_epoch * epochs) // grad_accum_steps
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            self.training_status["message"] = f"Training epoch {epoch+1}/{epochs}..."

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
                    input_ids.append(ids + [self.tokenizer.pad_id] * pad_len)
                    labels.append(ids + [-100] * pad_len)

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

                logits = self.model(input_ids_t)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_t[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config["vocab_size"]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / grad_accum_steps
                loss.backward()

                total_loss += loss.item() * grad_accum_steps
                n_batches += 1

                if n_batches % grad_accum_steps == 0:
                    cur_lr = get_lr(global_step, total_steps, warmup_steps, lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = cur_lr
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if n_batches % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
            self.training_status["log"].append(msg)
            self.training_status["message"] = msg

        self.model.eval()
        self.training_status["message"] = f"Training complete! Final loss: {avg_loss:.4f}"
        self.training_status["running"] = False

    # --------------------------------------------------------
    # Checkpoint save
    # --------------------------------------------------------

    def _save_checkpoint(self, datasets=None, extra_meta=None):
        """Save model checkpoint."""
        if not self.ckpt_path:
            self.ckpt_path = os.path.join(self.model_path or ".", "neuroq_checkpoint.pt")

        # Load existing checkpoint for training log history
        prev_log = []
        prev_datasets = []
        if os.path.isfile(self.ckpt_path):
            try:
                old_ckpt = torch.load(self.ckpt_path, map_location="cpu")
                prev_log = old_ckpt.get("training_log", [])
                prev_datasets = old_ckpt.get("datasets", [])
            except Exception:
                pass

        new_log_entries = [
            {"epoch": len(prev_log) + i + 1, "loss": float(l.split("Loss: ")[1])}
            for i, l in enumerate(self.training_status["log"]) if "Loss:" in l
        ]

        ds_list = list(set(prev_datasets + (datasets or [])))

        checkpoint = {
            "model_state": self.model.state_dict(),
            "config": self.config,
            "training_log": prev_log + new_log_entries,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "datasets": ds_list,
        }
        if extra_meta:
            checkpoint.update(extra_meta)

        torch.save(checkpoint, self.ckpt_path)
        self.training_status["log"].append(f"Checkpoint saved: {self.ckpt_path}")

        # Also save to network volume for persistence across pod restarts
        if os.path.isdir(NETWORK_VOLUME_PATH):
            nv_ckpt_path = os.path.join(NETWORK_VOLUME_PATH, os.path.basename(self.ckpt_path))
            try:
                shutil.copy2(self.ckpt_path, nv_ckpt_path)
                # Also copy tokenizer model to network volume if it exists
                tokenizer_src = os.path.join(self.model_path or ".", "neuroq_tokenizer.model")
                if os.path.isfile(tokenizer_src):
                    shutil.copy2(tokenizer_src, os.path.join(NETWORK_VOLUME_PATH, "neuroq_tokenizer.model"))
                self.training_status["log"].append(f"Checkpoint synced to network volume: {nv_ckpt_path}")
            except Exception as e:
                self.training_status["log"].append(f"Warning: failed to sync to network volume: {e}")

    # --------------------------------------------------------
    # Split training helpers
    # --------------------------------------------------------

    def _load_split_state(self):
        if os.path.exists(self.split_state_path):
            with open(self.split_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_split_state(self, state):
        with open(self.split_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_all_qa_texts(self, max_samples):
        """Load all QA texts from default datasets."""
        from dataset_utils import safe_load_dataset
        all_qa = []
        for ds_info in QA_DATASETS_INFO:
            ds_id = ds_info["id"]
            fmt = ds_info["format"]
            ms = min(1000, max_samples) if fmt == "izumi" else max_samples
            try:
                self.training_status["message"] = f"Loading {ds_id}..."
                ds = safe_load_dataset(ds_id, split="train")
                n = min(ms, len(ds))
                count = 0
                for row in ds.select(range(n)):
                    if fmt == "alpaca":
                        text = format_qa_alpaca(row)
                    elif fmt == "conversations":
                        text = format_qa_conversations(row)
                    elif fmt == "izumi":
                        text = format_qa_izumi(row)
                    else:
                        continue
                    if text and len(text) > 10:
                        all_qa.append(text)
                        count += 1
                self.training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
            except Exception as e:
                self.training_status["log"].append(f"Error loading {ds_id}: {e}")
        return all_qa

    def _load_all_general_texts(self, max_samples):
        """Load all general texts from default datasets."""
        from dataset_utils import safe_load_dataset
        all_texts = []
        for ds_info in DEFAULT_DATASETS:
            ds_id = ds_info["id"]
            col = ds_info["col"]
            try:
                self.training_status["message"] = f"Loading {ds_id}..."
                ds = safe_load_dataset(ds_id, split="train")
                texts = extract_texts(ds, col, max_samples)
                all_texts.extend(texts)
                self.training_status["log"].append(f"Loaded {ds_id}: {len(texts)} texts")
            except Exception as e:
                self.training_status["log"].append(f"Error loading {ds_id}: {e}")
        return all_texts

    def _load_custom_datasets(self, dataset_ids, max_samples, mode):
        """Load custom datasets by ID with auto-format detection.

        dataset_ids can be:
          - "owner/dataset" — loads default config
          - "owner/dataset:config" — loads specific config (e.g. "openai/gsm8k:main")
        """
        from dataset_utils import safe_load_dataset
        all_texts = []
        for ds_spec in dataset_ids:
            try:
                # Parse "owner/dataset:config" format
                if ":" in ds_spec and not ds_spec.startswith("http"):
                    ds_id, ds_config = ds_spec.rsplit(":", 1)
                else:
                    ds_id, ds_config = ds_spec, None

                self.training_status["message"] = f"Loading {ds_id}..."
                load_kwargs = {"split": "train"}
                if ds_config:
                    load_kwargs["name"] = ds_config
                try:
                    ds = safe_load_dataset(ds_id, **load_kwargs)
                    is_streaming = False
                except Exception:
                    load_kwargs["streaming"] = True
                    ds = safe_load_dataset(ds_id, **load_kwargs)
                    is_streaming = True

                count = 0
                iterator = ds if is_streaming else ds.select(range(min(max_samples, len(ds))))
                for row in iterator:
                    if count >= max_samples:
                        break
                    text = None
                    if mode == "qa":
                        q = (row.get("question") or row.get("instruction") or row.get("input") or "")
                        q = q.strip() if isinstance(q, str) else ""
                        a = (row.get("answer") or row.get("output") or row.get("response") or "")
                        a = a.strip() if isinstance(a, str) else ""
                        if q and a and len(q) > 2 and len(a) > 2:
                            text = f"質問: {q}\n回答: {a}"
                        elif not q and a and len(a) > 10:
                            text = f"回答: {a}"
                        if not text:
                            convs = row.get("conversations", [])
                            if isinstance(convs, list) and len(convs) >= 2:
                                text = format_qa_conversations(row)
                        if not text and row.get("instruction"):
                            text = format_qa_alpaca(row)
                        # Fallback: use plain text column (e.g. Wikipedia)
                        if not text:
                            val = row.get("text") or row.get("content") or ""
                            if isinstance(val, str) and len(val.strip()) > 20:
                                text = val.strip()
                    else:
                        for col in ["text", "content", "output", "sentence", "document"]:
                            val = row.get(col)
                            if isinstance(val, str) and len(val.strip()) > 10:
                                text = val.strip()
                                break
                        if not text:
                            convs = row.get("conversations", [])
                            if isinstance(convs, list) and convs:
                                parts = []
                                for turn in convs:
                                    if isinstance(turn, dict):
                                        parts.append(turn.get("value", turn.get("content", "")))
                                    elif isinstance(turn, str):
                                        parts.append(turn)
                                combined = "\n".join(parts)
                                if len(combined.strip()) > 10:
                                    text = combined.strip()
                    if text and len(text) > 10:
                        all_texts.append(text)
                        count += 1
                self.training_status["log"].append(f"Loaded {ds_id}: {count} texts")
            except Exception as e:
                self.training_status["log"].append(f"Error loading {ds_id}: {e}")
        return all_texts

    @staticmethod
    def _split_into_chunks(data, num_chunks, samples_per_batch=None):
        """Split data into chunks.

        Examples:
          [0,10000] with samples_per_batch=1000
            → chunk0=[0,999], chunk1=[1000,1999], ..., chunk9=[9000,9999]
        """
        random.shuffle(data)
        if samples_per_batch is not None and samples_per_batch > 0:
            chunk_size = samples_per_batch
        else:
            chunk_size = math.ceil(len(data) / num_chunks)
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

    def _run_chunk_training(self, sequences, epochs_per_chunk, lr, batch_size,
                            grad_accum_steps, warmup_steps, max_seq_len,
                            chunk_idx, total_chunks, max_minutes=None):
        """Train on a single chunk with optional timeout."""
        steps_per_epoch = len(sequences) // batch_size
        total_steps = (steps_per_epoch * epochs_per_chunk) // grad_accum_steps
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        min_lr_ratio = 0.1

        self.model.train()
        global_step = 0
        best_loss = float('inf')
        timed_out = False
        chunk_start_time = time.time()
        timeout_seconds = max_minutes * 60 if max_minutes else None

        for epoch in range(epochs_per_chunk):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()
            self.training_status["message"] = (
                f"Chunk {chunk_idx+1}/{total_chunks} | "
                f"Epoch {epoch+1}/{epochs_per_chunk}..."
            )

            for i in range(0, len(sequences), batch_size):
                if timeout_seconds and (time.time() - chunk_start_time) >= timeout_seconds:
                    elapsed = (time.time() - chunk_start_time) / 60
                    msg = f"TIMEOUT: Chunk {chunk_idx+1} stopped after {elapsed:.1f} min"
                    self.training_status["log"].append(msg)
                    self.training_status["message"] = msg
                    timed_out = True
                    break

                batch_seqs = sequences[i:i + batch_size]
                if not batch_seqs:
                    continue

                max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
                input_ids = []
                labels = []
                for s in batch_seqs:
                    ids = s[:max_len]
                    pad_len = max_len - len(ids)
                    input_ids.append(ids + [self.tokenizer.pad_id] * pad_len)
                    labels.append(ids + [-100] * pad_len)

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

                logits = self.model(input_ids_t)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels_t[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config["vocab_size"]),
                    shift_labels.view(-1),
                    ignore_index=-100,
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if timed_out:
                if n_batches % grad_accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                if n_batches > 0:
                    avg_loss = total_loss / n_batches
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                break

            if n_batches % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = (f"Chunk {chunk_idx+1}/{total_chunks} | "
                   f"Epoch {epoch+1}/{epochs_per_chunk} | Loss: {avg_loss:.4f}")
            self.training_status["log"].append(msg)
            self.training_status["message"] = msg

            if avg_loss < best_loss:
                best_loss = avg_loss

        self.model.eval()
        return best_loss, timed_out

    # --------------------------------------------------------
    # Split training (all chunks)
    # --------------------------------------------------------

    def _handle_train_split(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run split dataset training: divides data into chunks and trains sequentially.

        Example: 10000 samples with samples_per_batch=1000
          → [0,999], [1000,1999], ..., [9000,9999] の10チャンクで順次学習
        """
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        params = data.get("parameters", {})
        mode = str(params.get("mode", "qa"))
        num_chunks = int(params.get("num_chunks", 4))
        samples_per_batch = params.get("samples_per_batch", None)
        if samples_per_batch is not None:
            samples_per_batch = int(samples_per_batch)
        chunk_index = params.get("chunk_index", None)
        if chunk_index is not None:
            chunk_index = int(chunk_index)
        start_sample = params.get("start_sample", None)
        if start_sample is not None:
            start_sample = int(start_sample)
        end_sample = params.get("end_sample", None)
        if end_sample is not None:
            end_sample = int(end_sample)
        max_minutes_per_chunk = params.get("max_minutes_per_chunk", None)
        if max_minutes_per_chunk is not None:
            max_minutes_per_chunk = float(max_minutes_per_chunk)
        epochs_per_chunk = int(params.get("epochs_per_chunk", 5))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 20))
        max_samples = int(params.get("max_samples_per_dataset", 2000))
        crafted_repeat = int(params.get("crafted_repeat", 20))
        dataset_ids = params.get("dataset_ids", None)
        resume = bool(params.get("resume", False))

        self.training_status = {"running": True, "log": [], "message": "Loading datasets for split training..."}

        try:
            # Load data
            if dataset_ids:
                all_texts = self._load_custom_datasets(dataset_ids, max_samples, mode)
                if mode == "qa" and crafted_repeat > 0:
                    for _ in range(crafted_repeat):
                        all_texts.extend(CRAFTED_QA)
                    self.training_status["log"].append(
                        f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
                    )
                ds_names = dataset_ids
            elif mode == "qa":
                all_texts = self._load_all_qa_texts(max_samples)
                for _ in range(crafted_repeat):
                    all_texts.extend(CRAFTED_QA)
                self.training_status["log"].append(
                    f"Added {len(CRAFTED_QA) * crafted_repeat} crafted QA samples"
                )
                ds_names = [d["id"] for d in QA_DATASETS_INFO]
            else:
                all_texts = self._load_all_general_texts(max_samples)
                ds_names = [d["id"] for d in DEFAULT_DATASETS]

            self.training_status["log"].append(f"Total texts: {len(all_texts)}")

            if not all_texts:
                self.training_status["running"] = False
                self.training_status["message"] = "No texts loaded"
                return [{"status": "error", "message": "No texts loaded",
                         "log": self.training_status["log"]}]

            # Apply sample range filter
            if start_sample is not None or end_sample is not None:
                s = max(0, min(start_sample or 0, len(all_texts)))
                e = max(s, min(end_sample or len(all_texts), len(all_texts)))
                self.training_status["log"].append(f"Sample range: [{s}, {e}) = {e - s} samples")
                all_texts = all_texts[s:e]
                if not all_texts:
                    self.training_status["running"] = False
                    return [{"status": "error", "message": "No texts in specified range"}]

            # Split into chunks
            chunks = self._split_into_chunks(all_texts, num_chunks, samples_per_batch)
            actual_num_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                self.training_status["log"].append(f"Chunk {i}: {len(chunk)} texts")

            # Determine which chunks to train
            if chunk_index is not None:
                if chunk_index >= actual_num_chunks:
                    self.training_status["running"] = False
                    return [{"status": "error",
                             "message": f"chunk_index {chunk_index} >= {actual_num_chunks}"}]
                chunk_indices = [chunk_index]
            elif resume:
                state = self._load_split_state()
                if state and state.get("mode") == mode:
                    start_idx = state.get("last_completed_chunk", -1) + 1
                    if start_idx >= actual_num_chunks:
                        self.training_status["running"] = False
                        return [{"status": "completed",
                                 "message": "All chunks already completed!"}]
                    chunk_indices = list(range(start_idx, actual_num_chunks))
                else:
                    chunk_indices = list(range(actual_num_chunks))
            else:
                chunk_indices = list(range(actual_num_chunks))

            max_seq_len = self.config["max_seq_len"]

            # Train each chunk
            for cidx in chunk_indices:
                chunk_texts = chunks[cidx]
                self.training_status["message"] = f"Chunk {cidx+1}/{actual_num_chunks}: Tokenizing..."

                sequences = tokenize_texts(chunk_texts, self.tokenizer, max_seq_len)
                if not sequences:
                    # Debug: why no sequences?
                    sample_text = chunk_texts[0] if chunk_texts else ""
                    sample_ids = self.tokenizer.encode(sample_text[:100], add_special=False)
                    self.training_status["log"].append(
                        f"Chunk {cidx}: No sequences! sp={self.tokenizer.sp is not None} "
                        f"max_seq_len={max_seq_len} sample_len={len(sample_text)} "
                        f"sample_ids_len={len(sample_ids)} sample_ids={sample_ids[:10]} "
                        f"vocab_size={self.config.get('vocab_size')}"
                    )
                    self.training_status["log"].append(f"Chunk {cidx}: No sequences, skipping")
                    continue

                self.training_status["log"].append(f"Chunk {cidx}: {len(sequences)} sequences")

                best_loss, timed_out = self._run_chunk_training(
                    sequences, epochs_per_chunk, lr, batch_size,
                    grad_accum_steps, warmup_steps, max_seq_len,
                    cidx, actual_num_chunks, max_minutes_per_chunk,
                )

                # Save checkpoint after each chunk
                self._save_checkpoint(datasets=ds_names)
                self.training_status["log"].append(
                    f"Chunk {cidx+1}/{actual_num_chunks} complete, best loss: {best_loss:.4f}"
                )

                # Save split state
                self._save_split_state({
                    "mode": mode,
                    "num_chunks": actual_num_chunks,
                    "last_completed_chunk": cidx,
                    "best_loss": best_loss,
                    "timed_out": timed_out,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                if timed_out:
                    self.training_status["log"].append(
                        "Remaining chunks can be trained with resume=true or train_split_next"
                    )
                    break

            self.training_status["message"] = (
                f"Split training complete! {len(chunk_indices)} chunks trained"
            )

            # Run inference test on the same worker after training
            inference_results = {}
            test_prompts = [
                "日本の首都はどこですか？",
                "量子コンピュータとは何ですか？",
                "Pythonでリストをソートする方法を教えてください。",
            ]
            try:
                self.model.eval()
                for tp in test_prompts:
                    result = self._handle_inference({"inputs": tp, "parameters": {
                        "max_new_tokens": 200, "temperature": 0.7,
                        "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.3
                    }})
                    r = result[0] if result else {}
                    text = r.get("generated_text", "")
                    debug = r.get("debug", {})
                    inference_results[tp] = {"text": text, "debug": debug}
                    self.training_status["log"].append(
                        f"Inference: Q={tp} A={text[:80]} debug_tokens={debug.get('debug_tokens', [])[:3]}"
                    )
            except Exception as e:
                import traceback
                self.training_status["log"].append(f"Inference test error: {traceback.format_exc()}")

            self.training_status["running"] = False

            return [{"status": "success", "message": self.training_status["message"],
                     "chunks_total": actual_num_chunks,
                     "chunks_trained": len(chunk_indices),
                     "inference_test": inference_results,
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Split training (next chunk only — timeout-safe)
    # --------------------------------------------------------

    def _handle_train_split_next(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Train only the next unfinished chunk. Call repeatedly to train all chunks
        without hitting API timeouts."""
        if self.training_status["running"]:
            return [{"status": "error", "message": "Training already in progress"}]

        params = data.get("parameters", {})
        mode = str(params.get("mode", "qa"))
        num_chunks = int(params.get("num_chunks", 4))
        samples_per_batch = params.get("samples_per_batch", None)
        if samples_per_batch is not None:
            samples_per_batch = int(samples_per_batch)
        epochs_per_chunk = int(params.get("epochs_per_chunk", 3))
        lr = float(params.get("lr", 3e-5))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        warmup_steps = int(params.get("warmup_steps", 20))
        max_samples = int(params.get("max_samples_per_dataset", 2000))
        crafted_repeat = int(params.get("crafted_repeat", 20))
        dataset_ids = params.get("dataset_ids", None)

        self.training_status = {"running": True, "log": [], "message": "Loading datasets for batch training..."}

        try:
            # Determine next chunk from saved state
            state = self._load_split_state()
            if state and state.get("mode") == mode:
                next_chunk = state.get("last_completed_chunk", -1) + 1
            else:
                next_chunk = 0

            # Load data
            if dataset_ids:
                all_texts = self._load_custom_datasets(dataset_ids, max_samples, mode)
                if mode == "qa" and crafted_repeat > 0:
                    for _ in range(crafted_repeat):
                        all_texts.extend(CRAFTED_QA)
                ds_names = dataset_ids
            elif mode == "qa":
                all_texts = self._load_all_qa_texts(max_samples)
                for _ in range(crafted_repeat):
                    all_texts.extend(CRAFTED_QA)
                ds_names = [d["id"] for d in QA_DATASETS_INFO]
            else:
                all_texts = self._load_all_general_texts(max_samples)
                ds_names = [d["id"] for d in DEFAULT_DATASETS]

            if not all_texts:
                self.training_status["running"] = False
                return [{"status": "error", "message": "No texts loaded",
                         "log": self.training_status["log"]}]

            # Split into chunks
            chunks = self._split_into_chunks(all_texts, num_chunks, samples_per_batch)
            actual_num_chunks = len(chunks)

            if next_chunk >= actual_num_chunks:
                self.training_status["running"] = False
                return [{"status": "completed",
                         "message": "All chunks already completed. Use split_reset to retrain.",
                         "chunks_total": actual_num_chunks,
                         "chunks_completed": actual_num_chunks}]

            # Train this one chunk
            chunk_texts = chunks[next_chunk]
            self.training_status["log"].append(
                f"Training batch {next_chunk+1}/{actual_num_chunks} ({len(chunk_texts)} texts)"
            )

            max_seq_len = self.config["max_seq_len"]
            sequences = tokenize_texts(chunk_texts, self.tokenizer, max_seq_len)

            if not sequences:
                self._save_split_state({
                    "mode": mode,
                    "num_chunks": actual_num_chunks,
                    "last_completed_chunk": next_chunk,
                    "best_loss": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.training_status["running"] = False
                return [{"status": "skipped", "chunk_index": next_chunk,
                         "chunks_total": actual_num_chunks,
                         "chunks_remaining": actual_num_chunks - next_chunk - 1}]

            self.training_status["log"].append(f"Batch {next_chunk}: {len(sequences)} sequences")

            best_loss, timed_out = self._run_chunk_training(
                sequences, epochs_per_chunk, lr, batch_size,
                grad_accum_steps, warmup_steps, max_seq_len,
                next_chunk, actual_num_chunks,
            )

            # Save checkpoint and state
            self._save_checkpoint(datasets=ds_names)

            self._save_split_state({
                "mode": mode,
                "num_chunks": actual_num_chunks,
                "last_completed_chunk": next_chunk,
                "best_loss": best_loss,
                "timed_out": timed_out,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            remaining = actual_num_chunks - next_chunk - 1
            self.training_status["message"] = (
                f"Batch {next_chunk+1}/{actual_num_chunks} complete! "
                f"Loss: {best_loss:.4f}, remaining: {remaining}"
            )
            self.training_status["running"] = False

            return [{"status": "success",
                     "message": self.training_status["message"],
                     "chunk_index": next_chunk,
                     "chunks_total": actual_num_chunks,
                     "chunks_remaining": remaining,
                     "best_loss": best_loss,
                     "log": self.training_status["log"]}]

        except Exception as e:
            import traceback
            self.training_status["running"] = False
            self.training_status["message"] = f"Error: {e}"
            self.training_status["log"].append(traceback.format_exc())
            self.model.eval()
            return [{"status": "error", "message": str(e),
                     "log": self.training_status["log"]}]

    # --------------------------------------------------------
    # Split status / reset
    # --------------------------------------------------------

    def _handle_split_status(self) -> List[Dict[str, Any]]:
        """Return split training progress."""
        state = self._load_split_state()
        if state:
            return [{
                "status": "ok",
                "mode": state.get("mode"),
                "num_chunks": state.get("num_chunks"),
                "last_completed_chunk": state.get("last_completed_chunk"),
                "chunks_remaining": state.get("num_chunks", 0) - state.get("last_completed_chunk", -1) - 1,
                "best_loss": state.get("best_loss"),
                "timed_out": state.get("timed_out", False),
                "timestamp": state.get("timestamp"),
            }]
        return [{"status": "no_state", "message": "No split training state found"}]

    def _handle_split_reset(self) -> List[Dict[str, Any]]:
        """Reset split training state to start over."""
        if os.path.exists(self.split_state_path):
            os.remove(self.split_state_path)
            return [{"status": "ok", "message": "Split training state reset"}]
        return [{"status": "ok", "message": "No state to reset"}]

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def _handle_status(self) -> List[Dict[str, Any]]:
        """Return comprehensive model status information."""
        n_params = sum(p.numel() for p in self.model.parameters())

        # Checkpoint info
        ckpt_info = None
        training_history_count = 0
        training_history_latest = None
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            ckpt_stat = os.stat(self.ckpt_path)
            ckpt_info = {
                "path": self.ckpt_path,
                "size_mb": round(ckpt_stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    ckpt_stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            try:
                ckpt = torch.load(self.ckpt_path, map_location="cpu")
                trained_at = ckpt.get("trained_at")
                if trained_at:
                    ckpt_info["trained_at"] = trained_at
                training_log = ckpt.get("training_log", [])
                training_history_count = len(training_log)
                if training_log:
                    training_history_latest = training_log[-1]
            except Exception:
                pass

        # Split training state
        split_state = self._load_split_state()

        return [{
            "status": "running" if self.training_status["running"] else "idle",
            "message": self.training_status["message"],
            "log": self.training_status["log"],
            "architecture": self.config.get("architecture", "neuroquantum"),
            "model_params": n_params,
            "config": self.config,
            "checkpoint": ckpt_info,
            "training_history": {
                "count": training_history_count,
                "latest": training_history_latest,
            },
            "split_training": split_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]


# ============================================================
# RunPod Serverless Entry Point
# ============================================================

def _runpod_handler(event):
    """
    RunPod serverless handler function.

    Translates RunPod input format into EndpointHandler format and returns
    the result. Called by runpod.serverless.start() when this file is
    executed directly.

    Expected input format:
        {
            "input": {
                "prompt": "こんにちは",
                "action": "inference",       # optional
                "parameters": {              # optional
                    "temperature": 0.7,
                    "max_new_tokens": 100,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repetition_penalty": 1.3
                }
            }
        }

    Supported actions:
        inference (default), train, train_qa, train_qa_dataset,
        train_split, train_split_next, split_status, split_reset, status
    """
    job_input = event.get("input", {})

    # Translate RunPod input to EndpointHandler format
    data = {}

    # Action
    if "action" in job_input:
        data["action"] = job_input["action"]

    # Inputs (prompt text)
    if "prompt" in job_input:
        data["inputs"] = job_input["prompt"]
    elif "inputs" in job_input:
        data["inputs"] = job_input["inputs"]

    # Parameters
    if "parameters" in job_input:
        data["parameters"] = job_input["parameters"]

    # Pass through extra fields (for training payloads)
    for key in ("qa_pairs", "dataset_ids", "epochs", "lr", "batch_size",
                "mode", "num_chunks", "resume"):
        if key in job_input:
            data.setdefault("parameters", {})[key] = job_input[key]

    # Call the EndpointHandler
    result = _global_handler(data)

    # RunPod expects a dict or list, not wrapped in extra list
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


if __name__ == "__main__":
    import runpod

    MODEL_DIR = os.environ.get("MODEL_DIR", "/app")
    _global_handler = EndpointHandler(path=MODEL_DIR)
    print(f"[handler] RunPod serverless mode — model loaded from {MODEL_DIR}")

    runpod.serverless.start({"handler": _runpod_handler})
