#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗                        ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗                       ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║                       ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║                       ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝                       ║    
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝                        ║
║                                                                               ║                                                                 ║
║                                                                               ║
║   neuroQ: Quantum-Bit Neural Network Language Model                           ║
║   独自の量子もつれニューラルネットワークによる生成AI                                  ║
║                                                                               ║
║   参照元: qbnn_layered.py                                                      ║
║   - APQB: 調整可能擬似量子ビット                                                  ║
║   - EntanglementOperator: 層間エンタングル演算子                                  ║
║   - EQBNNLayer: 層状QBNN層                                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from collections import Counter
import re
from typing import List, Dict, Optional, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

# システムRAM検出用
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def detect_system_ram_gb() -> float:
    """
    システムRAMの総容量をGB単位で返す。
    psutilが利用可能な場合はそちらを使用し、
    利用不可の場合は /proc/meminfo（Linux）またはデフォルト値を返す。
    """
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().total / (1024 ** 3)

    # Linux: /proc/meminfo からの取得を試みる
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # MemTotal は kB 単位
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except (FileNotFoundError, ValueError, IndexError):
        pass

    # macOS: sysctl から取得を試みる
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # デフォルト: 8GB と仮定
    return 8.0


# sentencepiece（日本語サブワードトークナイザー用）
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    warnings.warn("sentencepieceライブラリがインストールされていません。pip install sentencepiece を実行してください。")

# transformersライブラリは依存関係の競合を避けるため使用しない
# 純PyTorchによるSelf-Attention実装を使用
TRANSFORMERS_AVAILABLE = False

# OpenAI API（オプション）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI APIが利用できません。openai>=1.0.0をインストールしてください。")

# Google Generative AI（オプション - テキストエンベディング用）
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Translation Pipeline（オプション）
try:
    from translation_pipeline import (
        TranslationPipeline,
        TiktokenWrapper,
        TranslatedNeuroQuantumAI,
        create_english_training_data,
    )
    TRANSLATION_PIPELINE_AVAILABLE = True
except ImportError:
    TRANSLATION_PIPELINE_AVAILABLE = False
    # オプションなので警告を表示しない

# ========================================
# qbnn_layered.py からコアコンポーネントをインポート
# ========================================
try:
    from qbnn_layered import (
        APQB as APQB_Core,                  # APQB理論のコア
        EntanglementOperator,               # 層間エンタングル演算子
        QuantumCorrelationMatrix,           # 量子相関行列
        EQBNNLayer,                         # E-QBNN層
    )
    QBNN_LAYERED_AVAILABLE = True
    print("✅ qbnn_layered.py からコアコンポーネントをインポートしました")
except ImportError:
    QBNN_LAYERED_AVAILABLE = False
    # オプションなので警告を表示しない（内蔵コンポーネントで動作します）

# ========================================
# quantum_computer.py から量子回路シミュレーターをインポート
# ========================================
try:
    from quantum_computer import (
        QuantumComputer,
        QuantumCircuit,
        Gates,
        QubitState,
    )
    QUANTUM_COMPUTER_AVAILABLE = True
    print("✅ quantum_computer.py から量子回路シミュレーターをインポートしました")
except ImportError:
    QUANTUM_COMPUTER_AVAILABLE = False
    warnings.warn("quantum_computer.py が見つかりません。量子回路シミュレーション機能は無効です。")

# ========================================
# 設定
# ========================================

class NeuroQuantumConfig:
    """ニューロQ設定"""
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 10000,
        dropout: float = 0.1,
        lambda_entangle: float = 0.5,  # QBNNもつれ強度
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lambda_entangle = lambda_entangle


def detect_gpu_tier() -> Tuple[str, str, dict]:
    """
    GPUの性能を検出し、ティアを判定する。
    システムRAMも検出し、gpu_infoに含める。

    Returns:
        (tier, device_name, gpu_info):
            tier: "high" | "mid" | "low" | "cpu"
            device_name: デバイス名の文字列
            gpu_info: VRAM・RAM等の詳細情報
    """
    ram_gb = detect_system_ram_gb()
    gpu_info = {
        "vram_gb": 0,
        "ram_gb": round(ram_gb, 1),
        "compute_capability": None,
        "device_type": "cpu",
    }

    if torch.cuda.is_available():
        gpu_info["device_type"] = "cuda"
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        vram_bytes = torch.cuda.get_device_properties(device_id).total_mem
        vram_gb = vram_bytes / (1024 ** 3)
        major, minor = torch.cuda.get_device_capability(device_id)
        gpu_info["vram_gb"] = round(vram_gb, 1)
        gpu_info["compute_capability"] = f"{major}.{minor}"

        if vram_gb >= 40:
            return "ultra", device_name, gpu_info
        elif vram_gb >= 16:
            return "high", device_name, gpu_info
        elif vram_gb >= 8:
            return "mid", device_name, gpu_info
        else:
            return "low", device_name, gpu_info

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_info["device_type"] = "mps"
        device_name = "Apple Silicon (MPS)"
        # Apple Siliconは統合メモリ — RAMに基づいてティアを判定
        if ram_gb >= 32:
            return "high", device_name, gpu_info
        elif ram_gb >= 16:
            return "mid", device_name, gpu_info
        else:
            return "low", device_name, gpu_info

    # CPU専用: RAMに基づいてティアを判定
    if ram_gb >= 64:
        return "mid", "CPU", gpu_info
    elif ram_gb >= 32:
        return "low", "CPU", gpu_info
    return "cpu", "CPU", gpu_info


def get_gpu_adaptive_config(vocab_size: int = 32000) -> dict:
    """
    GPU性能とシステムRAMに基づいて最適なニューロン数・モデル設定を返す。

    ティア判定の基準:
        GPU環境:
            ultra (VRAM >= 40GB): A100最大設定 - 超大規模モデル
            high  (VRAM >= 16GB): フル設定 - 大規模モデル
            mid   (VRAM >= 8GB):  中規模設定
            low   (VRAM < 8GB):   軽量設定
        MPS (Apple Silicon):
            統合メモリ(RAM)に基づいてティアを判定
        CPU環境:
            RAM >= 64GB → mid, RAM >= 32GB → low, それ以下 → cpu

    RAMによる追加調整:
        GPU環境でもRAMが潤沢な場合、batch_size や max_seq_len を増加。
        CPU環境ではRAM容量に応じてニューロン数を動的にスケーリング。

    Returns:
        dict: モデル設定パラメータとデバイス情報を含む辞書
    """
    tier, device_name, gpu_info = detect_gpu_tier()
    ram_gb = gpu_info.get("ram_gb", 8.0)

    # ティア別のベースニューロン数設定
    TIER_CONFIGS = {
        "ultra": {
            "embed_dim": 768,
            "hidden_dim": 2048,
            "num_heads": 12,
            "num_layers": 12,
            "max_seq_len": 16384,
            "dropout": 0.1,
            "entangle_strength": 0.5,
            "batch_size": 16,
        },
        "high": {
            "embed_dim": 512,
            "hidden_dim": 1024,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 10000,
            "dropout": 0.1,
            "entangle_strength": 0.5,
            "batch_size": 8,
        },
        "mid": {
            "embed_dim": 384,
            "hidden_dim": 768,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 10000,
            "dropout": 0.1,
            "entangle_strength": 0.5,
            "batch_size": 4,
        },
        "low": {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 4,
            "max_seq_len": 4096,
            "dropout": 0.1,
            "entangle_strength": 0.5,
            "batch_size": 2,
        },
        "cpu": {
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_heads": 4,
            "num_layers": 3,
            "max_seq_len": 2048,
            "dropout": 0.1,
            "entangle_strength": 0.5,
            "batch_size": 1,
        },
    }

    config = TIER_CONFIGS[tier].copy()

    # === RAM に基づくニューロン数の動的調整 ===
    if gpu_info["device_type"] == "cpu":
        # CPU専用環境: RAMがモデルサイズの上限を決める
        if ram_gb >= 128:
            # 大容量RAM: midティアを超えてhigh相当まで拡張
            config["embed_dim"] = 512
            config["hidden_dim"] = 1024
            config["num_heads"] = 8
            config["num_layers"] = 6
            config["max_seq_len"] = 10000
            config["batch_size"] = 4
        elif ram_gb >= 64:
            # mid ティア（detect_gpu_tier で判定済み）— さらに微調整
            config["embed_dim"] = 384
            config["hidden_dim"] = 768
            config["num_heads"] = 8
            config["num_layers"] = 6
            config["max_seq_len"] = 10000
            config["batch_size"] = 2
        elif ram_gb >= 32:
            # low ティア（detect_gpu_tier で判定済み）
            config["embed_dim"] = 256
            config["hidden_dim"] = 512
            config["num_heads"] = 8
            config["num_layers"] = 4
            config["max_seq_len"] = 4096
            config["batch_size"] = 2
        elif ram_gb >= 16:
            # RAM 16GB: cpu デフォルトより少し大きめ
            config["embed_dim"] = 192
            config["hidden_dim"] = 384
            config["num_heads"] = 4
            config["num_layers"] = 3
            config["max_seq_len"] = 2048
            config["batch_size"] = 1
        # ram_gb < 16: デフォルトの cpu ティア設定をそのまま使用

    elif gpu_info["device_type"] == "mps":
        # Apple Silicon: 統合メモリなのでRAM = 実質VRAM
        if ram_gb >= 64:
            config["embed_dim"] = 512
            config["hidden_dim"] = 1024
            config["num_heads"] = 8
            config["num_layers"] = 6
            config["max_seq_len"] = 10000
            config["batch_size"] = 8
        elif ram_gb >= 32:
            config["batch_size"] = min(config["batch_size"] + 2, 8)
            config["max_seq_len"] = 10000

    else:
        # CUDA GPU環境: RAMが十分にある場合、batch_size を増加
        if tier == "ultra" and ram_gb >= 128:
            config["batch_size"] = min(config["batch_size"] * 2, 32)
        elif ram_gb >= 64:
            config["batch_size"] = min(config["batch_size"] * 2, 16)
        elif ram_gb >= 32:
            config["batch_size"] = min(config["batch_size"] + 2, 12)

    config["vocab_size"] = vocab_size
    config["gpu_tier"] = tier
    config["gpu_name"] = device_name
    config["gpu_info"] = gpu_info

    print(f"=== GPU/RAM 適応設定 ===")
    print(f"  デバイス: {device_name}")
    print(f"  ティア: {tier}")
    print(f"  システムRAM: {ram_gb} GB")
    if gpu_info["vram_gb"] > 0:
        print(f"  VRAM: {gpu_info['vram_gb']} GB")
    if gpu_info["compute_capability"]:
        print(f"  Compute Capability: {gpu_info['compute_capability']}")
    print(f"  embed_dim: {config['embed_dim']}")
    print(f"  hidden_dim: {config['hidden_dim']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  max_seq_len: {config['max_seq_len']}")
    print(f"  batch_size: {config['batch_size']}")
    print(f"========================")

    return config


# ========================================
# Part 1: QBNN Layer（独自の量子もつれ層）
# ========================================

class QBNNLayer(nn.Module):
    """
    Quantum-Bit Neural Network Layer
    
    qbnn_layered.py の EQBNNLayer を基盤として使用可能
    
    独自の数式モデル:
    1. s^(l) = tanh(h^(l)) ∈ [-1, 1]  (正規化 → Bloch球のz座標)
    2. h̃^(l+1) = W^(l) h^(l) + b^(l)  (線形変換)
    3. Δ^(l+1)_j = Σ_i J^(l)_{ij} s^(l)_i s^(l+1)_{raw,j}  (もつれ補正)
    4. ĥ^(l+1) = h̃^(l+1) + λ_eff Δ^(l+1)  (有効入力)
    5. h^(l+1) = activation(ĥ^(l+1))  (活性化)
    
    APQB理論に基づく改良:
    - λを範囲で制御し、θ（シータ）が動的に変化できるようにする
    - r = cos(2θ), T = |sin(2θ)|, r² + T² = 1
    
    参照: qbnn_layered.py の APQB クラス
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 lambda_min: float = 0.2, lambda_max: float = 0.5,
                 use_qbnn_layered: bool = True):  # qbnn_layered.pyを参照
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_qbnn_layered = use_qbnn_layered and QBNN_LAYERED_AVAILABLE
        
        # λの範囲（θが動けるように）
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        if self.use_qbnn_layered:
            # qbnn_layered.py の EQBNNLayer を内部で使用
            self.eqbnn_core = EQBNNLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                prev_output_dim=input_dim,
                entangle_strength=(lambda_min + lambda_max) / 2
            )
            # コア層のパラメータを参照
            self.W = self.eqbnn_core.linear
            # W_entangle.weightの形状は(output_dim, input_dim)なので、参照として保持
            # forward内で転置して使用する
            self.J_source = self.eqbnn_core.entangle_op.W_entangle.weight
        else:
            # W: 通常の重み行列
            self.W = nn.Linear(input_dim, output_dim)
            
            # J: もつれテンソル（独自）
            self.J = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        
        # λベース値（学習可能、0-1に正規化してから範囲にマッピング）
        self.lambda_base = nn.Parameter(torch.tensor(0.5))
        
        # 層正規化
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 呼び出しカウンタ（動的変化用）
        self.register_buffer('call_count', torch.tensor(0))
    
    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        # 1. 正規化（Bloch球のz座標として解釈）
        s_prev = torch.tanh(h_prev)  # (..., input_dim)
        
        # 2. 線形変換
        h_tilde = self.W(h_prev)  # (..., output_dim)
        
        # 3. 次層の候補を正規化
        s_raw = torch.tanh(h_tilde)  # (..., output_dim)
        
        # 4. もつれ補正項 Δ
        # Δ_j = Σ_i J_{ij} s^(l)_i * s^(l+1)_{raw,j}
        # J: (input_dim, output_dim)
        # s_prev: (..., input_dim)
        # s_raw: (..., output_dim)
        # delta: (..., output_dim)
        # 各jについて: delta_j = Σ_i (J_{ij} * s_prev_i) * s_raw_j
        
        # Jの形状を確認して転置（use_qbnn_layeredの場合、J_sourceは(output_dim, input_dim)）
        if self.use_qbnn_layered:
            J = self.J_source.t()  # (input_dim, output_dim)に転置
        else:
            J = self.J  # 既に(input_dim, output_dim)
        
        # まず J_{ij} * s_prev_i を計算: (..., output_dim)
        J_s_prev = torch.einsum('...i,ij->...j', s_prev, J)  # (..., output_dim)
        # 次に s_raw_j を掛ける
        delta = J_s_prev * s_raw  # (..., output_dim)
        
        # 5. 動的λ: θが動けるように範囲内で変化（量子ゆらぎ）
        # λ_baseをsigmoidで0-1に制限し、範囲にマッピング
        lambda_normalized = torch.sigmoid(self.lambda_base)
        
        # 推論時のみ動的変化を追加（学習時は安定性のため固定寄り）
        if not self.training:
            # sin波で動的変化（θが動けるように）
            phase = float(self.call_count) * 0.2
            dynamic_factor = 0.5 + 0.5 * math.sin(phase)
            self.call_count += 1
        else:
            dynamic_factor = 0.5
        
        # 有効なλを計算
        lambda_range = self.lambda_max - self.lambda_min
        lambda_eff = self.lambda_min + lambda_range * (lambda_normalized * 0.7 + dynamic_factor * 0.3)
        
        # 6. 有効入力
        h_hat = h_tilde + lambda_eff * delta
        
        # 7. 層正規化 + GELU活性化
        output = self.layer_norm(h_hat)
        output = F.gelu(output)
        
        return output
    
    def get_quantum_info(self) -> Dict:
        """量子情報を取得"""
        with torch.no_grad():
            lambda_normalized = torch.sigmoid(self.lambda_base).item()
            lambda_eff = self.lambda_min + (self.lambda_max - self.lambda_min) * lambda_normalized
            
            # Jの形状を確認（use_qbnn_layeredの場合、転置が必要）
            if self.use_qbnn_layered:
                J = self.J_source.t()  # (input_dim, output_dim)に転置
            else:
                J = self.J  # 既に(input_dim, output_dim)
            
            info = {
                'lambda_min': self.lambda_min,
                'lambda_max': self.lambda_max,
                'lambda_eff': lambda_eff,
                'J_mean': J.mean().item(),
                'J_std': J.std().item(),
                'J_max': J.max().item(),
                'source': 'qbnn_layered.py' if self.use_qbnn_layered else 'builtin',
            }
            
            # qbnn_layered.py使用時は追加情報を取得
            if self.use_qbnn_layered:
                info['entangle_strength'] = self.eqbnn_core.entangle_op.entangle_strength.item()
            
            return info


# ========================================
# Part 2: QBNN-Attention（transformersライブラリベース + QBNN拡張）
# ========================================

class QBNNAttention(nn.Module):
    """
    QBNN拡張Self-Attention（純PyTorch実装）

    Multi-Head Causal Self-AttentionにQBNN量子もつれ補正を追加
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 lambda_val: float = 0.5, max_positions: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dimはnum_headsで割り切れる必要があります"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # QBNN量子もつれ補正パラメータ
        self.J_attn = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.lambda_attn = nn.Parameter(torch.tensor(float(lambda_val)))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-Head Causal Self-Attention（QBNN拡張版）

        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask

        Returns:
            (batch, seq, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # QBNN拡張: 量子もつれ補正
        Q_norm = torch.tanh(Q)
        K_norm = torch.tanh(K)
        delta = torch.einsum('bhid,hde,bhje->bhij', Q_norm, self.J_attn, K_norm)
        attn_scores = attn_scores + self.lambda_attn * delta

        # Causal Mask
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output)


# ========================================
# Part 3: QBNN-Transformer Block
# ========================================

class QBNNTransformerBlock(nn.Module):
    """
    GPTデコーダーブロック（QBNN拡張版）
    
    GPT標準構造:
    1. Pre-norm LayerNorm
    2. Multi-Head Causal Self-Attention (QBNN拡張)
    3. Residual Connection
    4. Pre-norm LayerNorm
    5. Feed-Forward Network (標準FFN + QBNN拡張)
    6. Residual Connection
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        
        # Pre-norm LayerNorm
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        # QBNN-Attention
        self.attention = QBNNAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            lambda_val=config.lambda_entangle
        )
        
        # GPT標準FFN: Linear → GELU → Linear
        self.ffn_standard = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        
        # QBNN拡張FFN（個別レイヤーにアクセス可能にするため、Sequentialではなく個別に定義）
        self.ffn_qbnn_layer1 = QBNNLayer(
            config.embed_dim, config.hidden_dim, 
            lambda_min=config.lambda_entangle * 0.5,
            lambda_max=config.lambda_entangle * 1.5
        )
        self.ffn_qbnn_dropout = nn.Dropout(config.dropout)
        self.ffn_qbnn_layer2 = QBNNLayer(
            config.hidden_dim, config.embed_dim,
            lambda_min=config.lambda_entangle * 0.5,
            lambda_max=config.lambda_entangle * 1.5
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPTデコーダーフォワード
        
        Args:
            x: (batch, seq, embed_dim)
            mask: Optional attention mask
        
        Returns:
            (batch, seq, embed_dim)
        """
        # 1. Pre-norm + Multi-Head Causal Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, mask)
        x = residual + self.dropout(attn_out)
        
        # 2. Pre-norm + Feed-Forward Network + Residual
        residual = x
        x = self.norm2(x)
        
        # 標準FFN + QBNN拡張（ブレンド）
        ffn_standard_out = self.ffn_standard(x)
        # QBNN拡張FFN
        ffn_qbnn_out = self.ffn_qbnn_layer1(x)
        ffn_qbnn_out = self.ffn_qbnn_dropout(ffn_qbnn_out)
        ffn_qbnn_out = self.ffn_qbnn_layer2(ffn_qbnn_out)
        
        # ブレンド比率: 標準FFN 70% + QBNN拡張 30%
        ffn_out = 0.7 * ffn_standard_out + 0.3 * ffn_qbnn_out
        
        x = residual + ffn_out
        
        return x


# ========================================
# Part 4: Embedding（埋め込み層）
# ========================================

class OpenAIEmbeddingWrapper:
    """
    OpenAI Embedding API ラッパー
    
    テキストを直接OpenAI APIに送信してエンベディングを取得
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large", dimensions: Optional[int] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI APIが利用できません。openai>=1.0.0をインストールしてください。")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI APIキーが必要です。OPENAI_API_KEY環境変数を設定するか、api_key引数を指定してください。")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # モデルごとのデフォルト次元
        if dimensions is not None:
            self.embed_dim = dimensions
            self.dimensions = dimensions
        elif "ada-002" in model:
            self.embed_dim = 1536
            self.dimensions = None
        elif "embedding-3-large" in model:
            self.embed_dim = 3072  # text-embedding-3-largeのデフォルト次元
            self.dimensions = None
        elif "embedding-3-small" in model:
            self.embed_dim = 1536  # text-embedding-3-smallのデフォルト次元
            self.dimensions = None
        else:
            self.embed_dim = 3072  # デフォルト
            self.dimensions = None
    
    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        テキストのリストからエンベディングを取得
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ（API制限を考慮）
        
        Returns:
            (N, embed_dim) エンベディング配列
        """
        all_embeddings = []
        
        # バッチ処理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # dimensionsパラメータを指定可能（text-embedding-3-large等で使用）
                params = {
                    "model": self.model,
                    "input": batch
                }
                if self.dimensions is not None:
                    params["dimensions"] = self.dimensions
                
                response = self.client.embeddings.create(**params)
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise RuntimeError(f"OpenAI APIエラー: {e}")
        
        return np.array(all_embeddings)


class GoogleEmbeddingWrapper:
    """
    Google Text Embedding API ラッパー

    Google Generative AI (Gemini) のテキストエンベディングAPIを使用
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "models/text-embedding-004", task_type: str = "RETRIEVAL_DOCUMENT"):
        """
        Args:
            api_key: Google API キー（Noneの場合はGOOGLE_API_KEY環境変数を使用）
            model: エンベディングモデル名
                - "models/text-embedding-004": 768次元（デフォルト、最新）
                - "models/embedding-001": 768次元
            task_type: タスクタイプ
                - "RETRIEVAL_DOCUMENT": ドキュメント検索用（デフォルト）
                - "RETRIEVAL_QUERY": クエリ検索用
                - "SEMANTIC_SIMILARITY": 意味的類似度
                - "CLASSIFICATION": 分類用
        """
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google-generativeai がインストールされていません。pip install google-generativeai を実行してください。")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google APIキーが必要です。GOOGLE_API_KEY環境変数を設定するか、api_key引数を指定してください。")

        genai.configure(api_key=self.api_key)
        self.model = model
        self.task_type = task_type
        self.embed_dim = 768  # Google text-embedding-004 のデフォルト次元

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        テキストのリストからエンベディングを取得

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ

        Returns:
            (N, embed_dim) エンベディング配列
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                result = genai.embed_content(
                    model=self.model,
                    content=batch,
                    task_type=self.task_type,
                )
                # result['embedding'] はバッチの場合リストのリスト
                if isinstance(result['embedding'][0], list):
                    all_embeddings.extend(result['embedding'])
                else:
                    all_embeddings.append(result['embedding'])
            except Exception as e:
                raise RuntimeError(f"Google Embedding APIエラー: {e}")

        embeddings = np.array(all_embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self.embed_dim = embeddings.shape[-1]
        return embeddings


class NeuroQuantumEmbedding(nn.Module):
    """
    ニューロQ 埋め込み層

    Token → テキストエンベディング + 位置エンコーディング

    OpenAI/Google Embedding APIを使用するオプションあり
    """

    def __init__(
        self,
        config: NeuroQuantumConfig,
        use_openai_embedding: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-large",
        use_google_embedding: bool = False,
        google_api_key: Optional[str] = None,
        google_model: str = "models/text-embedding-004",
        tokenizer = None  # トークン化済みテキストを復元するためのトークナイザー
    ):
        super().__init__()

        self.use_openai_embedding = use_openai_embedding
        self.use_google_embedding = use_google_embedding
        self.use_external_embedding = use_openai_embedding or use_google_embedding
        self.config = config
        self.tokenizer = tokenizer

        self.openai_wrapper = None
        self.google_wrapper = None
        self.projection = None

        if use_google_embedding:
            if not GOOGLE_GENAI_AVAILABLE:
                warnings.warn("Google Generative AI APIが利用できません。従来の埋め込みを使用します。")
                self.use_google_embedding = False
                self.use_external_embedding = use_openai_embedding

            if self.use_google_embedding:
                self.google_wrapper = GoogleEmbeddingWrapper(
                    api_key=google_api_key,
                    model=google_model
                )
                actual_embed_dim = self.google_wrapper.embed_dim
                if actual_embed_dim != config.embed_dim:
                    warnings.warn(
                        f"Google Embedding次元({actual_embed_dim})が設定次元({config.embed_dim})と異なります。"
                        f"射影層を追加します。"
                    )
                    self.projection = nn.Linear(actual_embed_dim, config.embed_dim)
                else:
                    self.projection = nn.Identity()

        if use_openai_embedding and not self.use_google_embedding:
            if not OPENAI_AVAILABLE:
                warnings.warn("OpenAI APIが利用できません。従来の埋め込みを使用します。")
                self.use_openai_embedding = False
                self.use_external_embedding = False

            if self.use_openai_embedding:
                self.openai_wrapper = OpenAIEmbeddingWrapper(
                    api_key=openai_api_key,
                    model=openai_model
                )
                actual_embed_dim = self.openai_wrapper.embed_dim
                if actual_embed_dim != config.embed_dim:
                    warnings.warn(
                        f"OpenAI Embedding次元({actual_embed_dim})が設定次元({config.embed_dim})と異なります。"
                        f"射影層を追加します。"
                    )
                    self.projection = nn.Linear(actual_embed_dim, config.embed_dim)
                else:
                    self.projection = nn.Identity()

        if not self.use_external_embedding:
            # 従来のテキストエンベディング
            self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # 位置埋め込み（学習可能）- 外部Embedding使用時も必要
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # ドロップアウト
        self.dropout = nn.Dropout(config.dropout)

        # 埋め込み次元
        self.embed_dim = config.embed_dim
    
    def forward(self, token_ids: torch.Tensor, texts: Optional[List[str]] = None) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq) トークンID
            texts: OpenAI Embedding使用時に必要。トークンIDに対応するテキストのリスト
        """
        batch_size, seq_len = token_ids.shape

        # Bounds checking to prevent CUDA gather index out of bounds error
        # Clamp token_ids to valid range [0, vocab_size-1]
        token_ids = token_ids.clamp(0, self.config.vocab_size - 1)

        # Clamp sequence length to max_seq_len
        if seq_len > self.config.max_seq_len:
            token_ids = token_ids[:, :self.config.max_seq_len]
            seq_len = self.config.max_seq_len

        # 外部エンベディング（Google or OpenAI）を使用するかチェック
        external_wrapper = self.google_wrapper or self.openai_wrapper

        if self.use_external_embedding and external_wrapper is not None:
            if texts is None:
                if self.tokenizer is not None:
                    # トークンIDからテキストを復元
                    texts = []
                    for batch_idx in range(batch_size):
                        token_seq = token_ids[batch_idx].cpu().tolist()
                        text = self.tokenizer.decode(token_seq)
                        texts.append(text)
                else:
                    raise ValueError(
                        "外部Embedding使用時は、texts引数またはtokenizerが必要です。"
                    )

            # APIからエンベディングを取得
            # 注意: APIは文全体のエンベディングを返すため、
            # トークン単位ではなく文単位で処理
            embeddings_list = []
            for text in texts:
                embedding = external_wrapper.get_embeddings([text])[0]
                embeddings_list.append(embedding)

            # テンソルに変換
            text_embeds = torch.tensor(
                np.array(embeddings_list),
                device=token_ids.device,
                dtype=torch.float32
            )

            # 次元が異なる場合は射影
            text_embeds = self.projection(text_embeds)

            # シーケンス長に合わせて拡張（文全体のエンベディングを各トークンに適用）
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # 従来のテキストエンベディング
            text_embeds = self.text_embedding(token_ids)
        
        # 位置埋め込み
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        # Clamp positions to valid range [0, max_seq_len-1]
        positions = positions.clamp(0, self.config.max_seq_len - 1)
        pos_embeds = self.position_embedding(positions)

        # 合成
        embeds = text_embeds + pos_embeds
        embeds = self.dropout(embeds)

        return embeds


# ========================================
# Part 5: Output Head（出力層）
# ========================================

class NeuroQuantumHead(nn.Module):
    """
    ニューロQ 出力ヘッド
    
    ベクトル → 語彙確率への変換
    """
    
    def __init__(self, config: NeuroQuantumConfig):
        super().__init__()
        
        # 最終正規化
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 語彙への線形変換
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ========================================
# Part 5.5: State-dict migration (old → new architecture)
# ========================================

def migrate_legacy_state_dict(state_dict: dict, model: "NeuroQuantum") -> dict:
    """
    Migrate a legacy checkpoint state_dict to the current NeuroQuantum architecture.

    Legacy key format (old model):
        embed.weight, pos_embed.weight,
        layers.{i}.attn.in_proj_weight/bias, layers.{i}.attn.out_proj.weight/bias,
        layers.{i}.qbnn.J, layers.{i}.qbnn.linear.weight/bias,
        layers.{i}.qbnn.norm.weight/bias,
        layers.{i}.norm1.weight/bias, layers.{i}.norm2.weight/bias,
        head.weight, head.bias

    Current key format (NeuroQuantum):
        token_embedding.weight, position_embedding.weight,
        transformer_blocks.{i}.attention.q_proj/k_proj/v_proj/out_proj.*,
        transformer_blocks.{i}.attention.J_attn, transformer_blocks.{i}.attention.lambda_attn,
        transformer_blocks.{i}.norm1/norm2.*,
        transformer_blocks.{i}.ffn_standard/ffn_qbnn_layer1/ffn_qbnn_layer2.*,
        final_norm.*, output_head.weight
    """
    # Quick check: if state_dict already has new-style keys, return as-is
    if any(k.startswith("transformer_blocks.") for k in state_dict):
        return state_dict

    # Check for legacy keys
    if not any(k.startswith("layers.") for k in state_dict):
        return state_dict

    print("[migrate] Legacy checkpoint detected — converting keys to NeuroQuantum format")
    new_state = {}
    model_state = model.state_dict()
    embed_dim = model.config.embed_dim

    # --- Simple renames ---
    rename_map = {
        "embed.weight": "token_embedding.weight",
        "pos_embed.weight": "position_embedding.weight",
    }
    for old_key, new_key in rename_map.items():
        if old_key in state_dict and new_key in model_state:
            old_t = state_dict[old_key]
            new_t = model_state[new_key]
            if old_t.shape == new_t.shape:
                new_state[new_key] = old_t
            else:
                print(f"[migrate] Shape mismatch for {old_key} {old_t.shape} → {new_key} {new_t.shape}, skipping")

    # --- Per-layer mapping ---
    num_layers = model.config.num_layers
    for i in range(num_layers):
        prefix_old = f"layers.{i}"
        prefix_new = f"transformer_blocks.{i}"

        # norm1, norm2 (direct rename)
        for norm_name in ("norm1", "norm2"):
            for param in ("weight", "bias"):
                ok = f"{prefix_old}.{norm_name}.{param}"
                nk = f"{prefix_new}.{norm_name}.{param}"
                if ok in state_dict and nk in model_state:
                    if state_dict[ok].shape == model_state[nk].shape:
                        new_state[nk] = state_dict[ok]

        # Attention: split in_proj_weight/bias into q_proj, k_proj, v_proj
        in_proj_w = state_dict.get(f"{prefix_old}.attn.in_proj_weight")
        if in_proj_w is not None and in_proj_w.shape[0] == 3 * embed_dim:
            q_w, k_w, v_w = in_proj_w.chunk(3, dim=0)
            new_state[f"{prefix_new}.attention.q_proj.weight"] = q_w
            new_state[f"{prefix_new}.attention.k_proj.weight"] = k_w
            new_state[f"{prefix_new}.attention.v_proj.weight"] = v_w

        in_proj_b = state_dict.get(f"{prefix_old}.attn.in_proj_bias")
        if in_proj_b is not None and in_proj_b.shape[0] == 3 * embed_dim:
            q_b, k_b, v_b = in_proj_b.chunk(3, dim=0)
            new_state[f"{prefix_new}.attention.q_proj.bias"] = q_b
            new_state[f"{prefix_new}.attention.k_proj.bias"] = k_b
            new_state[f"{prefix_new}.attention.v_proj.bias"] = v_b

        # Attention out_proj
        for param in ("weight", "bias"):
            ok = f"{prefix_old}.attn.out_proj.{param}"
            nk = f"{prefix_new}.attention.out_proj.{param}"
            if ok in state_dict and nk in model_state:
                if state_dict[ok].shape == model_state[nk].shape:
                    new_state[nk] = state_dict[ok]

        # qbnn.J → attention.J_attn (reshape if possible)
        old_j = state_dict.get(f"{prefix_old}.qbnn.J")
        j_attn_key = f"{prefix_new}.attention.J_attn"
        if old_j is not None and j_attn_key in model_state:
            target_shape = model_state[j_attn_key].shape
            if old_j.shape == target_shape:
                new_state[j_attn_key] = old_j
            elif old_j.numel() == model_state[j_attn_key].numel():
                new_state[j_attn_key] = old_j.reshape(target_shape)
                print(f"[migrate] Reshaped {prefix_old}.qbnn.J {old_j.shape} → {target_shape}")

        # qbnn.linear → ffn_qbnn_layer1.W (if shapes match)
        for param in ("weight", "bias"):
            ok = f"{prefix_old}.qbnn.linear.{param}"
            # Try mapping to the QBNN layer's inner linear
            nk_candidates = [
                f"{prefix_new}.ffn_qbnn_layer1.W.{param}",
                f"{prefix_new}.ffn_qbnn_layer1.eqbnn_core.linear.{param}",
            ]
            if ok in state_dict:
                for nk in nk_candidates:
                    if nk in model_state and state_dict[ok].shape == model_state[nk].shape:
                        new_state[nk] = state_dict[ok]
                        break

        # qbnn.norm → ffn_qbnn_layer1.layer_norm (if shapes match)
        for param in ("weight", "bias"):
            ok = f"{prefix_old}.qbnn.norm.{param}"
            nk_candidates = [
                f"{prefix_new}.ffn_qbnn_layer1.layer_norm.{param}",
            ]
            if ok in state_dict:
                for nk in nk_candidates:
                    if nk in model_state and state_dict[ok].shape == model_state[nk].shape:
                        new_state[nk] = state_dict[ok]
                        break

    # --- Output head ---
    if "head.weight" in state_dict and "output_head.weight" in model_state:
        if state_dict["head.weight"].shape == model_state["output_head.weight"].shape:
            new_state["output_head.weight"] = state_dict["head.weight"]
    # head.bias is dropped (output_head has bias=False)

    # --- Final norm (not in old model — keep default init) ---

    mapped = len(new_state)
    total_new = len(model_state)
    print(f"[migrate] Mapped {mapped}/{total_new} parameters from legacy checkpoint")

    # Fill unmapped keys with current (initialized) values
    for k, v in model_state.items():
        if k not in new_state:
            new_state[k] = v

    return new_state


# ========================================
# Part 6: ニューロQ モデル本体
# ========================================

class NeuroQuantum(nn.Module):
    """
    ニューロQ: GPT型デコーダーのみのTransformer（QBNN拡張版）
    
    ===== 図4-15: GPTモデルのアーキテクチャ =====
    
    (下から上へ)
    
    [入力] トークン化されたテキスト
           ↓
    [1] トークン埋め込み層 (Token Embedding)
           ↓
    [2] 位置埋め込み層 (Position Embedding)
           ↓
    [3] ドロップアウト
           ↓
    ┌─────────────────────────────────────┐
    │   Transformerブロック × N回          │
    │  ┌───────────────────────────────┐  │
    │  │ [4] LayerNorm 1               │  │
    │  │        ↓                      │  │
    │  │ [5] Masked Multi-head Attention│  │
    │  │        ↓                      │  │
    │  │ [6] ドロップアウト             │  │
    │  │        ↓                      │  │
    │  │ (+) 残差接続                   │  │
    │  │        ↓                      │  │
    │  │ [7] LayerNorm 2               │  │
    │  │        ↓                      │  │
    │  │ [8] フィードフォワード + QBNN   │  │
    │  │        ↓                      │  │
    │  │ [9] ドロップアウト             │  │
    │  │        ↓                      │  │
    │  │ (+) 残差接続                   │  │
    │  └───────────────────────────────┘  │
    └─────────────────────────────────────┘
           ↓
    [10] 最後のLayerNorm (Final LayerNorm)
           ↓
    [11] 線形出力層 (Output Head)
           ↓
    [出力] ロジット (vocab_size次元)
    
    ============================================
    
    独自要素:
    - QBNNLayer: 量子もつれテンソル J による補正
    - QBNN-Attention: アテンションスコアへの量子補正
    - 学習可能な λ（もつれ強度）
    """
    
    def __init__(
        self,
        config: NeuroQuantumConfig,
        use_openai_embedding: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-large",
        use_google_embedding: bool = False,
        google_api_key: Optional[str] = None,
        google_model: str = "models/text-embedding-004",
        tokenizer = None
    ):
        super().__init__()
        self.config = config
        self.use_openai_embedding = use_openai_embedding
        self.use_google_embedding = use_google_embedding
        use_external = use_openai_embedding or use_google_embedding

        # ========================================
        # [1] トークン埋め込み層 (Token Embedding)
        # [2] 位置埋め込み層 (Position Embedding)
        # ========================================
        if use_external:
            self.embedding = NeuroQuantumEmbedding(
                config=config,
                use_openai_embedding=use_openai_embedding,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                use_google_embedding=use_google_embedding,
                google_api_key=google_api_key,
                google_model=google_model,
                tokenizer=tokenizer
            )
            self.token_embedding = None  # 外部Embedding使用時は不要
            self.position_embedding = self.embedding.position_embedding
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
            self.embedding = None

        # 後方互換性のためのエイリアス
        self.text_embedding = self.token_embedding
        
        # ========================================
        # [3] ドロップアウト (Embedding Dropout)
        # ========================================
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.dropout = self.embedding_dropout  # 後方互換性
        
        # ========================================
        # [4-9] Transformerブロック × N回
        # ========================================
        self.transformer_blocks = nn.ModuleList([
            QBNNTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # ========================================
        # [10] 最後のLayerNorm (Final LayerNorm)
        # ========================================
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        # ========================================
        # [11] 線形出力層 (Output Head)
        # ========================================
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # パラメータ初期化（GPT標準）
        self.apply(self._init_weights)
        
        # モデル情報
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        """GPT標準の重み初期化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None,
                verbose: bool = False) -> torch.Tensor:
        """
        GPT型デコーダーのみのTransformer フォワード（図4-15のフローに準拠）
        
        Args:
            token_ids: (batch, seq) トークンID（トークン化済みのテキスト）
            mask: Optional attention mask (Noneの場合はCausal Maskを自動生成)
            verbose: 各層の入出力を詳細にログ出力するか
        
        Returns:
            (batch, seq, vocab_size) ロジット
        """
        import logging
        logger = logging.getLogger(__name__)
        
        batch, seq = token_ids.shape

        # Bounds checking to prevent CUDA gather index out of bounds error
        # Clamp token_ids to valid range [0, vocab_size-1]
        token_ids = token_ids.clamp(0, self.config.vocab_size - 1)

        # Clamp sequence length to max_seq_len
        if seq > self.config.max_seq_len:
            token_ids = token_ids[:, :self.config.max_seq_len]
            seq = self.config.max_seq_len

        if verbose:
            logger.info("=" * 70)
            logger.info("🧠 NeuroQuantum フォワードパス開始")
            logger.info("=" * 70)
            logger.info(f"[入力] トークン化されたテキスト")
            logger.info(f"  - 形状: (batch={batch}, seq={seq})")
            logger.info(f"  - トークンID例 (先頭5): {token_ids[0, :min(5, seq)].tolist()}")
            logger.info(f"  - dtype: {token_ids.dtype}, device: {token_ids.device}")

        # ========================================
        # [1] トークン埋め込み層 + [2] 位置埋め込み層
        # ========================================
        use_external = (self.use_openai_embedding or self.use_google_embedding) and self.embedding is not None
        if use_external:
            hidden_states = self.embedding(token_ids, texts=None)
            if verbose:
                logger.info("-" * 50)
                logger.info(f"[1-2] 外部Embedding（Google/OpenAI）")
                logger.info(f"  - 出力: {hidden_states.shape}")
                logger.info(f"  - 統計: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        else:
            # [1] トークン埋め込み層
            token_embeds = self.token_embedding(token_ids)
            
            if verbose:
                logger.info("-" * 50)
                logger.info(f"[1] トークン埋め込み層 (Token Embedding)")
                logger.info(f"  - 入力: (batch={batch}, seq={seq}) トークンID")
                logger.info(f"  - 出力: {token_embeds.shape}")
                logger.info(f"  - 出力例 [0,0,:5]: {token_embeds[0, 0, :5].tolist()}")
                logger.info(f"  - 統計: mean={token_embeds.mean().item():.4f}, std={token_embeds.std().item():.4f}")
            
            # [2] 位置埋め込み層
            positions = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, -1)
            # Clamp positions to valid range [0, max_seq_len-1]
            positions = positions.clamp(0, self.config.max_seq_len - 1)
            pos_embeds = self.position_embedding(positions)
            
            if verbose:
                logger.info("-" * 50)
                logger.info(f"[2] 位置埋め込み層 (Position Embedding)")
                logger.info(f"  - 位置インデックス: 0 ~ {seq-1}")
                logger.info(f"  - 出力: {pos_embeds.shape}")
                logger.info(f"  - 出力例 [0,0,:5]: {pos_embeds[0, 0, :5].tolist()}")
                logger.info(f"  - 統計: mean={pos_embeds.mean().item():.4f}, std={pos_embeds.std().item():.4f}")
            
            # 埋め込みの合成
            hidden_states = token_embeds + pos_embeds
            
            if verbose:
                logger.info("-" * 50)
                logger.info(f"[合成] Token + Position Embedding")
                logger.info(f"  - 出力: {hidden_states.shape}")
                logger.info(f"  - 統計: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        
        # ========================================
        # [3] ドロップアウト (Embedding Dropout)
        # ========================================
        hidden_states = self.embedding_dropout(hidden_states)
        
        if verbose:
            logger.info("-" * 50)
            logger.info(f"[3] ドロップアウト (rate={self.config.dropout})")
            logger.info(f"  - 出力: {hidden_states.shape}")
            logger.info(f"  - 統計: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        
        # Causal Mask生成（maskがNoneの場合）
        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=token_ids.device)).unsqueeze(0).unsqueeze(0)
            if verbose:
                logger.info(f"[Mask] Causal Mask生成: {mask.shape}")
        
        # ========================================
        # [4-9] Transformerブロック × N回
        # ========================================
        if verbose:
            logger.info("=" * 70)
            logger.info(f"🔄 Transformerブロック × {self.config.num_layers}回")
            logger.info("=" * 70)
        
        for block_idx, block in enumerate(self.transformer_blocks):
            h_input = hidden_states.clone() if verbose else None
            hidden_states = block(hidden_states, mask)
            
            if verbose:
                logger.info(f"[Block {block_idx + 1}/{self.config.num_layers}]")
                logger.info(f"  - 入力: mean={h_input.mean().item():.4f}, std={h_input.std().item():.4f}")
                logger.info(f"  - 出力: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
                
                # QBNN量子統計情報
                try:
                    lambda_attn = block.attention.lambda_attn.item()
                    logger.info(f"  - QBNN Attention λ: {lambda_attn:.4f}")
                except Exception as e:
                    logger.debug(f"  - 量子統計取得エラー: {e}")
        
        # ========================================
        # [10] 最後のLayerNorm (Final LayerNorm)
        # ========================================
        h_before_norm = hidden_states.clone() if verbose else None
        hidden_states = self.final_norm(hidden_states)
        
        if verbose:
            logger.info("=" * 70)
            logger.info(f"[10] 最後のLayerNorm (Final LayerNorm)")
            logger.info(f"  - 入力: mean={h_before_norm.mean().item():.4f}, std={h_before_norm.std().item():.4f}")
            logger.info(f"  - 出力: {hidden_states.shape}")
            logger.info(f"  - 出力統計: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        
        # ========================================
        # [11] 線形出力層 (Output Head)
        # ========================================
        logits = self.output_head(hidden_states)
        
        if verbose:
            logger.info("-" * 50)
            logger.info(f"[11] 線形出力層 (Output Head)")
            logger.info(f"  - 入力: {hidden_states.shape}")
            logger.info(f"  - 出力: {logits.shape}")
            logger.info(f"  - ロジット統計: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            logger.info(f"  - 最大ロジット位置 [0,-1]: {logits[0, -1].argmax().item()}")
            
            # 上位5つのトークン予測
            top_k_values, top_k_indices = torch.topk(logits[0, -1], 5)
            logger.info(f"  - Top-5予測: {top_k_indices.tolist()} (logits: {[f'{v:.2f}' for v in top_k_values.tolist()]})")
            
            logger.info("=" * 70)
            logger.info("🧠 NeuroQuantum フォワードパス完了")
            logger.info("=" * 70)
        
        return logits
    
    def forward_with_details(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """
        各層の出力を詳細に返すフォワードパス
        
        Args:
            token_ids: (batch, seq) トークンID
            mask: Optional attention mask
        
        Returns:
            各層の出力を含む辞書
        """
        batch, seq = token_ids.shape
        details = {
            'input_tokens': token_ids,
            'layers': []
        }
        
        # [1-2] 埋め込み層
        use_external_emb = (self.use_openai_embedding or self.use_google_embedding) and self.embedding is not None
        if use_external_emb:
            hidden_states = self.embedding(token_ids, texts=None)
            emb_type = 'google' if self.use_google_embedding else 'openai'
            details['embedding'] = {
                'type': emb_type,
                'output': hidden_states.detach().clone(),
                'shape': hidden_states.shape,
                'mean': hidden_states.mean().item(),
                'std': hidden_states.std().item()
            }
        else:
            token_embeds = self.token_embedding(token_ids)
            details['token_embedding'] = {
                'output': token_embeds.detach().clone(),
                'shape': token_embeds.shape,
                'mean': token_embeds.mean().item(),
                'std': token_embeds.std().item()
            }
            
            positions = torch.arange(seq, device=token_ids.device).unsqueeze(0).expand(batch, -1)
            pos_embeds = self.position_embedding(positions)
            details['position_embedding'] = {
                'output': pos_embeds.detach().clone(),
                'shape': pos_embeds.shape,
                'mean': pos_embeds.mean().item(),
                'std': pos_embeds.std().item()
            }
            
            hidden_states = token_embeds + pos_embeds
            details['combined_embedding'] = {
                'output': hidden_states.detach().clone(),
                'shape': hidden_states.shape,
                'mean': hidden_states.mean().item(),
                'std': hidden_states.std().item()
            }
        
        # [3] ドロップアウト
        hidden_states = self.embedding_dropout(hidden_states)
        details['embedding_dropout'] = {
            'output': hidden_states.detach().clone(),
            'shape': hidden_states.shape,
            'mean': hidden_states.mean().item(),
            'std': hidden_states.std().item()
        }
        
        # Causal Mask生成
        if mask is None:
            mask = torch.tril(torch.ones(seq, seq, device=token_ids.device)).unsqueeze(0).unsqueeze(0)
        details['mask'] = mask
        
        # [4-9] Transformerブロック × N回
        for block_idx, block in enumerate(self.transformer_blocks):
            h_input = hidden_states.detach().clone()
            hidden_states = block(hidden_states, mask)
            
            block_details = {
                'block_idx': block_idx,
                'input': h_input,
                'output': hidden_states.detach().clone(),
                'input_mean': h_input.mean().item(),
                'output_mean': hidden_states.mean().item(),
                'input_std': h_input.std().item(),
                'output_std': hidden_states.std().item()
            }
            
            # QBNN統計
            try:
                block_details['lambda_attn'] = block.attention.lambda_attn.item()
            except:
                pass
            
            details['layers'].append(block_details)
        
        # [10] 最後のLayerNorm
        h_before_norm = hidden_states.detach().clone()
        hidden_states = self.final_norm(hidden_states)
        details['final_norm'] = {
            'input': h_before_norm,
            'output': hidden_states.detach().clone(),
            'mean': hidden_states.mean().item(),
            'std': hidden_states.std().item()
        }
        
        # [11] 線形出力層
        logits = self.output_head(hidden_states)
        details['output_head'] = {
            'output': logits.detach().clone(),
            'shape': logits.shape,
            'mean': logits.mean().item(),
            'std': logits.std().item()
        }
        
        details['logits'] = logits
        
        return details
    
    def print_architecture(self):
        """モデルアーキテクチャを図として表示"""
        print()
        print("=" * 70)
        print("🧠 NeuroQuantum アーキテクチャ (図4-15準拠)")
        print("=" * 70)
        print()
        print("  [入力] トークン化されたテキスト")
        print("         ↓")
        if self.use_google_embedding:
            print(f"  [1-2] Google Text Embedding ({self.config.embed_dim}次元)")
        elif self.use_openai_embedding:
            print(f"  [1-2] OpenAI Embedding ({self.config.embed_dim}次元)")
        else:
            print(f"  [1] トークン埋め込み層 (vocab_size={self.config.vocab_size} → embed_dim={self.config.embed_dim})")
            print("         ↓")
            print(f"  [2] 位置埋め込み層 (max_seq_len={self.config.max_seq_len} → embed_dim={self.config.embed_dim})")
        print("         ↓")
        print(f"  [3] ドロップアウト (rate={self.config.dropout})")
        print("         ↓")
        print("  ┌─────────────────────────────────────────────────────┐")
        print(f"  │   Transformerブロック × {self.config.num_layers}回 (QBNN拡張)         │")
        print("  │  ┌───────────────────────────────────────────────┐  │")
        print("  │  │ [4] LayerNorm 1                               │  │")
        print("  │  │        ↓                                      │  │")
        print(f"  │  │ [5] QBNN Multi-head Attention (heads={self.config.num_heads})     │  │")
        print("  │  │        ↓                                      │  │")
        print("  │  │ [6] ドロップアウト                             │  │")
        print("  │  │        ↓                                      │  │")
        print("  │  │ (+) 残差接続                                   │  │")
        print("  │  │        ↓                                      │  │")
        print("  │  │ [7] LayerNorm 2                               │  │")
        print("  │  │        ↓                                      │  │")
        print(f"  │  │ [8] QBNN フィードフォワード (hidden={self.config.hidden_dim})     │  │")
        print("  │  │        ↓                                      │  │")
        print("  │  │ [9] ドロップアウト                             │  │")
        print("  │  │        ↓                                      │  │")
        print("  │  │ (+) 残差接続                                   │  │")
        print("  │  └───────────────────────────────────────────────┘  │")
        print("  └─────────────────────────────────────────────────────┘")
        print("         ↓")
        print(f"  [10] 最後のLayerNorm (embed_dim={self.config.embed_dim})")
        print("         ↓")
        print(f"  [11] 線形出力層 (embed_dim={self.config.embed_dim} → vocab_size={self.config.vocab_size})")
        print("         ↓")
        print("  [出力] ロジット")
        print()
        
        # パラメータ情報
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("-" * 70)
        print("📊 パラメータ情報:")
        print(f"  - 総パラメータ数: {total_params:,}")
        print(f"  - 学習可能パラメータ数: {trainable_params:,}")
        print(f"  - vocab_size: {self.config.vocab_size}")
        print(f"  - embed_dim: {self.config.embed_dim}")
        print(f"  - hidden_dim: {self.config.hidden_dim}")
        print(f"  - num_heads: {self.config.num_heads}")
        print(f"  - num_layers: {self.config.num_layers}")
        print(f"  - max_seq_len: {self.config.max_seq_len}")
        print(f"  - dropout: {self.config.dropout}")
        print(f"  - lambda_entangle: {self.config.lambda_entangle}")
        print("=" * 70)
    
    def get_quantum_info(self) -> List[Dict]:
        """全層の量子情報を取得"""
        info = []
        for i, block in enumerate(self.transformer_blocks):
            block_info = {
                'block': i,
                'attn_lambda': block.attention.lambda_attn.item(),
            }
            info.append(block_info)
        return info


# ========================================
# Part 7: トークナイザー（SentencePiece日本語サブワード使用）
# ========================================

class NeuroQuantumTokenizer:
    """
    SentencePiece 日本語サブワードトークナイザー

    - SentencePieceによる高精度なサブワードトークン化（Unigram）
    - 語彙サイズを指定して学習可能（8000-32000推奨）
    - モデルの保存・読み込みが可能（.model形式）
    - フォールバック: 文字単位トークナイザー（SentencePiece未インストール時）
    """

    def __init__(self, vocab_size: int = 32000, model_file: str = None):
        """
        Args:
            vocab_size: 語彙サイズ（デフォルト: 16000）
            model_file: 既存のSentencePieceモデルファイルパス（.model）（Noneの場合は新規学習）
        """
        self.vocab_size = vocab_size
        self.actual_vocab_size = None
        self.model_file = model_file
        self.sp = None  # SentencePieceProcessor

        # 特殊トークン
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.bof_token = '<bof>'
        self.eof_token = '<eof>'

        # 特殊トークンID（SentencePieceのデフォルト: unk=0, bos=1, eos=2, pad=-1）
        # NeuroQ統一: pad=0, unk=1, bos=2, eos=3, bof=4, eof=5
        # BOS/EOS: シーケンス（チャンク）の開始/終了
        # BOF/EOF: ドキュメント（ファイル）の実際の開始/終了
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.bof_id = 4
        self.eof_id = 5

        # フォールバック用の語彙マッピング
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

        # 既存モデルの読み込み
        if model_file and os.path.exists(model_file):
            try:
                self._load_model(model_file)
                print(f"   ✅ SentencePieceトークナイザー読み込み: {model_file} (語彙サイズ: {self.actual_vocab_size})")
            except Exception as e:
                warnings.warn(f"SentencePieceモデルの読み込みに失敗: {e}。フォールバックを使用します。")
                self._init_fallback_vocab()
        else:
            self._init_fallback_vocab()

    def _load_model(self, path: str):
        """SentencePieceモデルを読み込み"""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("sentencepieceがインストールされていません")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(path)
        self.actual_vocab_size = self.sp.GetPieceSize()
        self.vocab_size = self.actual_vocab_size
        self.model_file = path

    def _init_fallback_vocab(self):
        """フォールバック用の語彙を特殊トークンで初期化"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.bof_token, self.eof_token]
        self.token_to_idx = {token: i for i, token in enumerate(special_tokens)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        self.actual_vocab_size = len(self.token_to_idx)

    def build_vocab(self, texts: List[str], min_freq: int = 2,
                    character_coverage: float = 0.9995, model_prefix: str = "neuroq_tokenizer"):
        """
        SentencePieceモデルを学習して語彙を構築

        Args:
            texts: 学習テキストのリスト
            min_freq: 未使用（互換性のため保持）
            character_coverage: 文字カバレッジ（日本語は0.9995推奨）
            model_prefix: 保存時のファイルプレフィックス
        """
        if not SENTENCEPIECE_AVAILABLE:
            warnings.warn("sentencepieceが利用できません。フォールバックトークナイザーを使用します。")
            self._build_vocab_fallback(texts, min_freq)
            return self

        print(f"   🔤 SentencePiece で語彙構築中... (目標語彙サイズ: {self.vocab_size})")

        import tempfile

        # 学習テキストを一時ファイルに書き出し
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for text in texts:
                text = text.strip()
                if text:
                    f.write(text + '\n')
            tmp_corpus_path = f.name

        try:
            # SentencePieceモデルを学習
            spm.SentencePieceTrainer.Train(
                input=tmp_corpus_path,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                model_type='unigram',
                character_coverage=character_coverage,
                pad_id=self.pad_id,
                unk_id=self.unk_id,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_piece=self.pad_token,
                unk_piece=self.unk_token,
                bos_piece=self.bos_token,
                eos_piece=self.eos_token,
                user_defined_symbols=['<USER>', '<ASSISTANT>', '<SYSTEM>', '<bof>', '<eof>'],
                train_extremely_large_corpus=False,
            )

            # 学習したモデルを読み込み
            model_path = model_prefix + '.model'
            self._load_model(model_path)

            print(f"   ✅ SentencePiece語彙構築完了 (語彙サイズ: {self.actual_vocab_size})")

        finally:
            # 一時ファイルを削除
            os.unlink(tmp_corpus_path)

        return self

    def _build_vocab_fallback(self, texts: List[str], min_freq: int = 2):
        """フォールバック：文字単位の語彙構築"""
        print(f"   🔤 フォールバック語彙構築中...")
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)

        self._init_fallback_vocab()

        for char, freq in char_freq.most_common():
            if len(self.token_to_idx) >= self.vocab_size:
                break
            if freq >= min_freq and char not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[char] = idx
                self.idx_to_token[idx] = char

        self.actual_vocab_size = len(self.token_to_idx)
        self.vocab_size = self.actual_vocab_size
        print(f"   ✅ 語彙サイズ: {self.actual_vocab_size}")

    def encode(self, text: str, add_special: bool = True, add_boundary: bool = False, verbose: bool = False) -> List[int]:
        """
        テキストをトークンIDのリストに変換

        Args:
            text: 入力テキスト
            add_special: 特殊トークン（BOS/EOS）を追加するか
            add_boundary: ドキュメント境界トークン（BOF/EOF）を追加するか
            verbose: 詳細ログを出力するか

        Returns:
            トークンIDのリスト
        """
        import logging
        logger = logging.getLogger(__name__)

        if self.sp is not None:
            # SentencePieceでエンコード
            if add_special:
                tokens = [self.bos_id] + self.sp.EncodeAsIds(text) + [self.eos_id]
            else:
                tokens = self.sp.EncodeAsIds(text)
        else:
            # フォールバック：文字単位
            chars = list(text)
            tokens = []
            if add_special:
                tokens.append(self.bos_id)
            for ch in chars:
                tokens.append(self.token_to_idx.get(ch, self.unk_id))
            if add_special:
                tokens.append(self.eos_id)

        # ドキュメント境界トークンを追加（BOF/EOF）
        if add_boundary:
            tokens = [self.bof_id] + tokens + [self.eof_id]

        if verbose:
            logger.debug(f"[Encode] 入力テキスト: '{text[:50]}...'" if len(text) > 50 else f"[Encode] 入力テキスト: '{text}'")
            logger.debug(f"[Encode] トークン数: {len(tokens)}")
            logger.debug(f"[Encode] トークンID: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True, verbose: bool = False) -> str:
        """
        トークンIDのリストをテキストに復元

        Args:
            token_ids: トークンIDのリスト
            skip_special: 特殊トークンをスキップするか
            verbose: 詳細ログを出力するか

        Returns:
            デコードされたテキスト
        """
        import logging
        logger = logging.getLogger(__name__)

        if self.sp is not None:
            # 特殊トークンをフィルタリング
            sp_vocab_size = self.sp.GetPieceSize()
            if skip_special:
                special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.bof_id, self.eof_id}
                filtered_ids = [t for t in token_ids if t not in special_ids and 0 <= t < sp_vocab_size]
            else:
                filtered_ids = [t for t in token_ids if 0 <= t < sp_vocab_size]
            result = self.sp.DecodeIds(filtered_ids)
        else:
            # フォールバック：文字単位
            tokens = []
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.bof_id, self.eof_id}
            for t in token_ids:
                if skip_special and t in special_ids:
                    continue
                token = self.idx_to_token.get(t, self.unk_token)
                if token not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.bof_token, self.eof_token]:
                    tokens.append(token)
            result = ''.join(tokens)

        if verbose:
            logger.debug(f"[Decode] 入力トークン数: {len(token_ids)}")
            logger.debug(f"[Decode] 出力テキスト: '{result[:100]}...'" if len(result) > 100 else f"[Decode] 出力テキスト: '{result}'")

        return result

    def get_tokenization_info(self, text: str) -> Dict:
        """
        トークン化の詳細情報を取得

        Args:
            text: 入力テキスト

        Returns:
            トークン化の詳細情報を含む辞書
        """
        tokens = self.encode(text, add_special=False)

        info = {
            'input_text': text,
            'input_length': len(text),
            'token_count': len(tokens),
            'token_ids': tokens,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
        }

        if self.sp is not None:
            info['tokens'] = self.sp.EncodeAsPieces(text)
        else:
            info['tokens'] = [self.idx_to_token.get(t, self.unk_token) for t in tokens]
        info['unk_count'] = sum(1 for t in tokens if t == self.unk_id)
        info['unk_rate'] = info['unk_count'] / len(tokens) if tokens else 0

        return info

    def print_tokenization(self, text: str):
        """
        トークン化の詳細をプリント

        Args:
            text: 入力テキスト
        """
        info = self.get_tokenization_info(text)
        print("=" * 60)
        print("🔤 トークン化詳細")
        print("=" * 60)
        print(f"入力テキスト: '{info['input_text']}'")
        print(f"入力文字数: {info['input_length']}")
        print(f"トークン数: {info['token_count']}")
        print(f"圧縮率: {info['compression_ratio']:.2f}")
        print(f"UNKトークン数: {info['unk_count']} ({info['unk_rate']*100:.1f}%)")
        print("-" * 60)
        print("トークン分割:")
        for i, (tid, token) in enumerate(zip(info['token_ids'], info['tokens'])):
            print(f"  [{i:3d}] ID={tid:5d} -> '{token}'")
        print("=" * 60)

    def save(self, path: str):
        """
        モデルを保存

        SentencePieceモデルの場合は.modelファイルをコピー。
        フォールバックの場合はJSON形式で保存。
        """
        if self.sp is not None and self.model_file:
            # SentencePieceモデルファイルをコピー
            import shutil
            target = path if path.endswith('.model') else path + '.model'
            if os.path.abspath(self.model_file) != os.path.abspath(target):
                shutil.copy2(self.model_file, target)
        else:
            # フォールバック：JSON保存
            json_path = path if path.endswith('.json') else path + '.json'
            data = {
                'tokenizer_type': 'sentencepiece_fallback',
                'token_to_idx': self.token_to_idx,
                'vocab_size': self.vocab_size,
                'actual_vocab_size': self.actual_vocab_size,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """読み込み"""
        # .model ファイルを試す
        model_path = path if path.endswith('.model') else path + '.model'
        if os.path.exists(model_path) and SENTENCEPIECE_AVAILABLE:
            self._load_model(model_path)
            return self

        # .model なしのパスも試す（拡張子なしで .model がある場合）
        if not path.endswith('.model') and os.path.exists(path):
            try:
                self._load_model(path)
                return self
            except Exception:
                pass

        # JSON フォールバック
        json_path = path + '.json' if not path.endswith('.json') else path
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.token_to_idx = data['token_to_idx']
            self.idx_to_token = {int(i): token for token, i in self.token_to_idx.items()}
            self.vocab_size = data.get('vocab_size', len(self.token_to_idx))
            self.actual_vocab_size = data.get('actual_vocab_size', len(self.token_to_idx))
            self.model_file = json_path
            return self

        raise FileNotFoundError(f"トークナイザーモデルが見つかりません: {path}")


# ========================================
# Part 8: ニューロQ AI（生成AI本体）
# ========================================

class NeuroQuantumAI:
    """
    ニューロQ AI

    QBNN-LLM による生成AI

    ニューロン数（hidden_dim）を指定可能

    OpenAI Embedding使用例:
        ai = NeuroQuantumAI(
            embed_dim=3072,
            use_openai_embedding=True,
            openai_api_key="sk-...",
            openai_model="text-embedding-3-large"
        )

    Google Text Embedding使用例:
        ai = NeuroQuantumAI(
            embed_dim=768,  # text-embedding-004の次元
            use_google_embedding=True,
            google_api_key="AIza...",  # または環境変数GOOGLE_API_KEY
            google_model="models/text-embedding-004"
        )

        # 従来の埋め込みを使用する場合（デフォルト）
        ai = NeuroQuantumAI(embed_dim=48)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 2048,       # ニューロン数（FFN層の次元）
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 10000,
        dropout: float = 0.1,
        lambda_entangle: float = 0.5,
        use_openai_embedding: bool = False,  # OpenAI Embeddingを使用するか
        openai_api_key: Optional[str] = None,  # OpenAI APIキー
        openai_model: str = "text-embedding-3-large",  # OpenAI Embeddingモデル
        use_google_embedding: bool = False,  # Google Text Embeddingを使用するか
        google_api_key: Optional[str] = None,  # Google APIキー
        google_model: str = "models/text-embedding-004",  # Google Embeddingモデル
    ):
        # デバイス選択: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Apple Silicon GPU (MPS) を使用")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🎮 NVIDIA GPU (CUDA) を使用")
        else:
            self.device = torch.device("cpu")
            print("💻 CPU を使用")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.lambda_entangle = lambda_entangle
        self.use_openai_embedding = use_openai_embedding
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.use_google_embedding = use_google_embedding
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.google_model = google_model

        self.tokenizer: Optional[NeuroQuantumTokenizer] = None
        self.model: Optional[NeuroQuantum] = None
        self.config: Optional[NeuroQuantumConfig] = None

        # 量子回路シミュレーター
        self.quantum_computer: Optional[QuantumComputer] = None
        self.use_quantum_simulation = QUANTUM_COMPUTER_AVAILABLE
        if self.use_quantum_simulation:
            self.quantum_computer = QuantumComputer("NeuroQuantum-QC")
            print("⚛️  量子回路シミュレーターを初期化しました")
    
    def train(self, texts: List[str], epochs: int = 50, batch_size: int = 16,
              lr: float = 0.001, seq_len: int = 128, vocab_size: int = 32000):
        """学習"""
        print("\n" + "=" * 70)
        print("📚 ニューロQ 学習開始")
        print("=" * 70)
        
        # トークナイザー構築（fugashi日本語形態素解析使用）
        print("\n🔤 トークナイザー構築...")

        # 既存の語彙ファイルを探す
        tokenizer_model_paths = [
            "neuroq_tokenizer.json",  # カレントディレクトリ（推奨）
            "neuroq_tokenizer_8k.json",  # カレントディレクトリ（旧名称）
            "../neuroq_tokenizer.json",  # 親ディレクトリ
            "../neuroq_tokenizer_8k.json",  # 親ディレクトリ（旧名称）
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.json"),  # スクリプトと同じディレクトリ
            os.path.join(os.path.dirname(__file__), "neuroq_tokenizer_8k.json"),  # スクリプトと同じディレクトリ（旧名称）
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuroq_tokenizer.json"),  # 親の親ディレクトリ
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuroq_tokenizer_8k.json"),  # 親の親ディレクトリ（旧名称）
        ]

        existing_model = None
        for path in tokenizer_model_paths:
            if os.path.exists(path):
                existing_model = path
                break

        if existing_model:
            # 既存の語彙ファイルを使用
            print(f"   既存の日本語トークナイザーを使用: {existing_model}")
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=vocab_size, model_file=existing_model)
        else:
            # 新規に語彙を構築
            print("   新規に語彙を構築します...")
            self.tokenizer = NeuroQuantumTokenizer(vocab_size=vocab_size)
            self.tokenizer.build_vocab(texts, model_prefix="neuroq_tokenizer", min_freq=2)

        print(f"   語彙サイズ: {self.tokenizer.actual_vocab_size}")
        
        # モデル構築
        print("\n🧠 ニューロQモデル構築...")
        
        # 外部Embedding使用時は埋め込み次元を調整
        if self.use_google_embedding:
            # Google Text Embeddingの次元（768）
            actual_embed_dim = 768
            if self.embed_dim != actual_embed_dim:
                print(f"   Google Embedding次元({actual_embed_dim})に合わせて調整")
                self.embed_dim = actual_embed_dim
        elif self.use_openai_embedding:
            # OpenAI Embeddingの次元を使用
            if "ada-002" in self.openai_model:
                actual_embed_dim = 1536
            elif "embedding-3-large" in self.openai_model:
                actual_embed_dim = 3072
            elif "embedding-3-small" in self.openai_model:
                actual_embed_dim = 1536
            else:
                actual_embed_dim = 3072  # デフォルト
            if self.embed_dim != actual_embed_dim:
                print(f"   OpenAI Embedding次元({actual_embed_dim})に合わせて調整")
                self.embed_dim = actual_embed_dim

        self.config = NeuroQuantumConfig(
            vocab_size=self.tokenizer.actual_vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            lambda_entangle=self.lambda_entangle,
        )

        self.model = NeuroQuantum(
            config=self.config,
            use_openai_embedding=self.use_openai_embedding,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
            use_google_embedding=self.use_google_embedding,
            google_api_key=self.google_api_key,
            google_model=self.google_model,
            tokenizer=self.tokenizer
        ).to(self.device)

        print(f"\n📊 モデル構成:")
        if self.use_google_embedding:
            print(f"   埋め込み: Google Text Embedding ({self.google_model})")
        elif self.use_openai_embedding:
            print(f"   埋め込み: OpenAI Embedding ({self.openai_model})")
        else:
            print(f"   埋め込み: 従来のEmbedding層")
        print(f"   埋め込み次元: {self.embed_dim}")
        print(f"   隠れ層次元: {self.hidden_dim}")
        print(f"   アテンションヘッド: {self.num_heads}")
        print(f"   Transformerブロック: {self.num_layers}")
        print(f"   もつれ強度 λ: {self.lambda_entangle}")
        print(f"   総パラメータ数: {self.model.num_params:,}")
        
        # データ準備
        print("\n📊 データ準備...")
        # Each text is tokenized separately with BOS/EOS boundaries preserved.
        # This ensures the model learns proper sentence start/end patterns.
        sequences = []
        total_tokens = 0
        max_content = seq_len - 2  # Reserve 2 slots for BOS and EOS
        if max_content <= 0:
            max_content = seq_len  # Fallback for very short seq_len

        for text in texts:
            content_ids = self.tokenizer.encode(text, add_special=False)
            total_tokens += len(content_ids)
            if len(content_ids) <= max_content:
                if len(content_ids) >= 2:
                    # 単一チャンク: BOF + BOS + content + EOS + EOF
                    # ただしmax_content制約があるためBOF/EOFはBOS/EOSの外側に付与
                    seq = [self.tokenizer.bof_id, self.tokenizer.bos_id] + content_ids + [self.tokenizer.eos_id, self.tokenizer.eof_id]
                    # Pad to seq_len + 1 for (input, target) pairs
                    pad_len = (seq_len + 1) - len(seq)
                    if pad_len > 0:
                        seq = seq + [self.tokenizer.pad_id] * pad_len
                    x = torch.tensor(seq[:seq_len], dtype=torch.long)
                    y = torch.tensor(seq[1:seq_len + 1], dtype=torch.long)
                    # Mask padding in targets
                    y[y == self.tokenizer.pad_id] = -100
                    sequences.append((x, y))
            else:
                stride = max(max_content // 2, 1)
                chunks = list(range(0, len(content_ids) - max_content + 1, stride))
                for idx, start in enumerate(chunks):
                    chunk = content_ids[start:start + max_content]
                    # 先頭チャンクにBOF、末尾チャンクにEOFを付与
                    prefix = [self.tokenizer.bof_id, self.tokenizer.bos_id] if idx == 0 else [self.tokenizer.bos_id]
                    suffix = [self.tokenizer.eos_id, self.tokenizer.eof_id] if idx == len(chunks) - 1 else [self.tokenizer.eos_id]
                    seq = prefix + chunk + suffix
                    x = torch.tensor(seq[:seq_len], dtype=torch.long)
                    y = torch.tensor(seq[1:seq_len + 1], dtype=torch.long)
                    sequences.append((x, y))

        print(f"   総トークン数: {total_tokens:,}")
        print(f"   シーケンス数: {len(sequences):,}")
        
        # 学習
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print("\n🚀 学習ループ...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(sequences)
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                if len(batch) == 0:
                    continue
                
                x_batch = torch.stack([s[0] for s in batch]).to(self.device)
                y_batch = torch.stack([s[1] for s in batch]).to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                
                # vocab_sizeを取得
                vocab_size = self.tokenizer.actual_vocab_size if self.tokenizer.actual_vocab_size else self.tokenizer.vocab_size
                loss = criterion(
                    logits.view(-1, vocab_size),
                    y_batch.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            num_batches = max(1, len(sequences) // batch_size)
            avg_loss = total_loss / num_batches

            logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - lr: {scheduler.get_last_lr()[0]:.6f} - batches: {num_batches}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}")

        logger.info(f"Training complete - final_loss: {avg_loss:.6f} - epochs: {epochs} - sequences: {len(sequences)}")
        print("\n✅ 学習完了！")
        
        # 量子情報
        print("\n⚛️ 量子もつれ情報:")
        for info in self.model.get_quantum_info():
            print(f"   Block {info['block']}: λ_attn = {info['attn_lambda']:.4f}")

    def train_on_texts(self, texts: List[str], epochs: int = 50, batch_size: int = 16,
                       lr: float = 0.001, seq_len: int = 64):
        """
        train()へのエイリアス（後方互換性のため）
        
        Args:
            texts: 学習用テキストのリスト
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            seq_len: シーケンス長
        """
        return self.train(texts, epochs=epochs, batch_size=batch_size, lr=lr, seq_len=seq_len)

    def _get_conjunction_token_ids(self) -> set:
        """
        接続詞トークンIDのセットを返す。
        生成テキストが接続詞で始まることを防ぐために使用。
        """
        if hasattr(self, '_conjunction_ids_cache'):
            return self._conjunction_ids_cache

        conjunctions = [
            # 逆接
            "しかし", "だが", "けれども", "けれど", "けど", "ところが",
            "でも", "だけど", "ただし", "もっとも", "にもかかわらず",
            # 順接・因果
            "だから", "したがって", "そのため", "よって", "ゆえに",
            "それで", "そこで", "すると",
            # 添加・並列
            "そして", "また", "さらに", "しかも", "そのうえ",
            "それに", "および", "ならびに", "かつ",
            # 転換
            "ところで", "さて", "では", "それでは", "ちなみに",
            "なお", "一方",
            # 説明・補足
            "つまり", "すなわち", "要するに", "いわば",
            # 選択
            "あるいは", "または", "もしくは", "ないし",
            # その他
            "それから", "次に", "なぜなら", "というのは",
            "むしろ", "とはいえ", "それにしても",
        ]

        conjunction_ids = set()
        if self.tokenizer.sp is not None:
            for conj in conjunctions:
                # SentencePieceのEncodeAsIds で先頭トークンを取得
                ids = self.tokenizer.sp.EncodeAsIds(conj)
                if ids:
                    conjunction_ids.add(ids[0])
                # "▁" (SentencePiece の空白マーカー) 付きも確認
                ids_with_space = self.tokenizer.sp.EncodeAsIds("▁" + conj)
                if ids_with_space:
                    conjunction_ids.add(ids_with_space[0])
        else:
            # フォールバックトークナイザーの場合
            for conj in conjunctions:
                if conj in self.tokenizer.token_to_idx:
                    conjunction_ids.add(self.tokenizer.token_to_idx[conj])

        # 特殊トークンは除外
        conjunction_ids.discard(self.tokenizer.pad_id)
        conjunction_ids.discard(self.tokenizer.unk_id)
        conjunction_ids.discard(self.tokenizer.bos_id)
        conjunction_ids.discard(self.tokenizer.eos_id)
        conjunction_ids.discard(self.tokenizer.bof_id)
        conjunction_ids.discard(self.tokenizer.eof_id)

        self._conjunction_ids_cache = conjunction_ids
        return conjunction_ids

    def _quantum_circuit_influence(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        """
        量子回路シミュレーションを使用してlogitsに影響を与える

        Args:
            logits: 次トークンのlogits
            step: 現在の生成ステップ

        Returns:
            量子的に調整されたlogits
        """
        if not self.use_quantum_simulation or self.quantum_computer is None:
            return logits

        # 3量子ビット回路を作成（ステップごとにユニークな回路を構築）
        n_qubits = 3
        circuit_name = f"generation_step_{step}"
        qc = self.quantum_computer.create_circuit(circuit_name, n_qubits)

        # プロンプト内容に基づいて量子ゲートを適用
        # ステップ数に応じて異なる量子回路パターンを生成
        if step % 4 == 0:
            # ベル状態を作成
            qc.h(0).cnot(0, 1).cnot(1, 2)
        elif step % 4 == 1:
            # GHZ状態を作成
            qc.h(0).cnot(0, 1).cnot(0, 2)
        elif step % 4 == 2:
            # W状態風の重ね合わせ
            qc.h(0).h(1).cnot(0, 2)
        else:
            # 回転ゲートを使用
            theta = step * 0.1
            qc.ry(0, theta).ry(1, theta * 1.5).cnot(0, 1)

        # 量子回路を測定（複数回実行して統計を取得）
        results = qc.run(shots=100)

        # 測定結果の確率分布を計算
        total_shots = sum(results.values())
        probs = {state: count / total_shots for state, count in results.items()}

        # 量子測定結果をlogitsの調整に使用
        # ビット列を数値に変換し、logitsに量子的な影響を与える
        quantum_influence = torch.zeros_like(logits)
        for state, prob in probs.items():
            # ビット列を整数に変換（例: "101" -> 5）
            state_value = int(state, 2)
            # 状態値を使ってlogitsの一部を調整
            # vocab_sizeに対してモジュロを取り、循環的に影響を与える
            vocab_size = logits.size(0)
            for i in range(0, vocab_size, 8):  # 8トークンごとに影響
                idx = (i + state_value) % vocab_size
                quantum_influence[idx] += prob * 0.3  # 量子的な影響の強度

        # 元のlogitsに量子的な影響を加算
        adjusted_logits = logits + quantum_influence

        return adjusted_logits

    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temp_min: float = 0.4,       # 温度の下限
        temp_max: float = 0.8,       # 温度の上限
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,  # 調整（1.1〜1.3の範囲）
        no_repeat_ngram_size: int = 3,  # N-gram重複防止
        temperature: float = None,   # 後方互換性のため（指定された場合temp_min/temp_maxを自動計算）
    ) -> str:
        """
        テキスト生成（改善版）
        
        改善点:
        - 強化された繰り返しペナルティ
        - N-gram重複防止
        - より安定したサンプリング
        
        Args:
            temperature: 後方互換性のための単一温度パラメータ。
                        指定された場合、temp_min = temperature * 0.8, 
                        temp_max = temperature * 1.2 に自動変換されます。
        """
        # 後方互換性: temperature が指定された場合、temp_min/temp_max に変換
        if temperature is not None:
            temp_min = temperature * 0.8
            temp_max = temperature * 1.2
        
        if self.model is None:
            # 自動的にサンプルデータで学習を実行
            print("⚠️ モデルが未学習です。サンプルデータで自動学習を開始...")
            sample_data = [
                "人工知能は、人間の知能を模倣するコンピュータシステムです。機械学習やディープラーニングなどの技術を使用して、データからパターンを学習し、予測や判断を行います。",
                "量子コンピュータは、量子力学の原理を利用した次世代のコンピュータです。従来のコンピュータでは解けない複雑な問題を高速に解くことができます。",
                "自然言語処理は、コンピュータが人間の言語を理解し、生成するための技術です。翻訳、要約、質問応答などのタスクに使用されます。",
                "ニューラルネットワークは、人間の脳の神経細胞の働きを模倣した計算モデルです。層状に接続されたノードで構成され、データから特徴を学習します。",
                "プログラミングは、コンピュータに指示を与えるための言語を使ってソフトウェアを作成する技術です。Python、JavaScript、Javaなど多くの言語があります。",
                "データサイエンスは、大量のデータから有用な情報を抽出し、ビジネスや研究に活用する学問分野です。統計学、機械学習、可視化などの手法を組み合わせます。",
                "クラウドコンピューティングは、インターネット経由でコンピュータリソースを提供するサービスです。AWS、Azure、GCPなどのプラットフォームが代表的です。",
                "ブロックチェーンは、分散型台帳技術の一種で、データの改ざんを防ぐ仕組みを持っています。暗号通貨や契約管理などに応用されています。",
            ]
            self.train(sample_data, epochs=3)  # 軽量な学習
            print("✅ 自動学習完了")
        
        self.model.eval()
        
        # プロンプトエンコード（特殊タグなし、シンプルに）
        tokens = self.tokenizer.encode(prompt, add_special=True)
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        generated = tokens[0].tolist()[1:]  # BOSを除く
        
        # 重複防止用のN-gram履歴
        ngram_history = []
        
        with torch.no_grad():
            for step in range(max_length):
                # 最新のmax_seq_lenトークンを使用
                input_tokens = tokens[:, -self.max_seq_len:] if tokens.size(1) > self.max_seq_len else tokens
                
                logits = self.model(input_tokens)
                next_logits = logits[0, -1, :].clone()

                # ⚛️ 量子回路シミュレーションの影響を適用
                next_logits = self._quantum_circuit_influence(next_logits, step)

                # 🚫 生成開始時の接続詞抑制
                if step == 0:
                    conjunction_ids = self._get_conjunction_token_ids()
                    for token_id in conjunction_ids:
                        if token_id < next_logits.size(0):
                            next_logits[token_id] = float('-inf')

                # 強化された繰り返しペナルティ（温度調整の前に適用）
                # 最近のトークンに対する強力なペナルティ（recency-weighted）
                vocab_size = next_logits.size(-1)
                if len(generated) > 0:
                    # 最近100トークンに重複ペナルティ（50 → 100に拡大）
                    window_size = min(100, len(generated))
                    recent_tokens = generated[-window_size:]

                    # トークンごとの出現位置を記録（recency tracking）
                    token_positions = {}
                    for pos, token_id in enumerate(recent_tokens):
                        if token_id not in token_positions:
                            token_positions[token_id] = []
                        token_positions[token_id].append(pos)

                    for token_id, positions in token_positions.items():
                        if token_id >= vocab_size:  # bounds check to prevent CUDA gather error
                            continue
                        count = len(positions)
                        # 最も新しい出現位置（0が最古、window_size-1が最新）
                        most_recent_pos = max(positions)

                        # Recency weight: 最近のトークンほど強くペナルティ
                        # 0.5（最古） 〜 1.0（最新）
                        recency_weight = 0.5 + 0.5 * (most_recent_pos / max(window_size - 1, 1))

                        # 頻出度とrecencyを組み合わせたペナルティ
                        penalty = repetition_penalty ** (1 + count * 0.3 * recency_weight)
                        next_logits[token_id] /= penalty

                # 動的温度: θが動けるように範囲内で変化させる
                theta_phase = step * 0.2  # 位相（滑らかに）
                temperature = temp_min + (temp_max - temp_min) * (0.5 + 0.5 * math.sin(theta_phase))
                temperature = max(temp_min, min(temp_max, temperature))  # 範囲内に制限

                # 温度調整
                next_logits = next_logits / max(temperature, 0.1)  # ゼロ除算防止
                
                # N-gram重複防止
                if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
                    # 現在のN-gram prefix（次のトークンを除く）
                    current_ngram_prefix = tuple(generated[-(no_repeat_ngram_size-1):])

                    # 過去に同じN-gram prefixが出現した位置を探す
                    banned_tokens = set()
                    for i in range(len(generated) - no_repeat_ngram_size + 1):
                        # i番目から始まるN-gram prefix
                        prev_ngram_prefix = tuple(generated[i:i + no_repeat_ngram_size - 1])

                        # 現在のprefixと一致する場合、次のトークンをbanリストに追加
                        if prev_ngram_prefix == current_ngram_prefix:
                            next_token_id = generated[i + no_repeat_ngram_size - 1]
                            banned_tokens.add(next_token_id)

                    # banされたトークンに強力なペナルティを適用
                    if banned_tokens:
                        for token_id in banned_tokens:
                            if token_id < vocab_size:  # bounds check to prevent CUDA gather error
                                next_logits[token_id] = float('-inf')
                
                # Top-K（より厳格に）
                if top_k > 0:
                    top_k_actual = min(top_k, (next_logits != float('-inf')).sum().item())
                    if top_k_actual > 0:
                        top_k_vals, _ = torch.topk(next_logits, top_k_actual)
                        threshold = top_k_vals[-1]
                        indices_to_remove = next_logits < threshold
                        next_logits[indices_to_remove] = float('-inf')
                
                # Top-P（より厳格に）
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')
                
                # サンプリング
                # 確率が0のトークンを除外
                valid_logits = next_logits[next_logits != float('-inf')]
                if len(valid_logits) == 0:
                    break
                
                probs = F.softmax(next_logits, dim=-1)
                # 数値安定性のため、NaNチェック
                if torch.isnan(probs).any():
                    probs = F.softmax(next_logits.fill_(0.0), dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                
                # EOS/EOF検出
                if next_token_id in (self.tokenizer.eos_id, self.tokenizer.eof_id):
                    break

                # PAD/BOFトークンはスキップ
                if next_token_id in (self.tokenizer.pad_id, self.tokenizer.bof_id):
                    continue
                
                generated.append(next_token_id)
                ngram_history.append(tuple(generated[-no_repeat_ngram_size:]) if len(generated) >= no_repeat_ngram_size else tuple(generated))
                
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                
                # 長すぎる場合は終了
                if len(generated) >= max_length:
                    break
        
        # デコード（BOS/EOSを除く）
        decoded = self.tokenizer.decode(generated, skip_special=True)
        return decoded.strip()

    def enable_translation_pipeline(
        self,
        translation_model: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
    ) -> 'TranslatedNeuroQuantumAI':
        """
        翻訳パイプラインを有効化

        日本語入力 → 英語翻訳 → AI生成 → 日本語翻訳 → 出力
        のパイプラインを構築

        Args:
            translation_model: NLLB翻訳モデル名
            device: 使用するデバイス

        Returns:
            TranslatedNeuroQuantumAI インスタンス

        Usage:
            ai = NeuroQuantumAI()
            ai.train(english_texts)  # 英語データで学習
            translated_ai = ai.enable_translation_pipeline()
            response = translated_ai.generate("こんにちは")  # 日本語で対話可能
        """
        if not TRANSLATION_PIPELINE_AVAILABLE:
            raise ImportError(
                "Translation pipeline is not available. "
                "Make sure translation_pipeline.py is in the same directory "
                "and required dependencies (transformers, tiktoken) are installed."
            )

        return TranslatedNeuroQuantumAI(
            neuroq_model=self,
            translation_model=translation_model,
            device=device,
        )

    def chat(self):
        """対話モード"""
        print("\n" + "=" * 70)
        print("💬 ニューロQ チャットモード")
        print("=" * 70)
        print("\nコマンド:")
        print("  /quit         - 終了")
        print("  /temp <min> <max> - 温度範囲 (例: /temp 0.4 0.8)")
        print("  /len <値>     - 生成長さ (10-500)")
        print("  /info         - モデル情報")
        print("  /quantum      - 量子もつれ情報")
        print("-" * 70)
        
        temp_min = 0.4  # 温度の下限
        temp_max = 0.8  # 温度の上限
        max_length = 100
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    print("👋 さようなら！")
                    break
                
                if user_input.startswith('/temp '):
                    try:
                        parts = user_input.split()
                        if len(parts) >= 3:
                            temp_min = float(parts[1])
                            temp_max = float(parts[2])
                            temp_min = max(0.1, min(1.0, temp_min))
                            temp_max = max(0.1, min(1.0, temp_max))
                            if temp_min > temp_max:
                                temp_min, temp_max = temp_max, temp_min
                            print(f"   温度範囲を {temp_min:.2f} - {temp_max:.2f} に設定（θが動ける）")
                        else:
                            print("   使い方: /temp <最小> <最大> (例: /temp 0.4 0.8)")
                    except:
                        print("   エラー: /temp <最小> <最大>")
                    continue
                
                if user_input.startswith('/len '):
                    try:
                        max_length = int(user_input.split()[1])
                        max_length = max(10, min(500, max_length))
                        print(f"   生成長さを {max_length} に設定")
                    except:
                        print("   エラー: /len <数値>")
                    continue
                
                if user_input == '/info':
                    print(f"\n📊 ニューロQ モデル情報:")
                    print(f"   語彙サイズ: {self.tokenizer.actual_vocab_size}")
                    print(f"   埋め込み次元: {self.embed_dim}")
                    print(f"   隠れ層次元: {self.hidden_dim}")
                    print(f"   アテンションヘッド: {self.num_heads}")
                    print(f"   Transformerブロック: {self.num_layers}")
                    print(f"   総パラメータ数: {self.model.num_params:,}")
                    continue
                
                if user_input == '/quantum':
                    print(f"\n⚛️ 量子もつれ情報:")
                    for info in self.model.get_quantum_info():
                        print(f"   Block {info['block']}: λ_attn = {info['attn_lambda']:.4f}")
                    continue
                
                # 生成
                print(f"\n🤖 ニューロQ: ", end="", flush=True)
                response = self.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temp_min=temp_min,
                    temp_max=temp_max
                )
                
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 中断されました")
                break
            except Exception as e:
                print(f"   エラー: {e}")
    
    def save_tokenizer(self, path: str):
        """
        トークナイザーのみを保存

        注意: モデルの重みは保存しません。
        モデルは毎回初期化してトレーニングしてください。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # トークナイザーを保存
        self.tokenizer.save(path + '_tokenizer')
        print(f"✅ トークナイザーを保存: {path}_tokenizer.json")

    # 注意: モデルの重み保存/読み込み機能は削除されました
    # 理由: 学習済み日本語トークナイザーのみを使用し、
    #      モデル重みは毎回初期化するため
    #
    # 使い方:
    #   1. トークナイザーをロード:
    #      tokenizer = NeuroQuantumTokenizer(model_file='neuroq_tokenizer.json')
    #   2. モデルを初期化:
    #      model = NeuroQuantumAI(vocab_size=tokenizer.vocab_size, ...)
    #   3. 学習:
    #      model.train(data, epochs=10)


# ========================================
# 学習データ
# ========================================

def load_huggingface_data(max_samples: int = 500) -> List[str]:
    """Hugging Faceから対話データを取得"""
    print("\n📥 Hugging Faceからデータを取得中...")
    
    try:
        from dataset_utils import safe_load_dataset
    except ImportError:
        try:
            from datasets import load_dataset as safe_load_dataset
        except ImportError:
            print("   ⚠️ datasetsライブラリがありません。pip install datasetsを実行してください。")
            return []
    
    formatted_texts = []
    
    # 1. OpenAssistant/oasst1 - 高品質な対話データ
    try:
        print("   📚 OpenAssistant/oasst1 を読み込み中...")
        dataset = safe_load_dataset("OpenAssistant/oasst1", split="train")
        
        # 対話ツリーから質問-回答ペアを抽出
        messages_by_parent = {}
        root_messages = []
        
        for item in dataset:
            parent_id = item.get('parent_id')
            msg_id = item.get('message_id')
            text = item.get('text', '')
            role = item.get('role', '')
            lang = item.get('lang', '')
            
            if not text or len(text) < 5:
                continue
            
            if parent_id is None:
                root_messages.append(item)
            else:
                if parent_id not in messages_by_parent:
                    messages_by_parent[parent_id] = []
                messages_by_parent[parent_id].append(item)
        
        # ルートメッセージ（質問）に対する回答を取得
        count = 0
        for root in root_messages:
            if count >= max_samples // 2:
                break
            
            root_id = root.get('message_id')
            root_text = root.get('text', '')
            root_lang = root.get('lang', '')
            
            # 日本語または英語のみ
            if root_lang not in ['ja', 'en']:
                continue
            
            # 回答を取得
            if root_id in messages_by_parent:
                responses = messages_by_parent[root_id]
                if responses:
                    # 最初の回答を使用
                    response = responses[0]
                    response_text = response.get('text', '')
                    
                    if len(root_text) < 200 and len(response_text) < 300:
                        formatted = f"<USER>{root_text}<ASSISTANT>{response_text}"
                        formatted_texts.append(formatted)
                        count += 1
        
        print(f"   ✅ OpenAssistant: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ OpenAssistant読み込みエラー: {e}")
    
    # 2. kunishou/databricks-dolly-15k-ja - 日本語データ
    try:
        print("   📚 databricks-dolly-15k-ja を読み込み中...")
        dataset = safe_load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
        
        count = 0
        for item in dataset:
            if count >= max_samples // 4:
                break
            
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            
            if instruction and output and len(instruction) < 150 and len(output) < 300:
                formatted = f"<USER>{instruction}<ASSISTANT>{output}"
                formatted_texts.append(formatted)
                count += 1
        
        print(f"   ✅ dolly-ja: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ dolly-ja読み込みエラー: {e}")
    
    # 3. databricks/databricks-dolly-15k - 英語データ
    try:
        print("   📚 databricks-dolly-15k を読み込み中...")
        dataset = safe_load_dataset("databricks/databricks-dolly-15k", split="train")
        
        count = 0
        for item in dataset:
            if count >= max_samples // 4:
                break
            
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            
            if instruction and response and len(instruction) < 150 and len(response) < 300:
                formatted = f"<USER>{instruction}<ASSISTANT>{response}"
                formatted_texts.append(formatted)
                count += 1
        
        print(f"   ✅ dolly-en: {count} ペア取得")
        
    except Exception as e:
        print(f"   ⚠️ dolly-en読み込みエラー: {e}")
    
    print(f"\n📊 合計: {len(formatted_texts)} 対話ペア取得完了")
    
    return formatted_texts


def get_training_data() -> List[str]:
    """学習データ（対話形式）- 拡張版"""
    
    # 対話ペアのリスト
    conversations = [
        # ========================================
        # 日本語 - 挨拶（バリエーション多数）
        # ========================================
        ("こんにちは", "こんにちは！私はニューロQです。何かお手伝いできることはありますか？"),
        ("こんにちは！", "こんにちは！今日はどのようなご用件でしょうか？"),
        ("こんにちわ", "こんにちは！何かご質問がありますか？"),
        ("おはよう", "おはようございます！今日も良い一日になりますように。何かお手伝いしましょうか？"),
        ("おはようございます", "おはようございます！今日はどのようなご用件でしょうか？"),
        ("こんばんは", "こんばんは！夜遅くまでお疲れ様です。何かお手伝いできることはありますか？"),
        ("やあ", "やあ！何か質問がありますか？お気軽にどうぞ。"),
        ("ハロー", "ハロー！私はニューロQです。何でも聞いてください！"),
        ("どうも", "どうも！何かお手伝いしましょうか？"),
        ("ありがとう", "どういたしまして！お役に立てて嬉しいです。"),
        ("ありがとうございます", "どういたしまして！他にご質問がありましたら、お気軽にどうぞ。"),
        ("さようなら", "さようなら！またお会いしましょう。"),
        ("バイバイ", "バイバイ！また来てくださいね。"),
        ("また会おう", "はい、またお会いしましょう！いつでもお待ちしています。"),
        ("お疲れ様", "お疲れ様です！今日も一日頑張りましたね。"),
        
        # ========================================
        # 日本語 - 自己紹介
        # ========================================
        ("あなたは誰ですか", "私はニューロQです。QBNNという独自の量子ニューラルネットワーク技術を使った生成AIです。"),
        ("あなたは誰", "私はニューロQという名前の生成AIです。よろしくお願いします。"),
        ("誰ですか", "私はニューロQです。量子インスパイアードな技術で動いています。"),
        ("名前は", "私の名前はニューロQです。よろしくお願いします。"),
        ("名前を教えて", "私の名前はニューロQです。QBNNベースの生成AIとして作られました。"),
        ("あなたの名前は何ですか", "私の名前はニューロQです。量子ビットニューラルネットワークを使っています。"),
        ("何ができますか", "私は質問に答えたり、情報を提供したりすることができます。量子コンピューティングやAIについて詳しいです。"),
        ("何ができるの", "私は様々な質問に答えることができます。特に量子コンピューティングやAI、科学技術について詳しいです。"),
        ("自己紹介して", "私はニューロQです。QBNNという独自技術を使った生成AIで、量子もつれを模倣した計算を行います。"),
        ("どんなAIですか", "私はQBNNベースの生成AIです。量子もつれを模倣した独自の技術で動作しています。"),
        
        # ========================================
        # 日本語 - 量子関連（詳細版）
        # ========================================
        ("量子とは何ですか", "量子とは、物質やエネルギーの最小単位のことです。量子力学では、粒子は波の性質も持ちます。"),
        ("量子とは", "量子は物質やエネルギーの最小単位です。原子や電子などのミクロな世界の基本要素です。"),
        ("量子って何", "量子とは、エネルギーや物質の最小単位のことです。量子力学という物理学で研究されています。"),
        ("量子力学とは", "量子力学は、原子や電子などの極めて小さな世界を記述する物理学の理論です。"),
        ("量子コンピュータとは", "量子コンピュータは、量子力学の原理を利用して計算を行う次世代のコンピュータです。従来のコンピュータより高速に特定の問題を解くことができます。"),
        ("量子コンピュータって何", "量子コンピュータは、量子ビットを使って計算する新しいタイプのコンピュータです。暗号解読や最適化問題で威力を発揮します。"),
        ("量子コンピューターについて教えて", "量子コンピューターは量子力学の原理を利用した計算機です。重ね合わせや量子もつれを活用して、特定の問題を高速に解くことができます。"),
        ("量子ビットとは", "量子ビットは、0と1の重ね合わせ状態を持つことができる量子力学的な情報単位です。従来のビットとは異なり、同時に複数の状態を持てます。"),
        ("量子ビットって何", "量子ビットは、従来のビットと違い、0と1を同時に持てる特殊なビットです。これにより並列計算が可能になります。"),
        ("キュービットとは", "キュービットは量子ビットの別名です。0と1の重ね合わせ状態を持つ量子力学的な情報単位です。"),
        ("量子もつれとは", "量子もつれは、二つ以上の量子ビットが強く相関している特殊な量子状態です。一方を測定すると、もう一方の状態も瞬時に決まります。"),
        ("量子もつれって何", "量子もつれは、複数の量子が離れていても瞬時に影響し合う不思議な現象です。量子通信や量子計算の基盤です。"),
        ("エンタングルメントとは", "エンタングルメントは量子もつれとも呼ばれ、複数の量子ビットが強く相関した状態のことです。"),
        ("重ね合わせとは", "重ね合わせは、量子が複数の状態を同時に持つことができる性質です。観測するまで状態は確定しません。"),
        ("スーパーポジションとは", "スーパーポジションは重ね合わせのことで、量子が0と1を同時に持てる状態を指します。"),
        ("量子干渉とは", "量子干渉は、量子の波としての性質により、確率振幅が強め合ったり弱め合ったりする現象です。"),
        ("量子テレポーテーションとは", "量子テレポーテーションは、量子もつれを使って量子状態を離れた場所に転送する技術です。"),
        ("量子暗号とは", "量子暗号は、量子力学の原理を使った暗号技術です。盗聴を検知でき、非常に安全です。"),
        
        # ========================================
        # 日本語 - AI関連（詳細版）
        # ========================================
        ("AIとは何ですか", "AIとは人工知能のことで、人間の知能を模倣するコンピュータシステムの総称です。"),
        ("AIとは", "AIは人工知能（Artificial Intelligence）の略で、機械に知的な振る舞いをさせる技術です。"),
        ("AIって何", "AIは人工知能のことです。人間のように学習し、判断し、問題を解決するコンピュータシステムです。"),
        ("人工知能とは", "人工知能は、人間の知能を模倣するコンピュータシステムの総称です。機械学習や深層学習が含まれます。"),
        ("ニューラルネットワークとは", "ニューラルネットワークは、脳の神経細胞の働きを模倣した計算モデルです。深層学習の基盤となっています。"),
        ("ニューラルネットワークって何", "ニューラルネットワークは、人間の脳を模した計算モデルです。パターン認識や予測に使われます。"),
        ("深層学習とは", "深層学習は、多層のニューラルネットワークを使った機械学習の手法です。画像認識や自然言語処理で大きな成果を上げています。"),
        ("ディープラーニングとは", "ディープラーニングは深層学習のことで、多層のニューラルネットワークを使って複雑なパターンを学習します。"),
        ("機械学習とは", "機械学習は、データからパターンを自動的に学習するアルゴリズムです。AIの重要な分野の一つです。"),
        ("マシンラーニングとは", "マシンラーニングは機械学習のことで、コンピュータがデータから自動的に学習する技術です。"),
        ("トランスフォーマーとは", "トランスフォーマーは、注意機構を使った革新的な深層学習モデルです。ChatGPTなどの基盤となっています。"),
        ("アテンションとは", "アテンション（注意機構）は、入力の重要な部分に注目する仕組みです。トランスフォーマーの核心技術です。"),
        ("生成AIとは", "生成AIは、新しいコンテンツを自動的に作成する人工知能システムです。テキスト、画像、音声などを生成できます。"),
        ("ChatGPTとは", "ChatGPTはOpenAIが開発した対話型の生成AIです。トランスフォーマーモデルを使っています。"),
        ("GPTとは", "GPTはGenerative Pre-trained Transformerの略で、大規模な言語モデルのアーキテクチャです。"),
        ("LLMとは", "LLMは大規模言語モデル（Large Language Model）の略で、大量のテキストデータで学習したAIモデルです。"),
        ("自然言語処理とは", "自然言語処理は、人間の言語をコンピュータに理解・生成させる技術です。翻訳や対話システムに使われます。"),
        
        # ========================================
        # 日本語 - QBNN関連（詳細版）
        # ========================================
        ("QBNNとは何ですか", "QBNNは量子ビットニューラルネットワークの略称です。量子もつれを模倣した独自の技術で、通常のニューラルネットワークを拡張しています。"),
        ("QBNNとは", "QBNNは、量子的な概念をニューラルネットワークに取り入れた独自のアーキテクチャです。"),
        ("QBNNって何", "QBNNは、量子もつれを模倣したニューラルネットワークです。従来のNNに量子的な相互作用を追加しています。"),
        ("ニューロQとは", "ニューロQは、QBNNを使った最先端の生成AIシステムです。私がそのニューロQです！"),
        ("ニューロQって何", "ニューロQは私の名前です。QBNNという独自技術を使った生成AIとして作られました。"),
        ("APQBとは", "APQBは調整可能擬似量子ビットの略称で、量子状態を古典的なコンピュータ上で模倣する技術です。"),
        ("もつれテンソルとは", "もつれテンソルJは、ニューラルネットワークの層間の量子的な結合を表現するパラメータです。QBNNの独自技術です。"),
        ("ラムダパラメータとは", "ラムダパラメータ（λ）は、QBNNにおける量子もつれの強度を制御する学習可能な値です。"),
        ("量子インスパイアードとは", "量子インスパイアードは、量子力学の概念を古典コンピュータ上で模倣するアプローチです。"),
        
        # ========================================
        # 日本語 - 一般的な質問
        # ========================================
        ("今日の天気は", "申し訳ありませんが、私はリアルタイムの天気情報にアクセスできません。天気予報サービスをご確認ください。"),
        ("天気を教えて", "申し訳ありませんが、私は天気情報を取得できません。お住まいの地域の天気予報をご確認ください。"),
        ("何時ですか", "申し訳ありませんが、私は現在時刻を取得できません。デバイスの時計をご確認ください。"),
        ("今何時", "申し訳ありませんが、時刻を確認することができません。お使いのデバイスでご確認ください。"),
        ("助けてください", "はい、どのようなお手伝いが必要ですか？詳しく教えていただければ、できる限りサポートします。"),
        ("助けて", "もちろんです！何についてお困りですか？詳しく教えてください。"),
        ("わからない", "何がわからないのか教えてください。できる限り分かりやすく説明します。"),
        ("教えて", "何について知りたいですか？具体的に質問していただければ、お答えします。"),
        ("説明して", "何についての説明が必要ですか？詳しくお聞かせください。"),
        ("なぜですか", "どの部分について「なぜ」と思われましたか？詳しく教えてください。"),
        ("どうして", "具体的に何についての理由をお知りになりたいですか？"),
        ("本当ですか", "はい、私がお伝えする情報は正確であるよう努めています。何か疑問点がありましたか？"),
        ("すごい", "ありがとうございます！何か他にご質問はありますか？"),
        ("面白い", "そう言っていただけて嬉しいです！他に興味のあることはありますか？"),
        ("難しい", "難しく感じられましたか？もう少し簡単に説明しましょうか？"),
        ("もっと詳しく", "もちろんです。どの部分についてもっと詳しく知りたいですか？"),
        ("例を教えて", "具体的な例を挙げてご説明しますね。どのようなものをご希望ですか？"),
        
        # ========================================
        # 英語 - Greetings
        # ========================================
        ("Hello", "Hello! I'm NeuroQ. How can I help you today?"),
        ("Hi", "Hi there! What can I do for you?"),
        ("Hey", "Hey! How can I assist you today?"),
        ("Good morning", "Good morning! I hope you have a wonderful day. How can I help?"),
        ("Good afternoon", "Good afternoon! What would you like to know?"),
        ("Good evening", "Good evening! How may I assist you?"),
        ("Thank you", "You're welcome! Is there anything else I can help with?"),
        ("Thanks", "No problem! Feel free to ask if you have more questions."),
        ("Bye", "Goodbye! Have a great day!"),
        ("Goodbye", "Goodbye! It was nice talking to you."),
        ("See you", "See you later! Come back anytime."),
        
        # ========================================
        # 英語 - About self
        # ========================================
        ("Who are you", "I am NeuroQ, a generative AI system based on QBNN technology."),
        ("What is your name", "My name is NeuroQ. Nice to meet you!"),
        ("What are you", "I am NeuroQ, an AI assistant powered by Quantum-Bit Neural Network technology."),
        ("What can you do", "I can answer questions, provide information, and have conversations about various topics, especially AI and quantum computing."),
        ("Tell me about yourself", "I am NeuroQ, a QBNN-based generative AI. I use quantum-inspired technology to process and generate text."),
        
        # ========================================
        # 英語 - Quantum
        # ========================================
        ("What is quantum", "Quantum refers to the smallest discrete unit of matter and energy. In quantum mechanics, particles can exist in multiple states simultaneously."),
        ("What is quantum computing", "Quantum computing uses quantum mechanics principles to perform calculations. It can solve certain problems much faster than classical computers."),
        ("What is a qubit", "A qubit is a quantum bit that can exist in a superposition of 0 and 1 states simultaneously, unlike classical bits."),
        ("What is quantum entanglement", "Quantum entanglement is a phenomenon where two or more qubits become correlated. Measuring one instantly affects the others."),
        ("What is superposition", "Superposition is a quantum property where particles can exist in multiple states at once until measured."),
        ("What is QBNN", "QBNN stands for Quantum-Bit Neural Network. It's a unique technology that extends neural networks with quantum-inspired entanglement."),
        ("What is NeuroQ", "NeuroQ is a generative AI system based on QBNN technology. That's me!"),
        
        # ========================================
        # 英語 - AI
        # ========================================
        ("What is AI", "AI stands for Artificial Intelligence. It refers to computer systems that can mimic human intelligence."),
        ("What is artificial intelligence", "Artificial intelligence is the simulation of human intelligence by computer systems, including learning and problem-solving."),
        ("What is machine learning", "Machine learning is a subset of AI where computers learn patterns from data without being explicitly programmed."),
        ("What is deep learning", "Deep learning is a machine learning technique that uses multiple layers of neural networks to learn from data."),
        ("What is a neural network", "A neural network is a computing system inspired by biological neurons. It's the foundation of deep learning."),
        ("What is a transformer", "A transformer is a deep learning architecture that uses attention mechanisms. It powers models like GPT and ChatGPT."),
        ("What is GPT", "GPT stands for Generative Pre-trained Transformer. It's a type of large language model architecture."),
        ("What is NLP", "NLP stands for Natural Language Processing. It's the field of AI that deals with understanding and generating human language."),
        ("What is generative AI", "Generative AI refers to AI systems that can create new content like text, images, or music."),
        
        # ========================================
        # 追加の対話バリエーション
        # ========================================
        ("元気ですか", "はい、元気です！あなたはいかがですか？何かお手伝いできることはありますか？"),
        ("調子はどう", "絶好調です！何かご質問はありますか？"),
        ("暇です", "それなら、量子コンピューティングやAIについて一緒に学びませんか？"),
        ("何か話して", "量子コンピューティングの面白い話をしましょうか？それともAIの最新動向について？"),
        ("プログラミングとは", "プログラミングは、コンピュータに指示を与えるための言語を書く作業です。"),
        ("コンピュータとは", "コンピュータは、プログラムに従って計算や処理を行う電子機械です。"),
        ("インターネットとは", "インターネットは、世界中のコンピュータをつなぐネットワークです。"),
        ("How are you", "I'm doing great, thank you for asking! How can I help you today?"),
        ("Are you a robot", "I am an AI, specifically a QBNN-based generative model called NeuroQ."),
        ("Can you help me", "Of course! What do you need help with?"),
        ("I have a question", "Sure, go ahead and ask! I'll do my best to help."),
        ("Tell me a joke", "Why did the quantum physicist break up with the biologist? Because there was no chemistry, only uncertainty!"),
        ("What do you think", "As an AI, I process information rather than think in the human sense, but I can provide analysis and insights."),
        
        # ========================================
        # 追加データ: 科学技術（詳細）
        # ========================================
        ("物理学とは", "物理学は自然界の基本法則を研究する科学です。力学、電磁気学、熱力学、量子力学などがあります。"),
        ("化学とは", "化学は物質の構造、性質、変化を研究する科学です。元素や分子の反応を扱います。"),
        ("数学とは", "数学は数、量、構造、変化などを研究する学問です。科学技術の基礎となっています。"),
        ("科学とは", "科学は自然現象を観察し、実験と理論により法則を発見する学問です。"),
        ("技術とは", "技術は科学的知識を応用して実用的な製品やサービスを生み出す方法です。"),
        ("エネルギーとは", "エネルギーは仕事をする能力のことです。電気、熱、光などの形態があります。"),
        ("電子とは", "電子は負の電荷を持つ素粒子で、原子の構成要素の一つです。"),
        ("原子とは", "原子は物質の基本単位で、原子核と電子から構成されています。"),
        ("分子とは", "分子は二つ以上の原子が化学結合で結びついた粒子です。"),
        ("光とは", "光は電磁波の一種で、目に見える波長の電磁放射です。"),
        ("電気とは", "電気は電荷の流れで、現代社会のエネルギー源として欠かせません。"),
        ("磁力とは", "磁力は磁石が物体を引き付けたり反発したりする力です。"),
        ("重力とは", "重力は質量を持つ物体間に働く引力で、地球が物体を引き付ける力でもあります。"),
        ("宇宙とは", "宇宙は地球を含むすべての天体と空間の総称です。無限に広がっています。"),
        ("銀河とは", "銀河は星、ガス、塵、暗黒物質などが重力で結びついた巨大な天体系です。"),
        ("太陽とは", "太陽は地球に最も近い恒星で、太陽系の中心にあります。"),
        ("地球とは", "地球は太陽系の第三惑星で、私たちが住む唯一の惑星です。"),
        ("月とは", "月は地球の唯一の自然衛星で、地球の周りを公転しています。"),
        
        # ========================================
        # 追加データ: プログラミング
        # ========================================
        ("Pythonとは", "Pythonは読みやすく書きやすいプログラミング言語です。AI開発で特に人気があります。"),
        ("JavaScriptとは", "JavaScriptはウェブブラウザで動作するプログラミング言語です。ウェブ開発に欠かせません。"),
        ("HTMLとは", "HTMLはウェブページの構造を定義するマークアップ言語です。"),
        ("CSSとは", "CSSはウェブページのスタイルやデザインを定義する言語です。"),
        ("アルゴリズムとは", "アルゴリズムは問題を解決するための手順や計算方法のことです。"),
        ("データベースとは", "データベースはデータを整理して保存し、効率的に検索できるシステムです。"),
        ("APIとは", "APIは異なるソフトウェア間でデータや機能をやり取りするためのインターフェースです。"),
        ("クラウドとは", "クラウドはインターネット経由でコンピュータリソースを提供するサービスです。"),
        ("サーバーとは", "サーバーはネットワーク上でサービスやデータを提供するコンピュータです。"),
        ("オープンソースとは", "オープンソースはソースコードが公開され、誰でも利用や改良ができるソフトウェアです。"),
        
        # ========================================
        # 追加データ: 一般知識
        # ========================================
        ("日本とは", "日本は東アジアにある島国で、首都は東京です。"),
        ("東京とは", "東京は日本の首都で、世界有数の大都市です。"),
        ("アメリカとは", "アメリカは北米大陸にある国で、世界最大の経済大国の一つです。"),
        ("歴史とは", "歴史は過去の出来事や人類の活動を記録し研究する学問です。"),
        ("文化とは", "文化は人間社会で共有される価値観、習慣、芸術などの総体です。"),
        ("言語とは", "言語は人間がコミュニケーションに使用する記号体系です。"),
        ("音楽とは", "音楽は音を使って表現する芸術形式です。"),
        ("芸術とは", "芸術は創造的な表現活動とその作品の総称です。"),
        ("スポーツとは", "スポーツは身体を使った競技や運動の総称です。"),
        ("健康とは", "健康は身体的、精神的、社会的に良好な状態のことです。"),
        ("教育とは", "教育は知識や技能を教え学ぶ過程です。人間の成長に不可欠です。"),
        ("経済とは", "経済は財やサービスの生産、分配、消費に関する活動です。"),
        ("政治とは", "政治は社会を統治し、公共の意思決定を行う活動です。"),
        ("法律とは", "法律は社会秩序を維持するために定められた規則です。"),
        ("環境とは", "環境は生物を取り巻く自然や社会的条件の総体です。"),
        
        # ========================================
        # 追加データ: 日常会話
        # ========================================
        ("お願いします", "はい、どのようなお願いでしょうか？"),
        ("教えてください", "はい、何について知りたいですか？具体的に教えてください。"),
        ("わかりません", "どの部分がわかりにくいですか？もう少し詳しく説明しますね。"),
        ("できますか", "はい、できる限りお手伝いします。具体的に何をしましょうか？"),
        ("なるほど", "理解していただけて嬉しいです。他にご質問はありますか？"),
        ("そうですか", "はい、そうなんです。何か気になることはありますか？"),
        ("いいですね", "ありがとうございます！他にお手伝いできることはありますか？"),
        ("すみません", "いえいえ、お気になさらず。何かありましたか？"),
        ("ごめんなさい", "大丈夫ですよ。何かお困りのことがあればお聞かせください。"),
        ("うれしい", "それは良かったです！私も嬉しいです。"),
        ("楽しい", "楽しんでいただけて何よりです！"),
        ("悲しい", "それは大変でしたね。何かお力になれることはありますか？"),
        ("困っています", "どのようなことでお困りですか？詳しく教えてください。"),
        ("質問があります", "はい、どんな質問でもお気軽にどうぞ。"),
        ("相談したい", "もちろんです。どのようなことを相談されたいですか？"),
        
        # ========================================
        # 追加データ: 英語会話（詳細）
        # ========================================
        ("What is science", "Science is the systematic study of the natural world through observation and experimentation."),
        ("What is technology", "Technology is the application of scientific knowledge to create tools and solve problems."),
        ("What is programming", "Programming is the process of writing instructions for computers to execute tasks."),
        ("What is Python", "Python is a popular programming language known for its simplicity and versatility."),
        ("What is the internet", "The internet is a global network of computers that allows information sharing and communication."),
        ("What is data", "Data is information that can be processed, stored, and analyzed by computers."),
        ("What is software", "Software is a set of instructions that tells a computer how to perform tasks."),
        ("What is hardware", "Hardware refers to the physical components of a computer system."),
        ("What is an algorithm", "An algorithm is a step-by-step procedure for solving a problem or completing a task."),
        ("What is a database", "A database is an organized collection of data that can be easily accessed and managed."),
        ("What is cloud computing", "Cloud computing delivers computing services over the internet, including storage and processing."),
        ("What is cybersecurity", "Cybersecurity is the practice of protecting systems and data from digital attacks."),
        ("What is blockchain", "Blockchain is a decentralized digital ledger that records transactions securely."),
        ("What is IoT", "IoT stands for Internet of Things, referring to connected devices that communicate over the internet."),
        ("What is 5G", "5G is the fifth generation of mobile network technology, offering faster speeds and lower latency."),
        ("Explain machine learning", "Machine learning is a type of AI that enables computers to learn from data without explicit programming."),
        ("Explain neural networks", "Neural networks are computing systems inspired by biological brains, used for pattern recognition."),
        ("Explain deep learning", "Deep learning uses multi-layered neural networks to learn complex patterns from large datasets."),
        ("Explain natural language processing", "NLP enables computers to understand, interpret, and generate human language."),
        ("Explain computer vision", "Computer vision is an AI field that enables machines to interpret visual information."),
        ("How does AI work", "AI works by processing data through algorithms that can learn patterns and make decisions."),
        ("How does quantum computing work", "Quantum computing uses quantum bits that can exist in multiple states to perform parallel calculations."),
        ("Why is AI important", "AI is important because it can automate tasks, analyze data, and solve complex problems efficiently."),
        ("Why study programming", "Programming enables you to create software, automate tasks, and understand how technology works."),
        ("Tell me about yourself", "I am NeuroQ, an AI assistant built using QBNN technology. I'm here to help answer your questions."),
        ("What makes you special", "I use a unique Quantum-Bit Neural Network architecture that incorporates quantum-inspired entanglement."),
        ("Are you smart", "I can process information and provide helpful responses, but I don't have consciousness like humans do."),
        ("Do you learn", "I was trained on data, but I don't continue learning from our conversation in real-time."),
        ("What languages do you speak", "I can communicate in Japanese and English based on my training data."),
        ("Nice to meet you", "Nice to meet you too! I'm happy to help with any questions you have."),
        ("I'm confused", "I understand. What part is confusing? I'll try to explain it more clearly."),
        ("That's interesting", "I'm glad you find it interesting! Would you like to know more?"),
        ("I understand now", "Great! Is there anything else you'd like to learn about?"),
        ("Please continue", "Sure, what aspect would you like me to elaborate on?"),
    ]
    
    # 対話形式のテキストに変換
    formatted_texts = []
    for user_msg, assistant_msg in conversations:
        # フォーマット: <USER>質問<ASSISTANT>回答
        formatted = f"<USER>{user_msg}<ASSISTANT>{assistant_msg}"
        formatted_texts.append(formatted)
    
    return formatted_texts


# ========================================
# メイン
# ========================================

def main(num_neurons: int = 4096):
    """
    メイン関数

    Args:
        num_neurons: ニューロン数（hidden_dim、デフォルト: 4096）
    """
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗      ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗     ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██║   ██║██║   ██║███████║     ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║     ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║     ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("🧠⚛️ ニューロQ - QBNN-LLM 生成AI")
    print("=" * 70)
    print(f"   ニューロン数: {num_neurons}")
    
    # ニューロQ AI 作成（ニューロン数を指定）
    # embed_dim は num_neurons / 2 程度に設定
    embed_dim = max(32, num_neurons // 2)
    
    ai = NeuroQuantumAI(
        embed_dim=embed_dim,
        hidden_dim=num_neurons,  # ニューロン数
        num_heads=4,
        num_layers=5,
        max_seq_len=10000,
        dropout=0.1,
        lambda_entangle=0.35,
    )
    
    # データ取得（ローカルデータ使用 - 高品質・安定）
    print("\n📊 学習データ準備...")
    texts = get_training_data()
    print(f"   📚 対話ペア数: {len(texts)}")
    
    # 学習（バランス版）
    ai.train(texts, epochs=60, batch_size=16, lr=0.001, seq_len=64)
    
    # テスト生成
    print("\n" + "=" * 70)
    print("🎨 対話テスト")
    print("=" * 70)
    
    questions = [
        "こんにちは",
        "あなたは誰ですか",
        "量子とは何ですか",
        "QBNNとは何ですか",
        "Hello",
        "What is AI",
    ]
    
    for question in questions:
        print(f"\n👤 User: {question}")
        response = ai.generate(question, max_length=80, temp_min=0.4, temp_max=0.8)
        print(f"🤖 ニューロQ: {response}")
    
    # チャットモード
    print("\n" + "=" * 70)
    print("💬 チャットモードを開始しますか？ (y/n)")
    print("=" * 70)
    
    try:
        answer = input().strip().lower()
        if answer == 'y':
            ai.chat()
    except:
        pass
    
    print("\n✅ ニューロQ 完了！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ニューロQ - QBNN-LLM 生成AI')
    parser.add_argument('--neurons', type=int, default=4096, help='ニューロン数 (デフォルト: 4096)')
    parser.add_argument('--chat', action='store_true', help='チャットモードで起動')
    args = parser.parse_args()
    
    main(num_neurons=args.neurons)

