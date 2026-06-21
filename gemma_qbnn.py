#!/usr/bin/env python3
"""
Gemma + QBNN ハイブリッドモデル
================================

Gemma 系（NeuroQuantum）の Transformer バックボーンをベースに、
量子もつれ層（EQBNNLayer）を1層挿入したハイブリッドアーキテクチャ。

フロー:
    [トークン埋め込み + 位置埋め込み]
            ↓
    [Gemma 系 Transformer ブロック × N]   ← ベース
            ↓
    [QBNN 層（量子相関 + エンタングルメント補正）]  ← 追加
            ↓
    [Final LayerNorm]
            ↓
    [出力ヘッド]

追加された QBNN 層は以下の量子テンソルを保持する:
    - quantum_corr.* : 量子相関行列（APQB θ 投影）
    - entangle_op.*  : 層間エンタングルメント演算子
これらは GGUF エクスポート時に量子化されず保持され、量子メタデータとして
記録される（generate_gguf_models.py 参照）。
"""

import torch
import torch.nn as nn
from typing import Optional

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, get_model_config_by_size
from qbnn_layered import EQBNNLayer


class GemmaQBNN(nn.Module):
    """Gemma 系バックボーン + 挿入された QBNN 層のハイブリッドモデル。

    NeuroQuantum をベースとして内部に保持し、その Transformer ブロックの出力に
    対して EQBNNLayer による量子補正を残差接続で加える。
    """

    def __init__(self, config: NeuroQuantumConfig, entangle_strength: float = 0.5):
        super().__init__()
        self.config = config

        # ベース: Gemma 系 Transformer バックボーン
        self.base = NeuroQuantum(config=config)

        # 追加: 量子もつれ層（埋め込み次元を保ったまま補正を生成）
        self.qbnn_layer = EQBNNLayer(
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
            prev_output_dim=config.embed_dim,
            entangle_strength=entangle_strength,
        )

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.config
        base = self.base

        token_ids = token_ids.clamp(0, cfg.vocab_size - 1)
        batch, seq = token_ids.shape
        if seq > cfg.max_seq_len:
            token_ids = token_ids[:, : cfg.max_seq_len]
            seq = cfg.max_seq_len

        # [1-2] トークン埋め込み + 位置埋め込み（ベースの submodule を再利用）
        token_embeds = base.token_embedding(token_ids)
        positions = (
            torch.arange(seq, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .clamp(0, cfg.max_seq_len - 1)
        )
        hidden_states = token_embeds + base.position_embedding(positions)

        # [3] ドロップアウト
        hidden_states = base.embedding_dropout(hidden_states)

        # Causal Mask
        if mask is None:
            mask = torch.tril(
                torch.ones(seq, seq, device=token_ids.device)
            ).unsqueeze(0).unsqueeze(0)

        # [4-9] Gemma 系 Transformer ブロック × N
        for block in base.transformer_blocks:
            hidden_states = block(hidden_states, mask)

        # ====== 挿入された QBNN 層 ======
        # [batch, seq, dim] -> [batch*seq, dim] に展開して量子補正を計算
        flat = hidden_states.reshape(batch * seq, cfg.embed_dim)
        quantum_correction, _q = self.qbnn_layer(flat, q_prev=flat)
        hidden_states = hidden_states + quantum_correction.reshape(batch, seq, cfg.embed_dim)
        # ================================

        # [10] Final LayerNorm
        hidden_states = base.final_norm(hidden_states)

        # [11] 出力ヘッド
        logits = base.output_head(hidden_states)
        return logits

    def get_constraint_loss(self) -> torch.Tensor:
        """挿入された QBNN 層の幾何学的制約損失。"""
        return self.qbnn_layer.get_constraint_loss()


def create_gemma_qbnn_model(
    size: str = "medium",
    vocab_size: int = 32000,
) -> GemmaQBNN:
    """サイズ指定で Gemma+QBNN ハイブリッドモデルを生成する。"""
    config_dict = get_model_config_by_size(size=size, vocab_size=vocab_size)
    config = NeuroQuantumConfig(
        vocab_size=config_dict["vocab_size"],
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_seq_len=config_dict["max_seq_len"],
        lambda_entangle=config_dict["entangle_strength"],
        dropout=config_dict["dropout"],
    )
    return GemmaQBNN(config=config, entangle_strength=config_dict["entangle_strength"])
