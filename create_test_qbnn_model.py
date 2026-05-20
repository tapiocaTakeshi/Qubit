#!/usr/bin/env python3
"""
Create a minimal test QBNN model for conversion pipeline testing

テスト用の最小限の QBNN モデルを作成します。
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class SimpleQBNNLayer(nn.Module):
    """簡潔なテスト用 QBNN 層"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        # APQB theta パラメータ
        self.theta = nn.Parameter(torch.randn(output_dim) * np.pi / 2)

        # エンタングルメント用パラメータ
        self.phase = nn.Parameter(torch.randn(output_dim) * np.pi / 2)

    def forward(self, x):
        h = self.linear(x)

        # APQB 量子状態の計算
        r = torch.cos(2 * self.theta)
        T = torch.abs(torch.sin(2 * self.theta))

        # 量子補正を適用
        quantum_corr = r * torch.cos(self.phase)
        h = h + quantum_corr.unsqueeze(0)

        return h


class TestQBNNModel(nn.Module):
    """テスト用 QBNN モデル"""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # QBNN layers
        self.qbnn_layers = nn.ModuleList([
            SimpleQBNNLayer(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for layer in self.qbnn_layers:
            x = layer(x)
            x = torch.relu(x)

        logits = self.lm_head(x)
        return logits


def create_and_save_test_model(
    output_path: str = "test_qbnn_model.pt",
    vocab_size: int = 1000,
    hidden_size: int = 256,
    num_layers: int = 2,
):
    """テスト用 QBNN モデルを作成・保存"""

    print("🔧 Creating test QBNN model...")
    model = TestQBNNModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    # モデルの重みを初期化
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) >= 2:
            nn.init.kaiming_normal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    # モデル情報を表示
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Vocab size: {vocab_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Total parameters: {total_params:,}")

    # 状態辞書を作成
    state_dict = model.state_dict()

    # APQB theta パラメータを確認
    theta_params = [name for name in state_dict.keys() if "theta" in name]
    print(f"   APQB theta parameters: {len(theta_params)}")
    for name in theta_params[:3]:
        param = state_dict[name]
        print(f"     - {name}: shape={param.shape}, range=[{param.min():.4f}, {param.max():.4f}]")

    # checkpoint として保存
    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "theta_dim": hidden_size,
            "entangle_strength": 0.5,
            "use_entanglement": True,
        }
    }

    torch.save(checkpoint, output_path)
    print(f"\n✅ Test model saved to {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def test_model_structure():
    """モデルの構造をテスト"""
    print("\n🧪 Testing model structure...")

    model = TestQBNNModel(vocab_size=100, hidden_size=128, num_layers=2)
    model.eval()

    # ダミー入力でテスト
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

    # APQB パラメータを確認
    print(f"\n   APQB Parameters:")
    for name, param in model.named_parameters():
        if "theta" in name:
            print(f"     {name}: {param.shape} range=[{param.min():.4f}, {param.max():.4f}]")

    print("   ✓ Model structure test passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test QBNN model")
    parser.add_argument("--output", type=str, default="test_qbnn_model.pt",
                        help="Output path for model")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--test", action="store_true", help="Run model structure test")

    args = parser.parse_args()

    if args.test:
        test_model_structure()

    create_and_save_test_model(
        output_path=args.output,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
