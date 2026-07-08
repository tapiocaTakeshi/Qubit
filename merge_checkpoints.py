#!/usr/bin/env python3
"""
複数のエポックチェックポイントを統合する
重みを平均化して1つの統合モデルを生成
"""
import torch
import os
from pathlib import Path
from collections import defaultdict
import argparse

def merge_checkpoints(checkpoint_dir: str, output_name: str = "merged", pattern: str = None):
    """
    複数のチェックポイントを統合

    Args:
        checkpoint_dir: チェックポイント保存ディレクトリ
        output_name: 出力ファイル名（拡張子なし）
        pattern: ファイル名パターン（e.g., "japanese", "epoch"）
    """
    checkpoint_dir = Path(checkpoint_dir)

    print(f"\n{'='*80}")
    print(f"🔀 Merging Checkpoints")
    print(f"{'='*80}\n")

    # チェックポイントファイルを探す
    if pattern:
        checkpoint_files = sorted(checkpoint_dir.glob(f"*{pattern}*.pt"))
    else:
        checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return False

    print(f"📂 Found {len(checkpoint_files)} checkpoints:")
    for i, ckpt in enumerate(checkpoint_files, 1):
        size_mb = ckpt.stat().st_size / (1024**2)
        print(f"   {i}. {ckpt.name} ({size_mb:.1f}MB)")

    # 重みを平均化
    print(f"\n🔄 Averaging weights across {len(checkpoint_files)} checkpoints...")

    averaged_state = defaultdict(lambda: None)

    for idx, ckpt_path in enumerate(checkpoint_files):
        print(f"   Loading {ckpt_path.name}...", end=" ", flush=True)

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    if averaged_state[key] is None:
                        averaged_state[key] = value.float().clone()
                    else:
                        averaged_state[key] += value.float()

            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    # 平均を計算
    print(f"\n✓ Averaging weights...")
    for key in averaged_state:
        if averaged_state[key] is not None:
            averaged_state[key] = averaged_state[key] / len(checkpoint_files)

    # 出力ファイルのパスを決定
    output_path = checkpoint_dir / f"{output_name}_checkpoint.pt"

    # マージされたチェックポイントを保存
    print(f"\n💾 Saving merged checkpoint...")
    torch.save(dict(averaged_state), output_path)

    output_size_mb = output_path.stat().st_size / (1024**2)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   ✓ Size: {output_size_mb:.1f}MB")

    print(f"\n{'='*80}")
    print(f"✅ Merge Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"   Input files: {len(checkpoint_files)}")
    print(f"   Output file: {output_path.name}")
    print(f"   Strategy: Average weights")

    # 統計情報
    print(f"\n📈 Statistics:")
    total_params = sum(p.numel() for p in averaged_state.values() if isinstance(p, torch.Tensor))
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model layers: {len(averaged_state)}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Merge multiple epoch checkpoints")
    parser.add_argument("--dir", default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--output", default="merged", help="Output filename (without .pt)")
    parser.add_argument("--pattern", default=None, help="File pattern filter (e.g., 'japanese')")

    args = parser.parse_args()

    success = merge_checkpoints(
        checkpoint_dir=args.dir,
        output_name=args.output,
        pattern=args.pattern
    )

    if not success:
        exit(1)

if __name__ == "__main__":
    main()
