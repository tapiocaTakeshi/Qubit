#!/usr/bin/env python3
"""簡単な推論テスト"""
import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

def main():
    print("\n🤖 推論テスト\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデル初期化
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=8000)
    config = NeuroQuantumConfig(
        vocab_size=8000,
        embed_dim=1024,
        hidden_dim=2048,
        num_heads=16,
        num_layers=4,
        max_seq_len=256,
    )

    tokenizer = NeuroQuantumTokenizer(vocab_size=8000)
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    
    # チェックポイント読み込み
    ckpt = torch.load("./checkpoints/megabyte_100mb_stable.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("✓ モデル読み込み完了")
    print(f"✓ パラメータ: {sum(p.numel() for p in model.parameters()):,}")

    # テスト生成
    print("\n📝 テキスト生成テスト:")
    
    test_texts = [
        "こんにちは",
        "日本語",
        "AI",
    ]

    for text in test_texts:
        print(f"\n  入力: '{text}'")
        
        # トークン化
        ids = tokenizer.encode(text, add_special=False)
        print(f"  Token IDs: {ids}")
        
        # パディング
        seq = [tokenizer.bos_id] + ids
        pad_len = 256 - len(seq)
        padded = seq + [tokenizer.pad_id] * pad_len
        padded = padded[:256]
        
        # 推論
        with torch.no_grad():
            input_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            logits = model(input_tensor)
            
            # 最初のトークンの確率
            logits_first = logits[0, len(seq)-1, :]
            probs = torch.softmax(logits_first, dim=-1)
            top_tokens = torch.topk(probs, 5)
            
            print(f"  次のトークン候補（確率）:")
            for prob, token_id in zip(top_tokens.values, top_tokens.indices):
                print(f"    Token {token_id.item()}: {prob.item():.4f}")

    print("\n✅ テスト完了")

if __name__ == "__main__":
    main()
