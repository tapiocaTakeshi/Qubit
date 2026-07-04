#!/usr/bin/env python3
"""
NeuroQuantum Inference Script

訓練済みモデルを使用して推論を実行
"""

import torch
import argparse
from pathlib import Path
import logging
from typing import Optional, List
import json

from neuroquantum_layered import NeuroQuantum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuroQuantumInference:
    """NeuroQuantumモデルの推論"""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        max_length: int = 1024
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        self._load_checkpoint()

    def _load_checkpoint(self):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        config_dict = checkpoint.get('config', {})
        logger.info(f"Config: {config_dict}")

        self.model = NeuroQuantum(config_dict=config_dict)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def tokenize(self, text: str) -> List[int]:
        """テキストをトークン化"""
        tokens = []
        for char in text:
            token = min(ord(char) % (self.model.config.vocab_size - 1) + 1, self.model.config.vocab_size - 1)
            tokens.append(token)
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """トークンをテキストに戻す"""
        text = ""
        for token in tokens:
            if token > 0:
                text += chr(token % 256)
        return text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """テキスト生成"""
        # プロンプトをトークン化
        input_tokens = self.tokenize(prompt)
        input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)

        logger.info(f"Generating from prompt: '{prompt[:100]}'")

        generated_tokens = list(input_tokens)

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 最新のトークンシーケンスで予測
                current_ids = torch.tensor([generated_tokens[-self.max_length:]], device=self.device, dtype=torch.long)

                # Forward pass
                try:
                    logits = self.model(current_ids)  # (1, seq_len, vocab_size)
                except Exception as e:
                    logger.error(f"Error during forward pass: {e}")
                    break

                # 最後のトークンのロジットを取得
                next_logits = logits[0, -1, :]  # (vocab_size,)

                # Temperature scaling
                if temperature != 1.0:
                    next_logits = next_logits / temperature

                # Top-K filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                    next_logits[next_logits < top_k_logits[-1]] = float('-inf')

                # Top-P (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[0] = False  # 最初のトークンは必ず保持
                    next_logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

                # サンプリング
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated_tokens.append(next_token)

                if (step + 1) % 20 == 0:
                    logger.info(f"Generated {step + 1}/{max_new_tokens} tokens")

        # トークンをテキストに変換
        generated_text = self.detokenize(generated_tokens[len(input_tokens):])

        return generated_text

    def predict_next_tokens(self, text: str, num_predictions: int = 5) -> List[dict]:
        """次のトークンの確率を予測"""
        input_tokens = self.tokenize(text)
        input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(next_logits, dim=-1)

        # トップ-K確率を取得
        top_probs, top_indices = torch.topk(probs, num_predictions)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'token_id': idx.item(),
                'character': chr(idx.item() % 256) if idx.item() > 0 else '<UNK>',
                'probability': prob.item()
            })

        return predictions

    def get_embeddings(self, text: str) -> torch.Tensor:
        """テキストのエンベディングを取得"""
        input_tokens = self.tokenize(text)
        input_ids = torch.tensor([input_tokens], device=self.device, dtype=torch.long)

        with torch.no_grad():
            # モデルから隠れ状態を抽出
            # NeuroQuantumの場合、最後の層の出力がエンベディング
            logits = self.model(input_ids)  # (1, seq_len, vocab_size)

        # 最後のトークンのロジットを使用
        embeddings = logits[0, -1, :]  # (vocab_size,)

        return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="NeuroQuantum Inference"
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Input prompt'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Top-K filtering'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-P (nucleus) filtering'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    inferencer = NeuroQuantumInference(args.checkpoint, device=device)

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("NeuroQuantum Interactive Inference")
        print("=" * 80)
        print("Commands:")
        print("  - Type prompt and press Enter to generate")
        print("  - 'predict <text>': Predict next tokens")
        print("  - 'embed <text>': Get embeddings")
        print("  - 'quit': Exit")
        print("=" * 80 + "\n")

        while True:
            user_input = input("> ").strip()

            if user_input.lower() == 'quit':
                break

            if user_input.lower().startswith('predict'):
                text = user_input[8:].strip()
                predictions = inferencer.predict_next_tokens(text)
                print("\nNext token predictions:")
                for pred in predictions:
                    print(f"  {pred['character']}: {pred['probability']:.4f}")

            elif user_input.lower().startswith('embed'):
                text = user_input[6:].strip()
                embeddings = inferencer.get_embeddings(text)
                print(f"\nEmbeddings shape: {embeddings.shape}")
                print(f"Embedding (first 10 values): {embeddings[:10]}")

            else:
                generated = inferencer.generate(
                    user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(f"\nGenerated:\n{generated}\n")

    else:
        # Single inference
        if not args.prompt:
            args.prompt = input("Enter prompt: ").strip()

        generated = inferencer.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print("\n" + "=" * 80)
        print("Prompt:")
        print(args.prompt)
        print("\nGenerated:")
        print(generated)
        print("=" * 80)


if __name__ == "__main__":
    main()
