#!/usr/bin/env python3
"""
各データセット形式ごとの推論テスト。

モデルをロードし、各データセット（一般、QA、マークダウン、分割学習）に
適したプロンプトで推論を実行して、出力の基本的な品質を検証する。
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict


# ---------- 定数 ----------
CHECKPOINT_CANDIDATES = [
    os.environ.get("CHECKPOINT_PATH", ""),
    os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt"),
    os.path.join(os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume"), "qbnn_checkpoint.pt"),
    os.path.join(os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume"), "neuroq_checkpoint.pt"),
]

TOKENIZER_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model"),
    os.path.join(os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume"), "neuroq_tokenizer.model"),
]

DEFAULT_CONFIG = {
    "vocab_size": 8000,
    "embed_dim": 512,
    "hidden_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "max_seq_len": 10000,
    "entangle_strength": 0.5,
    "dropout": 0.1,
}


def _find_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _load_model():
    """チェックポイントからモデルとトークナイザーをロードする。"""
    device = torch.device("cpu")
    tok_path = _find_file(TOKENIZER_CANDIDATES)
    ckpt_path = _find_file(CHECKPOINT_CANDIDATES)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = checkpoint.get("config", dict(DEFAULT_CONFIG))
    else:
        checkpoint = None
        config = dict(DEFAULT_CONFIG)

    tokenizer = NeuroQuantumTokenizer(
        vocab_size=config["vocab_size"],
        model_file=tok_path if tok_path else None,
    )
    tok_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    if tok_vocab and tok_vocab != config["vocab_size"]:
        config["vocab_size"] = tok_vocab

    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)

    if checkpoint:
        migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
        model_state = model.state_dict()
        for key in list(migrated.keys()):
            if key in model_state and migrated[key].shape != model_state[key].shape:
                new_tensor = model_state[key].clone()
                slices = tuple(
                    slice(0, min(o, n))
                    for o, n in zip(migrated[key].shape, model_state[key].shape)
                )
                new_tensor[slices] = migrated[key][slices]
                migrated[key] = new_tensor
        model.load_state_dict(migrated)

    model.eval()
    return model, tokenizer, config, device


def generate_text(model, tokenizer, config, device, prompt,
                  max_new_tokens=50, temperature=0.7, top_k=40,
                  top_p=0.9, repetition_penalty=1.3):
    """テキスト生成（api.py の generate_text と同じロジック）。"""
    prompt_str = f"<s>{prompt}</s>"
    tokens = tokenizer.encode(prompt_str, add_special=True)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)
    max_seq_len = config["max_seq_len"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = input_tensor[:, -max_seq_len:]
            logits = model(seq)[:, -1, :] / max(temperature, 1e-5)

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
                        logits[0, prev] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt_id = nxt.item()

            if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                break
            if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                continue

            generated.append(nxt_id)
            input_tensor = torch.cat([input_tensor, nxt], dim=1)

    return tokenizer.decode(generated[len(tokens):], skip_special=True)


# ========================================
# テスト本体
# ========================================

class TestInference(unittest.TestCase):
    """各データセット形式でのモデル推論テスト。"""

    model = None
    tokenizer = None
    config = None
    device = None

    @classmethod
    def setUpClass(cls):
        print("\n=== モデルロード中 ===")
        cls.model, cls.tokenizer, cls.config, cls.device = _load_model()
        n_params = sum(p.numel() for p in cls.model.parameters())
        print(f"モデルパラメータ数: {n_params:,}")
        print(f"語彙サイズ: {cls.config['vocab_size']}")
        print(f"デバイス: {cls.device}")

    def _generate(self, prompt, max_new_tokens=50, temperature=0.7):
        return generate_text(
            self.model, self.tokenizer, self.config, self.device,
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        )

    # ------------------------------------------------------------------
    # 1. 一般データセット (DEFAULT_DATASETS) の推論テスト
    #    izumi-lab, oasst1-chat, japanese_alpaca, wikipedia_conversation, alpaca-gpt4
    # ------------------------------------------------------------------

    def test_general_japanese_text(self):
        """一般的な日本語テキスト生成テスト。"""
        prompt = "日本の文化について"
        result = self._generate(prompt)
        print(f"\n[一般] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_general_knowledge(self):
        """一般知識ベースの生成テスト。"""
        prompt = "プログラミング言語Pythonは"
        result = self._generate(prompt)
        print(f"\n[一般知識] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_general_conversation(self):
        """会話形式（wikipedia_conversation / oasst1-chat相当）の生成テスト。"""
        prompt = "こんにちは、今日は何について話しましょうか"
        result = self._generate(prompt)
        print(f"\n[会話] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # 2. QAデータセットの推論テスト
    #    japanese_alpaca, alpaca-gpt4-japanese, oasst1-chat, izumi-lab + CRAFTED_QA
    # ------------------------------------------------------------------

    def test_qa_basic_question(self):
        """基本的なQA形式の推論テスト。"""
        prompt = "質問: 日本の首都はどこですか？\n回答:"
        result = self._generate(prompt, max_new_tokens=80)
        print(f"\n[QA基本] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_qa_science_question(self):
        """科学分野のQA推論テスト。"""
        prompt = "質問: 量子コンピュータとは何ですか？\n回答:"
        result = self._generate(prompt, max_new_tokens=100)
        print(f"\n[QA科学] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_qa_programming_question(self):
        """プログラミングQA推論テスト。"""
        prompt = "質問: 機械学習とは何ですか？\n回答:"
        result = self._generate(prompt, max_new_tokens=100)
        print(f"\n[QAプログラミング] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_qa_alpaca_instruction(self):
        """Alpaca instruction形式のQA推論テスト。"""
        prompt = "質問: 東京タワーについて教えてください。\n回答:"
        result = self._generate(prompt, max_new_tokens=100)
        print(f"\n[QA Alpaca] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # 3. マークダウン形式の推論テスト
    #    MARKDOWN_QA に基づく構造化出力テスト
    # ------------------------------------------------------------------

    def test_markdown_structured_output(self):
        """マークダウン形式の構造化出力テスト。"""
        prompt = "質問: Pythonの基本を教えてください。\n回答: ## Pythonの基本"
        result = self._generate(prompt, max_new_tokens=150)
        print(f"\n[MD構造化] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_markdown_with_code_block(self):
        """コードブロック付きマークダウン推論テスト。"""
        prompt = "質問: Docker とは何ですか？\n回答: ## Docker\n\nアプリケーションを"
        result = self._generate(prompt, max_new_tokens=150)
        print(f"\n[MDコード] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_markdown_with_table(self):
        """テーブル付きマークダウン推論テスト。"""
        prompt = "質問: 富士山について教えてください。\n回答: ## 富士山\n\n日本の"
        result = self._generate(prompt, max_new_tokens=150)
        print(f"\n[MDテーブル] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_markdown_list_output(self):
        """リスト形式のマークダウン推論テスト。"""
        prompt = "質問: 健康的な食事について教えてください。\n回答: ## 健康的な食事ガイド\n\n### 五大栄養素\n1."
        result = self._generate(prompt, max_new_tokens=150)
        print(f"\n[MDリスト] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # 4. 分割学習（split learning）の推論テスト
    #    split_learning.py / train_split_learning.py のデータで学習したモデル用
    # ------------------------------------------------------------------

    def test_split_learning_general(self):
        """分割学習モデルでの一般的な推論テスト。"""
        prompt = "人工知能の未来について"
        result = self._generate(prompt, max_new_tokens=80)
        print(f"\n[分割学習・一般] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_split_learning_qa(self):
        """分割学習モデルでのQA推論テスト。"""
        prompt = "質問: ニューラルネットワークとは何ですか？\n回答:"
        result = self._generate(prompt, max_new_tokens=100)
        print(f"\n[分割学習・QA] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # 5. エッジケーステスト
    # ------------------------------------------------------------------

    def test_empty_prompt(self):
        """空プロンプトでエラーが発生しないことを確認。"""
        result = self._generate("", max_new_tokens=20)
        print(f"\n[空プロンプト] -> {result!r}")
        self.assertIsInstance(result, str)

    def test_long_prompt(self):
        """長いプロンプトでエラーが発生しないことを確認。"""
        prompt = "日本語のテスト文。" * 50
        result = self._generate(prompt, max_new_tokens=30)
        print(f"\n[長文プロンプト] len={len(prompt)} -> {result!r}")
        self.assertIsInstance(result, str)

    def test_special_characters(self):
        """特殊文字を含むプロンプトのテスト。"""
        prompt = "数式: x² + y² = z² について"
        result = self._generate(prompt, max_new_tokens=50)
        print(f"\n[特殊文字] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    def test_temperature_zero(self):
        """temperature=0（greedy decoding）での推論テスト。"""
        prompt = "質問: 日本の首都は？\n回答:"
        result = self._generate(prompt, max_new_tokens=30, temperature=0.01)
        print(f"\n[greedy] prompt={prompt!r}")
        print(f"  -> {result!r}")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # 6. モデル基本検証
    # ------------------------------------------------------------------

    def test_model_output_shape(self):
        """モデルの出力テンソル形状を検証。"""
        input_ids = torch.tensor([[2, 10, 20, 30, 3]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids)
        self.assertEqual(logits.dim(), 3)
        self.assertEqual(logits.shape[0], 1)  # batch
        self.assertEqual(logits.shape[1], 5)  # seq_len
        self.assertEqual(logits.shape[2], self.config["vocab_size"])
        print(f"\n[出力形状] {logits.shape} (batch=1, seq=5, vocab={self.config['vocab_size']})")

    def test_model_deterministic_eval(self):
        """eval モードでの出力が決定的であることを確認。"""
        input_ids = torch.tensor([[2, 10, 20, 30, 3]], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(input_ids)
            out2 = self.model(input_ids)
        max_diff = torch.max(torch.abs(out1 - out2)).item()
        self.assertLess(
            max_diff, 0.05,
            f"eval モードで出力差が大きすぎる (max diff={max_diff:.2e})",
        )
        print(f"\n[決定性] eval モードで出力が近似一致 (max diff={max_diff:.2e}) ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
