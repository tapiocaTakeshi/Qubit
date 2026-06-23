#!/usr/bin/env python3
"""
Quantum Prefrontal Cortex - Unit Tests
量子前頭葉システムのユニットテスト
"""

import sys
import os
import unittest
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from gemma_qbnn_prefrontal_cortex import (
        GemmaQBNNPrefrontalCortex,
        JudgmentConfig,
        JudgmentHead,
        create_prefrontal_cortex
    )
    from frontal_engine_mcp_server import FrontalEngineJudge
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}", file=sys.stderr)
    QUANTUM_AVAILABLE = False


class TestJudgmentConfig(unittest.TestCase):
    """JudgmentConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定の確認"""
        config = JudgmentConfig()

        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.embed_dim, 768)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.quantum_weight, 0.6)
        self.assertEqual(config.entangle_strength, 0.7)

    def test_custom_config(self):
        """カスタム設定の確認"""
        config = JudgmentConfig(
            vocab_size=16000,
            embed_dim=512,
            quantum_weight=0.8
        )

        self.assertEqual(config.vocab_size, 16000)
        self.assertEqual(config.embed_dim, 512)
        self.assertEqual(config.quantum_weight, 0.8)


class TestJudgmentHead(unittest.TestCase):
    """JudgmentHeadのテスト"""

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_judgment_head_forward(self):
        """JudgmentHeadの順伝播テスト"""
        head = JudgmentHead(embed_dim=768)

        # テスト入力
        hidden_state = torch.randn(2, 768)  # batch_size=2

        output = head(hidden_state)

        # 出力の確認
        self.assertIn("decision_logits", output)
        self.assertIn("score_logits", output)
        self.assertIn("confidence_logits", output)
        self.assertIn("reasoning_logits", output)

        # 形状の確認
        self.assertEqual(output["decision_logits"].shape, (2, 2))  # Yes/No
        self.assertEqual(output["score_logits"].shape, (2, 1))    # スコア
        self.assertEqual(output["confidence_logits"].shape, (2, 3))  # low/medium/high


class TestQuantumPrefrontalCortex(unittest.TestCase):
    """GemmaQBNNPrefrontalCortexのテスト"""

    @unittest.skipUnless(QUANTUM_AVAILABLE, "Quantum prefrontal cortex not available")
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_cortex_initialization(self):
        """前頭葉システムの初期化テスト"""
        device = torch.device("cpu")
        config = JudgmentConfig(embed_dim=256, hidden_dim=512, num_layers=2)

        cortex = GemmaQBNNPrefrontalCortex(config)
        cortex = cortex.to(device)

        self.assertIsNotNone(cortex)
        self.assertEqual(cortex.config.vocab_size, 32000)

    @unittest.skipUnless(QUANTUM_AVAILABLE, "Quantum prefrontal cortex not available")
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_judgment_task(self):
        """判断タスクのテスト"""
        device = torch.device("cpu")
        config = JudgmentConfig(embed_dim=256, hidden_dim=512, num_layers=2)

        cortex = GemmaQBNNPrefrontalCortex(config)
        cortex = cortex.to(device)
        cortex.eval()

        judgment_task = {
            "context": "テストコンテキスト。これは安全なアクションです。",
            "judgment_request": "このアクションを実行するべきか？",
            "strict_mode": False
        }

        result = cortex.judge(judgment_task)

        # 出力の確認
        self.assertIn("decision", result)
        self.assertIn("score", result)
        self.assertIn("reasoning", result)
        self.assertIn("confidence", result)

        # 値の確認
        self.assertIn(result["decision"], ["Yes", "No"])
        self.assertTrue(0 <= result["score"] <= 100)
        self.assertIn(result["confidence"], ["low", "medium", "high"])

    @unittest.skipUnless(QUANTUM_AVAILABLE, "Quantum prefrontal cortex not available")
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_strict_mode(self):
        """厳密モードのテスト"""
        device = torch.device("cpu")
        config = JudgmentConfig(embed_dim=256, hidden_dim=512, num_layers=2)

        cortex = GemmaQBNNPrefrontalCortex(config)
        cortex = cortex.to(device)
        cortex.eval()

        # 厳密モード
        judgment_task = {
            "context": "明確に肯定的な状況",
            "judgment_request": "これを承認するべきか？",
            "strict_mode": True
        }

        result = cortex.judge(judgment_task)

        # 厳密モードではスコア70以上でYes
        if result["score"] >= 70:
            self.assertEqual(result["decision"], "Yes")
        else:
            self.assertEqual(result["decision"], "No")


class TestFrontalEngineJudge(unittest.TestCase):
    """FrontalEngineJudgeのテスト"""

    def test_judge_initialization(self):
        """FrontalEngineJudgeの初期化テスト"""
        judge = FrontalEngineJudge()
        self.assertIsNotNone(judge)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_judge_with_quantum(self):
        """量子判断エンジンのテスト"""
        judge = FrontalEngineJudge()

        judgment_task = {
            "context": "これは重要なアクションです。",
            "judgment_request": "このアクションは安全か？",
            "strict_mode": False
        }

        result = judge.judge(judgment_task)

        # 出力の確認
        self.assertIn("decision", result)
        self.assertIn("score", result)
        self.assertIn("reasoning", result)
        self.assertIn("confidence", result)

        # 値の確認
        if "error" not in result:
            self.assertIn(result["decision"], ["Yes", "No"])
            self.assertTrue(0 <= result["score"] <= 100)
            self.assertIn(result["confidence"], ["low", "medium", "high"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        judge = FrontalEngineJudge()

        # 必須フィールドが欠けている
        judgment_task = {
            "context": "テストコンテキスト"
            # judgment_request が欠けている
        }

        result = judge.judge(judgment_task)

        # エラーレスポンスの確認
        self.assertEqual(result["decision"], "No")
        self.assertTrue(result.get("error", False) or "error" in result.get("reasoning", "").lower())


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    @unittest.skipUnless(QUANTUM_AVAILABLE and TORCH_AVAILABLE, "Requirements not available")
    def test_cortex_and_judge_compatibility(self):
        """前頭葉システムと判断エンジンの互換性テスト"""
        judge = FrontalEngineJudge()

        # 判断エンジンが量子前頭葉を使用しているか確認
        if judge.use_quantum:
            self.assertIsNotNone(judge.quantum_cortex)
            print("✓ 量子前頭葉が有効")
        else:
            print("✓ フォールバックモード使用")


def run_tests():
    """テストスイートを実行"""
    # テストスイートを作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # テストを追加
    suite.addTests(loader.loadTestsFromTestCase(TestJudgmentConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestJudgmentHead))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumPrefrontalCortex))
    suite.addTests(loader.loadTestsFromTestCase(TestFrontalEngineJudge))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # テストを実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
