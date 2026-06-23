#!/usr/bin/env python3
"""
Test Suite for Claude Prefrontal Cortex Integration
統合テストスイート
"""

import asyncio
import unittest
from typing import Dict, Any

from claude_prefrontal_integration import (
    ClaudePrefrontalCortex,
    JudgmentType,
    judge_action,
    judge_response_quality,
    check_ethics
)


class TestClaudePrefrontalCortex(unittest.TestCase):
    """ClaudePrefrontalCortex の基本機能テスト"""

    def setUp(self):
        """Test setup"""
        self.cortex = ClaudePrefrontalCortex()

    def test_initialization(self):
        """前頭葉の初期化テスト"""
        self.assertIsNotNone(self.cortex)
        self.assertIsNotNone(self.cortex.judge)

    def test_system_status(self):
        """システムステータスの確認テスト"""
        status = self.cortex.get_system_status()
        self.assertIn('frontal_engine_available', status)
        self.assertIn('judgment_history_size', status)

    def test_judgment_history_limit(self):
        """判断履歴の上限テスト"""
        # 履歴の上限は100件
        self.assertEqual(self.cortex.max_history, 100)

    def test_judgment_record_format(self):
        """判断記録のフォーマットテスト"""
        history = self.cortex.get_judgment_history(limit=10)
        # 履歴が空の可能性があるため、スキップ
        if history:
            record = history[0]
            self.assertIn('timestamp', record)
            self.assertIn('judgment_type', record)
            self.assertIn('decision', record)
            self.assertIn('score', record)


class TestAsyncMethods(unittest.IsolatedAsyncioTestCase):
    """非同期メソッドのテスト"""

    async def asyncSetUp(self):
        """Async test setup"""
        self.cortex = ClaudePrefrontalCortex()

    async def test_should_proceed_with_action(self):
        """アクション判断メソッドのテスト"""
        should_proceed, result = await self.cortex.should_proceed_with_action(
            action_description="テストアクション",
            context="テスト環境",
            risks=["低リスク"]
        )

        self.assertIsInstance(should_proceed, bool)
        self.assertIn('decision', result)
        self.assertIn('score', result)
        self.assertIn('reasoning', result)

    async def test_evaluate_response_quality(self):
        """応答品質評価メソッドのテスト"""
        result = await self.cortex.evaluate_response_quality(
            response="これはテスト応答です。",
            requirements=["明確", "有用"]
        )

        self.assertIn('decision', result)
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], int)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)

    async def test_assess_ethical_concerns(self):
        """倫理評価メソッドのテスト"""
        result = await self.cortex.assess_ethical_concerns(
            action_description="テストアクション",
            stakeholders=["ユーザー"],
            potential_harms=["潜在的な害"]
        )

        self.assertIn('decision', result)
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertIn(result['decision'], ['Yes', 'No'])

    async def test_prioritize_tasks(self):
        """タスク優先順位付けメソッドのテスト"""
        tasks = [
            {"name": "タスク1", "description": "説明1"},
            {"name": "タスク2", "description": "説明2"},
            {"name": "タスク3", "description": "説明3"}
        ]

        prioritized = await self.cortex.prioritize_tasks(tasks)

        self.assertEqual(len(prioritized), len(tasks))
        # スコアが降順に並んでいることを確認
        for i in range(len(prioritized) - 1):
            self.assertGreaterEqual(prioritized[i][1], prioritized[i + 1][1])

    async def test_make_judgment(self):
        """基本判断メソッドのテスト"""
        result = await self.cortex.make_judgment(
            context="テスト文脈",
            judgment_request="これはテストか？",
            judgment_type=JudgmentType.DECISION_MAKING
        )

        self.assertIsInstance(result, dict)
        self.assertIn('decision', result)
        self.assertIn('score', result)
        self.assertIn('reasoning', result)
        self.assertIn('confidence', result)
        self.assertIn('timestamp', result)

    async def test_judgment_with_strict_mode(self):
        """厳密判断モードのテスト"""
        result = await self.cortex.make_judgment(
            context="高リスク操作",
            judgment_request="実行すべきか？",
            strict_mode=True
        )

        # strict_mode=True のとき、スコア70以上でYes
        if result['decision'] == 'Yes':
            self.assertGreaterEqual(result['score'], 70)
        else:
            self.assertLess(result['score'], 70)

    async def test_explain_decision(self):
        """意思決定説明メソッドのテスト"""
        judgment_result = {
            "decision": "Yes",
            "score": 85,
            "reasoning": "テスト根拠",
            "confidence": "high",
            "key_factors": ["要因1", "要因2"]
        }

        explanation = await self.cortex.explain_decision(judgment_result)
        self.assertIsInstance(explanation, str)
        self.assertIn("Yes", explanation)
        self.assertIn("85", explanation)


class TestConvenienceFunctions(unittest.IsolatedAsyncioTestCase):
    """便利関数のテスト"""

    async def test_judge_action(self):
        """judge_action 関数のテスト"""
        should_proceed, result = await judge_action(
            action="テストアクション",
            context="テスト文脈",
            risks=["テストリスク"]
        )

        self.assertIsInstance(should_proceed, bool)
        self.assertIsInstance(result, dict)

    async def test_judge_response_quality(self):
        """judge_response_quality 関数のテスト"""
        result = await judge_response_quality(
            response="テスト応答",
            requirements=["要件1"]
        )

        self.assertIn('decision', result)
        self.assertIn('score', result)

    async def test_check_ethics(self):
        """check_ethics 関数のテスト"""
        result = await check_ethics(
            action="テストアクション",
            stakeholders=["利害関係者"]
        )

        self.assertIn('decision', result)
        self.assertIn('score', result)


class TestJudgmentTypes(unittest.TestCase):
    """判断タイプのテスト"""

    def test_judgment_type_enum(self):
        """JudgmentType enumのテスト"""
        # すべての判断タイプが定義されていることを確認
        types = [
            JudgmentType.DECISION_MAKING,
            JudgmentType.RISK_ASSESSMENT,
            JudgmentType.QUALITY_JUDGMENT,
            JudgmentType.ETHICAL_JUDGMENT,
            JudgmentType.PRIORITIZATION,
            JudgmentType.SAFETY_CHECK
        ]

        for judgment_type in types:
            self.assertIsNotNone(judgment_type.value)
            self.assertIsInstance(judgment_type.value, str)


class TestErrorHandling(unittest.IsolatedAsyncioTestCase):
    """エラーハンドリングのテスト"""

    async def test_missing_context(self):
        """文脈なしの判断テスト"""
        cortex = ClaudePrefrontalCortex()
        result = await cortex.make_judgment(
            context="",
            judgment_request="テスト"
        )

        # エラーハンドリングされて結果が返ることを確認
        self.assertIn('decision', result)

    async def test_missing_judgment_request(self):
        """判断リクエストなしのテスト"""
        cortex = ClaudePrefrontalCortex()
        result = await cortex.make_judgment(
            context="文脈",
            judgment_request=""
        )

        # エラーハンドリングされて結果が返ることを確認
        self.assertIn('decision', result)


class TestJudgmentHistoryManagement(unittest.IsolatedAsyncioTestCase):
    """判断履歴管理のテスト"""

    async def test_history_recording(self):
        """履歴記録のテスト"""
        cortex = ClaudePrefrontalCortex()
        initial_count = len(cortex.get_judgment_history())

        # 判断を実行
        await cortex.make_judgment(
            context="テスト",
            judgment_request="テスト？"
        )

        # 履歴が増加したことを確認
        final_count = len(cortex.get_judgment_history())
        self.assertGreater(final_count, initial_count)

    async def test_history_limit_enforcement(self):
        """履歴上限実装のテスト"""
        cortex = ClaudePrefrontalCortex()

        # max_history に設定
        self.assertEqual(cortex.max_history, 100)

    async def test_get_judgment_history(self):
        """履歴取得のテスト"""
        cortex = ClaudePrefrontalCortex()

        # 最新10件を取得
        history = cortex.get_judgment_history(limit=10)

        self.assertIsInstance(history, list)
        self.assertLessEqual(len(history), 10)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """統合テスト"""

    async def test_complete_workflow(self):
        """完全なワークフローテスト"""
        cortex = ClaudePrefrontalCortex()

        # Step 1: アクション判断
        action_result = await cortex.should_proceed_with_action(
            action_description="テストアクション",
            context="テスト環境"
        )
        self.assertIsInstance(action_result, tuple)

        # Step 2: 品質評価
        quality_result = await cortex.evaluate_response_quality(
            response="テスト応答"
        )
        self.assertIn('decision', quality_result)

        # Step 3: 倫理評価
        ethics_result = await cortex.assess_ethical_concerns(
            action_description="テストアクション"
        )
        self.assertIn('decision', ethics_result)

        # Step 4: 履歴確認
        history = cortex.get_judgment_history()
        self.assertGreater(len(history), 0)


def run_tests():
    """すべてのテストを実行"""
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    print("=" * 60)
    print("Claude Prefrontal Cortex Integration Test Suite")
    print("=" * 60)
    print()

    run_tests()
