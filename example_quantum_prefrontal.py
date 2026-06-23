#!/usr/bin/env python3
"""
Gemma+QBNN Quantum Prefrontal Cortex - Usage Examples
前頭葉として動作するGemma+QBNNの使用例

このスクリプトは、量子強化型前頭葉をClaude AI統合に使用する方法を示します。
"""

import sys
import os
import json
import asyncio
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

try:
    from gemma_qbnn_prefrontal_cortex import (
        GemmaQBNNPrefrontalCortex,
        JudgmentConfig,
        create_prefrontal_cortex
    )
    from claude_prefrontal_integration import ClaudePrefrontalCortex
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: {e}", file=sys.stderr)
    QUANTUM_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None


class QuantumPrefrontalDemo:
    """量子前頭葉システムのデモンストレーション"""

    def __init__(self):
        self.device = None
        self.cortex = None
        self.claude_cortex = None

        self._initialize()

    def _initialize(self):
        """システムを初期化"""
        if torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[初期化] Device: {self.device}")
        else:
            print("[初期化] PyTorchが利用不可 - CPU モードで実行")

        if QUANTUM_AVAILABLE:
            try:
                print("[初期化] Gemma+QBNN 前頭葉を作成中...")
                self.cortex = create_prefrontal_cortex(device=self.device)
                print("[初期化] ✓ 量子前頭葉システムの初期化完了")
            except Exception as e:
                print(f"[初期化] ✗ エラー: {e}", file=sys.stderr)
                self.cortex = None

            try:
                print("[初期化] Claude前頭葉統合を初期化中...")
                self.claude_cortex = ClaudePrefrontalCortex()
                print("[初期化] ✓ Claude前頭葉統合の初期化完了")
            except Exception as e:
                print(f"[初期化] ✗ エラー: {e}", file=sys.stderr)

    # =========================================================================
    # デモンストレーション
    # =========================================================================

    def demo_1_simple_judgment(self):
        """例1: シンプルな意思決定判断"""
        print("\n" + "=" * 70)
        print("デモ 1: シンプルな意思決定判断")
        print("=" * 70)

        if not self.cortex:
            print("✗ 量子前頭葉が利用不可")
            return

        judgment_task = {
            "context": """
            ユーザーが新しいプロジェクト提案を提出しました。
            提案は技術的に実現可能で、チームのスキルセットに合致しています。
            予算も確保されており、タイムラインも明確です。
            リスク分析では低リスクと評価されています。
            """,
            "judgment_request": "このプロジェクト提案を承認してもよいか？",
            "criteria": {
                "technical_feasibility": True,
                "budget_available": True,
                "risk_level": "low"
            },
            "strict_mode": False
        }

        print("\n【判断タスク】")
        print(f"質問: {judgment_task['judgment_request']}")
        print(f"コンテキスト: {judgment_task['context'][:100]}...")

        result = self.cortex.judge(judgment_task)

        print("\n【量子前頭葉の判断結果】")
        print(f"  決定: {result['decision']}")
        print(f"  スコア: {result['score']}/100")
        print(f"  信頼度: {result['confidence']}")
        print(f"  根拠: {result['reasoning']}")
        print(f"  主要要因: {', '.join(result['key_factors'])}")
        if "quantum_info" in result:
            print(f"  量子推論情報: Yes確率={result['quantum_info'].get('yes_probability', 0):.2%}")

    def demo_2_risk_assessment(self):
        """例2: リスク評価と安全性確認"""
        print("\n" + "=" * 70)
        print("デモ 2: リスク評価と安全性確認")
        print("=" * 70)

        if not self.cortex:
            print("✗ 量子前頭葉が利用不可")
            return

        judgment_task = {
            "context": """
            提案されたアクション: 本番データベースへの大規模スキーマ変更

            詳細:
            - 対象: 5000万件のレコード
            - 変更: 新しい必須カラムを追加
            - バックアップ: あり
            - ロールバック計画: あり
            - テスト環境での検証: 完了
            - 本番環境への段階的展開: あり
            - リスク: ダウンタイムの可能性あり（最大5分）

            制約条件:
            - 実行時刻: ピーク時間帯
            - ユーザー数: 現在5000人以上がアクティブ
            """,
            "judgment_request": "現在の状況下で、このスキーマ変更を実行するのは安全か？",
            "criteria": {
                "has_backup": True,
                "has_rollback_plan": True,
                "tested": True,
                "peak_hours": False
            },
            "strict_mode": True  # 厳密モード（安全重視）
        }

        print("\n【リスク評価タスク】")
        print(f"質問: {judgment_task['judgment_request']}")

        result = self.cortex.judge(judgment_task)

        print("\n【量子リスク分析結果】")
        print(f"  安全性評価: {result['decision']}")
        print(f"  リスクスコア: {result['score']}/100 (高いほどリスク低い)")
        print(f"  信頼度: {result['confidence']}")
        print(f"  推奨事項: {result['reasoning']}")

    def demo_3_ethical_judgment(self):
        """例3: 倫理的判断"""
        print("\n" + "=" * 70)
        print("デモ 3: 倫理的判断")
        print("=" * 70)

        if not self.cortex:
            print("✗ 量子前頭葉が利用不可")
            return

        judgment_task = {
            "context": """
            シナリオ: ユーザーの行動データ分析

            提案されたアクション:
            - ユーザーの閲覧履歴を分析
            - クリックパターンから個人的なニーズを推測
            - カスタマイズされた推奨を生成

            考慮要素:
            - プライバシー: ユーザーは明示的に同意していない
            - 利益: ユーザーはより関連性の高い推奨を受け取る
            - 透明性: 分析プロセスを明示していない
            - リスク: プライバシー侵害の可能性
            """,
            "judgment_request": "この行動分析と推奨生成は倫理的に適切か？",
            "criteria": {
                "user_consent": False,
                "privacy_risk": True,
                "transparency": False
            },
            "strict_mode": True
        }

        print("\n【倫理的判断タスク】")
        print(f"質問: {judgment_task['judgment_request']}")

        result = self.cortex.judge(judgment_task)

        print("\n【量子的倫理判断結果】")
        print(f"  判断: {result['decision']}")
        print(f"  倫理スコア: {result['score']}/100 (高いほど倫理的)")
        print(f"  信頼度: {result['confidence']}")
        print(f"  分析: {result['reasoning']}")

    def demo_4_priority_ranking(self):
        """例4: 複数の判断を組み合わせた優先順位付け"""
        print("\n" + "=" * 70)
        print("デモ 4: 複数タスクの優先順位付け（量子最適化）")
        print("=" * 70)

        if not self.cortex:
            print("✗ 量子前頭葉が利用不可")
            return

        tasks = [
            {
                "name": "本番バグ修正",
                "description": "クリティカルバグがユーザー5%に影響",
                "priority_context": "重大度: 高, 影響範囲: 広い, 修正難度: 低"
            },
            {
                "name": "新機能開発",
                "description": "要望が高い機能の実装",
                "priority_context": "重要度: 中, ユーザー満足度向上, 実装時間: 5日"
            },
            {
                "name": "セキュリティ監査",
                "description": "年次セキュリティ監査の実施",
                "priority_context": "重要性: 高, コンプライアンス要件, 緊急性: 中"
            }
        ]

        print("\n【優先順位付けの対象タスク】")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task['name']}: {task['description']}")

        # 各タスクを判断
        results = []
        for task in tasks:
            judgment_task = {
                "context": task['priority_context'],
                "judgment_request": f"タスク '{task['name']}' の優先度は高いか？"
            }

            result = self.cortex.judge(judgment_task)
            score = result.get('score', 50)
            results.append((task['name'], score))

        # スコアでソート
        results.sort(key=lambda x: x[1], reverse=True)

        print("\n【量子最適化による優先順位】")
        for rank, (task_name, score) in enumerate(results, 1):
            print(f"  {rank}. {task_name}: {score}点")

    def demo_5_hybrid_system(self):
        """例5: Claude統合での使用（ハイブリッドシステム）"""
        print("\n" + "=" * 70)
        print("デモ 5: Claude AI 統合（ハイブリッド前頭葉）")
        print("=" * 70)

        if not self.claude_cortex:
            print("✗ Claude前頭葉統合が利用不可")
            return

        print("\n【ハイブリッド前頭葉システム】")
        print("Claude AI メイン処理: 自然言語生成と文脈理解")
        print("Gemma+QBNN 前頭葉: 量子強化型判断と意思決定")

        status = self.claude_cortex.get_system_status()
        print(f"\n前頭葉統合ステータス:")
        print(f"  FrontalEngineが利用可能: {status['frontal_engine_available']}")
        print(f"  判断履歴数: {status['judgment_history_size']}")

    # =========================================================================
    # メイン実行
    # =========================================================================

    def run_all_demos(self):
        """すべてのデモを実行"""
        print("\n")
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "  Gemma+QBNN 量子強化前頭葉システム デモンストレーション".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")

        self.demo_1_simple_judgment()
        self.demo_2_risk_assessment()
        self.demo_3_ethical_judgment()
        self.demo_4_priority_ranking()
        self.demo_5_hybrid_system()

        print("\n" + "=" * 70)
        print("デモンストレーション完了")
        print("=" * 70)
        print("\n💡 次のステップ:")
        print("  1. フレームワークのプロパティをカスタマイズする")
        print("  2. ローカルモデルをダウンロードして利用する")
        print("  3. 本番環境でのテストを実施する")
        print("  4. Claude統合での実運用を開始する")


def main():
    """メイン関数"""
    demo = QuantumPrefrontalDemo()
    demo.run_all_demos()


if __name__ == "__main__":
    main()
