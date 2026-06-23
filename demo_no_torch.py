#!/usr/bin/env python3
"""
Quantum Prefrontal Cortex Demo - PyTorch不要版
簡単な動作確認とシミュレーション
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any


class QuantumSimulator:
    """量子前頭葉のシミュレータ（PyTorch不要版）"""

    def __init__(self):
        self.decisions_made = 0
        self.total_score = 0

    def simulate_quantum_judgment(
        self,
        context: str,
        judgment_request: str,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """量子判断をシミュレート"""

        # テキスト長に基づくスコア計算
        context_len = len(context)
        request_len = len(judgment_request)
        base_score = 50

        # キーワード分析
        positive_keywords = ["安全", "確認", "承認", "完了", "テスト済み", "バックアップ", "問題なし"]
        negative_keywords = ["危険", "リスク", "問題", "未確認", "テスト未了", "同意なし"]

        for keyword in positive_keywords:
            if keyword in context or keyword in judgment_request:
                base_score += 8

        for keyword in negative_keywords:
            if keyword in context or keyword in judgment_request:
                base_score -= 8

        # エンタングルメント効果（複雑性ボーナス）
        if context_len > 200:
            base_score += 5
        if request_len > 50:
            base_score += 3

        # スコアを制限
        score = max(0, min(100, int(base_score)))

        # 決定を決定
        if strict_mode:
            decision = "Yes" if score >= 70 else "No"
            confidence = "high" if score >= 80 or score <= 20 else "medium"
        else:
            decision = "Yes" if score >= 50 else "No"
            confidence = "high" if score >= 75 or score <= 25 else "medium"

        # 信頼度の調整
        if confidence == "high" and 40 <= score <= 60:
            confidence = "medium"

        # 根拠を生成
        if score >= 70:
            reasoning = "量子推論により、提供された情報は肯定的な判断を支持しています。"
        elif score >= 50:
            reasoning = "量子推論の結果、判断は不確定ですが、妥当な結論が導き出されます。"
        else:
            reasoning = "量子推論により、提供された情報は否定的な判断を支持しています。"

        # キーファクターを抽出
        key_factors = []
        if context_len > 200:
            key_factors.append("複雑なコンテキスト")
        if "テスト" in context:
            key_factors.append("テスト検証完了")
        if "バックアップ" in context:
            key_factors.append("バックアップ確保")
        if "リスク" in judgment_request:
            key_factors.append("リスク評価")

        key_factors.append("量子推論適用")

        self.decisions_made += 1
        self.total_score += score

        return {
            "decision": decision,
            "score": score,
            "reasoning": reasoning,
            "confidence": confidence,
            "key_factors": key_factors[:5],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "quantum_info": {
                "yes_probability": score / 100.0,
                "quantum_weight": 0.6,
                "entangle_strength": 0.7,
                "context_complexity": min(100, int(context_len / 10)),
            }
        }


def print_result(title: str, result: Dict[str, Any]):
    """結果をフォーマットして表示"""
    print(f"\n【{title}】")
    print(f"  決定: {result['decision']}")
    print(f"  スコア: {result['score']}/100")
    print(f"  信頼度: {result['confidence']}")
    print(f"  根拠: {result['reasoning']}")
    print(f"  主要要因: {', '.join(result['key_factors'])}")
    if "quantum_info" in result:
        print(f"  量子情報: Yes確率={result['quantum_info']['yes_probability']:.1%}, "
              f"複雑度={result['quantum_info']['context_complexity']}/100")


def main():
    """メイン実行"""
    print("\n" + "=" * 70)
    print("Gemma+QBNN 量子強化前頭葉システム - デモンストレーション")
    print("=" * 70)

    simulator = QuantumSimulator()

    # デモ 1: シンプルな意思決定
    print("\n【デモ1】シンプルな意思決定判断")
    result1 = simulator.simulate_quantum_judgment(
        context="""
        ユーザーが新しいプロジェクト提案を提出しました。
        提案は技術的に実現可能で、チームのスキルセットに合致しています。
        予算も確保されており、タイムラインも明確です。
        テスト環境での検証も完了しています。
        """,
        judgment_request="このプロジェクト提案を承認してもよいか？"
    )
    print_result("シンプル意思決定の結果", result1)

    # デモ 2: リスク評価
    print("\n【デモ2】リスク評価と安全性確認")
    result2 = simulator.simulate_quantum_judgment(
        context="""
        提案されたアクション: 本番データベースへのスキーマ変更
        - 対象: 5000万件のレコード
        - 変更: 新しいカラムを追加
        - バックアップ: あり
        - ロールバック計画: あり
        - テスト環境での検証: 完了
        - 本番環境への段階的展開: あり
        """,
        judgment_request="現在の状況下で、このスキーマ変更を実行するのは安全か？",
        strict_mode=True
    )
    print_result("リスク評価の結果", result2)

    # デモ 3: 倫理的判断
    print("\n【デモ3】倫理的判断")
    result3 = simulator.simulate_quantum_judgment(
        context="""
        シナリオ: ユーザーの行動データ分析
        - ユーザーの閲覧履歴を分析
        - クリックパターンから個人的なニーズを推測
        - プライバシー: ユーザーは明示的に同意していない
        - 透明性: 分析プロセスを明示していない
        - リスク: プライバシー侵害の可能性
        """,
        judgment_request="この行動分析は倫理的に適切か？",
        strict_mode=True
    )
    print_result("倫理判断の結果", result3)

    # デモ 4: 複数タスクの優先順位付け
    print("\n【デモ4】複数タスクの優先順位付け")
    tasks = [
        {
            "name": "本番バグ修正",
            "context": "クリティカルバグがユーザー5%に影響。テスト完了済み。"
        },
        {
            "name": "新機能開発",
            "context": "ユーザーから多くの要望。予算確保済み。スケジュール確認中。"
        },
        {
            "name": "セキュリティ監査",
            "context": "年次要件。リスク評価が必要。テスト環境で検証予定。"
        }
    ]

    print(f"\n対象タスク:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task['name']}")

    results_priority = []
    for task in tasks:
        result = simulator.simulate_quantum_judgment(
            context=task['context'],
            judgment_request=f"タスク '{task['name']}' の優先度は高いか？"
        )
        results_priority.append((task['name'], result['score']))

    results_priority.sort(key=lambda x: x[1], reverse=True)
    print(f"\n優先順位付け結果:")
    for rank, (task_name, score) in enumerate(results_priority, 1):
        print(f"  {rank}. {task_name}: {score}点")

    # 統計情報
    print("\n" + "=" * 70)
    print("統計情報")
    print("=" * 70)
    avg_score = simulator.total_score / simulator.decisions_made if simulator.decisions_made > 0 else 0
    print(f"判断実行数: {simulator.decisions_made}")
    print(f"平均スコア: {avg_score:.1f}/100")
    print(f"システム状態: ✓ 正常（量子シミュレータ稼働中）")

    # 最終メッセージ
    print("\n" + "=" * 70)
    print("デモンストレーション完了 ✨")
    print("=" * 70)
    print("""
このシミュレータは PyTorch 不要で Gemma+QBNN 前頭葉の動作を示します。

実際の PyTorch + CUDA 環境では:
  $ pip install torch
  $ python example_quantum_prefrontal.py

次のステップ:
  1. PyTorch をインストール
  2. 実際のモデルを読み込む
  3. MCP サーバーを起動
  4. Claude AI に統合
    """)


if __name__ == "__main__":
    main()
