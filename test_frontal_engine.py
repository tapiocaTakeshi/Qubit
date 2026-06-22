#!/usr/bin/env python3
"""
QBNN Frontal Engine MCP Server - テストスクリプト
"""

import json
import asyncio
from frontal_engine_mcp_server import judge

def test_basic_judgment():
    """基本的な判断テスト"""
    print("=" * 60)
    print("Test 1: 基本的なYes/No判断")
    print("=" * 60)

    task = {
        "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。チームは高い士気を持ち、リスク要因は特に報告されていません。",
        "judgment_request": "このプロジェクトをリリースしても安全か？"
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def test_judgment_with_criteria():
    """基準付き判断テスト"""
    print("=" * 60)
    print("Test 2: 基準付き判断")
    print("=" * 60)

    task = {
        "context": "提案者の信頼スコア: 85/100、過去の成功率: 80%、提案の複雑度: 中程度",
        "judgment_request": "この提案を承認すべきか？",
        "criteria": {
            "trust_score": 80,
            "success_rate": "80%",
            "complexity": "manageable"
        }
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def test_risk_assessment():
    """リスク評価テスト"""
    print("=" * 60)
    print("Test 3: リスク評価")
    print("=" * 60)

    task = {
        "context": "新技術の導入には以下のリスクが考えられます: 学習曲線が急、既存システムとの互換性問題の可能性、短期的には生産性低下の予測。",
        "judgment_request": "この新技術を導入することのリスクは許容可能か？",
        "strict_mode": True
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def test_decision_with_options():
    """オプション付き決定テスト"""
    print("=" * 60)
    print("Test 4: 複数選択肢からの意思決定")
    print("=" * 60)

    task = {
        "context": "3つのベンダー候補を評価しました。ベンダーA: 価格安い、サポート弱い。ベンダーB: 価格中程度、サポート強い。ベンダーC: 価格高い、実績豊富。",
        "judgment_request": "ベンダーBを選択することが最適な判断か？",
        "options": ["ベンダーA", "ベンダーB", "ベンダーC"]
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def test_quality_judgment():
    """品質判定テスト"""
    print("=" * 60)
    print("Test 5: 品質・品質判定")
    print("=" * 60)

    task = {
        "context": "コードレビュー: テストカバレッジ95%、重大なバグなし、ドキュメント完全、パフォーマンス最適化済み。",
        "judgment_request": "このコードの品質は本番環境への展開に十分か？"
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def test_error_handling():
    """エラーハンドリングテスト"""
    print("=" * 60)
    print("Test 6: エラーハンドリング（必須パラメータなし）")
    print("=" * 60)

    task = {
        "context": "これは不完全な入力です"
        # judgment_request がない
    }

    result = judge.judge(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def run_all_tests():
    """すべてのテストを実行"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  QBNN Frontal Engine MCP Server - テストスイート".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    test_basic_judgment()
    test_judgment_with_criteria()
    test_risk_assessment()
    test_decision_with_options()
    test_quality_judgment()
    test_error_handling()

    print("=" * 60)
    print("すべてのテストが完了しました")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
