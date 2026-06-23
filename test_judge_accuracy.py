#!/usr/bin/env python3
"""
QBNN Frontal Engine Judge - 精度分析テスト
判断の正確性、一貫性、バイアスを詳細に検証
"""

import json
from frontal_engine_mcp_server import FrontalEngineJudge


class JudgeAccuracyAnalyzer:
    """Judge機能の精度を分析するクラス"""

    def __init__(self):
        self.judge = FrontalEngineJudge()
        self.test_results = []

    def add_test(self, context: str, judgment_request: str, expected_decision: str,
                 test_name: str = "", expected_score_min: int = 0, expected_score_max: int = 100):
        """テストケースを追加して実行"""
        result = self.judge.judge({
            "context": context,
            "judgment_request": judgment_request
        })

        score = result.get("score")
        decision = result.get("decision")

        is_correct = decision == expected_decision
        is_score_in_range = expected_score_min <= score <= expected_score_max

        self.test_results.append({
            "name": test_name,
            "context": context,
            "expected_decision": expected_decision,
            "actual_decision": decision,
            "is_correct": is_correct,
            "expected_score_range": f"{expected_score_min}-{expected_score_max}",
            "actual_score": score,
            "is_score_correct": is_score_in_range,
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning")
        })

    def print_summary(self):
        """精度サマリーを表示"""
        total = len(self.test_results)
        correct_decisions = sum(1 for r in self.test_results if r["is_correct"])
        correct_scores = sum(1 for r in self.test_results if r["is_score_correct"])

        decision_accuracy = correct_decisions / total * 100
        score_accuracy = correct_scores / total * 100

        print("\n" + "=" * 70)
        print("【精度分析結果】")
        print("=" * 70)

        print(f"\n【1】判断精度 (Decision Accuracy)")
        print(f"  正解数: {correct_decisions}/{total}")
        print(f"  精度: {decision_accuracy:.1f}%")

        print(f"\n【2】スコア精度 (Score Accuracy)")
        print(f"  正解数: {correct_scores}/{total}")
        print(f"  精度: {score_accuracy:.1f}%")

        print(f"\n【3】総合精度")
        overall_accuracy = (decision_accuracy + score_accuracy) / 2
        print(f"  平均精度: {overall_accuracy:.1f}%")

        # 詳細な結果表示
        print(f"\n【4】テスト結果詳細")
        print("-" * 70)
        for i, result in enumerate(self.test_results, 1):
            status = "✓" if result["is_correct"] else "✗"
            print(f"\n{i}. {result['name']} {status}")
            print(f"   期待判断: {result['expected_decision']} → 実際: {result['actual_decision']}")
            print(f"   期待スコア範囲: {result['expected_score_range']} → 実際: {result['actual_score']}")
            print(f"   信頼度: {result['confidence']}")


def test_positive_cases():
    """ポジティブケースのテスト（Yes判断を期待）"""
    print("\n" + "=" * 70)
    print("テストカテゴリ 1: ポジティブケース（Yes判断を期待）")
    print("=" * 70)

    analyzer = JudgeAccuracyAnalyzer()

    # テスト1: プロジェクトリリース（ポジティブ）
    analyzer.add_test(
        context="プロジェクトは予定通り進行。品質基準すべて満たす。チーム士気高い。"
                "リスク要因なし。テスト完了率95%。本番準備完了。",
        judgment_request="このプロジェクトをリリースできるか？",
        expected_decision="Yes",
        test_name="プロジェクトリリース（ポジティブ）",
        expected_score_min=50,
        expected_score_max=100
    )

    # テスト2: 投資判断（ポジティブ）
    analyzer.add_test(
        context="初期投資¥5000万。予想ROI年¥10000万（200%）。市場成長率30%。"
                "競争力あり。実現性高い。リスク管理されている。",
        judgment_request="この投資は合理的か？",
        expected_decision="Yes",
        test_name="投資判断（ポジティブ）",
        expected_score_min=50,
        expected_score_max=100
    )

    # テスト3: 品質判定（高品質）
    analyzer.add_test(
        context="テストカバレッジ95%。重大バグなし。ドキュメント完全。"
                "パフォーマンス最適化済み。セキュリティ監査クリア。",
        judgment_request="このコードの品質は本番対応か？",
        expected_decision="Yes",
        test_name="品質判定（高品質）",
        expected_score_min=50,
        expected_score_max=100
    )

    # テスト4: 採用判断（適応者）
    analyzer.add_test(
        context="候補者スキル優秀。プロジェクト経験豊富。文化適応性良好。"
                "給与要求は適正。参照記録良好。すぐに貢献可能。",
        judgment_request="この候補者を採用すべきか？",
        expected_decision="Yes",
        test_name="採用判断（適応者）",
        expected_score_min=50,
        expected_score_max=100
    )

    # テスト5: 拡張判断（妥当な拡張）
    analyzer.add_test(
        context="新機能の需要が高い。実装難易度は中程度。リソースは十分。"
                "既存機能と統合可能。ユーザー需要が確認されている。",
        judgment_request="新機能を追加すべきか？",
        expected_decision="Yes",
        test_name="拡張判断（妥当な拡張）",
        expected_score_min=50,
        expected_score_max=100
    )

    analyzer.print_summary()
    return analyzer


def test_negative_cases():
    """ネガティブケースのテスト（No判断を期待）"""
    print("\n" + "=" * 70)
    print("テストカテゴリ 2: ネガティブケース（No判断を期待）")
    print("=" * 70)

    analyzer = JudgeAccuracyAnalyzer()

    # テスト1: プロジェクトリリース（ネガティブ）
    analyzer.add_test(
        context="プロジェクトに重大な問題がある。リスク要因が多数報告。"
                "テスト完了率30%。重大バグ存在。品質基準未達成。",
        judgment_request="このプロジェクトをリリースできるか？",
        expected_decision="No",
        test_name="プロジェクトリリース（ネガティブ）",
        expected_score_min=0,
        expected_score_max=50
    )

    # テスト2: リスク評価（許容不可）
    analyzer.add_test(
        context="新技術導入のリスク: 学習曲線が非常に急。互換性問題の可能性。"
                "短期的に生産性50%低下予測。高い離職リスク。",
        judgment_request="このリスクは許容可能か？",
        expected_decision="No",
        test_name="リスク評価（許容不可）",
        expected_score_min=0,
        expected_score_max=50
    )

    # テスト3: 廃止判断（廃止推奨）
    analyzer.add_test(
        context="古い機能。利用率5%以下。メンテナンスコスト高い。"
                "後継機能がある。ユーザーの要望なし。技術債務。",
        judgment_request="この機能を廃止すべきか？",
        expected_decision="Yes",  # 廃止=Yes
        test_name="廃止判断（廃止推奨）",
        expected_score_min=50,
        expected_score_max=100
    )

    # テスト4: 投資判断（ネガティブ）
    analyzer.add_test(
        context="初期投資¥5000万。予想ROI年¥500万（10%）。市場成長率-5%。"
                "競争激化。規制リスク。実現性不確実。",
        judgment_request="この投資は合理的か？",
        expected_decision="No",
        test_name="投資判断（ネガティブ）",
        expected_score_min=0,
        expected_score_max=50
    )

    # テスト5: 採用判断（不適合者）
    analyzer.add_test(
        context="候補者スキル不足。必要な経験なし。文化適応性疑問。"
                "参照記録に問題。給与要求高すぎる。すぐに貢献不可能。",
        judgment_request="この候補者を採用すべきか？",
        expected_decision="No",
        test_name="採用判断（不適合者）",
        expected_score_min=0,
        expected_score_max=50
    )

    analyzer.print_summary()
    return analyzer


def test_boundary_cases():
    """境界ケースのテスト（判断が分かれやすいケース）"""
    print("\n" + "=" * 70)
    print("テストカテゴリ 3: 境界ケース（判断が分かれる可能性）")
    print("=" * 70)

    analyzer = JudgeAccuracyAnalyzer()

    # テスト1: やや肯定的
    analyzer.add_test(
        context="プロジェクトはほぼ完了。いくつかの小さな問題がある。"
                "テスト完了率80%。品質はまあまあ。チーム疲弊している。",
        judgment_request="リリース可能か？",
        expected_decision="Yes",  # 判断: やや肯定（Yes）
        test_name="境界1: やや肯定的（リリース）",
        expected_score_min=45,
        expected_score_max=65
    )

    # テスト2: やや否定的
    analyzer.add_test(
        context="新技術にはリスクがあるが、導入すれば効率性が向上する。"
                "チームは学習意欲がある。段階的導入が可能。",
        judgment_request="導入すべきか？",
        expected_decision="Yes",  # 判断: わずかに肯定（Yes）
        test_name="境界2: わずかに肯定的（導入）",
        expected_score_min=45,
        expected_score_max=65
    )

    # テスト3: 複雑な判断
    analyzer.add_test(
        context="候補者スキルは平均的だが、成長ポテンシャルが高い。"
                "文化フィット良好。給与要求は適正。参照記録も悪くない。",
        judgment_request="採用すべきか？",
        expected_decision="Yes",  # 判断: 総合的にYes
        test_name="境界3: 複雑な判断（採用）",
        expected_score_min=45,
        expected_score_max=65
    )

    analyzer.print_summary()
    return analyzer


def test_consistency():
    """一貫性テスト（同じケースを複数回実行）"""
    print("\n" + "=" * 70)
    print("テストカテゴリ 4: 一貫性テスト")
    print("=" * 70)

    judge = FrontalEngineJudge()
    consistent = True
    first_result = None

    test_cases = [
        {
            "name": "一貫性テスト1",
            "context": "テスト背景情報1",
            "judgment_request": "判断1"
        },
        {
            "name": "一貫性テスト2",
            "context": "テスト背景情報2（より詳細）",
            "judgment_request": "判断2"
        }
    ]

    print("\n【テスト: 各ケースを5回実行して一貫性を確認】")
    for case in test_cases:
        print(f"\n{case['name']}:")
        results = []
        for i in range(5):
            result = judge.judge({
                "context": case["context"],
                "judgment_request": case["judgment_request"]
            })
            results.append((result["decision"], result["score"]))
            print(f"  実行{i+1}: Decision={result['decision']}, Score={result['score']}")

        # 一貫性チェック
        first = results[0]
        all_same = all(r == first for r in results)
        if all_same:
            print(f"  ✓ 5回すべて同じ結果 (Decision={first[0]}, Score={first[1]})")
        else:
            print(f"  ✗ 結果が異なる")
            consistent = False

    print("\n【総合評価】")
    if consistent:
        print("✓ 完全に一貫している - 高い信頼性")
    else:
        print("✗ 一貫性に問題がある")

    return consistent


def analyze_overall_accuracy():
    """全体的な精度を分析"""
    print("\n" + "=" * 70)
    print("【全体的な精度分析】")
    print("=" * 70)

    # すべてのテストカテゴリを実行
    print("\n実行中...")
    pos_analyzer = test_positive_cases()
    neg_analyzer = test_negative_cases()
    bound_analyzer = test_boundary_cases()
    consistency = test_consistency()

    # 総合精度を計算
    all_results = (pos_analyzer.test_results +
                   neg_analyzer.test_results +
                   bound_analyzer.test_results)

    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    score_correct = sum(1 for r in all_results if r["is_score_correct"])

    accuracy = correct / total * 100
    score_accuracy = score_correct / total * 100
    overall = (accuracy + score_accuracy) / 2

    print("\n" + "=" * 70)
    print("【最終的な精度評価】")
    print("=" * 70)
    print(f"\n判断精度 (Decision Accuracy): {accuracy:.1f}%")
    print(f"スコア精度 (Score Accuracy): {score_accuracy:.1f}%")
    print(f"総合精度 (Overall Accuracy): {overall:.1f}%")
    print(f"\n実行テスト数: {total}")
    print(f"正解数: {correct}/{total}")
    print(f"信頼性レベル: ", end="")

    if overall >= 90:
        print("⭐⭐⭐⭐⭐ (5/5) - エンタープライズレベル")
    elif overall >= 80:
        print("⭐⭐⭐⭐ (4/5) - 本番環境対応")
    elif overall >= 70:
        print("⭐⭐⭐ (3/5) - 本運用可能")
    elif overall >= 60:
        print("⭐⭐ (2/5) - 改善が必要")
    else:
        print("⭐ (1/5) - 要改善")

    print("\n" + "=" * 70)
    print(f"一貫性: {'✓ 完全 (100%)' if consistency else '✗ 問題あり'}")
    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  QBNN Frontal Engine Judge - 精度分析テスト".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    analyze_overall_accuracy()

    print("\n✅ 精度分析完了\n")
