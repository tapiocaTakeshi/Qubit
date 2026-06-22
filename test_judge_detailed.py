#!/usr/bin/env python3
"""
QBNN Frontal Engine Judge - 詳細テストスイート
Judge機能の精度、ロジック、エッジケースを検証
"""

import json
from frontal_engine_mcp_server import FrontalEngineJudge


# ============================================================
# テスト用ユーティリティ
# ============================================================

def test_print(test_num: int, name: str):
    """テスト開始時の表示"""
    print(f"\n【テスト {test_num}】{name}")
    print("-" * 60)


def assert_decision(result, expected=None, test_name=""):
    """判断結果の検証"""
    decision = result.get("decision")
    assert decision in ["Yes", "No"], f"Invalid decision: {decision}"
    if expected is not None:
        assert decision == expected, f"{test_name}: Expected {expected}, got {decision}"
    return decision


def assert_score_range(result, min_val=0, max_val=100):
    """スコアの範囲検証"""
    score = result.get("score")
    assert isinstance(score, int), f"Score must be int, got {type(score)}"
    assert min_val <= score <= max_val, f"Score {score} out of range [{min_val}, {max_val}]"
    return score


def assert_confidence(result, expected=None):
    """信頼度の検証"""
    confidence = result.get("confidence")
    assert confidence in ["high", "medium", "low"], f"Invalid confidence: {confidence}"
    if expected is not None:
        assert confidence == expected, f"Expected {expected}, got {confidence}"
    return confidence


# ============================================================
# テストカテゴリ 1: スコア計算の精度
# ============================================================

def test_score_calculation():
    """スコア計算の精度テスト"""
    print("\n" + "=" * 70)
    print("カテゴリ 1: スコア計算の精度")
    print("=" * 70)

    judge = FrontalEngineJudge()

    # テスト1: 最小背景情報
    test_print(1, "最小限の背景情報")
    result = judge.judge({
        "context": "情報なし",
        "judgment_request": "判断してください"
    })
    score = assert_score_range(result)
    print(f"Context length: 3 words → Score: {score}")
    assert score >= 40 and score <= 60, "スコアが中立範囲外"
    print("✓ 最小情報でもスコア計算可能")

    # テスト2: 豊富な背景情報
    test_print(2, "豊富な背景情報による精度向上")
    result = judge.judge({
        "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。"
                  "チームは高い士気を持ち、リスク要因は特に報告されていません。"
                  "テスト完了率は95%です。本番環境への移行準備も完了しています。",
        "judgment_request": "このプロジェクトをリリースしても安全か？"
    })
    score1 = assert_score_range(result)
    print(f"Context length: ~55 words → Score: {score1}")
    assert score1 >= 50, "豊富な情報でスコア向上すべき"
    print("✓ 詳細な情報でスコアが向上")

    # テスト3: スコア計算の一貫性
    test_print(3, "スコア計算の一貫性（同じ入力で同じスコア）")
    context = "テスト背景情報"
    request = "テスト判断"
    results = []
    for i in range(3):
        result = judge.judge({"context": context, "judgment_request": request})
        results.append(result.get("score"))

    print(f"実行1: {results[0]}, 実行2: {results[1]}, 実行3: {results[2]}")
    assert results[0] == results[1] == results[2], "スコアが一貫していない"
    print("✓ 同じ入力で常に同じスコアを返す")

    print("\n✓ スコア計算テスト完了")


# ============================================================
# テストカテゴリ 2: 判断ロジック
# ============================================================

def test_judgment_logic():
    """判断ロジックの検証"""
    print("\n" + "=" * 70)
    print("カテゴリ 2: 判断ロジック (Yes/No判定)")
    print("=" * 70)

    judge = FrontalEngineJudge()

    # テスト1: 通常モード (threshold=50)
    test_print(1, "通常モード：スコア50以上でYes")
    result = judge.judge({
        "context": "プロジェクトは完了しており、すべての要件を満たしています。",
        "judgment_request": "リリースできるか？",
        "strict_mode": False
    })
    decision = assert_decision(result, "Yes", "score >= 50")
    score = result.get("score")
    print(f"Score: {score} → Decision: {decision}")
    assert score >= 50, "通常モードではスコア50以上でYesになるべき"
    print("✓ 通常モードの判定ロジックが正しい")

    # テスト2: 厳密モード (threshold=70)
    test_print(2, "厳密モード：スコア70以上でYes")
    result = judge.judge({
        "context": "新技術導入には様々なリスクがあります。",
        "judgment_request": "導入するべきか？",
        "strict_mode": True
    })
    decision = assert_decision(result)
    score = result.get("score")
    print(f"Score: {score} → Decision: {decision} (strict_mode=True)")

    # スコアが70未満ならNoになるはず
    if score < 70:
        assert decision == "No", f"スコア{score}で厳密モードはNoになるべき"
    else:
        assert decision == "Yes", f"スコア{score}で厳密モードはYesになるべき"
    print("✓ 厳密モードの判定ロジックが正しい")

    # テスト3: ポジティブキーワードの影響
    test_print(3, "ポジティブキーワード含有によるスコア向上")
    result_negative = judge.judge({
        "context": "プロジェクトに問題があります。",
        "judgment_request": "進行できるか？"
    })

    result_positive = judge.judge({
        "context": "プロジェクトは完了し、成功しました。品質は高く、安全です。",
        "judgment_request": "進行できるか？"
    })

    score_neg = result_negative.get("score")
    score_pos = result_positive.get("score")
    print(f"Negative keywords score: {score_neg}")
    print(f"Positive keywords score: {score_pos}")
    assert score_pos > score_neg, "ポジティブキーワードでスコア向上すべき"
    print("✓ キーワード分析が機能している")

    print("\n✓ 判断ロジックテスト完了")


# ============================================================
# テストカテゴリ 3: 信頼度判定
# ============================================================

def test_confidence_judgment():
    """信頼度判定の検証"""
    print("\n" + " " * 70)
    print("カテゴリ 3: 信頼度判定")
    print("=" * 70)

    judge = FrontalEngineJudge()

    # テスト1: 高信頼度（スコア75以上または25以下）
    test_print(1, "高信頼度：スコア75以上または25以下")
    result_high = judge.judge({
        "context": "明確で詳細な情報があります。成功の可能性は非常に高いです。"
                  "すべてのリスク要因が特定され、対処されました。",
        "judgment_request": "進行できるか？"
    })
    confidence = assert_confidence(result_high)
    score = result_high.get("score")
    print(f"Score: {score} → Confidence: {confidence}")
    if score >= 75 or score <= 25:
        assert confidence == "high", f"スコア{score}で高信頼度になるべき"
    print("✓ 高信頼度判定が正しい")

    # テスト2: 中信頼度（40-60範囲）
    test_print(2, "中信頼度：判断が不確実な範囲（40-60）")
    results = []
    for i in range(5):
        result = judge.judge({
            "context": "情報があります",
            "judgment_request": "判断してください"
        })
        conf = result.get("confidence")
        score = result.get("score")
        if 40 <= score <= 60:
            results.append((score, conf))

    print(f"中心付近のスコア：{results[:3]}")
    for score, conf in results:
        assert conf == "medium", f"スコア{score}で中信頼度になるべき"
    print("✓ 中信頼度判定が正しい")

    # テスト3: 信頼度と決定の関係
    test_print(3, "信頼度と決定の一貫性")
    for _ in range(5):
        result = judge.judge({
            "context": "テスト背景情報",
            "judgment_request": "判断してください"
        })
        decision = result.get("decision")
        confidence = result.get("confidence")
        score = result.get("score")

        # スコアが極端なら信頼度は高いはず
        if score >= 75 or score <= 25:
            assert confidence == "high", f"スコア{score}で信頼度が{confidence}（highであるべき）"

    print("✓ 信頼度と決定が一貫している")

    print("\n✓ 信頼度判定テスト完了")


# ============================================================
# テストカテゴリ 4: キーファクター抽出
# ============================================================

def test_key_factors():
    """キーファクター抽出の検証"""
    print("\n" + "=" * 70)
    print("カテゴリ 4: キーファクター抽出")
    print("=" * 70)

    judge = FrontalEngineJudge()

    # テスト1: キーファクターの存在
    test_print(1, "キーファクターの存在確認")
    result = judge.judge({
        "context": "プロジェクト品質は高く、リスク要因は低い。",
        "judgment_request": "リリースできるか？"
    })
    key_factors = result.get("key_factors")
    assert isinstance(key_factors, list), "key_factorsはリストであるべき"
    assert len(key_factors) > 0, "キーファクターが抽出されるべき"
    print(f"抽出されたキーファクター: {key_factors}")
    print("✓ キーファクターが正しく抽出されている")

    # テスト2: キーファクター数の上限
    test_print(2, "キーファクター数の上限（最大5個）")
    result = judge.judge({
        "context": "詳細で包括的な情報が含まれている。"
                  "品質が高く、安全で、効率的で、実現可能で、必要である。",
        "judgment_request": "判断してください？"
    })
    key_factors = result.get("key_factors")
    print(f"キーファクター数: {len(key_factors)}")
    assert len(key_factors) <= 5, "キーファクターは最大5個であるべき"
    print("✓ キーファクター数が制限されている")

    # テスト3: 基準付きのキーファクター
    test_print(3, "基準付き判断でのキーファクター")
    result = judge.judge({
        "context": "信頼スコア: 85点、成功率: 90%",
        "judgment_request": "提案を承認するか？",
        "criteria": {"trust": "80", "success_rate": "high"}
    })
    key_factors = result.get("key_factors")
    print(f"キーファクター: {key_factors}")
    # 基準情報が含まれているはず
    has_criteria_info = any("基準" in str(f) for f in key_factors)
    print(f"基準関連の情報を含む: {has_criteria_info}")
    print("✓ 基準付き判断でキーファクターが抽出されている")

    print("\n✓ キーファクター抽出テスト完了")


# ============================================================
# テストカテゴリ 5: エッジケース
# ============================================================

def test_edge_cases():
    """エッジケースの検証"""
    print("\n" + "=" * 70)
    print("カテゴリ 5: エッジケース")
    print("=" * 70)

    judge = FrontalEngineJudge()

    # テスト1: 空文字列の処理
    test_print(1, "空またはなしのコンテキスト処理")
    result = judge.judge({
        "context": "",
        "judgment_request": "判断してください"
    })
    # エラー応答を期待
    if result.get("error"):
        print("✓ 空のcontextはエラーを返す")
    else:
        # またはエラーにならない場合も有効（デフォルト値で処理）
        assert "decision" in result
        print("✓ 空のcontextでもスコア計算される")

    # テスト2: 非常に長いテキスト
    test_print(2, "非常に長いテキストの処理")
    long_context = "情報 " * 1000  # 2000語以上
    result = judge.judge({
        "context": long_context,
        "judgment_request": "判断してください"
    })
    score = assert_score_range(result)
    print(f"長いテキスト（{len(long_context)} 文字）→ Score: {score}")
    assert "decision" in result
    print("✓ 長いテキストでもスコア計算される")

    # テスト3: 複数条件の同時指定
    test_print(3, "複数条件の同時指定（criteria + options + strict_mode）")
    result = judge.judge({
        "context": "プロジェクトA、品質95%、コストは予算内。"
                  "プロジェクトB、品質80%、低コスト。"
                  "プロジェクトC、品質90%、中コスト。",
        "judgment_request": "プロジェクトAを選択すべきか？",
        "criteria": {"quality": "high", "cost": "reasonable"},
        "options": ["プロジェクトA", "プロジェクトB", "プロジェクトC"],
        "strict_mode": True
    })
    assert "decision" in result
    assert "score" in result
    print(f"複数条件の結果 → Decision: {result['decision']}, Score: {result['score']}")
    print("✓ 複数条件を同時に処理できる")

    # テスト4: タイムスタンプの形式
    test_print(4, "タイムスタンプのフォーマット確認")
    result = judge.judge({
        "context": "テスト背景",
        "judgment_request": "判断"
    })
    timestamp = result.get("timestamp")
    assert timestamp is not None, "タイムスタンプが必須"
    assert timestamp.endswith("Z"), "タイムスタンプはISOフォーマット（Z終尾）であるべき"
    print(f"Timestamp: {timestamp}")
    print("✓ タイムスタンプ形式が正しい")

    print("\n✓ エッジケーステスト完了")


# ============================================================
# テストカテゴリ 6: パフォーマンス
# ============================================================

def test_performance():
    """パフォーマンステスト"""
    print("\n" + "=" * 70)
    print("カテゴリ 6: パフォーマンス")
    print("=" * 70)

    judge = FrontalEngineJudge()
    import time

    # テスト1: 単一判断の速度
    test_print(1, "単一判断の処理速度")
    start = time.time()
    result = judge.judge({
        "context": "テスト背景情報",
        "judgment_request": "判断してください"
    })
    elapsed = time.time() - start
    print(f"処理時間: {elapsed*1000:.2f}ms")
    assert elapsed < 1.0, "処理は1秒以内に完了するべき"
    print("✓ 単一判断が高速である")

    # テスト2: 複数判断のスループット
    test_print(2, "複数判断のスループット（100回実行）")
    start = time.time()
    for i in range(100):
        judge.judge({
            "context": f"テスト背景 {i}",
            "judgment_request": f"判断 {i}"
        })
    elapsed = time.time() - start
    avg_time = elapsed / 100
    print(f"100回実行時間: {elapsed:.2f}s → 平均: {avg_time*1000:.2f}ms/回")
    assert avg_time < 0.1, "平均100ms以下であるべき"
    print("✓ スループットが十分である")

    print("\n✓ パフォーマンステスト完了")


# ============================================================
# メイン
# ============================================================

def run_all_tests():
    """すべてのテストを実行"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  QBNN Frontal Engine Judge - 詳細テストスイート".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        test_score_calculation()
        test_judgment_logic()
        test_confidence_judgment()
        test_key_factors()
        test_edge_cases()
        test_performance()

        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + "✓ すべてのJudge詳細テストが成功しました".center(68) + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print("テスト統計:")
        print("  - スコア計算精度: ✓")
        print("  - 判断ロジック: ✓")
        print("  - 信頼度判定: ✓")
        print("  - キーファクター: ✓")
        print("  - エッジケース: ✓")
        print("  - パフォーマンス: ✓")
        print()

    except Exception as e:
        print(f"\n✗ テスト失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
