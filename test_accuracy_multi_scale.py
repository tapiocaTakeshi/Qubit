#!/usr/bin/env python3
"""
QBNN Frontal Engine Judge - 複数スケールでの精度測定
10, 50, 100, 200, 500, 1000回での精度測定
"""

import random
import time
from collections import defaultdict
from frontal_engine_mcp_server import FrontalEngineJudge


class MultiScaleAccuracyTest:
    """複数スケールでの精度テスト"""

    def __init__(self):
        self.judge = FrontalEngineJudge()
        self.results = {}

        # キーワードベース
        self.positive_keywords = [
            "完璧", "優秀", "高い", "成功", "安全", "効率的", "良好",
            "満たし", "達成", "承認", "推奨", "OK", "可能", "確実",
            "メリット", "利益", "効果", "向上", "改善", "拡大"
        ]

        self.negative_keywords = [
            "不良", "低い", "失敗", "危険", "非効率", "悪い", "問題",
            "不足", "未達成", "拒否", "非推奨", "NG", "不可能", "不確実",
            "デメリット", "損失", "負作用", "低下", "悪化", "縮小"
        ]

        self.domains = [
            "リリース", "投資", "採用", "品質", "リスク", "承認",
            "拡張", "廃止", "導入", "実施", "推奨", "開始"
        ]

        self.judgments = [
            "可能か？", "すべきか？", "推奨か？", "対応か？", "許容可能か？",
            "実施すべきか？", "承認すべきか？", "採用すべきか？"
        ]

    def generate_random_context(self, expected_type: str) -> tuple:
        """ランダムなコンテキストを生成"""
        if expected_type == "yes":
            num_positive = random.randint(2, 5)
            keywords = random.sample(self.positive_keywords, min(num_positive, len(self.positive_keywords)))
            context = "、".join(keywords) + "。プロジェクトは" + random.choice(keywords) + "です。"
            expected = "Yes"
        elif expected_type == "no":
            num_negative = random.randint(2, 5)
            keywords = random.sample(self.negative_keywords, min(num_negative, len(self.negative_keywords)))
            context = "、".join(keywords) + "。プロジェクトは" + random.choice(keywords) + "です。"
            expected = "No"
        else:  # boundary
            pos_count = random.randint(1, 3)
            neg_count = random.randint(1, 3)
            pos_keywords = random.sample(self.positive_keywords, min(pos_count, len(self.positive_keywords)))
            neg_keywords = random.sample(self.negative_keywords, min(neg_count, len(self.negative_keywords)))
            all_keywords = pos_keywords + neg_keywords
            random.shuffle(all_keywords)
            context = "、".join(all_keywords) + "。"
            expected = "Yes"

        return context, expected

    def generate_random_request(self) -> str:
        """ランダムなジャッジメントリクエストを生成"""
        domain = random.choice(self.domains)
        judgment = random.choice(self.judgments)
        return f"{domain}を{judgment}"

    def run_test_at_scale(self, test_num: int) -> dict:
        """指定回数でテストを実行"""
        print(f"\n【{test_num}回テスト実行中...】", end=" ", flush=True)

        start_time = time.time()
        test_results = []

        for i in range(test_num):
            # ランダムなテストタイプを選択（70% positive, 20% negative, 10% boundary）
            rand = random.random()
            if rand < 0.7:
                test_type = "yes"
            elif rand < 0.9:
                test_type = "no"
            else:
                test_type = "boundary"

            context, expected_decision = self.generate_random_context(test_type)
            request = self.generate_random_request()

            result = self.judge.judge({
                "context": context,
                "judgment_request": request
            })

            actual_decision = result.get("decision")
            is_correct = actual_decision == expected_decision

            test_results.append({
                "type": test_type,
                "expected": expected_decision,
                "actual": actual_decision,
                "correct": is_correct
            })

        elapsed = time.time() - start_time

        # 精度計算
        total = len(test_results)
        correct = sum(1 for r in test_results if r["correct"])
        accuracy = correct / total * 100

        # タイプ別精度計算
        types = defaultdict(list)
        for result in test_results:
            types[result["type"]].append(result)

        type_accuracy = {}
        for test_type in ["yes", "no", "boundary"]:
            if test_type in types:
                type_results = types[test_type]
                type_correct = sum(1 for r in type_results if r["correct"])
                type_acc = type_correct / len(type_results) * 100
                type_accuracy[test_type] = type_acc

        print(f"完了！ ({elapsed:.2f}秒)")

        return {
            "test_num": test_num,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "elapsed": elapsed,
            "type_accuracy": type_accuracy
        }

    def run_all_scales(self):
        """すべてのスケールでテスト実行"""
        print("\n" + "=" * 70)
        print("【複数スケール精度測定】")
        print("=" * 70)

        scales = [10, 50, 100, 200, 500, 1000]

        for scale in scales:
            result = self.run_test_at_scale(scale)
            self.results[scale] = result

        # 結果表示
        print("\n" + "=" * 70)
        print("【測定結果サマリー】")
        print("=" * 70)
        print(f"\n{'回数':>6} | {'精度':>7} | {'正解数':>10} | {'Yes精度':>8} | {'No精度':>8} | {'Boundary':>8}")
        print("-" * 70)

        for scale in scales:
            result = self.results[scale]
            yes_acc = result["type_accuracy"].get("yes", 0)
            no_acc = result["type_accuracy"].get("no", 0)
            boundary_acc = result["type_accuracy"].get("boundary", 0)
            print(f"{result['test_num']:6d} | {result['accuracy']:7.2f}% | {result['correct']:4d}/{result['total']:4d} | "
                  f"{yes_acc:7.2f}% | {no_acc:7.2f}% | {boundary_acc:7.2f}%")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  複数スケール精度測定テスト".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        tester = MultiScaleAccuracyTest()
        tester.run_all_scales()

        # 結果をJSONで保存
        import json
        with open("accuracy_results.json", "w") as f:
            results = {}
            for scale, data in tester.results.items():
                results[scale] = {
                    "test_num": data["test_num"],
                    "accuracy": data["accuracy"],
                    "correct": data["correct"],
                    "total": data["total"],
                    "elapsed": data["elapsed"],
                    "type_accuracy": data["type_accuracy"]
                }
            json.dump(results, f, indent=2)

        print("\n✅ 結果を accuracy_results.json に保存しました")
        print("✅ グラフ化スクリプトを実行してください: python plot_multi_scale_accuracy.py\n")

    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
