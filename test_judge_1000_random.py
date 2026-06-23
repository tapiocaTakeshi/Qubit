#!/usr/bin/env python3
"""
QBNN Frontal Engine Judge - 1000回ランダムテスト
ランダムに生成された1000個のテストケースで精度を測定
"""

import random
import time
from collections import defaultdict
from frontal_engine_mcp_server import FrontalEngineJudge


class Judge1000RandomTest:
    """1000回ランダムテスト"""

    def __init__(self):
        self.judge = FrontalEngineJudge()
        self.results = []

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
        """
        ランダムなコンテキストを生成

        Args:
            expected_type: "yes", "no", "boundary"

        Returns:
            (context, expected_decision)
        """
        if expected_type == "yes":
            # ポジティブコンテキスト
            num_positive = random.randint(2, 5)
            keywords = random.sample(self.positive_keywords, min(num_positive, len(self.positive_keywords)))
            context = "、".join(keywords) + "。プロジェクトは" + random.choice(keywords) + "です。"
            expected = "Yes"

        elif expected_type == "no":
            # ネガティブコンテキスト
            num_negative = random.randint(2, 5)
            keywords = random.sample(self.negative_keywords, min(num_negative, len(self.negative_keywords)))
            context = "、".join(keywords) + "。プロジェクトは" + random.choice(keywords) + "です。"
            expected = "No"

        else:  # boundary
            # 境界ケース：ポジティブとネガティブを混在
            pos_count = random.randint(1, 3)
            neg_count = random.randint(1, 3)
            pos_keywords = random.sample(self.positive_keywords, min(pos_count, len(self.positive_keywords)))
            neg_keywords = random.sample(self.negative_keywords, min(neg_count, len(self.negative_keywords)))
            all_keywords = pos_keywords + neg_keywords
            random.shuffle(all_keywords)
            context = "、".join(all_keywords) + "。"
            expected = "Yes"  # 境界ケースはYes期待

        return context, expected

    def generate_random_request(self) -> str:
        """ランダムなジャッジメントリクエストを生成"""
        domain = random.choice(self.domains)
        judgment = random.choice(self.judgments)
        return f"{domain}を{judgment}"

    def run_test(self, test_num: int):
        """1000回テストを実行"""
        print(f"\n【実行中】1000個のランダムテストケースを実行...")
        print("-" * 70)

        start_time = time.time()

        for i in range(test_num):
            # ランダムなテストタイプを選択（70% positive, 20% negative, 10% boundary）
            rand = random.random()
            if rand < 0.7:
                test_type = "yes"
            elif rand < 0.9:
                test_type = "no"
            else:
                test_type = "boundary"

            # ランダムなコンテキストとリクエストを生成
            context, expected_decision = self.generate_random_context(test_type)
            request = self.generate_random_request()

            # Judge実行
            result = self.judge.judge({
                "context": context,
                "judgment_request": request
            })

            actual_decision = result.get("decision")
            is_correct = actual_decision == expected_decision

            self.results.append({
                "index": i + 1,
                "type": test_type,
                "expected": expected_decision,
                "actual": actual_decision,
                "score": result.get("score"),
                "correct": is_correct
            })

            # 進捗表示
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (test_num - i - 1)
                print(f"進捗: {i + 1}/{test_num} 完了 | "
                      f"経過: {elapsed:.1f}s | "
                      f"残時間: {remaining:.1f}s")

        elapsed = time.time() - start_time
        print(f"進捗: {test_num}/{test_num} 完了")
        print(f"総処理時間: {elapsed:.2f}秒 (平均 {elapsed/test_num*1000:.2f}ms/回)")

    def analyze_results(self):
        """結果を分析"""
        print("\n" + "=" * 70)
        print("【1000回ランダムテスト精度分析結果】")
        print("=" * 70)

        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        accuracy = correct / total * 100

        print(f"\n【総合精度】")
        print(f"  正解数: {correct}/{total}")
        print(f"  精度: {accuracy:.2f}%")

        # テストタイプ別
        print(f"\n【テストタイプ別精度】")
        types = defaultdict(list)
        for result in self.results:
            types[result["type"]].append(result)

        for test_type in ["yes", "no", "boundary"]:
            if test_type in types:
                type_results = types[test_type]
                type_correct = sum(1 for r in type_results if r["correct"])
                type_accuracy = type_correct / len(type_results) * 100
                print(f"  {test_type:10s}: {type_correct:4d}/{len(type_results):4d} ({type_accuracy:6.2f}%)")

        # スコア分布
        print(f"\n【スコア分布】")
        scores = [r["score"] for r in self.results]
        score_ranges = {
            "0-10": sum(1 for s in scores if 0 <= s <= 10),
            "11-20": sum(1 for s in scores if 11 <= s <= 20),
            "21-30": sum(1 for s in scores if 21 <= s <= 30),
            "31-40": sum(1 for s in scores if 31 <= s <= 40),
            "41-50": sum(1 for s in scores if 41 <= s <= 50),
            "51-60": sum(1 for s in scores if 51 <= s <= 60),
            "61-70": sum(1 for s in scores if 61 <= s <= 70),
            "71-80": sum(1 for s in scores if 71 <= s <= 80),
            "81-90": sum(1 for s in scores if 81 <= s <= 90),
            "91-100": sum(1 for s in scores if 91 <= s <= 100),
        }

        print(f"  平均スコア: {sum(scores)/len(scores):.2f}")
        print(f"  最小スコア: {min(scores)}")
        print(f"  最大スコア: {max(scores)}")

        print(f"\n  スコア範囲分布:")
        for score_range, count in score_ranges.items():
            if count > 0:
                percent = count / len(scores) * 100
                bar = "█" * (count // 10)
                print(f"    {score_range:7s}: {count:4d} ({percent:5.1f}%) {bar}")

        # 信頼性レベル
        print(f"\n【信頼性レベル評価】")
        if accuracy >= 95:
            level = "⭐⭐⭐⭐⭐ (5/5) - 超高精度"
        elif accuracy >= 90:
            level = "⭐⭐⭐⭐⭐ (5/5) - エンタープライズレベル"
        elif accuracy >= 85:
            level = "⭐⭐⭐⭐ (4/5) - 本番環境対応"
        elif accuracy >= 80:
            level = "⭐⭐⭐⭐ (4/5) - 実用的"
        elif accuracy >= 70:
            level = "⭐⭐⭐ (3/5) - 改善の余地あり"
        else:
            level = "⭐⭐ (2/5) - 要改善"

        print(f"  精度: {accuracy:.2f}%")
        print(f"  レベル: {level}")

        # Yes/No判定分布
        print(f"\n【判定分布】")
        yes_count = sum(1 for r in self.results if r["actual"] == "Yes")
        no_count = sum(1 for r in self.results if r["actual"] == "No")
        print(f"  Yes判定: {yes_count} ({yes_count/total*100:.1f}%)")
        print(f"  No判定:  {no_count} ({no_count/total*100:.1f}%)")

        print("\n" + "=" * 70)
        print("✅ 1000回ランダムテスト完了")
        print("=" * 70)

        return accuracy


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  QBNN Frontal Engine Judge - 1000回ランダムテスト".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        tester = Judge1000RandomTest()
        tester.run_test(1000)
        tester.analyze_results()
        print("\n✅ 1000回テスト完了\n")
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
