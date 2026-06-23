#!/usr/bin/env python3
"""
Gemma+QBNN Frontal - シンプル推論 (Pure Python版)
外部ライブラリ不要で日常的な判断を実行
"""

import math
from typing import Dict, Any, List


class SimpleQuantumFrontal:
    """シンプルな量子前頭葉（Pure Python版）"""

    def __init__(self):
        """初期化"""
        self.theta = 0.3  # 量子ビットのパラメータ

    @staticmethod
    def apqb_judgment(context: str, question: str) -> Dict[str, Any]:
        """APQB層による判断"""
        # キーワード分析で基本スコアを決定
        base_score = 50

        # ポジティブキーワード
        positive = ["必要", "yes", "OK", "賛成", "おすすめ", "良い", "メリット", "利益", "成功", "安全"]
        negative = ["no", "危険", "反対", "リスク", "問題", "損失", "失敗", "不要"]

        combined = (context + " " + question).lower()

        for word in positive:
            if word in combined:
                base_score += 10

        for word in negative:
            if word in combined:
                base_score -= 10

        # スコア制限
        return max(0, min(100, base_score))

    def judge(self, context: str, question: str) -> Dict[str, Any]:
        """判断を実行"""
        # APQB量子ビット計算
        r = math.cos(2 * self.theta)  # 相関係数
        T = abs(math.sin(2 * self.theta))  # 温度

        # スコア計算
        base_score = self.apqb_judgment(context, question)

        # 量子補正を適用
        context_len = len(context)
        question_len = len(question)

        if context_len > 50:
            base_score += 5
        if question_len > 20:
            base_score += 3

        # 量子的な補正
        quantum_factor = (r * 0.3 + T * 0.2)
        final_score = int(base_score + quantum_factor * 10)
        final_score = max(0, min(100, final_score))

        # 決定を決定
        decision = "Yes" if final_score >= 50 else "No"

        # 信頼度を決定
        if final_score >= 70:
            confidence = "high"
        elif final_score >= 40:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "decision": decision,
            "score": final_score,
            "confidence": confidence,
            "quantum_info": {
                "r": round(r, 3),
                "T": round(T, 3),
                "constraint": round(r**2 + T**2, 6)
            }
        }


def run_simple_inference():
    """シンプル推論を実行"""

    print("\n" + "="*70)
    print("Gemma+QBNN Frontal - シンプル推論 (Pure Python版)")
    print("="*70)
    print("外部ライブラリ不要での量子推論実行\n")

    # フロンタルシステム初期化
    frontal = SimpleQuantumFrontal()

    # シンプルな判断タスク
    tasks = [
        {
            "title": "1. スマートフォンを新しく買うべきか？",
            "context": "現在3年前のモデルを使用。バッテリーが1日もたない。新モデルは高い。",
        },
        {
            "title": "2. 転職すべきか？",
            "context": "給与は100万円UP。ただし起業1年の小さな会社。安定性が不確定。",
        },
        {
            "title": "3. 大学院に進学すべきか？",
            "context": "給与は100万円UP。だが2年かかり学費200万円必要。直結性は不明確。",
        },
        {
            "title": "4. 結婚式は盛大にするべきか？",
            "context": "資金あり。相手も希望。準備は大変。人生の重要イベント。",
        },
        {
            "title": "5. 引っ越すべきか？",
            "context": "家賃は月4万円上昇。駅に近い。通勤が20分短縮。利便性向上。",
        },
        {
            "title": "6. ジムに入会すべきか？",
            "context": "月1万円。使用頻度は月2回。月5000円の計算だが運動は必要。",
        },
        {
            "title": "7. 今日会社を休むべきか？",
            "context": "少し風邪気味で37.2度。重要会議が3つ。代替がいない。",
        },
        {
            "title": "8. この投資話に乗るべきか？",
            "context": "年15%のリターン。詳細説明なし。知人からの勧誘。危険な可能性。",
        }
    ]

    results = []

    # 各タスク実行
    for task in tasks:
        print(f"【{task['title']}】")
        print(f"  コンテキスト: {task['context'][:50]}...")

        result = frontal.judge(task['context'], task['title'])

        print(f"  → 判断: {result['decision']} | スコア: {result['score']:3}/100 | 信頼度: {result['confidence']}")

        qi = result['quantum_info']
        print(f"    量子情報: r={qi['r']:.3f}, T={qi['T']:.3f}, 制約={qi['constraint']:.6f}")
        print()

        results.append((task['title'].split('. ')[1][:15], result['decision'], result['score']))

    # サマリー
    print("="*70)
    print("【推論結果サマリー】")
    print("="*70)

    yes_count = sum(1 for r in results if r[1] == "Yes")
    no_count = len(results) - yes_count
    avg_score = sum(r[2] for r in results) / len(results)

    print(f"\n実行数: {len(results)}")
    print(f"Yes判定: {yes_count}件")
    print(f"No判定: {no_count}件")
    print(f"平均スコア: {avg_score:.1f}/100")

    print("\n【判断一覧】")
    for title, decision, score in results:
        print(f"  {title:12} → {decision:3} (スコア: {score:3}/100)")

    print("\n" + "="*70)
    print("✨ Pure Python版推論完了")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_simple_inference()
