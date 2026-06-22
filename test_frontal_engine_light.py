#!/usr/bin/env python3
"""
QBNN Frontal Engine - ライト版テスト（torch不要）
"""

import json
import sys
from pathlib import Path

# torch のインポートをスキップして判断エンジンのみをテスト
sys.path.insert(0, str(Path(__file__).parent))

# MCPサーバーから判断エンジンのみを抽出
class FrontalEngineJudge:
    """
    前頭葉の判断エンジン（ライト版）
    """

    def __init__(self):
        pass

    def judge(self, judgment_task: dict) -> dict:
        """判断タスクを実行"""
        try:
            context = judgment_task.get("context", "")
            judgment_request = judgment_task.get("judgment_request", "")
            criteria = judgment_task.get("criteria", {})
            options = judgment_task.get("options", [])
            strict_mode = judgment_task.get("strict_mode", False)

            if not context or not judgment_request:
                return self._error_response("context と judgment_request は必須です")

            decision, score, reasoning, confidence, key_factors = self._analyze_judgment(
                context=context,
                judgment_request=judgment_request,
                criteria=criteria,
                options=options,
                strict_mode=strict_mode
            )

            from datetime import datetime
            return {
                "decision": decision,
                "score": score,
                "reasoning": reasoning,
                "confidence": confidence,
                "key_factors": key_factors,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            return self._error_response(f"判断処理エラー: {str(e)}")

    def _analyze_judgment(self, context, judgment_request, criteria, options, strict_mode):
        """判断分析を行う"""
        score, reasoning, key_factors = self._compute_score(
            context, judgment_request, criteria, options
        )

        if strict_mode:
            decision = "Yes" if score >= 70 else "No"
            confidence = "high" if score >= 80 or score <= 20 else "medium"
        else:
            decision = "Yes" if score >= 50 else "No"
            confidence = "high" if score >= 75 or score <= 25 else "medium"

        if confidence == "high" and 40 <= score <= 60:
            confidence = "medium"

        return decision, score, reasoning, confidence, key_factors

    def _compute_score(self, context, judgment_request, criteria, options):
        """スコアを計算"""
        score = 50
        key_factors = []

        # テキスト長分析
        context_words = len(context.split())
        if context_words > 100:
            score += 5
            key_factors.append("十分な背景情報がある")

        # キーワード分析
        positive_keywords = ["重要", "必須", "確認", "承認", "安全", "有効", "高い", "良い", "正しい"]
        negative_keywords = ["危険", "リスク", "問題", "失敗", "低い", "悪い", "不正", "禁止"]

        for keyword in positive_keywords:
            if keyword in context or keyword in judgment_request:
                score += 3
                key_factors.append(f"ポジティブ要因: {keyword}")

        for keyword in negative_keywords:
            if keyword in context or keyword in judgment_request:
                score -= 3
                key_factors.append(f"ネガティブ要因: {keyword}")

        # 基準評価
        if criteria:
            criteria_score = self._evaluate_criteria(context, criteria)
            score = (score + criteria_score) / 2
            key_factors.append(f"基準評価: {criteria_score}点")

        # オプション評価
        if options:
            options_score = self._evaluate_options(context, options)
            score = (score + options_score) / 2
            key_factors.append(f"選択肢評価: {options_score}点")

        score = max(0, min(100, int(score)))

        # 根拠説明
        if score >= 70:
            reasoning = "指定された基準と文脈に基づいて、肯定的な判断が支持されています。"
        elif score >= 50:
            reasoning = "判断は不確実ですが、利用可能な情報から妥当な結論が導き出されます。"
        else:
            reasoning = "利用可能な情報に基づいて、否定的な判断が支持されています。"

        if key_factors:
            reasoning += f" 主要な要因: {', '.join(key_factors[:3])}"

        return score, reasoning, key_factors[:5]

    def _evaluate_criteria(self, context, criteria):
        """基準評価"""
        score = 50.0
        for criterion_name, criterion_value in criteria.items():
            if isinstance(criterion_value, str):
                if criterion_value.lower() in context.lower():
                    score += 10
                else:
                    score -= 5
            elif isinstance(criterion_value, bool):
                score += 15 if criterion_value else -15
            elif isinstance(criterion_value, (int, float)):
                score += min(20, max(-20, criterion_value / 10))
        return max(0, min(100, score))

    def _evaluate_options(self, context, options):
        """オプション評価"""
        if not options:
            return 50.0
        score = 50.0
        matched = sum(1 for opt in options if isinstance(opt, str) and opt.lower() in context.lower())
        if matched > 0:
            score = 50 + (matched / len(options)) * 30
        return score

    def _error_response(self, error_msg):
        """エラーレスポンス"""
        from datetime import datetime
        return {
            "decision": "No",
            "score": 0,
            "reasoning": error_msg,
            "confidence": "low",
            "key_factors": ["エラーが発生しました"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": True
        }


# テストケース実行
def run_tests():
    print("\n╔" + "=" * 58 + "╗")
    print("║" + "  QBNN Frontal Engine - ライト版テスト".center(58) + "║")
    print("╚" + "=" * 58 + "╝\n")

    judge = FrontalEngineJudge()

    # Test 1
    print("Test 1: 基本的なYes/No判断")
    print("-" * 60)
    task1 = {
        "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。チームは高い士気を持ち、リスク要因は特に報告されていません。",
        "judgment_request": "このプロジェクトをリリースしても安全か？"
    }
    result1 = judge.judge(task1)
    print(f"Decision: {result1['decision']}")
    print(f"Score: {result1['score']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"Reasoning: {result1['reasoning']}\n")

    # Test 2
    print("Test 2: リスク評価（厳密モード）")
    print("-" * 60)
    task2 = {
        "context": "新技術の導入には以下のリスクが考えられます: 学習曲線が急、既存システムとの互換性問題の可能性、短期的には生産性低下の予測。",
        "judgment_request": "このリスクは許容可能か？",
        "strict_mode": True
    }
    result2 = judge.judge(task2)
    print(f"Decision: {result2['decision']}")
    print(f"Score: {result2['score']}")
    print(f"Confidence: {result2['confidence']}\n")

    # Test 3
    print("Test 3: 基準付き判断")
    print("-" * 60)
    task3 = {
        "context": "提案者の信頼スコア: 85/100、過去の成功率: 80%、提案の複雑度: 中程度",
        "judgment_request": "この提案を承認すべきか？",
        "criteria": {
            "trust_score": 80,
            "success_rate": "80%"
        }
    }
    result3 = judge.judge(task3)
    print(f"Decision: {result3['decision']}")
    print(f"Score: {result3['score']}\n")

    # Test 4
    print("Test 4: エラーハンドリング")
    print("-" * 60)
    task4 = {"context": "不完全な入力"}
    result4 = judge.judge(task4)
    print(f"Decision: {result4['decision']}")
    print(f"Error present: {'error' in result4}\n")

    print("=" * 60)
    print("✓ すべてのテストが完了しました")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
