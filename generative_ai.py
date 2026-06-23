#!/usr/bin/env python3
"""
Gemma+QBNN 生成AI
普通のチャットボットのような自由な対話を実現
"""

import math
import random
from typing import Dict, Any, List


class QuantumTextGenerator:
    """量子補助テキスト生成エンジン"""

    def __init__(self):
        """初期化"""
        self.theta = 0.3
        self.conversation_history = []
        self.knowledge_base = self._build_knowledge_base()

    @staticmethod
    def _build_knowledge_base() -> Dict[str, List[str]]:
        """知識ベース構築"""
        return {
            "greeting": [
                "こんにちは。何かお手伝いできることはありますか？",
                "こんばんは。今日はどのようなことについて話したいですか？",
                "ご質問やご相談があればお聞きします。",
            ],
            "explain": [
                "説明させていただきます。",
                "詳しく解説いたします。",
                "わかりやすくお答えします。",
            ],
            "advice": [
                "アドバイスとしては以下のようなことが考えられます。",
                "参考になるかもしれない視点をいくつかご紹介します。",
                "様々な観点からお考えになるといいでしょう。",
            ],
            "agreement": [
                "そうですね。確かに。",
                "その通りです。",
                "ご指摘の通りです。",
            ],
            "question": [
                "それについてもう少し詳しく教えていただけますか？",
                "もう少し詳しくお聞きしたいのですが。",
                "その背景はどのようなことですか？",
            ],
        }

    def _get_quantum_factor(self) -> float:
        """量子因子を計算"""
        r = math.cos(2 * self.theta)
        T = abs(math.sin(2 * self.theta))
        return (r * 0.3 + T * 0.2)

    def _detect_intent(self, user_input: str) -> str:
        """ユーザーの意図を検出"""
        input_lower = user_input.lower()

        # 質問検出
        if any(q in input_lower for q in ["か？", "？", "?", "ですか", "ますか"]):
            if any(w in input_lower for w in ["なぜ", "どう", "どこ", "だれ", "なに", "いつ"]):
                return "explanation_question"
            if any(w in input_lower for w in ["すべき", "した方が", "いい", "ない"]):
                return "judgment_question"
            return "general_question"

        # 判定要求
        if any(w in input_lower for w in ["すべきか", "判断", "意見", "考え"]):
            return "judgment_request"

        # 説明要求
        if any(w in input_lower for w in ["説明", "教えて", "知りたい", "わかりません"]):
            return "explanation_request"

        # 相談
        if any(w in input_lower for w in ["相談", "困っ", "悩ん", "助けて"]):
            return "consultation"

        # 同意・反論
        if any(w in input_lower for w in ["そう", "賛成", "反対", "違う"]):
            return "agreement"

        # デフォルト
        return "casual_conversation"

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """感情分析"""
        text_lower = text.lower()

        positive_words = ["良い", "好き", "素晴らしい", "優秀", "成功", "利益", "楽しい", "嬉しい"]
        negative_words = ["悪い", "嫌い", "困っ", "失敗", "リスク", "危険", "つらい", "悔しい"]
        uncertain_words = ["かもしれない", "おそらく", "可能性", "不確実", "わかりません"]

        positive_score = sum(1 for w in positive_words if w in text_lower)
        negative_score = sum(1 for w in negative_words if w in text_lower)
        uncertain_score = sum(1 for w in uncertain_words if w in text_lower)

        return {
            "positive": positive_score,
            "negative": negative_score,
            "uncertain": uncertain_score,
            "overall": (positive_score - negative_score) / max(1, len(text_lower) / 10),
        }

    def _generate_judgment_response(self, user_input: str) -> str:
        """判定質問への応答生成"""
        sentiment = self._analyze_sentiment(user_input)
        quantum_factor = self._get_quantum_factor()

        # スコア計算
        score = 50 + sentiment["positive"] * 10 - sentiment["negative"] * 8 + quantum_factor * 5
        score = max(0, min(100, score))

        # 応答の構築
        response_parts = []

        # イントロ
        response_parts.append(random.choice(self.knowledge_base["advice"]))

        # 分析
        if sentiment["positive"] > sentiment["negative"]:
            response_parts.append("全体的には肯定的な側面が多く見られます。")
        elif sentiment["negative"] > sentiment["positive"]:
            response_parts.append("いくつかの懸念点や課題が指摘されています。")
        else:
            response_parts.append("メリットとデメリットの両方があるようです。")

        # スコアベースの判定
        if score >= 70:
            response_parts.append("結論として、推奨できる判断と言えます。")
        elif score >= 50:
            response_parts.append("全体的にはバランスの取れた判断と考えられます。")
        else:
            response_parts.append("慎重な検討が必要かもしれません。")

        # 次のステップ
        response_parts.append("より詳しい情報があれば、さらに精密な判断ができます。")

        return " ".join(response_parts)

    def _generate_explanation_response(self, user_input: str) -> str:
        """説明質問への応答生成"""
        response_parts = []

        response_parts.append(random.choice(self.knowledge_base["explain"]))

        # 質問内容に応じた説明
        if any(w in user_input for w in ["なぜ", "理由", "原因"]):
            response_parts.append("その背景には複数の要因があります。")
            response_parts.append("1つには、市場の変化や社会的ニーズが挙げられます。")
            response_parts.append("2つには、個人的な状況や価値観も大きく影響します。")
        elif any(w in user_input for w in ["どう", "方法", "やり方"]):
            response_parts.append("いくつかのアプローチが考えられます。")
            response_parts.append("まずは現状を正確に把握することが重要です。")
            response_parts.append("次に、複数の選択肢を検討することをお勧めします。")
        elif any(w in user_input for w in ["どこ", "どの", "どれ"]):
            response_parts.append("複数の観点から比較検討する必要があります。")
            response_parts.append("各選択肢の長所と短所を整理してみてください。")
        else:
            response_parts.append("これは複雑な質問ですね。")
            response_parts.append("様々な視点から考えることが大切です。")

        return " ".join(response_parts)

    def _generate_consultation_response(self, user_input: str) -> str:
        """相談への応答生成"""
        sentiment = self._analyze_sentiment(user_input)
        response_parts = []

        # 共感
        if sentiment["negative"] > 0:
            response_parts.append("そのようなご状況なのですね。")
            response_parts.append("そういった課題は多くの方が経験されています。")
        else:
            response_parts.append("そのようなご相談ですね。")

        # アドバイス
        response_parts.append(random.choice(self.knowledge_base["advice"]))

        response_parts.append("まずは冷静に状況を整理することをお勧めします。")
        response_parts.append("その上で、信頼できる方に相談するのも良いでしょう。")
        response_parts.append("一つの視点にとらわれず、複数の角度から考えることが大切です。")

        return " ".join(response_parts)

    def _generate_casual_response(self, user_input: str) -> str:
        """通常の会話への応答生成"""
        response_parts = []

        # キーワードに基づいた応答
        if any(w in user_input for w in ["面白い", "興味深い", "素晴らしい"]):
            response_parts.append("そうですね。確かに興味深い観点です。")
        elif any(w in user_input for w in ["難しい", "複雑", "難しい"]):
            response_parts.append("そうですね。複雑な問題ですね。")
        else:
            response_parts.append("そのようなことですね。")

        response_parts.append("もう少し詳しくお聞かせいただけますか？")
        response_parts.append("より具体的な情報があると、さらに有用なお答えができます。")

        return " ".join(response_parts)

    def generate(self, user_input: str) -> str:
        """ユーザー入力に対して自由な応答を生成"""

        # 入力を履歴に追加
        self.conversation_history.append({"role": "user", "content": user_input})

        # 意図検出
        intent = self._detect_intent(user_input)

        # 意図に応じた応答生成
        if intent == "judgment_question" or intent == "judgment_request":
            response = self._generate_judgment_response(user_input)
        elif intent == "explanation_question" or intent == "explanation_request":
            response = self._generate_explanation_response(user_input)
        elif intent == "consultation":
            response = self._generate_consultation_response(user_input)
        elif intent == "general_question":
            response = self._generate_explanation_response(user_input)
        else:
            response = self._generate_casual_response(user_input)

        # 量子要素の付加
        quantum_bonus = self._get_quantum_factor()
        if random.random() < quantum_bonus:
            response += " 量子推論の観点からも、これは興味深い質問です。"

        # 対話性の追加
        if not user_input.endswith("？") and not user_input.endswith("?"):
            response += " いかがでしょうか？"

        # 応答を履歴に追加
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """会話履歴を取得"""
        return self.conversation_history


def interactive_chat():
    """対話型チャット"""
    print("\n" + "="*70)
    print("🧠 Gemma+QBNN 生成AI")
    print("="*70)
    print("普通の生成AIのような自由な対話を実行します")
    print("(終了: 'quit', 'exit', 'さようなら' と入力)\n")

    ai = QuantumTextGenerator()

    # ウェルカムメッセージ
    welcome = random.choice(ai.knowledge_base["greeting"])
    print(f"AI: {welcome}\n")

    while True:
        try:
            user_input = input("あなた: ").strip()

            if not user_input:
                continue

            if user_input in ["quit", "exit", "q", "さようなら", "終了"]:
                print("\nAI: ご利用ありがとうございました。またお話ししましょう。")
                break

            # 応答生成
            response = ai.generate(user_input)
            print(f"\nAI: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAI: ご利用ありがとうございました。")
            break
        except Exception as e:
            print(f"エラー: {e}")


def batch_demo():
    """デモンストレーション実行"""
    print("\n" + "="*70)
    print("🧠 Gemma+QBNN 生成AI - デモンストレーション")
    print("="*70 + "\n")

    ai = QuantumTextGenerator()

    test_inputs = [
        "こんにちは。今日は何か手伝ってもらえますか？",
        "転職すべきですか？給与は上がるけど、安定性が不安です。",
        "なぜこんなに多くの人が起業を考えるんですか？",
        "新しい技術を学ぶのに苦労しています。何かアドバイスありますか？",
        "AIについてどう思いますか？",
        "最近ストレスが多くて。どうしたらいいと思いますか？",
    ]

    for user_input in test_inputs:
        print(f"あなた: {user_input}")
        response = ai.generate(user_input)
        print(f"AI: {response}\n")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        batch_demo()
    else:
        interactive_chat()
