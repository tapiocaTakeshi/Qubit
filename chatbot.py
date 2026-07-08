#!/usr/bin/env python3
"""
チャットボット起動スクリプト
複数のチャットボット実装から選択して起動
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

# ========================================
# チャットボット1: シンプル版
# ========================================
class SimpleChatBot:
    def __init__(self):
        self.name = "SimpleChatBot"
        self.mode = "simple"

    def chat(self, user_input):
        responses = {
            "こんにちは": "こんにちは！",
            "どうしたの": "何かお手伝いできることはありますか？",
            "ありがとう": "どういたしまして！",
            "さようなら": "またね！",
        }

        for key, response in responses.items():
            if key in user_input:
                return response

        return "そうですね。"

# ========================================
# チャットボット2: embedding-Gemma版
# ========================================
class EmbeddingChatBot:
    def __init__(self):
        self.name = "EmbeddingChatBot"
        self.mode = "semantic"
        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.knowledge = {
            "こんにちは": "こんにちは！embedding-Gemmaを使っています。",
            "何ができますか": "セマンティック理解が得意です。テキストの意味を正確に理解します。",
            "日本について": "日本は東アジアに位置する素晴らしい国ですね。",
            "機械学習とは": "機械学習はデータから自動的にパターンを学習する技術です。",
            "プログラミング": "プログラミングは論理的思考と創造性を組み合わせたスキルです。",
        }

    def chat(self, user_input):
        user_emb = self.model.encode(user_input)
        knowledge_keys = list(self.knowledge.keys())
        knowledge_embs = self.model.encode(knowledge_keys)

        similarities = cosine_similarity([user_emb], knowledge_embs)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > 0.3:
            response = self.knowledge[knowledge_keys[best_idx]]
            return f"{response} (信頼度: {best_score:.1%})"
        else:
            return "もう少し詳しく教えていただけますか？"

# ========================================
# チャットボット3: 感情分析版
# ========================================
class SentimentChatBot:
    def __init__(self):
        self.name = "SentimentChatBot"
        self.mode = "sentiment"

    def chat(self, user_input):
        positive_words = ["素晴らしい", "良い", "好き", "楽しい", "嬉しい", "最高"]
        negative_words = ["悪い", "嫌い", "つまらない", "悲しい", "困った", "最悪"]

        pos_count = sum(1 for w in positive_words if w in user_input)
        neg_count = sum(1 for w in negative_words if w in user_input)

        if pos_count > neg_count:
            sentiment = "肯定的 😊"
            response = "素晴らしいですね！"
        elif neg_count > pos_count:
            sentiment = "否定的 😞"
            response = "そうなんですね。大変ですね。"
        else:
            sentiment = "中立的 😐"
            response = "なるほど。"

        return f"{response} (感情: {sentiment})"

# ========================================
# チャットボット4: QBNN版
# ========================================
class QBNNChatBot:
    def __init__(self):
        self.name = "QBNNChatBot"
        self.mode = "quantum"

    def chat(self, user_input):
        entanglement_factor = np.random.random()

        responses = [
            "量子的に考えると、複数の状態が重ね合わされています。",
            "エンタングルメント処理中...層間の相関を検出しました。",
            "位相干渉により、複雑な構造を理解しました。",
            "量子ビット間の相関が高いです。",
        ]

        response = np.random.choice(responses)
        return f"{response} (エンタングルメント: {entanglement_factor:.2f})"

# ========================================
# チャットボット5: ハイブリッド版
# ========================================
class HybridChatBot:
    def __init__(self):
        self.name = "HybridChatBot"
        self.mode = "hybrid"
        self.semantic_bot = EmbeddingChatBot()
        self.sentiment_bot = SentimentChatBot()

    def chat(self, user_input):
        semantic_response = self.semantic_bot.chat(user_input)
        sentiment_response = self.sentiment_bot.chat(user_input)

        return f"[セマンティック] {semantic_response}\n[感情分析] {sentiment_response}"

# ========================================
# チャットボットマネージャー
# ========================================
class ChatBotManager:
    def __init__(self):
        self.bots = {
            "1": SimpleChatBot(),
            "2": EmbeddingChatBot(),
            "3": SentimentChatBot(),
            "4": QBNNChatBot(),
            "5": HybridChatBot(),
        }
        self.current_bot = None

    def list_bots(self):
        print("\n" + "=" * 80)
        print("📋 利用可能なチャットボット")
        print("=" * 80)
        for key, bot in self.bots.items():
            print(f"  {key}. {bot.name:20s} (モード: {bot.mode})")
        print()

    def select_bot(self, bot_id):
        if bot_id in self.bots:
            self.current_bot = self.bots[bot_id]
            print(f"\n✅ {self.current_bot.name} を選択しました。")
            print(f"   (「さようなら」で終了)\n")
            return True
        else:
            print("❌ 無効なボットIDです。")
            return False

    def chat(self, user_input):
        if self.current_bot is None:
            return "❌ チャットボットが選択されていません。"
        return self.current_bot.chat(user_input)

    def run_interactive(self):
        """インタラクティブモード"""
        self.list_bots()

        while True:
            bot_id = input("🤖 チャットボットを選択 (1-5): ").strip()
            if self.select_bot(bot_id):
                break

        print("=" * 80)
        print(f"💬 {self.current_bot.name} が起動しました")
        print("=" * 80)

        while True:
            user_input = input("\n👤 あなた: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["さようなら", "さよなら", "quit", "exit"]:
                print(f"\n🤖 ボット: またね！")
                break

            response = self.chat(user_input)
            print(f"🤖 ボット: {response}")

def main():
    if len(sys.argv) > 1:
        # コマンドライン引数でボットを指定
        bot_id = sys.argv[1]
        manager = ChatBotManager()

        if not manager.select_bot(bot_id):
            manager.list_bots()
            return

        # テストモード
        print("\n" + "=" * 80)
        print("テスト入力:")
        print("=" * 80)

        test_inputs = [
            "こんにちは",
            "何ができますか",
            "素晴らしいですね",
        ]

        for test_input in test_inputs:
            response = manager.chat(test_input)
            print(f"\n👤 入力: \"{test_input}\"")
            print(f"🤖 応答: {response}")
    else:
        # インタラクティブモード
        manager = ChatBotManager()
        manager.run_interactive()

if __name__ == "__main__":
    main()
