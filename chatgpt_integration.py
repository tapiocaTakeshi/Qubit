#!/usr/bin/env python3
"""
QBNN Frontal Engine - ChatGPT Integration
OpenAI Function Calling を使用した統合
"""

import json
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from frontal_engine_mcp_server import FrontalEngineJudge


# ============================================================
# Judge Tool Definition
# ============================================================

JUDGE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "judge",
        "description": "前頭葉の判断機能: あらゆる判断タスクを処理。Yes/No、スコア(0-100)、根拠説明を返します。意思決定、リスク評価、品質判定、倫理的判断などに対応。",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "判断の背景情報・文脈。詳細に提供するほど判断精度が向上します。"
                },
                "judgment_request": {
                    "type": "string",
                    "description": "何を判断したいかを具体的に明記。例: 'このプロジェクトをリリースできるか？'"
                },
                "criteria": {
                    "type": "object",
                    "description": "判断時に考慮する基準。キー: 基準名、値: 基準値（文字列、数値、真偽値）",
                    "additionalProperties": True
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "検討対象となる選択肢。複数選択肢の比較判断時に使用。"
                },
                "strict_mode": {
                    "type": "boolean",
                    "description": "厳密モード。true: スコア70以上でYes、false: スコア50以上でYes（デフォルト: false）",
                    "default": False
                }
            },
            "required": ["context", "judgment_request"]
        }
    }
}


# ============================================================
# ChatGPT Integration
# ============================================================

class ChatGPTFrontalEngine:
    """ChatGPT統合用のFrontal Engine ラッパー"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初期化

        Args:
            api_key: OpenAI APIキー。Noneの場合は環境変数から取得
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.judge_engine = FrontalEngineJudge()
        self.conversation_history = []

    def call_judge(self, **kwargs) -> Dict[str, Any]:
        """
        Judge関数を呼び出し

        Args:
            context: 背景情報
            judgment_request: 判断内容
            criteria: 判断基準（オプション）
            options: 選択肢（オプション）
            strict_mode: 厳密モード（オプション）

        Returns:
            判断結果
        """
        task = {
            "context": kwargs.get("context", ""),
            "judgment_request": kwargs.get("judgment_request", ""),
            "criteria": kwargs.get("criteria", {}),
            "options": kwargs.get("options", []),
            "strict_mode": kwargs.get("strict_mode", False)
        }
        return self.judge_engine.judge(task)

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        チャットを実行。必要に応じてJudgeツールを呼び出す

        Args:
            user_message: ユーザーメッセージ
            system_prompt: システムプロンプト（オプション）

        Returns:
            アシスタントレスポンス
        """
        # システムプロンプト（デフォルト）
        if system_prompt is None:
            system_prompt = """あなたは判断を支援するAIアシスタントです。
ユーザーが判断や意思決定に関する質問をしたときは、
Judgeツールを使用して前頭葉的な判断を提供してください。
判断結果に基づいて、詳しく説明してください。"""

        # 会話履歴に追加
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 初回リクエスト
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=[JUDGE_TOOL_DEFINITION],
            tool_choice="auto"
        )

        # ツール呼び出しの処理
        while response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]

            # Judgeツール呼び出し
            if tool_call.function.name == "judge":
                tool_args = json.loads(tool_call.function.arguments)
                judge_result = self.call_judge(**tool_args)

                # 会話履歴に追加
                self.conversation_history.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(judge_result, ensure_ascii=False)
                })

                # 次のリクエスト
                messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=[JUDGE_TOOL_DEFINITION],
                    tool_choice="auto"
                )

        # 最終レスポンス
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self):
        """会話履歴をクリア"""
        self.conversation_history = []


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    """使用例"""
    print("=" * 60)
    print("ChatGPT + Frontal Engine 統合デモ")
    print("=" * 60)

    # ChatGPT統合の初期化
    engine = ChatGPTFrontalEngine()

    # 質問
    questions = [
        "新規プロジェクトの開始を検討しています。背景: 予算十分、チーム経験豊富、市場需要高い。開始すべきですか？",
        "3つのベンダー候補があります。ベンダーA: 安い、サポート弱い。ベンダーB: 中程度、サポート強い。ベンダーC: 高い、実績豊富。どれが最適ですか？"
    ]

    for question in questions:
        print(f"\n📝 質問: {question}\n")
        response = engine.chat(question)
        print(f"💭 レスポンス:\n{response}\n")
        print("-" * 60)


if __name__ == "__main__":
    example_usage()
