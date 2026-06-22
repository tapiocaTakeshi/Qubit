#!/usr/bin/env python3
"""
QBNN Frontal Engine - Google Gemini Integration
Google Generative AI を使用した統合
"""

import json
from typing import Optional, Dict, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from frontal_engine_mcp_server import FrontalEngineJudge


# ============================================================
# Judge Tool Definition for Gemini
# ============================================================

JUDGE_TOOL_DEFINITION = {
    "name": "judge",
    "description": "前頭葉の判断機能: あらゆる判断タスクを処理。Yes/No、スコア(0-100)、根拠説明を返します。意思決定、リスク評価、品質判定、倫理的判断などに対応。",
    "input_schema": {
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
                "description": "厳密モード。true: スコア70以上でYes、false: スコア50以上でYes（デフォルト: false）"
            }
        },
        "required": ["context", "judgment_request"]
    }
}


# ============================================================
# Gemini Integration
# ============================================================

class GeminiFrontalEngine:
    """Gemini統合用のFrontal Engine ラッパー"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        初期化

        Args:
            api_key: Google API キー。Noneの場合は環境変数から取得
            model: 使用するGeminiモデル
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library is required. "
                "Install: pip install google-generativeai"
            )

        if api_key:
            genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=model,
            tools=[{"function_declarations": [JUDGE_TOOL_DEFINITION]}]
        )
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

    def chat(self, user_message: str, system_instruction: Optional[str] = None) -> str:
        """
        チャットを実行。必要に応じてJudgeツールを呼び出す

        Args:
            user_message: ユーザーメッセージ
            system_instruction: システムプロンプト（オプション）

        Returns:
            アシスタントレスポンス
        """
        # システムプロンプト（デフォルト）
        if system_instruction is None:
            system_instruction = """あなたは判断を支援するAIアシスタントです。
ユーザーが判断や意思決定に関する質問をしたときは、
Judgeツールを使用して前頭葉的な判断を提供してください。
判断結果に基づいて、詳しく説明してください。"""

        # 会話履歴に追加
        self.conversation_history.append({
            "role": "user",
            "parts": [user_message]
        })

        # チャットセッション作成（システムプロンプト付き）
        chat = self.model.start_chat(history=self.conversation_history)

        # ツール呼び出しループ
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # リクエスト送信
            response = chat.send_message(
                user_message if iteration == 1 else "Continue processing",
                stream=False
            )

            # ツール呼び出し処理
            if response.parts and hasattr(response.parts[0], 'function_call'):
                func_call = response.parts[0].function_call

                # Judge ツール呼び出し
                if func_call.name == "judge":
                    tool_args = {k: v for k, v in func_call.args.items()}
                    judge_result = self.call_judge(**tool_args)

                    # レスポンスを履歴に追加
                    self.conversation_history.append({
                        "role": "model",
                        "parts": response.parts
                    })

                    # ツール結果を履歴に追加
                    self.conversation_history.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": "judge",
                                "response": judge_result
                            }
                        }]
                    })

                    # 次のリクエスト
                    response = chat.send_message("Process the tool response")
                else:
                    break
            else:
                # テキストレスポンス取得
                break

        # 最終レスポンス
        final_response = response.text if response.text else "No response generated"

        # 会話履歴に追加
        self.conversation_history.append({
            "role": "model",
            "parts": [final_response]
        })

        return final_response

    def clear_history(self):
        """会話履歴をクリア"""
        self.conversation_history = []


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    """使用例"""
    print("=" * 60)
    print("Gemini + Frontal Engine 統合デモ")
    print("=" * 60)

    # Gemini統合の初期化
    engine = GeminiFrontalEngine()

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
