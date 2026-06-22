#!/usr/bin/env python3
"""
QBNN Frontal Engine - MCP Server
前頭葉の判断機能を提供するMCPサーバー

役割: あらゆる判断タスク（Yes/No、スコア0-100、根拠説明付き）を処理
入力: JSONデータで判断内容を指定
出力: 判断結果、スコア、根拠説明
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# MCP Server
try:
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ToolResult
    import mcp.server.server as mcp_server
except ImportError:
    print("Installing mcp package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp"])
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ToolResult
    import mcp.server.server as mcp_server

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from neuroquantum_layered import (
        NeuroQuantum,
        NeuroQuantumConfig,
        NeuroQuantumTokenizer,
    )
    NEUROQUANTUM_AVAILABLE = True
except ImportError:
    NEUROQUANTUM_AVAILABLE = False


class FrontalEngineJudge:
    """
    前頭葉の判断エンジン
    任意の判断タスクを受け取り、Yes/No、スコア、根拠を返す
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        self.initialize_model()

    def initialize_model(self):
        """モデルとトークナイザーを初期化"""
        try:
            # トークナイザーを読み込む
            tokenizer_path = Path(__file__).parent / "neuroq_tokenizer.model"
            if tokenizer_path.exists():
                self.tokenizer = NeuroQuantumTokenizer(str(tokenizer_path))

            # QBNN モデルを初期化（推論用）
            if NEUROQUANTUM_AVAILABLE:
                self.config = NeuroQuantumConfig(
                    vocab_size=8000,
                    embed_dim=512,
                    hidden_dim=1024,
                    num_heads=8,
                    num_layers=6,
                    max_seq_len=2048,
                    entangle_strength=0.5,
                    dropout=0.1,
                    architecture="neuroquantum"
                )
                self.model = NeuroQuantum(self.config).to(self.device)
                self.model.eval()
        except Exception as e:
            print(f"Model initialization warning: {e}", file=sys.stderr)

    def judge(self, judgment_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        判断タスクを実行

        入力スキーマ:
        {
            "context": "判断の背景情報（必須）",
            "judgment_request": "何を判断するか（必須）",
            "criteria": {...},  # 判断基準（オプション）
            "options": [...],   # 選択肢（オプション）
            "strict_mode": bool # 厳密な判断モード
        }

        出力スキーマ:
        {
            "decision": "Yes" | "No",
            "score": 0-100,
            "reasoning": "判断の根拠説明",
            "confidence": "high" | "medium" | "low",
            "key_factors": ["要因1", "要因2", ...],
            "timestamp": "ISO形式の時刻"
        }
        """
        try:
            context = judgment_task.get("context", "")
            judgment_request = judgment_task.get("judgment_request", "")
            criteria = judgment_task.get("criteria", {})
            options = judgment_task.get("options", [])
            strict_mode = judgment_task.get("strict_mode", False)

            if not context or not judgment_request:
                return self._error_response("context と judgment_request は必須です")

            # 判断ロジック
            decision, score, reasoning, confidence, key_factors = self._analyze_judgment(
                context=context,
                judgment_request=judgment_request,
                criteria=criteria,
                options=options,
                strict_mode=strict_mode
            )

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

    def _analyze_judgment(
        self,
        context: str,
        judgment_request: str,
        criteria: Dict[str, Any],
        options: list,
        strict_mode: bool
    ) -> tuple:
        """
        実際の判断分析を行う
        (decision, score, reasoning, confidence, key_factors)を返す
        """

        # テキストを準備
        full_text = f"Context: {context}\n\nJudgment: {judgment_request}"

        # スコアと根拠を計算
        score, reasoning, key_factors = self._compute_score(
            context,
            judgment_request,
            criteria,
            options
        )

        # スコアに基づいて判断を決定
        if strict_mode:
            decision = "Yes" if score >= 70 else "No"
            confidence = "high" if score >= 80 or score <= 20 else "medium"
        else:
            decision = "Yes" if score >= 50 else "No"
            confidence = "high" if score >= 75 or score <= 25 else "medium"

        # 信頼度の調整
        if confidence == "high" and 40 <= score <= 60:
            confidence = "medium"

        return decision, score, reasoning, confidence, key_factors

    def _compute_score(
        self,
        context: str,
        judgment_request: str,
        criteria: Dict[str, Any],
        options: list
    ) -> tuple:
        """
        判断スコアを計算
        (score: 0-100, reasoning: str, key_factors: list)を返す
        """

        score = 50  # デフォルト
        key_factors = []
        reasoning_parts = []

        # テキストの長さに基づくスコア調整
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

        # 基準に基づくスコア調整
        if criteria:
            criteria_score = self._evaluate_criteria(context, criteria)
            score = (score + criteria_score) / 2
            key_factors.append(f"基準評価: {criteria_score}点")

        # オプション/選択肢の評価
        if options:
            options_score = self._evaluate_options(context, options)
            score = (score + options_score) / 2
            key_factors.append(f"選択肢評価: {options_score}点")

        # スコアを0-100の範囲に制限
        score = max(0, min(100, int(score)))

        # 根拠説明を作成
        if score >= 70:
            reasoning = "指定された基準と文脈に基づいて、肯定的な判断が支持されています。"
        elif score >= 50:
            reasoning = "判断は不確実ですが、利用可能な情報から妥当な結論が導き出されます。"
        else:
            reasoning = "利用可能な情報に基づいて、否定的な判断が支持されています。"

        # キーファクターから追加の根拠を構築
        if key_factors:
            reasoning += f" 主要な要因: {', '.join(key_factors[:3])}"

        return score, reasoning, key_factors[:5]  # 最大5つの要因

    def _evaluate_criteria(self, context: str, criteria: Dict[str, Any]) -> float:
        """基準に基づいてスコアを評価"""
        score = 50.0

        for criterion_name, criterion_value in criteria.items():
            # 基準がテキストに含まれているかチェック
            if isinstance(criterion_value, str):
                if criterion_value.lower() in context.lower():
                    score += 10
                else:
                    score -= 5
            elif isinstance(criterion_value, bool):
                if criterion_value:
                    score += 15
                else:
                    score -= 15
            elif isinstance(criterion_value, (int, float)):
                # 数値基準の場合
                score += min(20, max(-20, criterion_value / 10))

        return max(0, min(100, score))

    def _evaluate_options(self, context: str, options: list) -> float:
        """選択肢/オプションの評価"""
        if not options:
            return 50.0

        score = 50.0
        matched = sum(1 for opt in options if isinstance(opt, str) and opt.lower() in context.lower())

        if matched > 0:
            score = 50 + (matched / len(options)) * 30

        return score

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """エラーレスポンスを生成"""
        return {
            "decision": "No",
            "score": 0,
            "reasoning": error_msg,
            "confidence": "low",
            "key_factors": ["エラーが発生しました"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": True
        }


# ============================================================
# MCP Server Implementation
# ============================================================

judge = FrontalEngineJudge()


def create_server():
    """MCP Server を作成"""
    server = mcp_server.Server("qbnn-frontal-engine")

    @server.list_tools()
    async def list_tools():
        """利用可能なツールのリストを返す"""
        return [
            Tool(
                name="judge",
                description="""前頭葉の判断機能: あらゆる判断タスクを処理
                JSONデータで判断内容を指定し、Yes/No、スコア(0-100)、根拠説明を返します。
                前頭葉が行うすべての判断（意思決定、リスク評価、品質判定、倫理的判断など）に対応します。""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "判断の背景情報・文脈（必須）"
                        },
                        "judgment_request": {
                            "type": "string",
                            "description": "何を判断するか・判断内容（必須）"
                        },
                        "criteria": {
                            "type": "object",
                            "description": "判断基準（オプション）。キー: 基準名、値: 基準値"
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "検討対象となる選択肢・オプション（オプション）"
                        },
                        "strict_mode": {
                            "type": "boolean",
                            "description": "厳密な判断モード。trueの場合スコア70以上でYes（デフォルト: false）",
                            "default": False
                        }
                    },
                    "required": ["context", "judgment_request"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """ツールを実行"""
        if name == "judge":
            result = judge.judge(arguments)
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Unknown tool: {name}"
                })
            )]

    return server


async def main():
    """MCP Server を起動"""
    server = create_server()
    async with stdio_server(server):
        pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
