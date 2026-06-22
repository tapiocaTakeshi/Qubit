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
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Optional torch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# MCP Server
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

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
        self.device = None
        if TORCH_AVAILABLE:
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
        判断スコアを計算（QBNN モデルベース）
        (score: 0-100, reasoning: str, key_factors: list)を返す
        """

        key_factors = []

        # QBNN モデルがあれば推論スコアを取得
        qbnn_score = self._get_qbnn_judgment_score(context, judgment_request)

        # ハイブリッドスコア計算：QBNN (70%) + 従来的分析 (30%)
        traditional_score = self._compute_traditional_score(context, judgment_request)

        score = int(qbnn_score * 0.7 + traditional_score * 0.3)
        key_factors.append("QBNN推論" if self.model else "従来的分析")

        # 基準に基づくスコア調整
        if criteria:
            criteria_score = self._evaluate_criteria(context, criteria)
            score = (score + criteria_score) / 2
            key_factors.append(f"基準マッチ度: {criteria_score}%")

        # オプション/選択肢の評価
        if options:
            options_score = self._evaluate_options(context, options)
            score = (score + options_score) / 2
            key_factors.append(f"選択肢マッチ度: {options_score}%")

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
            reasoning += f" 分析手法: {', '.join(key_factors[:3])}"

        return score, reasoning, key_factors[:5]  # 最大5つの要因

    def _get_qbnn_judgment_score(self, context: str, judgment_request: str) -> float:
        """QBNN モデルを使った判断スコアを取得"""
        if not self.model or not self.tokenizer:
            return 50.0  # モデルがない場合は中立

        try:
            # テキストを結合して処理
            full_text = f"{context} [SEP] {judgment_request}"

            # テキストをトークナイズ
            tokens = self._encode_text(full_text)
            if not tokens:
                return 50.0

            # テンソルに変換
            input_ids = torch.tensor([tokens[:512]], device=self.device)  # max_seq_len 対応

            # QBNN モデルで推論
            with torch.no_grad():
                outputs = self.model(input_ids)

            # 出力から判断スコアを抽出
            if isinstance(outputs, torch.Tensor):
                # ロジット or スコアから判断値を計算
                logits = outputs[:, -1, :2] if outputs.shape[-1] > 2 else outputs[:, 0, :]
                scores = torch.softmax(logits, dim=-1)
                positive_score = scores[0, min(1, scores.shape[-1] - 1)].item()
                return positive_score * 100

            return 50.0
        except Exception as e:
            print(f"QBNN推論エラー: {e}", file=sys.stderr)
            return 50.0

    def _encode_text(self, text: str) -> list:
        """テキストをトークン列に変換"""
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode_as_ids(text)
                return tokens if isinstance(tokens, list) else [tokens]
            else:
                # フォールバック: 簡単なトークナイズ
                import hashlib
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                return [hash_val % 7999, (hash_val // 7999) % 7999]  # vocab_size: 8000
        except Exception:
            return []

    def _compute_traditional_score(self, context: str, judgment_request: str) -> float:
        """従来的な規則ベースの判断スコアを計算"""
        score = 50

        # テキストの長さに基づくスコア調整
        context_words = len(context.split())
        if context_words > 100:
            score += 5

        # キーワード分析
        positive_keywords = ["重要", "必須", "確認", "承認", "安全", "有効", "高い", "良い", "正しい", "成功"]
        negative_keywords = ["危険", "リスク", "問題", "失敗", "低い", "悪い", "不正", "禁止"]

        for keyword in positive_keywords:
            if keyword in context or keyword in judgment_request:
                score += 3

        for keyword in negative_keywords:
            if keyword in context or keyword in judgment_request:
                score -= 3

        return max(0, min(100, score))

    def _evaluate_criteria(self, context: str, criteria: Dict[str, Any]) -> float:
        """基準に基づいてスコアを評価"""
        score = 50.0

        for criterion_name, criterion_value in criteria.items():
            # 基準がテキストに含まれているかチェック
            if isinstance(criterion_value, str):
                criterion_lower = criterion_value.lower()
                if criterion_lower in context.lower():
                    score += 15  # マッチ時はより高く評価
                else:
                    score -= 5
            elif isinstance(criterion_value, bool):
                if criterion_value:
                    score += 20
                else:
                    score -= 20
            elif isinstance(criterion_value, (int, float)):
                # 数値基準の場合
                score += min(25, max(-25, criterion_value / 5))

        return max(0, min(100, score))

    def _evaluate_options(self, context: str, options: list) -> float:
        """選択肢/オプションの評価"""
        if not options:
            return 50.0

        score = 50.0
        matched = sum(1 for opt in options if isinstance(opt, str) and opt.lower() in context.lower())

        if matched > 0:
            match_ratio = matched / len(options)
            score = 50 + match_ratio * 40  # より高く評価

        return min(100, score)

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
# MCP Server Implementation (FastMCP)
# ============================================================

if MCP_AVAILABLE and FastMCP is not None:
    mcp = FastMCP("qbnn-frontal-engine")
    judge = FrontalEngineJudge()

    @mcp.tool()
    def judge_tool(
        context: str,
        judgment_request: str,
        criteria: Optional[Dict[str, Any]] = None,
        options: Optional[list] = None,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        前頭葉の判断機能: あらゆる判断タスクを処理
        JSONデータで判断内容を指定し、Yes/No、スコア(0-100)、根拠説明を返します。
        前頭葉が行うすべての判断（意思決定、リスク評価、品質判定、倫理的判断など）に対応します。

        Args:
            context: 判断の背景情報・文脈（必須）
            judgment_request: 何を判断するか・判断内容（必須）
            criteria: 判断基準（オプション）。キー: 基準名、値: 基準値
            options: 検討対象となる選択肢・オプション（オプション）
            strict_mode: 厳密な判断モード。trueの場合スコア70以上でYes（デフォルト: false）

        Returns:
            判断結果、スコア、根拠説明を含む辞書
        """
        task = {
            "context": context,
            "judgment_request": judgment_request,
            "criteria": criteria or {},
            "options": options or [],
            "strict_mode": strict_mode
        }
        return judge.judge(task)

else:
    # Fallback when MCP is not available
    judge = FrontalEngineJudge()
    mcp = None


def main():
    """MCP Server を起動"""
    if not MCP_AVAILABLE or FastMCP is None:
        print("Error: MCP package is required to run the server.", file=sys.stderr)
        print("Please install it: pip install 'mcp>=0.5.0'", file=sys.stderr)
        sys.exit(1)

    if mcp is None:
        print("Error: MCP server initialization failed.", file=sys.stderr)
        sys.exit(1)

    mcp.run()


if __name__ == "__main__":
    main()
