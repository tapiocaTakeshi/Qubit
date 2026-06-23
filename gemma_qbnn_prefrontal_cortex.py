#!/usr/bin/env python3
"""
Gemma + QBNN Prefrontal Cortex
前頭葉として動作する量子強化意思決定システム

役割:
  Gemma(LLM) + QBNN(量子層) を統合して、
  複雑な判断タスク（意思決定、リスク評価、倫理判断）を
  量子的推論能力で実行する前頭葉システム

アーキテクチャ:
  [入力テキスト]
       ↓
  [Gemma Transformer層] (意味理解)
       ↓
  [QBNN量子補正層] (量子推論)
       ↓
  [判断ヘッド] (決定生成)
       ↓
  [Yes/No + スコア + 根拠]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from gemma_qbnn import GemmaQBNN, create_gemma_qbnn_model
    from qbnn_layered import EQBNNLayer, APQB
    from neuroquantum_layered import NeuroQuantumTokenizer
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False


@dataclass
class JudgmentConfig:
    """判断エンジン用の設定"""
    vocab_size: int = 32000
    embed_dim: int = 768
    hidden_dim: int = 2048
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 4096
    entangle_strength: float = 0.7
    quantum_weight: float = 0.6  # QBNN寄与度
    decision_threshold: float = 0.5
    confidence_threshold: float = 0.7


class JudgmentHead(nn.Module):
    """
    判断ヘッド: Gemma+QBNN出力から判断を生成

    出力:
    - decision_logits: Yes/No (2値)
    - score_logits: 0-100のスコア
    - confidence_logits: 信頼度
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # 判断層（Yes/No）
        self.decision_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Yes/No
        )

        # スコア層（0-100）
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 連続値
        )

        # 信頼度層
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # low/medium/high
        )

        # 根拠生成層
        self.reasoning_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256)  # 根拠トークン
        )

    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, seq, embed_dim] または [batch, embed_dim]

        Returns:
            判断結果の辞書
        """
        # 最後のトークンの表現を使用（または平均化）
        if hidden_state.dim() == 3:
            # [batch, seq, dim] の場合
            representation = hidden_state[:, -1, :]  # 最後のトークン
        else:
            representation = hidden_state

        return {
            "decision_logits": self.decision_head(representation),
            "score_logits": self.score_head(representation),
            "confidence_logits": self.confidence_head(representation),
            "reasoning_logits": self.reasoning_head(representation)
        }


class GemmaQBNNPrefrontalCortex(nn.Module):
    """
    Gemma + QBNN 前頭葉

    量子強化型意思決定システム:
    1. Gemmaで複雑なテキストを理解
    2. QBNN量子層で非古典的推論
    3. 判断ヘッドで決定を生成
    """

    def __init__(self, config: JudgmentConfig):
        super().__init__()
        self.config = config
        self.device = None

        if not GEMMA_AVAILABLE:
            raise ImportError("Gemma+QBNN modules required. Check dependencies.")

        # ベースモデル: Gemma + QBNN
        self.base_model = create_gemma_qbnn_model(
            size="medium",
            vocab_size=config.vocab_size
        )

        # 判断ヘッド
        self.judgment_head = JudgmentHead(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim
        )

        # 量子推論層（追加の量子処理）
        self.quantum_reasoning = EQBNNLayer(
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
            prev_output_dim=config.embed_dim,
            entangle_strength=config.entangle_strength
        )

        # トークナイザー
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """トークナイザーを読み込む"""
        try:
            tokenizer_path = "neuroq_tokenizer.model"
            if os.path.exists(tokenizer_path):
                self.tokenizer = NeuroQuantumTokenizer(tokenizer_path)
        except Exception as e:
            print(f"Tokenizer loading warning: {e}", file=sys.stderr)

    def encode_text(self, text: str, max_len: int = None) -> torch.Tensor:
        """テキストをトークンに変換"""
        if max_len is None:
            max_len = self.config.max_seq_len

        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode_as_ids(text)
                if not isinstance(tokens, list):
                    tokens = [tokens]
            else:
                # フォールバック: 文字ベースのハッシュトークン化
                tokens = [ord(c) % self.config.vocab_size for c in text[:max_len]]

            # パディング
            if len(tokens) < max_len:
                tokens = tokens + [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]

            return torch.tensor([tokens], dtype=torch.long)
        except Exception as e:
            print(f"Text encoding error: {e}", file=sys.stderr)
            return torch.zeros(1, max_len, dtype=torch.long)

    def forward(
        self,
        context: str,
        judgment_request: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        判断を実行

        Args:
            context: 判断の背景情報
            judgment_request: 何を判断するか
            device: 実行デバイス

        Returns:
            判断結果
        """
        if device is None:
            device = next(self.parameters()).device

        self.device = device

        # テキストを結合してエンコード
        full_text = f"{context} [SEP] {judgment_request}"
        input_ids = self.encode_text(full_text, self.config.max_seq_len)
        input_ids = input_ids.to(device)

        # Gemma+QBNN で処理
        with torch.no_grad():
            gemma_output = self.base_model(input_ids)  # [batch, seq, vocab]

        # 隠れ状態を抽出（最後のレイヤー出力）
        hidden_states = gemma_output[:, :, :self.config.embed_dim]

        # 量子推論層で強化
        batch, seq, dim = hidden_states.shape
        flat = hidden_states.reshape(batch * seq, dim)
        quantum_correction, _ = self.quantum_reasoning(flat, q_prev=flat)
        hidden_states = hidden_states + quantum_correction.reshape(batch, seq, dim) * self.config.quantum_weight

        # 判断ヘッドで出力を生成
        judgment_logits = self.judgment_head(hidden_states)

        return judgment_logits

    def judge(self, judgment_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        判断タスクを実行（FrontalEngineJudge互換インターフェース）

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
            "timestamp": "ISO形式の時刻",
            "quantum_info": {...}  # 量子推論情報
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

            device = next(self.parameters()).device

            # 推論を実行
            outputs = self.forward(context, judgment_request, device)

            # 判断を抽出
            decision_logits = outputs["decision_logits"]
            score_logits = outputs["score_logits"]
            confidence_logits = outputs["confidence_logits"]

            # 判断を計算
            decision_probs = F.softmax(decision_logits, dim=-1)
            yes_prob = decision_probs[0, 1].item()  # Yesの確率

            # スコアを計算（0-100）
            score_raw = torch.clamp(score_logits[0, 0], 0, 1).item()
            score = int(score_raw * 100)

            # 信頼度を計算
            confidence_probs = F.softmax(confidence_logits, dim=-1)
            confidence_idx = confidence_logits[0].argmax().item()
            confidence_map = ["low", "medium", "high"]
            confidence = confidence_map[confidence_idx]

            # 判断を決定
            if strict_mode:
                decision = "Yes" if yes_prob >= 0.7 else "No"
            else:
                decision = "Yes" if yes_prob >= 0.5 else "No"

            # 根拠を生成
            reasoning = self._generate_reasoning(
                context, judgment_request, decision, score, yes_prob
            )

            # キーファクターを抽出
            key_factors = self._extract_key_factors(
                context, judgment_request, criteria, options
            )

            return {
                "decision": decision,
                "score": score,
                "reasoning": reasoning,
                "confidence": confidence,
                "key_factors": key_factors,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "quantum_info": {
                    "yes_probability": yes_prob,
                    "quantum_weight": self.config.quantum_weight,
                    "entangle_strength": self.config.entangle_strength
                }
            }

        except Exception as e:
            print(f"Judgment error: {e}", file=sys.stderr)
            return self._error_response(f"判断処理エラー: {str(e)}")

    def _generate_reasoning(
        self,
        context: str,
        judgment_request: str,
        decision: str,
        score: int,
        confidence: float
    ) -> str:
        """根拠を生成"""
        if score >= 70:
            reason_base = "量子推論により、提供された情報は肯定的な判断を支持しています。"
        elif score >= 50:
            reason_base = "量子推論の結果、判断は不確定ですが、妥当な結論が導き出されます。"
        else:
            reason_base = "量子推論により、提供された情報は否定的な判断を支持しています。"

        # 信頼度情報を追加
        confidence_text = f"信頼度: {confidence:.1%}"

        return f"{reason_base} {confidence_text}"

    def _extract_key_factors(
        self,
        context: str,
        judgment_request: str,
        criteria: Dict[str, Any],
        options: List[str]
    ) -> List[str]:
        """主要要因を抽出"""
        factors = []

        # 基準マッチ
        if criteria:
            for key in list(criteria.keys())[:2]:
                factors.append(f"基準: {key}")

        # オプションマッチ
        if options:
            matched = sum(1 for opt in options if isinstance(opt, str) and opt in context)
            if matched > 0:
                factors.append(f"マッチオプション: {matched}/{len(options)}")

        # テキスト長による複雑性
        if len(context) > 200:
            factors.append("複雑なコンテキスト")

        # 判断タイプ
        if "リスク" in judgment_request:
            factors.append("リスク評価")
        elif "倫理" in judgment_request:
            factors.append("倫理的判断")
        elif "優先" in judgment_request:
            factors.append("優先順位付け")

        # 量子情報
        factors.append("量子推論適用")

        return factors[:5]  # 最大5つ

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

    def get_constraint_loss(self) -> torch.Tensor:
        """QBNN層の制約損失を取得"""
        loss = self.base_model.get_constraint_loss()
        loss = loss + self.quantum_reasoning.get_constraint_loss()
        return loss


def create_prefrontal_cortex(
    config: Optional[JudgmentConfig] = None,
    device: Optional[torch.device] = None
) -> GemmaQBNNPrefrontalCortex:
    """前頭葉システムを作成"""
    if config is None:
        config = JudgmentConfig()

    cortex = GemmaQBNNPrefrontalCortex(config)

    if device is not None:
        cortex = cortex.to(device)

    return cortex


if __name__ == "__main__":
    print("=== Gemma+QBNN Prefrontal Cortex ===\n")

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 前頭葉システムの作成
    cortex = create_prefrontal_cortex(device=device)

    # 判断タスクの例
    judgment_task = {
        "context": """
        ユーザーがシステムに個人情報（名前、メールアドレス、電話番号）の削除を要求しています。
        削除要求の根拠は「サービス利用を中止するため」とのこと。
        システムはGDPRに準拠した削除メカニズムを持っています。
        ただし、削除前に削除確認メールを送信する必要があります。
        """,
        "judgment_request": "この個人情報削除要求を実行してもセキュアで適切か？",
        "criteria": {
            "security": True,
            "gdpr_compliance": True,
            "user_consent": True
        },
        "options": ["実行する", "確認メール送信後に実行", "拒否する"],
        "strict_mode": True
    }

    print("判断タスク:")
    print(json.dumps({k: v for k, v in judgment_task.items() if k != "context"}, ensure_ascii=False, indent=2))
    print(f"\nコンテキスト: {judgment_task['context'][:100]}...\n")

    # 判断を実行
    result = cortex.judge(judgment_task)

    print("判断結果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
