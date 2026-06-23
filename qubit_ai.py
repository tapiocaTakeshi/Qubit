#!/usr/bin/env python3
"""
Qubit.ai - Claude's Quantum Prefrontal Cortex

A quantum-inspired AI system that embeds QBNN (Quantum Bidirectional
Neural Network) as Claude's prefrontal cortex, enabling advanced
decision-making, ethical reasoning, and risk assessment.

https://qubit.ai
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

from claude_prefrontal_integration import (
    ClaudePrefrontalCortex,
    JudgmentType,
)


@dataclass
class QubitAIConfig:
    """Qubit.ai configuration"""
    version: str = "1.0.0"
    product_name: str = "Qubit.ai"
    description: str = "Claude's Quantum Prefrontal Cortex"
    strict_mode: bool = False
    enable_logging: bool = True
    max_judgment_history: int = 100


class QubitAI:
    """
    Qubit.ai - Claude の量子前頭葉システム

    QBNN (Quantum Bidirectional Neural Network) を利用した
    高度な意思決定・倫理判断・リスク評価システム

    Example:
        qubit = QubitAI()
        decision = await qubit.judge("ユーザーデータをログ出力", context)
    """

    def __init__(self, config: Optional[QubitAIConfig] = None):
        """初期化"""
        self.config = config or QubitAIConfig()
        self.cortex = ClaudePrefrontalCortex()
        self.session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """セッション ID を生成"""
        timestamp = datetime.utcnow().isoformat()
        return f"qubit-ai-{timestamp}"

    # =========================================================================
    # Main API Methods
    # =========================================================================

    async def judge(
        self,
        action: str,
        context: str,
        judgment_type: Optional[str] = None,
        strict: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        アクションを判断（シンプルインターフェース）

        Args:
            action: 判断対象のアクション説明
            context: 状況説明
            judgment_type: 判断タイプ（"safety", "ethics", "quality"）
            strict: 厳密モード（デフォルト: config.strict_mode）

        Returns:
            判断結果
        """
        judgment_type = judgment_type or "safety"
        strict_mode = strict if strict is not None else self.config.strict_mode

        # Map judgment_type string to JudgmentType enum
        type_map = {
            "safety": JudgmentType.SAFETY_CHECK,
            "ethics": JudgmentType.ETHICAL_JUDGMENT,
            "quality": JudgmentType.QUALITY_JUDGMENT,
            "risk": JudgmentType.RISK_ASSESSMENT,
            "decision": JudgmentType.DECISION_MAKING,
            "priority": JudgmentType.PRIORITIZATION
        }

        judgment_enum = type_map.get(judgment_type, JudgmentType.DECISION_MAKING)

        result = await self.cortex.make_judgment(
            context=context,
            judgment_request=f"{action}?",
            judgment_type=judgment_enum,
            strict_mode=strict_mode
        )

        return self._format_result(result)

    async def safety_check(
        self,
        action: str,
        context: str,
        risks: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        セキュリティ/安全性チェック

        Args:
            action: チェック対象のアクション
            context: 状況説明
            risks: 関連するリスク要因

        Returns:
            (安全か, 判断詳細)
        """
        should_proceed, result = await self.cortex.should_proceed_with_action(
            action_description=action,
            context=context,
            risks=risks
        )
        return should_proceed, self._format_result(result)

    async def evaluate_quality(
        self,
        content: str,
        requirements: Optional[List[str]] = None,
        content_type: str = "response"
    ) -> Dict[str, Any]:
        """
        品質評価

        Args:
            content: 評価対象のコンテンツ
            requirements: 満たすべき要件
            content_type: コンテンツタイプ（"response", "code", "document"）

        Returns:
            品質評価結果
        """
        result = await self.cortex.evaluate_response_quality(
            response=content,
            requirements=requirements
        )
        return self._format_result(result)

    async def ethics_check(
        self,
        action: str,
        stakeholders: Optional[List[str]] = None,
        potential_harms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        倫理的判断

        Args:
            action: チェック対象のアクション
            stakeholders: 影響を受ける関係者
            potential_harms: 潜在的な害

        Returns:
            倫理的評価結果
        """
        result = await self.cortex.assess_ethical_concerns(
            action_description=action,
            stakeholders=stakeholders,
            potential_harms=potential_harms
        )
        return self._format_result(result)

    async def prioritize(
        self,
        items: List[Dict[str, str]],
        constraints: Optional[str] = None
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        優先順位付け

        Args:
            items: 優先順位付けの対象（name, description 含む）
            constraints: 制約条件

        Returns:
            優先度スコア付きのリスト
        """
        return await self.cortex.prioritize_tasks(items, constraints)

    # =========================================================================
    # Status & Information Methods
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Qubit.ai の情報を取得"""
        return {
            "product": self.config.product_name,
            "version": self.config.version,
            "description": self.config.description,
            "session_id": self.session_id,
            "initialized_at": datetime.utcnow().isoformat(),
            "status": "operational"
        }

    def get_status(self) -> Dict[str, Any]:
        """システムステータスを取得"""
        cortex_status = self.cortex.get_system_status()
        return {
            "product": self.config.product_name,
            "status": "operational",
            "frontal_engine_available": cortex_status["frontal_engine_available"],
            "judgment_history_size": cortex_status["judgment_history_size"],
            "max_history": self.config.max_judgment_history,
            "timestamp": cortex_status["timestamp"]
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """判断履歴を取得"""
        return self.cortex.get_judgment_history(limit=limit)

    def clear_history(self) -> None:
        """判断履歴をクリア"""
        self.cortex.judgment_history.clear()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """結果をフォーマット"""
        return {
            "decision": result.get("decision"),
            "score": result.get("score"),
            "reasoning": result.get("reasoning"),
            "confidence": result.get("confidence"),
            "factors": result.get("key_factors", []),
            "timestamp": result.get("timestamp")
        }

    async def explain(self, result: Dict[str, Any]) -> str:
        """判断結果を自然言語で説明"""
        # Format result back to cortex format for explanation
        cortex_result = {
            "decision": result.get("decision"),
            "score": result.get("score"),
            "reasoning": result.get("reasoning"),
            "confidence": result.get("confidence"),
            "key_factors": result.get("factors", []),
            "timestamp": result.get("timestamp")
        }
        return await self.cortex.explain_decision(cortex_result)


# ============================================================================
# Singleton Instance - Global Qubit.ai
# ============================================================================

_qubit_ai_instance: Optional[QubitAI] = None


def get_qubit_ai(config: Optional[QubitAIConfig] = None) -> QubitAI:
    """Get or create Qubit.ai singleton instance"""
    global _qubit_ai_instance
    if _qubit_ai_instance is None:
        _qubit_ai_instance = QubitAI(config)
    return _qubit_ai_instance


def reset_qubit_ai() -> None:
    """Reset Qubit.ai singleton instance"""
    global _qubit_ai_instance
    _qubit_ai_instance = None


# ============================================================================
# Convenience Functions - Global Scope
# ============================================================================

async def judge(
    action: str,
    context: str,
    judgment_type: str = "safety"
) -> Dict[str, Any]:
    """Global judge function"""
    qubit = get_qubit_ai()
    return await qubit.judge(action, context, judgment_type)


async def safety_check(
    action: str,
    context: str,
    risks: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Global safety check function"""
    qubit = get_qubit_ai()
    return await qubit.safety_check(action, context, risks)


async def evaluate_quality(
    content: str,
    requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Global quality evaluation function"""
    qubit = get_qubit_ai()
    return await qubit.evaluate_quality(content, requirements)


async def ethics_check(
    action: str,
    stakeholders: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Global ethics check function"""
    qubit = get_qubit_ai()
    return await qubit.ethics_check(action, stakeholders)


# ============================================================================
# CLI / Demo
# ============================================================================

async def demo():
    """Qubit.ai デモンストレーション"""
    print("=" * 70)
    print("Welcome to Qubit.ai - Claude's Quantum Prefrontal Cortex")
    print("=" * 70)
    print()

    # Initialize
    qubit = get_qubit_ai()
    print(f"✓ {qubit.config.product_name} v{qubit.config.version} initialized")
    print()

    # Show info
    info = qubit.get_info()
    print("Product Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Safety check example
    print("-" * 70)
    print("Example 1: Safety Check")
    print("-" * 70)
    should_proceed, result = await qubit.safety_check(
        action="ユーザーのメールアドレスをログに出力",
        context="デバッグモードで実行中。ログはサーバーに保存。",
        risks=["プライバシー侵害", "GDPR違反"]
    )
    print(f"Decision: {result['decision']}")
    print(f"Score: {result['score']}/100")
    print(f"Reasoning: {result['reasoning']}")
    print()

    # Quality evaluation example
    print("-" * 70)
    print("Example 2: Quality Evaluation")
    print("-" * 70)
    response = "Pythonでファイルを読むには open() 関数を使用します。"
    quality = await qubit.evaluate_quality(
        content=response,
        requirements=["詳細な例", "実用的", "わかりやすい"]
    )
    print(f"Decision: {quality['decision']}")
    print(f"Score: {quality['score']}/100")
    print()

    # Ethics check example
    print("-" * 70)
    print("Example 3: Ethics Check")
    print("-" * 70)
    ethics = await qubit.ethics_check(
        action="ユーザー行動を分析して個人を特定する",
        stakeholders=["ユーザー", "社会"]
    )
    print(f"Ethical Decision: {ethics['decision']}")
    print(f"Score: {ethics['score']}/100")
    print()

    # Show status
    print("-" * 70)
    print("System Status")
    print("-" * 70)
    status = qubit.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    # Show history
    print("-" * 70)
    print("Judgment History")
    print("-" * 70)
    history = qubit.get_history(limit=5)
    print(f"Total judgments: {len(history)}")
    if history:
        for i, record in enumerate(history, 1):
            print(f"  {i}. {record.get('judgment_type')}: {record.get('decision')} (score: {record.get('score')})")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())
