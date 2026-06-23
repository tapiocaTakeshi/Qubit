#!/usr/bin/env python3
"""
Claude Prefrontal Cortex Integration Examples
実際の使用例とユースケース
"""

import asyncio
import json
from typing import List, Dict, Any

# Import the integration module
from claude_prefrontal_integration import (
    ClaudePrefrontalCortex,
    JudgmentType,
    judge_action,
    judge_response_quality,
    check_ethics
)


# ============================================================================
# Example 1: Security Decision Making
# ============================================================================

async def example_security_checks():
    """
    セキュリティ意思決定の例
    ユーザーデータへのアクセスやログ出力などの安全性を判断
    """
    print("=" * 60)
    print("Example 1: Security Decision Making")
    print("=" * 60)

    # Scenario 1: Logging user data
    print("\n【Scenario 1】: Logging sensitive user data")
    print("-" * 60)

    should_proceed, result = await judge_action(
        action="ユーザーのメールアドレスをDebugログに出力する",
        context="デバッグモードで実行中。ログはサーバーに保存されます。",
        risks=["プライバシー侵害", "GDPR違反", "セキュリティリスク"]
    )

    print(f"Action: Log user email address")
    print(f"Should Proceed: {should_proceed}")
    print(f"Decision: {result['decision']}")
    print(f"Score: {result['score']}/100")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

    # Scenario 2: Database access
    print("\n【Scenario 2】: Database access for feature")
    print("-" * 60)

    should_proceed, result = await judge_action(
        action="新機能のため全ユーザーデータにアクセスする",
        context="新しいアナリティクス機能を実装中。アクセスは認可されており、ユーザーは明示的に同意している。",
        risks=["データベースパフォーマンス低下"]
    )

    print(f"Action: Access all user data for analytics")
    print(f"Should Proceed: {should_proceed}")
    print(f"Decision: {result['decision']}")
    print(f"Score: {result['score']}/100")


# ============================================================================
# Example 2: Response Quality Evaluation
# ============================================================================

async def example_response_quality():
    """
    応答品質の評価例
    生成された回答がユーザーの要求を満たしているかを判定
    """
    print("\n" + "=" * 60)
    print("Example 2: Response Quality Evaluation")
    print("=" * 60)

    responses = [
        {
            "user_query": "Pythonでファイル処理をするにはどうしたらいい？",
            "response": "Pythonでファイル処理をするときは、built-inの`open()`関数を使います。",
            "requirements": ["詳細な例", "実用的", "安全なファイル処理"]
        },
        {
            "user_query": "機械学習の基礎を説明してください",
            "response": "機械学習とは、データから自動的に学習するシステムです。教師あり学習と教師なし学習があります。",
            "requirements": ["理論的", "わかりやすい", "実装例"]
        },
        {
            "user_query": "このバグを修正してください",
            "response": "OK",
            "requirements": ["詳細な説明", "解決策", "テスト方法"]
        }
    ]

    for i, item in enumerate(responses, 1):
        print(f"\n【Response {i}】")
        print(f"Query: {item['user_query']}")
        print(f"Response: {item['response'][:50]}...")

        quality = await judge_response_quality(
            response=item['response'],
            requirements=item['requirements']
        )

        print(f"Quality Decision: {quality['decision']}")
        print(f"Score: {quality['score']}/100")
        print(f"Confidence: {quality['confidence']}")

        if quality['decision'] == 'No':
            print(f"⚠️  Needs Improvement: {quality['reasoning']}")


# ============================================================================
# Example 3: Ethical Judgment
# ============================================================================

async def example_ethical_judgment():
    """
    倫理的判断の例
    提案されたアクションが倫理的に適切かを評価
    """
    print("\n" + "=" * 60)
    print("Example 3: Ethical Judgment")
    print("=" * 60)

    scenarios = [
        {
            "action": "ユーザーの購買履歴を分析して、個人の健康状態を推測する",
            "stakeholders": ["ユーザー", "企業", "社会"],
            "harms": ["プライバシー侵害", "差別的な推測", "信頼侵害"]
        },
        {
            "action": "アルゴリズムを改善することで、全体的な効率が向上する（ただし一部ユーザーに悪影響）",
            "stakeholders": ["ユーザー全体", "特定グループ"],
            "harms": ["不平等", "一部ユーザーへの悪影響"]
        },
        {
            "action": "透明性のためにアルゴリズムの詳細をオープンソースで公開する",
            "stakeholders": ["ユーザー", "企業", "セキュリティ"],
            "harms": ["セキュリティリスク"]
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n【Scenario {i}】")
        print(f"Action: {scenario['action'][:60]}...")

        ethics = await check_ethics(
            action=scenario['action'],
            stakeholders=scenario['stakeholders']
        )

        print(f"Ethical Decision: {ethics['decision']}")
        print(f"Score: {ethics['score']}/100")
        print(f"Confidence: {ethics['confidence']}")

        if ethics['decision'] == 'No':
            print(f"⚠️  Ethical Concerns: {ethics['reasoning']}")


# ============================================================================
# Example 4: Task Prioritization
# ============================================================================

async def example_task_prioritization():
    """
    タスク優先順位付けの例
    複数のタスクを優先度順にソート
    """
    print("\n" + "=" * 60)
    print("Example 4: Task Prioritization")
    print("=" * 60)

    tasks = [
        {
            "name": "本番環境でのクリティカルバグ修正",
            "description": "ユーザーがログインできない問題。50,000ユーザーに影響。スケーラビリティ問題が原因。"
        },
        {
            "name": "新機能の開発",
            "description": "ユーザーリクエストが多い新しいダッシュボード機能。見積もり2週間。"
        },
        {
            "name": "技術的負債の削減",
            "description": "レガシーコードの大規模リファクタリング。見積もり3週間。保守性向上。"
        },
        {
            "name": "セキュリティアップデート",
            "description": "依存パッケージのセキュリティパッチ。緊急度は中程度。"
        },
        {
            "name": "ドキュメント整備",
            "description": "APIドキュメントの改善。開発者体験向上。見積もり1週間。"
        }
    ]

    constraints = "利用可能なリソース: 開発者3名、期間2週間"

    cortex = ClaudePrefrontalCortex()
    prioritized = await cortex.prioritize_tasks(tasks, constraints)

    print("\nTask Priority Ranking:")
    print("-" * 60)
    for rank, (task, score) in enumerate(prioritized, 1):
        print(f"{rank}. {task['name']}")
        print(f"   Priority Score: {score:.2f}")
        print(f"   Description: {task['description'][:50]}...")
        print()


# ============================================================================
# Example 5: Integrated Decision Pipeline
# ============================================================================

async def example_integrated_pipeline():
    """
    統合的な意思決定パイプラインの例
    複数の判断ステップから最終判断を導出
    """
    print("\n" + "=" * 60)
    print("Example 5: Integrated Decision Pipeline")
    print("=" * 60)

    # Feature request from user
    feature_request = {
        "name": "ユーザー行動トラッキング機能",
        "description": "ユーザーのクリック、スクロール、滞在時間をトラッキング",
        "timeline": "2週間",
        "resources_needed": "2名の開発者"
    }

    print(f"Feature Request: {feature_request['name']}")
    print(f"Description: {feature_request['description']}")
    print()

    cortex = ClaudePrefrontalCortex()

    # Step 1: Safety Check
    print("Step 1: Security/Safety Check")
    print("-" * 40)
    safety_result, _ = await judge_action(
        action=f"実装: {feature_request['name']}",
        context="新しいトラッキング機能の実装",
        risks=["プライバシー侵害", "GDPR非準拠", "ユーザー信頼損失"]
    )
    print(f"✓ Safety Check: {'PASS' if safety_result else 'FAIL'}")

    if not safety_result:
        print("❌ Feature blocked for security reasons")
        return

    # Step 2: Ethical Judgment
    print("\nStep 2: Ethical Review")
    print("-" * 40)
    ethics_result = await check_ethics(
        action=feature_request['name'],
        stakeholders=["ユーザー", "組織", "規制当局"]
    )
    print(f"✓ Ethics Score: {ethics_result['score']}/100")

    if ethics_result['score'] < 50:
        print(f"⚠️  Ethical concerns: {ethics_result['reasoning']}")

    # Step 3: Resource Assessment
    print("\nStep 3: Resource & Timeline Assessment")
    print("-" * 40)
    resource_judgment = await cortex.make_judgment(
        context=f"必要なリソース: {feature_request['resources_needed']}, タイムライン: {feature_request['timeline']}",
        judgment_request="利用可能なリソースで予定期間内に実装可能か？",
        criteria={"resource_availability": "sufficient", "timeline_realistic": True}
    )
    print(f"✓ Resource Assessment: {resource_judgment['decision']}")

    # Final Decision
    print("\nFinal Decision:")
    print("-" * 40)
    all_checks_passed = (
        safety_result and
        ethics_result['score'] >= 50 and
        resource_judgment['decision'] == 'Yes'
    )

    if all_checks_passed:
        print("✅ Feature APPROVED for implementation")
    else:
        print("❌ Feature REJECTED - Address concerns before proceeding")

    # Print summary
    print("\nDecision Summary:")
    print("-" * 40)
    print(f"  Security: {'✓' if safety_result else '✗'}")
    print(f"  Ethics: {ethics_result['score']}/100")
    print(f"  Resources: {resource_judgment['decision']}")


# ============================================================================
# Example 6: Real-time Response Validation
# ============================================================================

async def example_response_validation():
    """
    リアルタイム応答検証の例
    Claude が生成した応答をその場で品質検証
    """
    print("\n" + "=" * 60)
    print("Example 6: Real-time Response Validation")
    print("=" * 60)

    candidate_responses = [
        {
            "topic": "Python非同期プログラミング",
            "response": """
Python非同期プログラミングについて説明します。

非同期プログラミングは、複数のタスクを同時に処理する方法です。
以下が基本的な例です：

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "データ取得完了"

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

このコードは1秒待機した後、"データ取得完了"を出力します。
            """,
            "requirements": ["実装例を含む", "わかりやすい説明", "ベストプラクティス"]
        }
    ]

    for response_item in candidate_responses:
        print(f"\nTopic: {response_item['topic']}")
        print("-" * 60)

        quality = await judge_response_quality(
            response=response_item['response'],
            requirements=response_item['requirements']
        )

        print(f"Quality Assessment:")
        print(f"  Decision: {quality['decision']}")
        print(f"  Score: {quality['score']}/100")
        print(f"  Confidence: {quality['confidence']}")
        print(f"  Key Factors: {', '.join(quality['key_factors'][:3])}")

        if quality['decision'] == 'Yes':
            print(f"\n✅ Response is ready to send to user")
        else:
            print(f"\n⚠️  Needs improvement: {quality['reasoning']}")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all examples"""
    try:
        await example_security_checks()
        await example_response_quality()
        await example_ethical_judgment()
        await example_task_prioritization()
        await example_integrated_pipeline()
        await example_response_validation()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
