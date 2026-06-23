#!/usr/bin/env python3
"""
FrontalEngineJudge の動作確認（PyTorch不要版）
MCPサーバーの判断エンジンをテスト
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

# MCPサーバーをインポート（PyTorchなしでも基本機能は動作）
try:
    from frontal_engine_mcp_server import FrontalEngineJudge
    print("✓ FrontalEngineJudge をインポート成功\n")
except Exception as e:
    print(f"✗ インポート失敗: {e}")
    sys.exit(1)


def test_judge():
    """FrontalEngineJudgeの動作テスト"""

    print("=" * 70)
    print("FrontalEngineJudge - MCP サーバー判断エンジン テスト")
    print("=" * 70)

    # Judge エンジンを初期化
    print("\n[1] Judge エンジン初期化中...")
    try:
        judge = FrontalEngineJudge()
        print("✓ Judge エンジン初期化完了")
        print(f"  - 量子前頭葉有効: {judge.use_quantum}")
        print(f"  - デバイス: {judge.device}")
    except Exception as e:
        print(f"✗ 初期化失敗: {e}")
        return

    # テストケース
    test_cases = [
        {
            "name": "セキュリティ判断",
            "task": {
                "context": """
                ユーザーの個人情報削除リクエスト。
                GDPRに準拠した削除メカニズムあり。
                削除確認メール機能あり。
                バックアップシステムで復旧可能。
                """,
                "judgment_request": "個人情報削除を実行してもセキュアか？",
                "criteria": {
                    "gdpr_compliance": True,
                    "backup_available": True
                },
                "strict_mode": True
            }
        },
        {
            "name": "プロジェクト承認判断",
            "task": {
                "context": """
                新プロジェクト提案：
                - 予算: 承認済み
                - リスク分析: 低リスク
                - 技術的実現性: 確認完了
                - チーム配置: 決定済み
                """,
                "judgment_request": "このプロジェクトを開始してもよいか？",
                "criteria": {
                    "budget_approved": True,
                    "risk_level": "low"
                },
                "strict_mode": False
            }
        },
        {
            "name": "倫理的懸念評価",
            "task": {
                "context": """
                ユーザーの行動データを分析して推奨を生成。
                問題点:
                - ユーザー明示同意なし
                - プライバシー侵害リスク
                - 透明性欠如
                """,
                "judgment_request": "このデータ分析は倫理的に適切か？",
                "strict_mode": True
            }
        }
    ]

    # 各テストを実行
    print("\n" + "=" * 70)
    print("判断タスクテスト実行")
    print("=" * 70)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n【テスト{i}】{test_case['name']}")
        print("-" * 70)

        try:
            result = judge.judge(test_case['task'])

            # 結果を表示
            print(f"決定: {result.get('decision', 'N/A')}")
            print(f"スコア: {result.get('score', 'N/A')}/100")
            print(f"信頼度: {result.get('confidence', 'N/A')}")
            print(f"根拠: {result.get('reasoning', 'N/A')[:100]}...")

            if 'quantum_info' in result:
                print(f"量子情報:")
                for key, value in result['quantum_info'].items():
                    if isinstance(value, float):
                        print(f"  - {key}: {value:.2f}")
                    else:
                        print(f"  - {key}: {value}")

            results.append({
                'name': test_case['name'],
                'success': True,
                'result': result
            })
            print("✓ テスト成功")

        except Exception as e:
            print(f"✗ テスト失敗: {e}")
            results.append({
                'name': test_case['name'],
                'success': False,
                'error': str(e)
            })

    # サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['name']}")

    print(f"\n成功: {success_count}/{total_count}")

    # 判断の平均スコア
    if success_count > 0:
        valid_results = [r for r in results if r['success']]
        avg_score = sum(r['result'].get('score', 0) for r in valid_results) / len(valid_results)
        print(f"平均スコア: {avg_score:.1f}/100")

    # システム情報
    print("\n" + "=" * 70)
    print("システム情報")
    print("=" * 70)
    print(f"Judge エンジンタイプ: {'量子前頭葉 (Gemma+QBNN)' if judge.use_quantum else 'フォールバック (従来型)'}")
    print(f"トークナイザー: {'有効' if judge.tokenizer else '無効'}")
    print(f"モデル: {'有効' if (judge.quantum_cortex or judge.fallback_model) else '無効'}")

    print("\n" + "=" * 70)
    print("✓ テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_judge()
