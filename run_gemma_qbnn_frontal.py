#!/usr/bin/env python3
"""
Gemma + QBNN as Frontal Cortex - 実行スクリプト
実際のGemmaモデル + QBNN量子層を使って前頭葉として動作させる
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    print("✓ PyTorch 読み込み成功")
except ImportError as e:
    print(f"✗ PyTorch エラー: {e}")
    print("  pip install torch を実行してください")
    sys.exit(1)

try:
    from gemma_qbnn import create_gemma_qbnn_model
    from gemma_qbnn_prefrontal_cortex import (
        GemmaQBNNPrefrontalCortex,
        JudgmentConfig,
        create_prefrontal_cortex
    )
    print("✓ Gemma+QBNN モデル読み込み成功")
except Exception as e:
    print(f"✗ モデル読み込みエラー: {e}")
    sys.exit(1)


class GemmaQBNNFrontalDemo:
    """実際のGemma+QBNN前頭葉デモンストレーション"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n【システム初期化】")
        print(f"  デバイス: {self.device}")
        print(f"  CUDA利用可能: {torch.cuda.is_available()}")

        # 前頭葉システムを作成
        print(f"\n  Gemma+QBNN 前頭葉を初期化中...")
        try:
            # 小さいモデル設定で高速実行
            config = JudgmentConfig(
                vocab_size=8000,
                embed_dim=256,
                hidden_dim=512,
                num_heads=4,
                num_layers=2,
                max_seq_len=256,
                entangle_strength=0.7,
                quantum_weight=0.6
            )

            self.cortex = create_prefrontal_cortex(config=config, device=self.device)
            self.cortex.eval()

            # パラメータ数を表示
            total_params = sum(p.numel() for p in self.cortex.parameters())
            print(f"  ✓ 初期化完了")
            print(f"  モデルパラメータ: {total_params:,}")

        except Exception as e:
            print(f"  ✗ 初期化失敗: {e}")
            raise

    def run_judgment(self, context: str, judgment_request: str, strict_mode: bool = False) -> dict:
        """判断を実行"""
        print(f"\n【判断実行】")
        print(f"  質問: {judgment_request}")
        print(f"  コンテキスト: {context[:80]}...")

        try:
            with torch.no_grad():
                result = self.cortex.judge({
                    "context": context,
                    "judgment_request": judgment_request,
                    "strict_mode": strict_mode
                })

            print(f"\n【判断結果】")
            print(f"  ✓ 決定: {result['decision']}")
            print(f"  スコア: {result['score']}/100")
            print(f"  信頼度: {result['confidence']}")
            print(f"  根拠: {result['reasoning'][:100]}...")

            if "quantum_info" in result:
                qi = result["quantum_info"]
                print(f"  量子情報:")
                print(f"    - Yes確率: {qi.get('yes_probability', 0):.1%}")
                print(f"    - 量子重み: {qi.get('quantum_weight', 0):.1f}")

            return result

        except Exception as e:
            print(f"  ✗ 判断実行エラー: {e}")
            raise

    def demo_1_security_judgment(self):
        """デモ1: セキュリティ判断（本番デプロイ）"""
        print("\n" + "="*70)
        print("デモ 1: セキュリティ判断 - 本番デプロイの可否判定")
        print("="*70)

        result = self.run_judgment(
            context="""
            本番環境へのコード配置を検討中。
            - コードレビュー: 完了
            - ユニットテスト: 成功 (98% カバレッジ)
            - 統合テスト: 成功
            - セキュリティスキャン: 問題なし
            - ロールバック計画: あり
            - モニタリング: 構成済み
            """,
            judgment_request="このコードは本番環境にデプロイ可能か？",
            strict_mode=True
        )

        return result

    def demo_2_risk_assessment(self):
        """デモ2: リスク評価（大規模スキーマ変更）"""
        print("\n" + "="*70)
        print("デモ 2: リスク評価 - データベーススキーマ変更")
        print("="*70)

        result = self.run_judgment(
            context="""
            本番PostgreSQLの大規模スキーマ変更。
            - 対象レコード数: 1000万件
            - 変更内容: 新カラム追加（デフォルト値あり）
            - ダウンタイム予想: 2-5分
            - バックアップ: フルバックアップ + WALアーカイブ
            - テスト: 本番と同じスケールで検証済み
            - ロールバック: 自動スクリプト用意
            - 実行時刻: オフピーク時間帯
            """,
            judgment_request="このスキーマ変更を本番環境で実行しても安全か？",
            strict_mode=True
        )

        return result

    def demo_3_ethical_judgment(self):
        """デモ3: 倫理的判断（プライバシー）"""
        print("\n" + "="*70)
        print("デモ 3: 倫理的判断 - ユーザープライバシー評価")
        print("="*70)

        result = self.run_judgment(
            context="""
            ユーザーのデータ処理に関する提案：
            - 処理内容: クリックストリーム分析で個人の好みを推測
            - ユーザー同意: 利用規約の改定が必要（未実施）
            - データ使用: 第三者と共有予定なし
            - 透明性: ユーザーに通知する計画がない
            - 削除権: ユーザーが削除を要求できるか不明確
            - リスク: GDPRやCCPA違反の可能性あり
            """,
            judgment_request="このデータ処理はプライバシー的に適切か？",
            strict_mode=True
        )

        return result

    def demo_4_decision_making(self):
        """デモ4: 意思決定（アーキテクチャ選択）"""
        print("\n" + "="*70)
        print("デモ 4: 意思決定 - マイクロサービスへの移行")
        print("="*70)

        result = self.run_judgment(
            context="""
            モノリシックアーキテクチャからマイクロサービスへの移行を検討。
            メリット:
            - 独立したスケーリング
            - デプロイメント頻度向上
            - 技術スタック選択の自由度

            デメリット:
            - 分散システムの複雑性
            - 運用コスト増加（初期段階）
            - チームスキル要件が高い

            現在の状況:
            - チームサイズ: 8名
            - 現在の課題: デプロイ頻度 (2週間に1回)
            - インフラ成熟度: Kubernetes経験あり
            """,
            judgment_request="マイクロサービス化に移行すべきか？",
            strict_mode=False
        )

        return result

    def run_all_demos(self):
        """すべてのデモを実行"""
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*68 + "║")
        print("║" + "Gemma+QBNN as Frontal Cortex - 実行デモ".center(68) + "║")
        print("║" + " "*68 + "║")
        print("╚" + "="*68 + "╝")

        results = []

        # 各デモを実行
        try:
            results.append(("セキュリティ判断", self.demo_1_security_judgment()))
        except Exception as e:
            print(f"✗ デモ1 失敗: {e}")

        try:
            results.append(("リスク評価", self.demo_2_risk_assessment()))
        except Exception as e:
            print(f"✗ デモ2 失敗: {e}")

        try:
            results.append(("倫理的判断", self.demo_3_ethical_judgment()))
        except Exception as e:
            print(f"✗ デモ3 失敗: {e}")

        try:
            results.append(("意思決定", self.demo_4_decision_making()))
        except Exception as e:
            print(f"✗ デモ4 失敗: {e}")

        # サマリー
        self._print_summary(results)

    def _print_summary(self, results):
        """結果サマリーを表示"""
        print("\n" + "="*70)
        print("実行サマリー")
        print("="*70)

        for name, result in results:
            decision = result.get('decision', 'N/A')
            score = result.get('score', 0)
            confidence = result.get('confidence', 'N/A')
            print(f"  {name:15} → {decision} (スコア: {score}/100, 信頼度: {confidence})")

        # 統計
        if results:
            avg_score = sum(r[1].get('score', 0) for r in results) / len(results)
            print(f"\n  平均スコア: {avg_score:.1f}/100")
            print(f"  判断実行数: {len(results)}")

        print("\n" + "="*70)
        print("✓ Gemma+QBNN 前頭葉システム実行完了")
        print("="*70)


def main():
    """メイン実行"""
    try:
        demo = GemmaQBNNFrontalDemo()
        demo.run_all_demos()

        print("""
次のステップ:
  1. より大規模なモデルで実行: embed_dim=768, num_layers=12
  2. MCPサーバーとして展開: python frontal_engine_mcp_server.py
  3. Claude統合: from claude_prefrontal_integration import claude_prefrontal_cortex
        """)

    except KeyboardInterrupt:
        print("\n\n✗ ユーザーによる中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
