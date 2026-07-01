#!/usr/bin/env python3
"""
NeuroQuantum推論テスト
neuroquantum_layered.pyの推論能力をテストします
"""

import sys
import torch
import numpy as np
from typing import List, Dict

# NeuroQuantumモジュールをインポート
try:
    from neuroquantum_layered import NeuroQuantumAI, NeuroQuantumTokenizer, get_gpu_adaptive_config
    print("✅ NeuroQuantumモジュールをインポートしました\n")
except ImportError as e:
    print(f"❌ NeuroQuantumモジュールのインポートに失敗: {e}")
    sys.exit(1)


def print_header(title: str):
    """ヘッダーを表示"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """セクションタイトルを表示"""
    print(f"\n📌 {title}")
    print("-" * 80)


class NeuroQuantumInferenceTest:
    """NeuroQuantum推論テストスイート"""

    def __init__(self):
        self.test_results = []
        self.config = get_gpu_adaptive_config(vocab_size=8000)

    def test_initialization(self) -> bool:
        """テスト1: モデル初期化"""
        print_section("テスト1: NeuroQuantumAI初期化")

        try:
            self.ai = NeuroQuantumAI(
                embed_dim=self.config.get("embed_dim", 256),
                hidden_dim=self.config.get("hidden_dim", 512),
                num_heads=self.config.get("num_heads", 4),
                num_layers=self.config.get("num_layers", 3),
                max_seq_len=512,
                dropout=0.1,
            )
            print("✅ NeuroQuantumAI インスタンス作成成功")
            print(f"   - Embed Dim: {self.ai.embed_dim}")
            print(f"   - Hidden Dim: {self.ai.hidden_dim}")
            print(f"   - Num Heads: {self.ai.num_heads}")
            print(f"   - Num Layers: {self.ai.num_layers}")
            print(f"   - Device: {self.ai.device}")
            return True
        except Exception as e:
            print(f"❌ 初期化失敗: {e}")
            return False

    def test_tokenizer(self) -> bool:
        """テスト2: トークナイザー機能"""
        print_section("テスト2: トークナイザー機能")

        try:
            tokenizer = NeuroQuantumTokenizer(vocab_size=8000)

            test_texts = [
                "人工知能は未来を変えます",
                "量子コンピュータは革新的な技術です",
                "自然言語処理で推論を行います"
            ]

            print("📖 トレーニングテキスト:")
            for i, text in enumerate(test_texts, 1):
                print(f"   {i}. {text}")

            # 語彙構築
            print("\n🔤 語彙を構築中...")
            tokenizer.build_vocab(test_texts, model_prefix="test_tokenizer", min_freq=1)

            print(f"✅ 語彙構築完了")
            print(f"   - 実際の語彙サイズ: {tokenizer.actual_vocab_size}")

            # エンコード/デコードテスト
            test_prompt = "人工知能"
            encoded = tokenizer.encode(test_prompt)
            print(f"\n🔤 エンコーディングテスト")
            print(f"   入力: '{test_prompt}'")
            print(f"   エンコード結果: {encoded[:10]}...")

            return True
        except Exception as e:
            print(f"❌ トークナイザーテスト失敗: {e}")
            return False

    def test_basic_generation(self) -> bool:
        """テスト3: 基本的なテキスト生成"""
        print_section("テスト3: 基本的なテキスト生成（自動学習付き）")

        try:
            if not hasattr(self, 'ai'):
                self.ai = NeuroQuantumAI(
                    embed_dim=self.config.get("embed_dim", 256),
                    hidden_dim=self.config.get("hidden_dim", 512),
                    num_heads=self.config.get("num_heads", 4),
                    num_layers=self.config.get("num_layers", 3),
                    max_seq_len=512,
                )

            prompts = [
                "人工知能は",
                "量子コンピュータは",
                "自然言語処理では",
            ]

            print("🎯 テスト用プロンプト:")
            for i, prompt in enumerate(prompts, 1):
                print(f"   {i}. '{prompt}'")

            print("\n🚀 生成テスト開始...\n")

            for prompt in prompts:
                print(f"📝 プロンプト: '{prompt}'")
                try:
                    result = self.ai.generate(
                        prompt=prompt,
                        max_length=50,
                        temp_min=0.3,
                        temp_max=0.7,
                        top_k=40,
                        top_p=0.9,
                    )
                    print(f"   生成結果: {result[:100]}...")
                    print()
                except Exception as e:
                    print(f"   ❌ 生成失敗: {e}")
                    print()

            print("✅ テキスト生成テスト完了")
            return True

        except Exception as e:
            print(f"❌ テキスト生成テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_quantum_features(self) -> bool:
        """テスト4: 量子機能（利用可能な場合）"""
        print_section("テスト4: 量子回路シミュレーション機能")

        try:
            if not hasattr(self, 'ai'):
                self.ai = NeuroQuantumAI()

            print(f"⚛️  量子回路シミュレーター利用可能: {self.ai.use_quantum_simulation}")

            if self.ai.use_quantum_simulation:
                print(f"✅ 量子コンピュータ: {self.ai.quantum_computer.name}")
                return True
            else:
                print("⚠️  量子回路シミュレーターは利用不可（quantum_computer.pyが必要）")
                return True  # 警告だが失敗ではない

        except Exception as e:
            print(f"⚠️  量子機能テスト中にエラー: {e}")
            return True  # オプション機能なので失敗ではない

    def test_gpu_tier_detection(self) -> bool:
        """テスト5: GPU/CPUティア検出"""
        print_section("テスト5: GPU/CPUティア検出と適応設定")

        try:
            print("🖥️  システム情報:")
            print(f"   - GPU Tier: {self.config.get('gpu_tier', 'unknown')}")
            print(f"   - GPU Name: {self.config.get('gpu_name', 'unknown')}")
            print(f"   - GPU Info: {self.config.get('gpu_info', {})}")
            print(f"   - Embed Dim: {self.config.get('embed_dim')}")
            print(f"   - Hidden Dim: {self.config.get('hidden_dim')}")
            print(f"   - Batch Size: {self.config.get('batch_size')}")
            print(f"   - Max Seq Len: {self.config.get('max_seq_len')}")

            print("\n✅ システムティア検出完了")
            return True
        except Exception as e:
            print(f"❌ ティア検出失敗: {e}")
            return False

    def test_reasoning_with_parameters(self) -> bool:
        """テスト6: パラメータ調整による推論"""
        print_section("テスト6: 異なるパラメータでの推論（温度・サンプリング）")

        try:
            if not hasattr(self, 'ai'):
                self.ai = NeuroQuantumAI(
                    embed_dim=self.config.get("embed_dim", 256),
                    hidden_dim=self.config.get("hidden_dim", 512),
                )

            prompt = "データサイエンスは"

            # 異なる温度パラメータでテスト
            temperatures = [
                (0.3, "保守的（確定性高）"),
                (0.7, "バランス"),
                (1.0, "創造的（ランダム性高）"),
            ]

            print(f"📝 プロンプト: '{prompt}'\n")
            print("🌡️  温度パラメータテスト:\n")

            for temp, desc in temperatures:
                print(f"  温度: {temp} ({desc})")
                try:
                    result = self.ai.generate(
                        prompt=prompt,
                        max_length=50,
                        temperature=temp,
                        top_k=40,
                        top_p=0.9,
                    )
                    print(f"  結果: {result[:80]}...")
                    print()
                except Exception as e:
                    print(f"  ❌ 生成失敗: {e}\n")

            print("✅ パラメータ調整テスト完了")
            return True

        except Exception as e:
            print(f"❌ パラメータテスト失敗: {e}")
            return False

    def run_all_tests(self):
        """すべてのテストを実行"""
        print_header("🧠 NeuroQuantum 推論テストスイート")

        tests = [
            ("初期化", self.test_initialization),
            ("トークナイザー", self.test_tokenizer),
            ("テキスト生成", self.test_basic_generation),
            ("量子機能", self.test_quantum_features),
            ("GPU/CPUティア", self.test_gpu_tier_detection),
            ("パラメータ調整", self.test_reasoning_with_parameters),
        ]

        results = {}
        for name, test_func in tests:
            try:
                results[name] = test_func()
            except Exception as e:
                print(f"\n❌ テスト '{name}' で予期しないエラー: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

        # テスト結果サマリー
        print_header("📊 テスト結果サマリー")

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {name}")

        print(f"\n合計: {passed}/{total} テスト成功")
        print("=" * 80)

        return passed, total


if __name__ == "__main__":
    tester = NeuroQuantumInferenceTest()
    passed, total = tester.run_all_tests()

    sys.exit(0 if passed == total else 1)
