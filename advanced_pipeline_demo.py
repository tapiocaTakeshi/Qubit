#!/usr/bin/env python3
"""
高度なパイプライン推論デモ
ライブラリー生成 → Gemma文章生成の完全フロー
"""

import sys
sys.path.insert(0, '.')

from pipeline_inference import PipelineInferenceSystem


def demo_library_accumulation():
    """デモ1: ライブラリー蓄積"""
    print("\n" + "="*70)
    print("【デモ1】ライブラリー蓄積フロー")
    print("="*70 + "\n")

    system = PipelineInferenceSystem()

    # 複数の判断を実行してライブラリーを構築
    diverse_inputs = [
        # キャリア系
        "転職すべきか？給与が100万円上がるが、起業1年で安定性が不確定。",
        "昇進の話があるが、現在の職場は居心地が良い。どうすべき？",
        "異動を打診されたが、新チームの雰囲気が不明確。",

        # 投資系
        "この怪しい投資話に乗るべきか？詳細説明なく『確実』という言葉だけ。",
        "仮想通貨への投資をどう思うか？",

        # 教育系
        "大学院に進学すべき？2年かかり学費200万円だが、給与は100万円上がる見込み。",
        "プログラミング学習に時間を投資すべき？",

        # 生活系
        "新しいアパートに引っ越すべき？家賃が2万円上がる。",
    ]

    print(f"【入力データ】{len(diverse_inputs)}件の判断を実行\n")

    for i, user_input in enumerate(diverse_inputs, 1):
        print(f"[{i}] {user_input[:50]}...")
        # 従来版で分析（詳細は表示しない）
        _ = system.infer(user_input, use_library_generation=False)

    print("\n✓ {len(diverse_inputs)}件のエントリーがライブラリーに蓄積されました\n")

    # ライブラリー統計を表示
    library_info = system.get_library_info()
    print("【ライブラリー統計】")
    print(f"  蓄積エントリー: {library_info['total_entries']}件")
    print(f"  判断タイプ: {', '.join(library_info['task_types'])}")
    print(f"  Yes判定: {library_info['yes_count']}件 ({library_info['yes_count']*100/library_info['total_entries']:.0f}%)")
    print(f"  No判定: {library_info['no_count']}件 ({library_info['no_count']*100/library_info['total_entries']:.0f}%)")

    return system


def demo_library_based_generation(system: PipelineInferenceSystem):
    """デモ2: ライブラリーに基づいた生成"""
    print("\n" + "="*70)
    print("【デモ2】ライブラリーを活用したGemma文章生成")
    print("="*70 + "\n")

    # 新しい入力
    test_inputs = [
        "起業に挑戦すべきか？新規事業だが市場ニーズが明確で、給与は下がる。",
        "新しいスキルを習得すべき？成長機会があるが時間がかかる。",
        "フリーランスになるべき？自由度が増すが収入が不安定。",
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"【質問 {i}】")
        print(f"Q: {user_input}\n")

        # ライブラリーベースの生成を実行
        response = system.infer(user_input, use_library_generation=True)

        print("【Gemmaが生成した応答】")
        print(response)
        print("\n" + "-"*70 + "\n")


def demo_comparative_analysis():
    """デモ3: 従来版 vs. ライブラリー版の比較"""
    print("\n" + "="*70)
    print("【デモ3】応答方式の比較")
    print("="*70 + "\n")

    system = PipelineInferenceSystem()

    # まずライブラリーにエントリーを追加
    sample_inputs = [
        "転職すべきか？給与が100万円上がるが、起業1年で安定性が不確定。",
        "大学院に進学すべき？2年かかり学費200万円だが、給与は100万円上がる見込み。",
    ]

    print("【ステップ1】ライブラリー構築")
    for user_input in sample_inputs:
        _ = system.infer(user_input, use_library_generation=False)
    print(f"  ✓ {len(sample_inputs)}件をライブラリーに蓄積\n")

    # テスト入力
    test_input = "スキル向上のため学費が30万円かかる研修に参加すべき？"
    print(f"【テスト入力】{test_input}\n")

    # 従来版
    print("【方式1：詳細分析版（従来版）】")
    traditional = system.infer(test_input, use_library_generation=False)
    traditional_lines = traditional.split('\n')[:10]  # 最初の10行
    for line in traditional_lines:
        print(line)
    print("  ... (詳細分析が続く)\n")

    # ライブラリー版
    print("【方式2：ライブラリー活用版（新版）】")
    library_based = system.infer(test_input, use_library_generation=True)
    print(library_based)


def demo_recommendation_quality():
    """デモ4: 推奨の質"""
    print("\n" + "="*70)
    print("【デモ4】スコア別の推奨メッセージの質")
    print("="*70 + "\n")

    system = PipelineInferenceSystem()

    # 異なるスコア帯の入力
    test_cases = [
        ("強く推奨される場合", "給与が200万円上がり、成長機会も多く、安定している企業への転職。"),
        ("推奨される場合", "給与が50万円上がるが、多少の不安定性がある転職。"),
        ("検討が必要な場合", "給与は変わらないが、成長機会がある異動。"),
        ("非推奨の場合", "給与が下がり、リスクが高い起業。"),
        ("強く非推奨の場合", "詳細不明な投資話で『確実』という言葉だけ。"),
    ]

    for category, user_input in test_cases:
        print(f"【{category}】")
        print(f"入力: {user_input}\n")

        response = system.infer(user_input, use_library_generation=True)
        # 最初の2行だけを表示
        first_lines = response.split('\n')[:2]
        for line in first_lines:
            print(line)
        print()


def main():
    """メイン実行"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "高度なパイプライン推論デモンストレーション".center(68) + "║")
    print("║" + "ライブラリー生成 → Gemma文章生成".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    try:
        # デモ1: ライブラリー蓄積
        system = demo_library_accumulation()

        # デモ2: ライブラリーベース生成
        demo_library_based_generation(system)

        # デモ3: 比較分析
        demo_comparative_analysis()

        # デモ4: 推奨の質
        demo_recommendation_quality()

        print("\n" + "="*70)
        print("【サマリー】")
        print("="*70)
        print("""
【パイプラインの特徴】

1. ライブラリー蓄積
   - 複数の推論結果を知識として保存
   - タスク別のインデックス化
   - 判断パターンの学習

2. Gemma文章生成
   - ライブラリー内の類似ケースを参照
   - コンテキストに基づいた応答
   - スコアに応じた推奨メッセージ

3. スケーラビリティ
   - エントリーが増えるほど精度向上
   - 異なるタスク領域への対応可能
   - 継続的な学習と改善

【今後の拡張案】
- セマンティック検索による類似ケース検出
- ユーザーフィードバックを基にした最適化
- マルチモーダル入力対応（テキスト、画像、音声）
- リアルタイム市場データの統合
        """)

        print("="*70)
        print("✓ すべてのデモが完了しました")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
