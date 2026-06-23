#!/usr/bin/env python3
"""
Gemma+QBNN Frontal Cortex - シンプル推論
普通の判断タスクを実行
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    from gemma_qbnn_prefrontal_cortex import create_prefrontal_cortex, JudgmentConfig
except ImportError as e:
    print(f"✗ エラー: {e}")
    sys.exit(1)


def simple_inference():
    """シンプルな推論を実行"""

    print("\n" + "="*70)
    print("Gemma+QBNN Frontal - シンプル推論モード")
    print("="*70)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nデバイス: {device}")

    # 前頭葉初期化
    print("前頭葉システム初期化中...")
    config = JudgmentConfig(
        vocab_size=8000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=4,
        num_layers=2,
        entangle_strength=0.7,
        quantum_weight=0.6
    )
    cortex = create_prefrontal_cortex(config=config, device=device)
    cortex.eval()
    print("✓ 初期化完了\n")

    # シンプルな判断タスク
    tasks = [
        {
            "title": "1. 新しいスマートフォンを買うべきか",
            "context": "現在使っているスマートフォンは3年前のモデル。バッテリーが1日もたない。最新モデルは高い。",
            "question": "新しいスマートフォンを買うべきか？"
        },
        {
            "title": "2. 転職すべきか",
            "context": "現在の給与は年600万円。新しい会社の提示は700万円。ただし転職先は起業1年の小さな会社。",
            "question": "転職すべきか？"
        },
        {
            "title": "3. 大学院に進学すべきか",
            "context": "学部卒業後の給与は平均450万円。大学院卒業後は550万円。進学に2年かかる。学費は200万円。",
            "question": "大学院に進学すべきか？"
        },
        {
            "title": "4. 結婚式は盛大にするべきか",
            "context": "資金は十分にある。結婚相手も盛大な式を希望。ただし親戚が多く、準備が大変。",
            "question": "盛大な結婚式をするべきか？"
        },
        {
            "title": "5. 引っ越すべきか",
            "context": "現在の家賃は8万円で十分。新しい物件は12万円だが駅に近い。通勤時間が20分短縮。",
            "question": "新しい物件に引っ越すべきか？"
        },
        {
            "title": "6. ジムに入会すべきか",
            "context": "月額1万円。月2回しか行かないペースだから、1回5000円の計算。でも運動は必要。",
            "question": "ジムに入会すべきか？"
        },
        {
            "title": "7. 今日は会社を休むべきか",
            "context": "朝起きたら少し風邪気味。熱は37.2度。重要な会議が3つある。代わりがいない。",
            "question": "会社を休むべきか？"
        },
        {
            "title": "8. この投資話に乗るべきか",
            "context": "知人から『確実に年15%のリターンが得られる投資話』と言われた。詳細は説明されていない。",
            "question": "この投資話に乗るべきか？"
        }
    ]

    results = []

    # 各タスクで推論実行
    for i, task in enumerate(tasks, 1):
        print(f"\n{'─'*70}")
        print(task['title'])
        print(f"{'─'*70}")
        print(f"コンテキスト: {task['context']}")
        print(f"判断内容: {task['question']}")

        try:
            with torch.no_grad():
                result = cortex.judge({
                    "context": task['context'],
                    "judgment_request": task['question'],
                    "strict_mode": False
                })

            # 結果を表示
            decision = result['decision']
            score = result['score']
            confidence = result['confidence']

            print(f"\n結果:")
            print(f"  判断: {decision}")
            print(f"  スコア: {score}/100")
            print(f"  信頼度: {confidence}")

            results.append((task['title'], decision, score, confidence))

        except Exception as e:
            print(f"✗ エラー: {e}")

    # サマリー表示
    print("\n" + "="*70)
    print("【推論結果サマリー】")
    print("="*70)

    yes_count = sum(1 for r in results if r[1] == "Yes")
    no_count = sum(1 for r in results if r[1] == "No")
    avg_score = sum(r[2] for r in results) / len(results) if results else 0

    print(f"\n【判断一覧】")
    for title, decision, score, confidence in results:
        # タイトルを短縮
        short_title = title.split('. ')[1][:20]
        print(f"  {short_title:22} → {decision:3} (スコア: {score:3}/100)")

    print(f"\n【統計】")
    print(f"  実行数: {len(results)}")
    print(f"  Yes判定: {yes_count}件")
    print(f"  No判定: {no_count}件")
    print(f"  平均スコア: {avg_score:.1f}/100")

    print("\n" + "="*70)
    print("✓ 推論完了")
    print("="*70 + "\n")


if __name__ == "__main__":
    simple_inference()
