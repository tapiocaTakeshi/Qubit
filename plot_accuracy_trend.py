#!/usr/bin/env python3
"""
Judge精度トレンドグラフ
テスト回数と精度の関係を可視化
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# テストデータ
test_data = {
    'cases': [100, 1000],
    'overall': [69.3, 87.2],
    'yes': [95.0, 95.91],
    'no': [35.0, 59.54],
    'boundary': [85.0, 75.42]
}

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# グラフ1: 総合精度トレンド
ax1.plot(test_data['cases'], test_data['overall'], 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Overall')
ax1.fill_between(test_data['cases'], test_data['overall'], alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Test Cases (回)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Accuracy Trend', fontsize=13, fontweight='bold')
ax1.set_ylim(60, 100)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# データポイントにラベル付け
for i, (cases, acc) in enumerate(zip(test_data['cases'], test_data['overall'])):
    ax1.annotate(f'{acc:.1f}%', xy=(cases, acc), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# グラフ2: テストタイプ別精度
x_pos = range(len(test_data['cases']))
width = 0.25

bars1 = ax2.bar([p - width for p in x_pos], test_data['yes'], width, label='Yes Cases', color='#06A77D')
bars2 = ax2.bar(x_pos, test_data['no'], width, label='No Cases', color='#D62828')
bars3 = ax2.bar([p + width for p in x_pos], test_data['boundary'], width, label='Boundary Cases', color='#F77F00')

ax2.set_xlabel('Test Cases (回)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy by Test Type', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(test_data['cases'])
ax2.set_ylim(0, 105)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# バーにラベル付け
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_trend.png', dpi=150, bbox_inches='tight')
print("\n✅ グラフを保存しました: accuracy_trend.png")
print("\n📊 精度トレンド分析")
print("=" * 70)
print(f"【100-knock → 1000-random】")
print(f"  総合精度: 69.3% → 87.2% (+17.9%)")
print(f"  Yes精度: 95.0% → 95.91% (+0.91%)")
print(f"  No精度: 35.0% → 59.54% (+24.54%) ⬆️ 大幅改善")
print(f"  Boundary精度: 85.0% → 75.42% (-9.58%)")
print("=" * 70)
print("\n結論:")
print("✓ 1000 random test で大規模データでの安定性を確認")
print("✓ No判定精度が大幅に改善（35% → 59.54%）")
print("✓ 総合精度 87.2% で本番環境対応レベル")
print()

plt.show()
