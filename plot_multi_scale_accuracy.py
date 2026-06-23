#!/usr/bin/env python3
"""
複数スケール精度測定結果のグラフ化
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 結果を読み込み
with open("accuracy_results.json", "r") as f:
    results = json.load(f)

# データを整理
scales = sorted([int(k) for k in results.keys()])
overall_accuracy = [results[str(s)]["accuracy"] for s in scales]
yes_accuracy = [results[str(s)]["type_accuracy"].get("yes", 0) for s in scales]
no_accuracy = [results[str(s)]["type_accuracy"].get("no", 0) for s in scales]
boundary_accuracy = [results[str(s)]["type_accuracy"].get("boundary", 0) for s in scales]

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# グラフ1: 総合精度トレンド
ax1.plot(scales, overall_accuracy, 'o-', linewidth=2.5, markersize=10,
         color='#2E86AB', label='Overall Accuracy', zorder=3)
ax1.fill_between(scales, overall_accuracy, alpha=0.2, color='#2E86AB')

ax1.set_xlabel('Test Cases (回)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Accuracy vs Test Scale', fontsize=13, fontweight='bold')
ax1.set_ylim(60, 100)
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=11, loc='lower right')

# データポイントにラベル付け
for scale, acc in zip(scales, overall_accuracy):
    ax1.annotate(f'{acc:.1f}%', xy=(scale, acc), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# グラフ2: テストタイプ別精度トレンド
ax2.plot(scales, yes_accuracy, 'o-', linewidth=2, markersize=8,
         color='#06A77D', label='Yes Cases', zorder=3)
ax2.plot(scales, no_accuracy, 's-', linewidth=2, markersize=8,
         color='#D62828', label='No Cases', zorder=3)
ax2.plot(scales, boundary_accuracy, '^-', linewidth=2, markersize=8,
         color='#F77F00', label='Boundary Cases', zorder=3)

ax2.set_xlabel('Test Cases (回)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy by Test Type vs Scale', fontsize=13, fontweight='bold')
ax2.set_ylim(30, 105)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('multi_scale_accuracy.png', dpi=150, bbox_inches='tight')
print("\n✅ グラフを保存しました: multi_scale_accuracy.png")

# 統計分析
print("\n" + "=" * 70)
print("【精度変化分析】")
print("=" * 70)

print(f"\n【総合精度の変化】")
print(f"  10回:   {overall_accuracy[0]:6.2f}%")
print(f"  50回:   {overall_accuracy[1]:6.2f}% ({overall_accuracy[1]-overall_accuracy[0]:+.2f}%)")
print(f"  100回:  {overall_accuracy[2]:6.2f}% ({overall_accuracy[2]-overall_accuracy[0]:+.2f}%)")
print(f"  200回:  {overall_accuracy[3]:6.2f}% ({overall_accuracy[3]-overall_accuracy[0]:+.2f}%)")
print(f"  500回:  {overall_accuracy[4]:6.2f}% ({overall_accuracy[4]-overall_accuracy[0]:+.2f}%)")
print(f"  1000回: {overall_accuracy[5]:6.2f}% ({overall_accuracy[5]-overall_accuracy[0]:+.2f}%)")

print(f"\n【テストタイプ別の変化】")
print(f"\nYes Cases (肯定判定):")
for scale, acc in zip(scales, yes_accuracy):
    print(f"  {scale:5d}回: {acc:6.2f}%")

print(f"\nNo Cases (否定判定):")
for scale, acc in zip(scales, no_accuracy):
    print(f"  {scale:5d}回: {acc:6.2f}%")

print(f"\nBoundary Cases (境界判定):")
for scale, acc in zip(scales, boundary_accuracy):
    print(f"  {scale:5d}回: {acc:6.2f}%")

print("\n" + "=" * 70)
print("【結論】")
print("=" * 70)

# 安定性判定
yes_std = np.std(yes_accuracy)
no_std = np.std(no_accuracy)
boundary_std = np.std(boundary_accuracy)

print(f"\n安定性 (標準偏差が小さいほど安定):")
print(f"  Yes Cases: σ = {yes_std:.2f} {'✓ 安定' if yes_std < 5 else '⚠️ 変動'}")
print(f"  No Cases:  σ = {no_std:.2f} {'✓ 安定' if no_std < 10 else '⚠️ 変動'}")
print(f"  Boundary:  σ = {boundary_std:.2f} {'✓ 安定' if boundary_std < 10 else '⚠️ 変動'}")

avg_accuracy = np.mean(overall_accuracy)
print(f"\n全スケールでの平均精度: {avg_accuracy:.2f}%")

if avg_accuracy >= 85:
    print("評価: ⭐⭐⭐⭐⭐ (本番環境対応レベル)")
elif avg_accuracy >= 80:
    print("評価: ⭐⭐⭐⭐ (実用的レベル)")
else:
    print("評価: ⭐⭐⭐ (改善の余地あり)")

print("\n" + "=" * 70)
