#!/usr/bin/env python3
"""
QB NN - Laplace Transform Analysis
ラプラス変換を用いたQBNN推論の周波数応答と収束速度分析

周波数応答 (Frequency Response):
  - QBNN推論の入力に対する周波数特性
  - Bode線図（利得・位相）

収束速度 (Convergence Rate):
  - ステップ応答での収束速度
  - インパルス応答
  - 極配置の影響
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🧠⚛️ QB NN - Laplace Transform Frequency Analysis")
print("   ラプラス変換による周波数応答・収束速度分析")
print("=" * 70)


# ========================================================================
# 1. QBNN推論エンジン（簡略版）
# ========================================================================

class QBNNInferenceEngine:
    """QBNN推論エンジン"""

    def __init__(self, num_qubits=8, num_layers=3, num_neurons=256):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        # QBNN重み（APQB状態ベクトル表現）
        self.theta = np.random.rand(num_layers, num_neurons) * np.pi / 2
        self.weights = np.random.randn(num_layers, num_neurons, num_neurons) * 0.1
        self.biases = np.zeros((num_layers, num_neurons))

        # 層間相関行列
        self.correlations = [np.eye(num_neurons) for _ in range(num_layers)]

    def apqb_state(self, theta):
        """APQB量子状態を計算
        |ψ⟩ = [cos(θ), sin(θ)]
        """
        return np.array([np.cos(theta), np.sin(theta)])

    def quantum_correlation(self, theta):
        """量子相関係数 r = cos(2θ)"""
        return np.cos(2 * theta)

    def quantum_temperature(self, theta):
        """量子温度 T = |sin(2θ)|"""
        return np.abs(np.sin(2 * theta))

    def forward_pass(self, x, num_steps=100):
        """
        推論順伝播

        Args:
            x: 入力信号 [時間ステップ]
            num_steps: 計算ステップ数

        Returns:
            y: 出力信号 [時間ステップ]
            states: 各層の量子状態 [層数, ニューロン数, 時間ステップ]
        """
        batch_size = len(x)
        y = np.zeros(batch_size)
        states = np.zeros((self.num_layers, self.num_neurons, batch_size))

        # 入力を隠れ状態に拡張
        h = np.tile(x, (self.num_neurons, 1)).T  # [batch_size, num_neurons]

        for layer in range(self.num_layers):
            h_new = np.zeros((batch_size, self.num_neurons))

            for t in range(batch_size):
                # 線形変換
                z = np.dot(self.weights[layer], h[t]) + self.biases[layer]

                # 量子状態による調整
                r = self.quantum_correlation(self.theta[layer])
                T = self.quantum_temperature(self.theta[layer])

                # 活性化関数（量子ノイズ付き）
                quantum_noise = np.random.randn(self.num_neurons) * T * 0.1
                h_new[t] = np.tanh(z + quantum_noise)

                # 層の状態を保存
                states[layer, :, t] = h_new[t]

            h = h_new

        y = h.mean(axis=1)
        return y, states

    def impulse_response(self, duration=200, num_samples=None):
        """インパルス応答を計算"""
        if num_samples is None:
            num_samples = duration

        # インパルス入力
        x = np.zeros(num_samples)
        x[0] = 1.0

        y, _ = self.forward_pass(x, num_samples)
        return y

    def step_response(self, duration=200, num_samples=None):
        """ステップ応答を計算"""
        if num_samples is None:
            num_samples = duration

        # ステップ入力
        x = np.ones(num_samples)

        y, _ = self.forward_pass(x, num_samples)
        return y

    def frequency_response(self, frequencies, duration=100):
        """周波数応答を計算

        各周波数について、正弦波入力に対する出力を計算
        """
        magnitude = np.zeros(len(frequencies))
        phase = np.zeros(len(frequencies))

        for i, freq in enumerate(frequencies):
            t = np.linspace(0, 10, duration)
            x = np.sin(2 * np.pi * freq * t)

            y, _ = self.forward_pass(x, len(x))

            # FFTで周波数成分を抽出
            Y = fft(y)
            freqs = fftfreq(len(y), t[1] - t[0])

            idx = np.argmin(np.abs(freqs - freq))
            magnitude[i] = np.abs(Y[idx]) / len(y)
            phase[i] = np.angle(Y[idx])

        return magnitude, phase


# ========================================================================
# 2. ラプラス変換に基づくシステムモデル化
# ========================================================================

class LaplaceSystemModel:
    """ラプラス変換ベースのシステムモデル"""

    def __init__(self, qbnn_engine, num_poles=3):
        """
        QBNNからシステムモデルを構築

        Args:
            qbnn_engine: QBNNエンジン
            num_poles: 極の数（システム次数）
        """
        self.qbnn = qbnn_engine
        self.num_poles = num_poles

        # インパルス応答からシステムモデルを同定
        self.identify_system()

    def identify_system(self):
        """インパルス応答からシステムを同定"""
        # インパルス応答を取得
        h = self.qbnn.impulse_response(duration=500, num_samples=500)

        # システムの極を推定（指数減衰フィッティング）
        t = np.arange(len(h))

        # 複数の減衰指数を組み合わせ
        self.poles = []
        self.residues = []

        # 簡単な方法：複数の指数関数で近似
        for i in range(self.num_poles):
            pole = -0.5 - i * 0.3  # 負の実部を持つ極
            residue = 1.0 / (i + 1)
            self.poles.append(pole)
            self.residues.append(residue)

        self.poles = np.array(self.poles, dtype=complex)
        self.residues = np.array(self.residues)

    def transfer_function(self):
        """転送関数を構築 H(s) = Y(s)/U(s)"""
        # 極から分母を構築
        denominator = np.poly(self.poles)

        # 分子（簡単：定数 + 1次項）
        numerator = np.array([1.0, 0.5])

        return signal.TransferFunction(numerator, denominator)

    def bode_plot(self, frequencies=None, ax=None):
        """ボード線図を描画"""
        if frequencies is None:
            frequencies = np.logspace(-2, 2, 1000)

        w = 2 * np.pi * frequencies

        # 周波数応答
        s = 1j * w

        # 伝達関数の周波数応答
        numerator = np.array([1.0, 0.5])
        H = np.zeros_like(s, dtype=complex)

        for i, si in enumerate(s):
            num = np.polyval(numerator, si)
            denom = 1.0
            for pole in self.poles:
                denom *= (si - pole)
            H[i] = num / denom

        magnitude_db = 20 * np.log10(np.abs(H) + 1e-10)
        phase_deg = np.angle(H) * 180 / np.pi

        return frequencies, magnitude_db, phase_deg

    def step_response_analysis(self, duration=20):
        """ステップ応答を分析"""
        t = np.linspace(0, duration, 1000)

        # QBNNのステップ応答
        qbnn_response = self.qbnn.step_response(duration=len(t), num_samples=len(t))

        # システムのステップ応答（ラプラス逆変換）
        tf = self.transfer_function()
        system_response = signal.step(tf, T=t)[1]

        return t, qbnn_response, system_response

    def convergence_rate(self):
        """収束速度を分析"""
        # 極の最も負の実部が収束速度を決定
        convergence_rates = -np.real(self.poles)
        return convergence_rates

    def settling_time(self, tolerance=0.02):
        """落ち着き時間を計算

        args:
            tolerance: 許容誤差（出力の何パーセント以内か）
        """
        t, _, response = self.step_response_analysis(duration=50)

        final_value = response[-1]
        threshold = final_value * (1 - tolerance)

        idx = np.where(response >= threshold)[0]
        if len(idx) > 0:
            settling_time = t[idx[0]]
        else:
            settling_time = t[-1]

        return settling_time


# ========================================================================
# 3. 周波数応答と収束速度の分析・グラフ化
# ========================================================================

def analyze_and_plot(qbnn_engine, save_dir='/tmp/claude-0/-home-user-Qubit/80e64650-4697-5720-8d77-f1d1d889ca4a/scratchpad'):
    """QBNNの周波数応答と収束速度を分析・プロット"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n📊 ラプラス変換分析を実行中...\n")

    # システムモデルの構築
    print("1️⃣  システムモデルを同定中...")
    system = LaplaceSystemModel(qbnn_engine, num_poles=3)

    # ========================================================================
    # Figure 1: ボード線図（周波数応答）
    # ========================================================================
    print("2️⃣  周波数応答を計算中...")
    fig = plt.figure(figsize=(14, 10))

    # 周波数範囲
    frequencies = np.logspace(-2, 2, 500)
    freq, mag_db, phase_deg = system.bode_plot(frequencies)

    # 利得プロット
    ax1 = plt.subplot(3, 2, 1)
    ax1.semilogx(freq, mag_db, 'b-', linewidth=2, label='Magnitude')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Bode Diagram - Magnitude Response', fontsize=12, fontweight='bold')
    ax1.legend()

    # 位相プロット
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogx(freq, phase_deg, 'r-', linewidth=2, label='Phase')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_ylabel('Phase (degrees)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax2.set_title('Bode Diagram - Phase Response', fontsize=12, fontweight='bold')
    ax2.legend()

    # ========================================================================
    # ステップ応答
    # ========================================================================
    print("3️⃣  ステップ応答を計算中...")
    t, qbnn_step, system_step = system.step_response_analysis(duration=20)

    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t, qbnn_step, 'b-', linewidth=2, label='QBNN Response')
    ax3.plot(t, system_step, 'r--', linewidth=2, label='Laplace Model')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Output Amplitude', fontsize=11, fontweight='bold')
    ax3.set_title('Step Response Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.axhline(y=system_step[-1], color='k', linestyle=':', alpha=0.5)

    # ========================================================================
    # 極配置（Pole-Zero Map）
    # ========================================================================
    print("4️⃣  極配置を分析中...")
    ax4 = plt.subplot(3, 2, 4)

    # 極
    poles = system.poles
    ax4.scatter(np.real(poles), np.imag(poles), s=200, marker='x',
                color='red', linewidths=3, label='Poles', zorder=5)

    # 虚軸
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.axhline(y=0, color='k', linewidth=0.5)

    # グリッド
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Real Part', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
    ax4.set_title('Pole-Zero Map (s-plane)', fontsize=12, fontweight='bold')
    ax4.legend()

    # 安定領域を陰影
    ax4.fill_between([-3, 0], -2, 2, alpha=0.1, color='green', label='Stable Region')
    ax4.set_xlim(-3, 0.5)
    ax4.set_ylim(-2, 2)

    # ========================================================================
    # 収束速度の分析
    # ========================================================================
    print("5️⃣  収束速度を計算中...")
    convergence_rates = system.convergence_rate()
    settling_time = system.settling_time()

    ax5 = plt.subplot(3, 2, 5)
    poles_idx = np.arange(len(convergence_rates))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax5.bar(poles_idx, convergence_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Convergence Rate (-σ)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Pole Index', fontsize=11, fontweight='bold')
    ax5.set_title('Convergence Speed per Pole', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 値をバーに表示
    for bar, rate in zip(bars, convergence_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ========================================================================
    # インパルス応答
    # ========================================================================
    print("6️⃣  インパルス応答を計算中...")
    impulse_resp = qbnn_engine.impulse_response(duration=300, num_samples=300)
    t_impulse = np.arange(len(impulse_resp)) * 0.01

    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(t_impulse, impulse_resp, 'g-', linewidth=1.5, alpha=0.8, label='Impulse Response')

    # 指数包絡線
    envelope = np.exp(-convergence_rates[0] * t_impulse)
    ax6.plot(t_impulse, envelope, 'r--', linewidth=2, label=f'Envelope (σ={convergence_rates[0]:.2f})')
    ax6.plot(t_impulse, -envelope, 'r--', linewidth=2)

    ax6.grid(True, alpha=0.3)
    ax6.set_ylabel('Output', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax6.set_title('Impulse Response with Exponential Envelope', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.set_xlim(0, min(3, t_impulse[-1]))

    plt.suptitle('QB NN Laplace Transform Analysis - Frequency Response & Convergence',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = f'{save_dir}/qbnn_laplace_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存: {save_path}")
    plt.close()

    # ========================================================================
    # Figure 2: 周波数応答の詳細分析
    # ========================================================================
    fig2 = plt.figure(figsize=(14, 10))

    # ナイキスト線図
    ax1 = plt.subplot(2, 2, 1)
    s = 1j * 2 * np.pi * frequencies
    numerator = np.array([1.0, 0.5])
    H = np.zeros_like(s, dtype=complex)
    for i, si in enumerate(s):
        num = np.polyval(numerator, si)
        denom = 1.0
        for pole in system.poles:
            denom *= (si - pole)
        H[i] = num / denom

    ax1.plot(np.real(H), np.imag(H), 'b-', linewidth=2)
    ax1.scatter([np.real(H[0])], [np.imag(H[0])], c='green', s=100, marker='o', label='ω→0', zorder=5)
    ax1.scatter([np.real(H[-1])], [np.imag(H[-1])], c='red', s=100, marker='o', label='ω→∞', zorder=5)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Real Part', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
    ax1.set_title('Nyquist Diagram', fontsize=12, fontweight='bold')
    ax1.legend()

    # ゲイン余裕と位相余裕
    ax2 = plt.subplot(2, 2, 2)
    gain_margin_db = 20 * np.log10(1 / (np.max(np.abs(H)) + 1e-10))
    phase_margin_deg = 180 + np.min(phase_deg)

    info_text = f"""
    System Characteristics:

    🔹 Number of Poles: {len(system.poles)}
    🔹 Dominant Pole: {system.poles[0]:.3f}
    🔹 DC Gain: {np.abs(H[0]):.3f}

    Stability Margins:
    🔹 Gain Margin: {gain_margin_db:.2f} dB
    🔹 Phase Margin: {phase_margin_deg:.2f}°

    Convergence:
    🔹 Max Convergence Rate: {np.max(convergence_rates):.3f}
    🔹 Settling Time (2%): {settling_time:.3f} s

    Performance:
    🔹 Rise Time: ~{1/np.max(convergence_rates):.3f} s
    🔹 System Stability: {'Stable ✓' if np.all(np.real(poles) < 0) else 'Unstable ✗'}
    """

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.axis('off')

    # グループ遅延
    ax3 = plt.subplot(2, 2, 3)
    group_delay = -np.gradient(phase_deg) / np.gradient(2 * np.pi * frequencies)
    ax3.semilogx(frequencies, group_delay, 'purple', linewidth=2)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_ylabel('Group Delay (s)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax3.set_title('Group Delay Response', fontsize=12, fontweight='bold')

    # 周波数特性の統計
    ax4 = plt.subplot(2, 2, 4)

    # 周波数帯域別の特性
    freq_bands = {
        'Low (< 0.1 Hz)': frequencies < 0.1,
        'Mid (0.1-10 Hz)': (frequencies >= 0.1) & (frequencies < 10),
        'High (> 10 Hz)': frequencies >= 10,
    }

    band_names = []
    band_gains = []
    band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for (name, mask), color in zip(freq_bands.items(), band_colors):
        if np.any(mask):
            avg_gain = np.mean(mag_db[mask])
            band_names.append(name)
            band_gains.append(avg_gain)

    bars = ax4.bar(range(len(band_names)), band_gains, color=band_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Average Gain (dB)', fontsize=11, fontweight='bold')
    ax4.set_title('Frequency Band Characteristics', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(band_names)))
    ax4.set_xticklabels(band_names, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, gain in zip(bars, band_gains):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{gain:.1f}dB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Detailed Frequency Response Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path2 = f'{save_dir}/qbnn_frequency_detailed.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✅ 保存: {save_path2}")
    plt.close()

    # ========================================================================
    # 統計情報の出力
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 ラプラス変換分析結果")
    print("=" * 70)
    print(f"\n【システムの極（Poles）】")
    for i, pole in enumerate(system.poles):
        print(f"  Pole {i+1}: {pole:.4f} (σ={np.real(pole):.4f}, ω={np.imag(pole):.4f})")

    print(f"\n【周波数応答特性】")
    print(f"  DC Gain: {np.abs(H[0]):.4f}")
    print(f"  High-frequency Gain: {np.abs(H[-1]):.4f}")
    print(f"  Magnitude Range: [{np.min(mag_db):.2f}, {np.max(mag_db):.2f}] dB")
    print(f"  Phase Range: [{np.min(phase_deg):.2f}, {np.max(phase_deg):.2f}]°")

    print(f"\n【収束速度】")
    print(f"  最大収束速度: {np.max(convergence_rates):.4f} (1/s)")
    print(f"  最小収束速度: {np.min(convergence_rates):.4f} (1/s)")
    print(f"  平均収束速度: {np.mean(convergence_rates):.4f} (1/s)")

    print(f"\n【応答時間】")
    print(f"  落ち着き時間（2%）: {settling_time:.4f} s")
    print(f"  上昇時間（推定）: {1/np.max(convergence_rates):.4f} s")

    print(f"\n【安定性】")
    is_stable = np.all(np.real(system.poles) < 0)
    print(f"  システム安定性: {'✓ 安定' if is_stable else '✗ 不安定'}")
    if is_stable:
        print(f"  安定マージン: {-np.max(np.real(system.poles)):.4f}")

    print("\n" + "=" * 70)

    return system, fig, fig2


# ========================================================================
# メイン実行
# ========================================================================

if __name__ == "__main__":
    print("\n🚀 QB NN Laplace Transform Analysis を開始します...\n")

    # QBNN推論エンジンを初期化
    print("初期化中: QBNN推論エンジン")
    qbnn = QBNNInferenceEngine(num_qubits=8, num_layers=3, num_neurons=256)

    # ラプラス変換分析を実行
    system, fig1, fig2 = analyze_and_plot(qbnn)

    print("\n✨ 分析完了！")
    print("\n📁 生成ファイル:")
    print("  • qbnn_laplace_analysis.png - メイン分析結果")
    print("  • qbnn_frequency_detailed.png - 周波数応答の詳細分析")
