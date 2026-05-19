# QBNN to GGUF Conversion Guide
# QBNNからGGUF形式への変換ガイド

Entangled Quantum Bit Neural Network (QBNN/E-QBNN) モデルを llama.cpp 互換の GGUF 形式に変換し、量子特性を保持したまま利用できるようにします。

## 概要

このガイドでは、以下の2つの方法でQBNNモデルをGGUF形式に変換できます：

1. **`export_qbnn_gguf.py`** - 単体のQBNNチェックポイントを変換
2. **`generate_gguf_models.py`** - 複数サイズ・複数アーキテクチャのバルク変換

## 特徴

### 量子特性の保持
- **APQB (Adjustable Pseudo Quantum Bit)** パラメータの保存
- **エンタングルメント層** 情報の記録
- **量子相関行列** データの保存
- **幾何学的制約** 情報の保持

### llama.cpp 互換性
- 標準 GGUF フォーマット対応
- 複数の量子化レベルをサポート (Q4_K_M, Q5_K_M, Q8_0, F32など)
- メタデータとして量子情報を埋め込み

## インストール

```bash
# 必要なライブラリをインストール
pip install torch numpy gguf

# 推奨：GPU サポート用
pip install torch-cuda  # NVIDIA GPU の場合
```

## 使用方法

### 方法1: 単体モデルの変換 (`export_qbnn_gguf.py`)

単一の QBNN チェックポイントを GGUF に変換します。

```bash
# 基本的な使用法
python export_qbnn_gguf.py checkpoint.pt

# カスタマイズ例
python export_qbnn_gguf.py checkpoint.pt \
  --output-file my_model.gguf \
  --model-name "MyQBNN" \
  --model-size medium \
  --quantization Q5_K_M \
  --preserve-quantum
```

#### オプション

| オプション | 説明 | デフォルト |
|----------|------|----------|
| `input_file` | 入力 .pt ファイル | - |
| `--output-file, -o` | 出力 GGUF ファイル | auto-generated |
| `--model-name, -n` | モデル名 | Qubit-QBNN |
| `--model-size, -s` | モデルサイズ (small/medium/large/unknown) | unknown |
| `--quantization, -q` | 量子化タイプ | Q4_K_M |
| `--output-dir, -d` | 出力ディレクトリ | gguf_models |
| `--device` | 処理デバイス (cpu/cuda) | cpu |
| `--preserve-quantum` | 量子特性を保持 | True |

#### 出力例

```
🚀 QBNN to GGUF Converter
   Input: qbnn_checkpoint.pt
   Output: gguf_models/qbnn_checkpoint_Q4_K_M.gguf
   Model: Qubit-QBNN (medium)
   Quantization: Q4_K_M
   Preserve Quantum: True

📥 Loading qbnn_checkpoint.pt...
✨ Extracting quantum characteristics...
📝 Writing GGUF to gguf_models/qbnn_checkpoint_Q4_K_M.gguf...
📦 Adding tensors (256 total)...
✅ Successfully exported 256 tensors
   - Total parameters: 125,000,000
   - Quantum tensors preserved: 24
   - File size: 485.32MB
   - Quantization: Q4_K_M
   - Quantum correlation: ✓
   - Entanglement layers: 12
   - APQB theta params: 24
```

### 方法2: バルク生成 (`generate_gguf_models.py`)

複数サイズと複数アーキテクチャのモデルを一括生成します。

```bash
# NeuroQuantum モデルのみ
python generate_gguf_models.py --architectures neuroquantum

# QBNN モデルのみ
python generate_gguf_models.py --architectures qbnn --sizes small medium large

# 両方のアーキテクチャ + すべてのサイズ
python generate_gguf_models.py \
  --architectures neuroquantum qbnn \
  --sizes small medium large \
  --quantization Q5_K_M

# カスタムデバイスと出力ディレクトリ
python generate_gguf_models.py \
  --architectures qbnn \
  --sizes medium \
  --device cuda \
  --output-dir ./my_gguf_models \
  --quantization Q4_K_M
```

#### オプション

| オプション | 説明 | デフォルト |
|----------|------|----------|
| `--output-dir` | 出力ディレクトリ | gguf_models |
| `--architectures` | 生成するアーキテクチャ | neuroquantum |
| `--sizes` | 生成するサイズ | small medium large |
| `--device` | 処理デバイス | cpu |
| `--quantization, -q` | 量子化タイプ | Q4_K_M |
| `--skip-checkpoint-cleanup` | チェックポイント削除をスキップ | (flag) |

## 量子化オプション

異なる量子化レベルでのトレードオフ：

| 量子化 | ファイルサイズ | 精度 | 推奨用途 |
|-------|-------------|------|---------|
| **F32** | 最大 | 最高 | 参照実装、デバッグ |
| **F16** | ~50% | 高 | 高精度が必要な場合 |
| **Q8_0** | ~25% | 高 | バランス重視 |
| **Q6_K** | ~18% | 中〜高 | 標準的な用途 |
| **Q5_K_M** | ~15% | 中 | 一般的な推奨 |
| **Q5_K_S** | ~15% | 中 | Q5_K_M よりメモリ効率 |
| **Q4_K_M** | ~13% | 中 | 推奨（低メモリ） |
| **Q4_K_S** | ~13% | 中 | Q4_K_M よりメモリ効率 |

## QBNN特有の情報

### 保存される量子特性

GGUF ファイルには以下の量子特有情報が保存されます：

```json
{
  "model.is_quantum": true,
  "model.has_quantum_correlation": true,
  "model.has_entanglement": true,
  "model.apqb_theta_count": 24,
  "model.entangle_layer_count": 12,
  "model.quantum_metadata": {
    "type": "qbnn",
    "has_quantum_correlation": true,
    "has_entanglement": true,
    "apqb_theta_parameters": 24,
    "entanglement_layers": 12
  }
}
```

### 量子テンソルの処理

QBNN の量子特有テンソルは**量子化されずに保持**されます：

- `quantum_corr_*`: 量子相関行列
- `entangle_*`: エンタングル層の重み
- `*theta*`: APQB パラメータ

これにより、llama.cpp での推論時に正確な量子特性が再現されます。

## 実例

### 例1: 中規模QBNNモデルの変換

```bash
# チェックポイントから直接変換
python export_qbnn_gguf.py qbnn_medium_trained.pt \
  --model-name "Qubit-QBNN-Medium" \
  --model-size medium \
  --quantization Q5_K_M \
  --preserve-quantum
```

### 例2: すべてのサイズのQBNNモデルを生成

```bash
# 小、中、大サイズのモデルを生成
python generate_gguf_models.py \
  --architectures qbnn \
  --sizes small medium large \
  --quantization Q4_K_M \
  --output-dir ./qbnn_models_q4
```

### 例3: GPU での高速生成

```bash
# CUDA デバイスで実行（高速）
python generate_gguf_models.py \
  --architectures qbnn \
  --sizes small medium large \
  --device cuda \
  --quantization Q5_K_M
```

## トラブルシューティング

### エラー: "gguf module not available"

```bash
pip install gguf
```

### エラー: "EQBNNGenerativeAI not available"

```bash
# qbnn_layered.py が正しくインポートされているか確認
python -c "from qbnn_layered import EQBNNGenerativeAI; print('OK')"
```

### メモリ不足エラー

```bash
# CPU で実行（デフォルト）
python export_qbnn_gguf.py checkpoint.pt --device cpu

# または、より小さいモデルサイズを使用
python generate_gguf_models.py --architectures qbnn --sizes small
```

### 生成が遅い

```bash
# GPU を使用
python generate_gguf_models.py --device cuda --architectures qbnn

# またはより高い量子化を使用
python generate_gguf_models.py --quantization Q4_K_M
```

## llama.cpp での使用

変換されたGGUFファイルは標準的な llama.cpp で使用できます：

```bash
# llama-cpp-python を使用
pip install llama-cpp-python

# 推論スクリプト例
python -c "
from llama_cpp import Llama

model = Llama(model_path='gguf_models/qbnn_medium_Q4_K_M.gguf')
output = model('Hello, how are you?', max_tokens=100)
print(output['choices'][0]['text'])
"
```

## パフォーマンス情報

### 生成時間の目安（CPU）

| サイズ | NeuroQuantum | QBNN |
|-------|-------------|------|
| small | ~30秒 | ~45秒 |
| medium | ~90秒 | ~150秒 |
| large | ~300秒+ | ~500秒+ |

GPU（NVIDIA）を使用すると約 3-5 倍高速化されます。

### ファイルサイズの目安

| アーキテクチャ | サイズ | Q4_K_M | Q5_K_M | F32 |
|-------------|-------|--------|--------|------|
| QBNN | small | ~120MB | ~150MB | ~900MB |
| QBNN | medium | ~480MB | ~600MB | ~3.6GB |
| QBNN | large | ~1.8GB | ~2.3GB | ~14GB |

## 開発・カスタマイズ

### カスタム量子化レベルの追加

`export_qbnn_gguf.py` の `quantization_map` を拡張：

```python
quantization_map = {
    "Q4_K_M": GGMLQuantizationType.Q4_K,
    "CUSTOM": custom_quantization_type,  # 追加
    ...
}
```

### メタデータの拡張

`add_quantum_metadata_to_gguf` メソッドを拡張してカスタム情報を追加：

```python
writer.add_float32("model.custom_param", value)
```

## 参考資料

- [GGUF フォーマット仕様](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [QBNN (量子ビットニューラルネットワーク) 理論](./qbnn_layered.py)

## サポート

問題が発生した場合：

1. ログを確認
2. 依存ライブラリを更新: `pip install --upgrade gguf torch`
3. GitHub Issue として報告: https://github.com/tapiocatakeshi/qubit/issues

## ライセンス

Qubit プロジェクトと同じライセンスに従います。
