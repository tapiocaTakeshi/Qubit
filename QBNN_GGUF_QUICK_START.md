# QBNN to GGUF - クイックスタートガイド

QBNNモデルをllama.cpp互換のGGUF形式に変換し、量子特性を完全に保持します。

## 30秒で開始

```bash
# 1. 必要なライブラリをインストール
pip install torch numpy gguf

# 2. QBNN チェックポイントを GGUF に変換
python export_qbnn_gguf.py your_model.pt

# 3. 変換ファイルを使用
# gguf_models/ ディレクトリに your_model_Q4_K_M.gguf が生成されます
```

## よくあるコマンド

### 単一モデルの変換

```bash
# デフォルト設定で変換（Q4_K_M量子化）
python export_qbnn_gguf.py checkpoint.pt

# カスタム設定で変換
python export_qbnn_gguf.py checkpoint.pt \
  -o my_model.gguf \
  -n "MyQBNN" \
  -s medium \
  -q Q5_K_M
```

### 複数サイズのモデルを一括生成

```bash
# QBNN: small, medium, large を生成
python generate_gguf_models.py --architectures qbnn

# 別の量子化レベルで生成
python generate_gguf_models.py \
  --architectures qbnn \
  --quantization Q5_K_M \
  --output-dir ./my_models
```

### GPU で高速化（NVIDIA）

```bash
python export_qbnn_gguf.py checkpoint.pt --device cuda

# または
python generate_gguf_models.py --device cuda --architectures qbnn
```

## 量子化レベルの選択

| 量子化 | サイズ削減 | 精度 | 用途 |
|-------|----------|------|-----|
| **F32** | なし | 最高 | デバッグ・参照 |
| **Q8_0** | 75% | 高 | 高精度が必要 |
| **Q5_K_M** | 85% | 中 | **推奨** |
| **Q4_K_M** | 87% | 中 | 低メモリ推奨 |

## 出力ファイル

変換完了後、`gguf_models/` ディレクトリに GGUF ファイルが生成されます。

```
gguf_models/
├── model_name_Q4_K_M.gguf      ← 最小サイズ
├── model_name_Q5_K_M.gguf      ← バランス重視
└── model_name_F32.gguf         ← 最高精度
```

## llama.cpp で使用

```bash
# llama-cpp-python をインストール
pip install llama-cpp-python

# Python で使用
python << 'EOF'
from llama_cpp import Llama

model = Llama(model_path='gguf_models/model_Q4_K_M.gguf')
response = model("こんにちは", max_tokens=100)
print(response['choices'][0]['text'])
EOF

# または llama.cpp CLI で使用
./main -m gguf_models/model_Q4_K_M.gguf --prompt "こんにちは"
```

## 量子特性の保存確認

変換時に以下の情報が表示されます：

```
✅ Successfully exported 256 tensors
   - Total parameters: 125,000,000
   - Quantum tensors preserved: 24      ← 量子テンソルが保持されている
   - File size: 485.32MB
   - Quantization: Q4_K_M
   - Quantum correlation: ✓             ← 量子相関が保存されている
   - Entanglement layers: 12            ← エンタングルメント層
   - APQB theta params: 24              ← APQB パラメータ
```

## トラブルシューティング

### "gguf module not available"
```bash
pip install gguf
```

### メモリ不足
```bash
# CPU で実行
python export_qbnn_gguf.py model.pt --device cpu

# または小さいモデルを使用
python generate_gguf_models.py --sizes small
```

### 変換が遅い
```bash
# GPU を使用
python export_qbnn_gguf.py model.pt --device cuda

# または Q4_K_M 量子化を使用
python export_qbnn_gguf.py model.pt -q Q4_K_M
```

## パフォーマンス

| タスク | 時間（CPU） | 時間（GPU） |
|-------|-----------|-----------|
| small モデル | ~30秒 | ~10秒 |
| medium モデル | ~90秒 | ~30秒 |
| large モデル | ~300秒+ | ~100秒 |

*時間は環境により異なります*

## 詳細ガイド

より詳しい説明は [QBNN_GGUF_GUIDE.md](./QBNN_GGUF_GUIDE.md) を参照してください。

## サポート

- 問題が発生した場合: [GitHub Issues](https://github.com/tapiocaTakeshi/Qubit/issues)
- 質問がある場合: [Discussions](https://github.com/tapiocaTakeshi/Qubit/discussions)

---

**Happy quantum modeling! 🚀⚛️**
