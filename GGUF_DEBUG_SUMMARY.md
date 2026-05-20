# GGUF Loading Debug Summary
# GGUFロード問題デバッグ概要

## 問題の診断結果

### 主な課題

1. **メタデータ不足**: export_qbnn_gguf.py がランタイムパラメータ（llm.gpu_layers など）をGGUFに記録していなかった
   - ✅ **修正済み**: 実行時パラメータメタデータを追加

2. **クライアント側の問題**: PocketPal などアプリケーションがGGUFメタデータを読み込まずハードコードされたデフォルト値を使用
   - ✅ **対策**: クライアント実装例を提供

3. **アーキテクチャ互換性**: llama.cpp は custom アーキテクチャ（neuroquantum、qbnn）をサポートしていない
   - ✅ **対策**: 代替ローダーと詳細なドキュメントを提供

4. **診断ツール不足**: ロード問題のデバッグに必要なツールがなかった
   - ✅ **追加**: 複数の診断・検証ツール

## 実装された修正

### 1. メタデータの追加 (`export_qbnn_gguf.py`)

**変更内容**:
```python
# 以前: なし
# 現在: 以下のフィールドを自動で追加
writer.add_int32("llm.context_length", self.gguf_params.get("n_ctx", 512))
writer.add_int32("llm.batch_size", self.gguf_params.get("n_batch", 64))
writer.add_int32("llm.ubatch_size", self.gguf_params.get("n_ubatch", 64))
writer.add_int32("llm.threads", self.gguf_params.get("n_threads", 4))
writer.add_int32("llm.gpu_layers", self.gguf_params.get("n_gpu_layers", 0))
writer.add_string("llm.cache_type_k", self.gguf_params.get("cache_type_k", "f16"))
writer.add_string("llm.cache_type_v", self.gguf_params.get("cache_type_v", "f16"))
writer.add_string("model.gguf_params", json.dumps(self.gguf_params))
```

**使用方法**:
```bash
# デフォルトパラメータで変換
python export_qbnn_gguf.py checkpoint.pt --model-size medium

# カスタムランタイムパラメータで変換
python export_qbnn_gguf.py checkpoint.pt \
  --model-size medium \
  --gguf-params '{"n_ctx": 2048, "n_gpu_layers": 10, "n_batch": 128}'
```

### 2. 診断ツール

#### check_gguf_params.py - メタデータ確認ツール
```bash
# メタデータを表示
python check_gguf_params.py model.gguf

# 互換性診断を実行
python check_gguf_params.py model.gguf --diagnose

# JSON 形式で出力
python check_gguf_params.py model.gguf --json
```

**出力例**:
```
📋 GGUF File: neuroquantum_medium_Q4_K_M.gguf
   File Size: 0.54 GB

🏗️  Model Info:
   Architecture: neuroquantum
   Size: medium
   Quantization: Q4_K_M

⚙️  Runtime Parameters:
   Context Length: 512
   Batch Size: 64
   GPU Layers: 0
   Threads: 4
```

#### validate_gguf_metadata.py - メタデータ検証ツール
```bash
# メタデータを検証
python validate_gguf_metadata.py model.gguf

# 複数ファイルを検証
python validate_gguf_metadata.py model1.gguf model2.gguf model3.gguf
```

**検証項目**:
- ✅ 必須フィールド（7個）
- ✅ 推奨フィールド（6個）
- ✅ データ型の正確性
- ✅ パラメータ値の妥当性

### 3. クライアント実装例 (`examples_gguf_client.py`)

**QubitGGUFClient クラス**:
```python
from examples_gguf_client import QubitGGUFClient

client = QubitGGUFClient("model.gguf")

# 1. メタデータを読み込む
metadata = client.load_metadata()
print(f"Architecture: {metadata['architecture']}")
print(f"GPU Layers: {client.runtime_params['n_gpu_layers']}")

# 2. モデルをロード（自動フォールバック付き）
model = client.load_with_llama_cpp()

# 3. 推論実行
output = client.generate("Hello", max_tokens=100)
```

**特徴**:
- GGUF メタデータの自動読み込み
- GPU メモリ不足時に自動的に CPU にフォールバック
- PyTorch チェックポイント形式へのフォールバック
- 詳細なログ出力

### 4. トラブルシューティングガイド

ファイル: `GGUF_LOADING_TROUBLESHOOTING.md`

**含まれる内容**:
- よくある問題と解決方法
- アーキテクチャ互換性の説明
- メモリ管理のベストプラクティス
- 複数の代替ローダー実装例
- エラーハンドリングパターン

## 使用手順

### 1. GGUF ファイルの正しい作成

```bash
# オプション A: export_qbnn_gguf.py を使用
python export_qbnn_gguf.py checkpoint.pt \
  --model-size medium \
  --quantization Q4_K_M \
  --gguf-params '{"n_ctx": 2048, "n_batch": 512, "n_gpu_layers": 0}'

# オプション B: generate_gguf_models.py を使用
python generate_gguf_models.py \
  --architectures qbnn \
  --sizes medium \
  --quantization Q4_K_M
```

### 2. GGUF ファイルの検証

```bash
# メタデータを確認
python check_gguf_params.py output_model.gguf

# メタデータを検証
python validate_gguf_metadata.py output_model.gguf

# 互換性を診断
python check_gguf_params.py output_model.gguf --diagnose
```

### 3. クライアント側での読み込み

```python
# Python: llama-cpp-python を使用
python examples_gguf_client.py model.gguf

# 以下の設定を自動で読み込み:
# - Context length (n_ctx)
# - Batch size (n_batch)
# - GPU layers (n_gpu_layers)
# - Thread count (n_threads)
```

### 4. アプリケーション側（PocketPal など）

**実装すべき機能**:
```javascript
// JavaScript/TypeScript の例
async function loadGGUFModel(modelPath) {
  // 1. GGUF メタデータを読み込む
  const metadata = await fetchGGUFMetadata(modelPath);
  
  // 2. ランタイムパラメータを取得
  const params = {
    n_ctx: metadata.llm.context_length || 512,
    n_batch: metadata.llm.batch_size || 64,
    n_gpu_layers: metadata.llm.gpu_layers || 0,
    n_threads: metadata.llm.threads || 4,
  };
  
  // 3. メタデータに基づいてモデルを初期化
  return initializeModel(modelPath, params);
}
```

## テスト方法

### テスト 1: メタデータが正しく記録されている

```bash
python check_gguf_params.py model.gguf
# 出力に llm.gpu_layers, llm.context_length などが含まれていることを確認
```

### テスト 2: メタデータ検証

```bash
python validate_gguf_metadata.py model.gguf
# 終了コード 0: 成功
# 終了コード 1: エラー（必須フィールド不足）
# 終了コード 2: 警告のみ（推奨フィールド不足）
```

### テスト 3: クライアント読み込み

```bash
python examples_gguf_client.py model.gguf
# メタデータが読み込まれ、モデルがロードされることを確認
```

### テスト 4: エンドツーエンド

```python
from examples_gguf_client import QubitGGUFClient

client = QubitGGUFClient("model.gguf")
metadata = client.load_metadata()

# メタデータが正しく読み込まれていることを確認
assert metadata['architecture'] in ['neuroquantum', 'qbnn']
assert client.runtime_params['n_gpu_layers'] == 0

# モデルをロード
model = client.load_with_llama_cpp()

# 推論テスト
output = client.generate("テスト", max_tokens=10)
assert len(output) > 0
```

## ファイル一覧

### 新規追加ファイル
1. `GGUF_LOADING_TROUBLESHOOTING.md` - トラブルシューティングガイド
2. `GGUF_DEBUG_SUMMARY.md` - このファイル
3. `check_gguf_params.py` - メタデータ確認ツール
4. `validate_gguf_metadata.py` - メタデータ検証ツール
5. `examples_gguf_client.py` - クライアント実装例

### 修正ファイル
1. `export_qbnn_gguf.py` - ランタイムパラメータメタデータを追加

### 既存ファイル
- `export_gguf.py` - 既に正しいメタデータを含む
- `generate_gguf_models.py` - 既に正しいメタデータを含む

## トラブルシューティング

### 問題: "Failed to load model"

**解決方法**:
```bash
# 1. メタデータを確認
python check_gguf_params.py model.gguf --diagnose

# 2. アーキテクチャを確認（neuroquantum/qbnn の場合は標準 llama.cpp では非サポート）
# → examples_gguf_client.py または PyTorch ローダーを使用

# 3. ランタイムパラメータを調整
python examples_gguf_client.py model.gguf
```

### 問題: "メタデータが読み込まれていない"

**解決方法**:
```bash
# 1. ファイルを検証
python validate_gguf_metadata.py model.gguf

# 2. 必要に応じて GGUF ファイルを再生成
python export_qbnn_gguf.py checkpoint.pt \
  --model-size medium \
  --gguf-params '{"n_ctx": 512, "n_batch": 64, "n_gpu_layers": 0}'

# 3. 再度検証
python validate_gguf_metadata.py model.gguf
```

### 問題: "GPU メモリ不足"

**解決方法**:
```python
# examples_gguf_client.py が自動的に CPU にフォールバック
# または手動で設定:
client = QubitGGUFClient("model.gguf")
model = client.load_with_llama_cpp(override_params={'n_gpu_layers': 0})
```

## 次のステップ

1. **CI/CD パイプライン統合**:
   ```bash
   # GGUF 生成後に自動検証
   python validate_gguf_metadata.py *.gguf || exit 1
   ```

2. **Hugging Face へのアップロード時に検証**:
   ```python
   # upload_to_huggingface.py に検証ステップを追加
   ```

3. **ドキュメント更新**:
   - README.md に GGUF 使用方法を追加
   - QBNN_GGUF_GUIDE.md に新しいツール情報を追加

4. **モバイルアプリ対応**:
   - PocketPal などで GGUF メタデータを読み込む実装
   - メタデータに基づいた自動設定

## 関連リソース

- [GGUF 仕様](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Qubit QBNN ガイド](./QBNN_GGUF_GUIDE.md)
- [トラブルシューティングガイド](./GGUF_LOADING_TROUBLESHOOTING.md)

## まとめ

これらの修正により：

✅ **GGUF ファイルは完全なメタデータを含む**
- すべてのランタイムパラメータが記録される
- 量子特性（QBNN）が保持される

✅ **クライアント開発者は正しく実装できる**
- 実装例が提供される
- メタデータの読み込み方がドキュメント化されている

✅ **問題のデバッグが容易**
- 複数の診断ツール
- 詳細なトラブルシューティングガイド

✅ **複数のローダーがサポートされる**
- llama-cpp-python
- PyTorch
- カスタムローダー

---

**質問や問題がある場合**: https://github.com/tapiocatakeshi/qubit/issues
