# パイプライン推論システム：ライブラリー統合版

**更新日**: 2025-06-23  
**ブランチ**: `claude/gemma-qbnn-frontal-5s9etz`

---

## 概要

`pipeline_inference.py` に以下の機能を追加し、推論結果を知識ライブラリーに蓄積し、Gemmaが生成した応答を活用するシステムを構築しました。

### パイプラインの構成

```
ユーザー入力
    ↓
[ステップ1] Gemma言語処理エンジン
    ├─ 課題タイプの検出
    ├─ キーワード抽出
    └─ メリット・デメリット分析
    ↓
[ステップ2] QBNN量子推論エンジン
    ├─ APQB計算
    ├─ スコア算出
    └─ 信頼度計算
    ↓
[ステップ3] 知識ライブラリー
    ├─ エントリー蓄積
    ├─ タスク別インデックス化
    └─ 判断パターン学習
    ↓
[ステップ4] Gemma文章生成エンジン
    ├─ 類似ケース検索
    ├─ コンテキスト構築
    └─ 推奨メッセージ生成
    ↓
最終応答
```

---

## 新規追加コンポーネント

### 1. KnowledgeLibrary（知識ライブラリー）

**責務**: 推論結果の蓄積と検索

**主要メソッド**:
- `add_entry()`: 推論結果をライブラリーに追加
- `get_similar_cases()`: タスクタイプ別に類似ケースを検索
- `get_library_summary()`: ライブラリーの統計情報を取得
- `format_for_context()`: 文章生成用のコンテキストフォーマット

**機能**:
```python
# タスク別インデックス
task_type_index = {
    "キャリア": [0, 2, 5, ...],
    "投資": [1, 4, 7, ...],
    "教育": [3, 6, 8, ...],
}

# 判断パターン学習
judgment_patterns = {
    "キャリア_Yes_60": [...],
    "投資_No_40": [...],
}
```

**統計情報**:
```python
{
    "total_entries": 8,
    "task_types": ["キャリア", "投資", "教育", "一般判断"],
    "judgment_patterns": ["キャリア_Yes_60", "投資_No_40", ...],
    "yes_count": 5,
    "no_count": 3,
}
```

---

### 2. GemmaTextGenerator（Gemma文章生成エンジン）

**責務**: 知識ライブラリーに基づいて自然言語応答を生成

**主要メソッド**:
- `generate_response()`: タスクと判断に基づいて応答を生成
- `_generate_recommendation()`: 推奨メッセージを作成

**生成ロジック**:

1. **ライブラリー参照**
   - 同じタスクタイプの類似ケースを検索
   - 判断パターンから学習

2. **コンテキスト構築**
   ```
   - ライブラリー統計（全体的な傾向）
   - 参考事例（具体的な類似ケース）
   - 現在の分析（今回の判定）
   ```

3. **推奨生成**
   - スコア帯に応じた強度調整
   - メリット・デメリット列挙
   - アクションアイテム提示

**推奨メッセージの段階**:

| スコア帯 | 推奨 | メッセージ |
|---------|------|-----------|
| 75+ | 強く推奨 | メリットが明確でリスクは限定的 |
| 60-75 | 推奨 | メリットがやや上回っている |
| 50-60 | おおむね推奨 | バランスが取れている |
| 40-50 | 検討必要 | メリット・デメリットが拮抗 |
| 30-40 | 非推奨 | デメリットが目立つ |
| -30 | 強く非推奨 | リスクが高すぎる |

---

## 使用例

### 基本的な使用方法

```python
from pipeline_inference import PipelineInferenceSystem

# システム初期化
system = PipelineInferenceSystem()

# ユーザー入力を処理
user_input = "転職すべきか？給与が上がるが安定性が不確定。"

# 従来版（詳細分析）
result_traditional = system.infer(user_input, use_library_generation=False)

# ライブラリー版（Gemma文章生成）
result_library = system.infer(user_input, use_library_generation=True)
```

### ライブラリー統計の確認

```python
library_info = system.get_library_info()
print(f"蓄積エントリー: {library_info['total_entries']}")
print(f"Yes判定: {library_info['yes_count']}")
print(f"No判定: {library_info['no_count']}")
```

---

## 実行例

### デモ実行

```bash
# 基本的なパイプラインデモ
python3 pipeline_inference.py

# 高度な機能デモ
python3 advanced_pipeline_demo.py
```

### 出力例

**質問**: 起業に挑戦すべきか？新規事業だが市場ニーズが明確で、給与は下がる。

**応答**:
```
✓ この判断は【おおむね推奨】できます。

【利点】
  • 給与・報酬

【課題】
  • 複数の課題

【分析スコア】
  総合判定スコア: 55/100
  解釈: バランスが取れています。慎重に計画を進めることをお勧めします。

【推奨アクション】
  1. 追加の情報収集を行う
  2. メンターや専門家に相談する
  3. 小さなステップから試す
```

---

## アーキテクチャ利点

### 1. スケーラビリティ
- ライブラリーエントリーが増えるほど精度向上
- 新しいタスク領域への対応が容易
- 判断パターンの統計的学習

### 2. 応答の多様性
- 完全にテンプレートに頼らない生成
- コンテキストベースの推奨
- スコアに応じた段階的なアドバイス

### 3. 説明可能性
- 類似ケースの参照による根拠提示
- 統計情報による信頼度表示
- メリット・デメリットの明示

### 4. 継続的改善
- ユーザーフィードバックの取り込み容易
- パターン学習による最適化
- A/Bテスト対応可能

---

## 技術仕様

### KnowledgeLibraryのデータ構造

```python
entries = [
    {
        "task": {
            "original_input": str,
            "task_type": str,
            "main_question": str,
            "keywords": List[str],
            "merits": List[str],
            "demerits": List[str],
            "context_length": int,
        },
        "judgment": {
            "base_score": float,
            "quantum_bonus": float,
            "final_score": float,
            "decision": str,  # "Yes" or "No"
            "confidence": str,  # "High", "Medium", "Low"
            "quantum_info": {...},
            "analysis": {...},
        },
        "timestamp": int,
    },
    ...
]
```

### インデックス構造

**task_type_index**:
```python
{
    "キャリア": [0, 2, 5],
    "投資": [1, 4],
    "教育": [3, 6],
}
```

**judgment_patterns**:
```python
{
    "キャリア_Yes_60": [{...}, {...}],
    "投資_No_40": [{...}],
}
```

---

## パフォーマンス特性

| 操作 | 時間複雑度 | 説明 |
|------|----------|------|
| add_entry() | O(1) | 定数時間で追加 |
| get_similar_cases() | O(k) | タスク別インデックスで高速検索 |
| get_library_summary() | O(n) | 全エントリーをスキャン |
| generate_response() | O(k log k) | 類似ケース検索とソート |

（k = ライブラリーエントリー数、n = 平均エントリー数）

---

## 今後の拡張案

### 短期（1-2週間）
- [ ] セマンティック検索の実装（ベクトル化）
- [ ] ユーザーフィードバック機構の追加
- [ ] キャッシング機能によるパフォーマンス最適化

### 中期（1ヶ月）
- [ ] 外部データソースの統合（市場データ等）
- [ ] マルチモーダル入力対応
- [ ] リアルタイムライブラリー更新

### 長期（3ヶ月+）
- [ ] 強化学習による推奨精度向上
- [ ] 自動カテゴリ化エンジン
- [ ] 多言語対応

---

## テスト実行

### テストケース

**テストケース1: ライブラリー蓄積**
```bash
python3 -c "
from pipeline_inference import PipelineInferenceSystem
system = PipelineInferenceSystem()
for i in range(8):
    system.infer('テスト入力 ' + str(i))
info = system.get_library_info()
assert info['total_entries'] == 8
print('✓ テスト1合格: ライブラリー蓄積')
"
```

**テストケース2: Gemma文章生成**
```bash
python3 -c "
from pipeline_inference import PipelineInferenceSystem
system = PipelineInferenceSystem()
result = system.infer('転職すべきか？給与が上がるが不確定。', use_library_generation=True)
assert '推奨' in result
print('✓ テスト2合格: 文章生成')
"
```

---

## まとめ

このパイプラインシステムは、以下の特性を持つエンタープライズグレードの推論システムです：

✅ **スケーラブル**: ライブラリーベースの学習  
✅ **説明可能**: 根拠を示した推奨  
✅ **柔軟**: テンプレートフリーの生成  
✅ **改善可能**: 継続的な最適化が可能  

Gemmaの言語能力とQBNNの量子推論を組み合わせることで、単純な判断エンジンを超えた、コンテキストを理解する意思決定支援システムを実現しました。
