# QBNN Frontal Engine - MCP Server

## 概要

**QBNN Frontal Engine** は、脳の前頭葉の役割をシミュレートする MCP (Model Context Protocol) サーバーです。

### 前頭葉の機能
- 🧠 **意思決定**: 複数の選択肢から最適な選択を判断
- ⚖️ **リスク評価**: 状況のリスクを定量的に評価
- ✅ **品質判定**: 成果物の品質を客観的に判断
- 🎯 **優先順位付け**: タスク間の優先度を判定
- 📋 **倫理的判断**: 行動の倫理性や適切性を評価

---

## 機能と仕様

### Tool: `judge`

汎用的な判断機能。あらゆる判断タスクに対応します。

#### 入力スキーマ

```json
{
  "context": "判断の背景情報・文脈（必須）",
  "judgment_request": "何を判断するか・判断内容（必須）",
  "criteria": {
    "criterion_name": "criterion_value"
  },
  "options": ["option1", "option2", "option3"],
  "strict_mode": false
}
```

**パラメータ詳細:**
- **context** (string, 必須)
  - 判断に必要な背景情報や文脈
  - できるだけ詳細に提供することで判断精度が向上
  
- **judgment_request** (string, 必須)
  - 具体的に何を判断したいかを明記
  - 例: "このプロジェクトをリリースできるか？"
  
- **criteria** (object, オプション)
  - 判断時に考慮する基準
  - キー: 基準名, 値: 基準値（文字列、数値、真偽値）
  
- **options** (array, オプション)
  - 検討対象となる選択肢
  - 複数選択肢の比較判断時に使用
  
- **strict_mode** (boolean, デフォルト: false)
  - true: スコア70以上でYes（厳密モード）
  - false: スコア50以上でYes（通常モード）

#### 出力スキーマ

```json
{
  "decision": "Yes",
  "score": 75,
  "reasoning": "判断の根拠説明",
  "confidence": "high",
  "key_factors": ["要因1", "要因2", "要因3"],
  "timestamp": "2026-06-22T12:34:56.789Z"
}
```

**出力フィールド:**
- **decision** (string)
  - 最終判断: "Yes" または "No"
  
- **score** (integer, 0-100)
  - 判断の強度をスコア化
  - 0に近い: 否定的, 100に近い: 肯定的
  
- **reasoning** (string)
  - 判断の理由・根拠説明
  
- **confidence** (string)
  - 判断の確信度: "high" / "medium" / "low"
  
- **key_factors** (array)
  - 判断に影響を与えた主要な要因（最大5個）
  
- **timestamp** (string)
  - ISO 8601形式のタイムスタンプ

---

## 使用例

### 例1: プロジェクトのリリース判断

```json
{
  "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。チームは高い士気を持ち、リスク要因は特に報告されていません。テスト完了率は95%です。",
  "judgment_request": "このプロジェクトをリリースしても安全か？"
}
```

**期待される出力:**
```json
{
  "decision": "Yes",
  "score": 82,
  "reasoning": "指定された基準と文脈に基づいて、肯定的な判断が支持されています。 主要な要因: 十分な背景情報がある, ポジティブ要因: 品質, ポジティブ要因: 安全",
  "confidence": "high",
  "key_factors": ["十分な背景情報がある", "ポジティブ要因: 品質", "ポジティブ要因: 安全"],
  "timestamp": "2026-06-22T12:34:56.789Z"
}
```

### 例2: リスク評価（厳密モード）

```json
{
  "context": "新技術の導入には以下のリスクが考えられます: 学習曲線が急、既存システムとの互換性問題の可能性、短期的には生産性低下の予測。一方、導入後の効率性向上は30%と見積もられています。",
  "judgment_request": "このリスクは許容可能か？",
  "strict_mode": true
}
```

### 例3: 複数ベンダーからの選択

```json
{
  "context": "3つのベンダーを評価しました。\nベンダーA: 価格¥100万、サポート弱、実績少\nベンダーB: 価格¥150万、サポート強い、実績豊富\nベンダーC: 価格¥200万、サポート最高、実績多数",
  "judgment_request": "ベンダーBを選択することが最適な判断か？",
  "options": ["ベンダーA", "ベンダーB", "ベンダーC"],
  "criteria": {
    "budget": "¥150万以下",
    "support": "strong",
    "track_record": "proven"
  }
}
```

### 例4: 品質判定

```json
{
  "context": "コードレビュー結果: テストカバレッジ95%、重大なバグなし、ドキュメント完全、パフォーマンス最適化済み、セキュリティ監査クリア。",
  "judgment_request": "このコードの品質は本番環境への展開に十分か？"
}
```

---

## インストールと実行

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. MCP サーバーの起動

```bash
python frontal_engine_mcp_server.py
```

### 3. テストの実行

```bash
python test_frontal_engine.py
```

---

## Claude Code での統合

Claude Code で QBNN Frontal Engine を使用するには、`settings.json` に以下を追加してください：

```json
{
  "mcp": {
    "servers": {
      "qbnn-frontal-engine": {
        "command": "python",
        "args": ["/path/to/frontal_engine_mcp_server.py"],
        "env": {
          "PYTHONUNBUFFERED": "1"
        }
      }
    }
  }
}
```

---

## 判断アルゴリズム

### スコア計算ロジック

1. **初期スコア**: 50（中立）
2. **テキスト分析**: 背景情報の量に基づいて調整
3. **キーワード分析**: ポジティブ/ネガティブキーワードの検出
4. **基準評価**: 指定された基準に対する適合性を評価
5. **オプション評価**: 選択肢が文脈にどの程度マッチするか
6. **最終スコア**: 0-100の範囲に正規化

### 判断ルール

- **strict_mode = false（デフォルト）**
  - score >= 50: **Yes**
  - score < 50: **No**
  
- **strict_mode = true**
  - score >= 70: **Yes**
  - score < 70: **No**

### 信頼度の判定

- **高信頼度 (high)**: score >= 75 または score <= 25
- **中信頼度 (medium)**: 25 < score < 75（ただし40-60の範囲は medium）
- **低信頼度 (low)**: エラー時

---

## 応用例

### 1. CI/CDパイプラインでの自動判断
```python
# デプロイメント判定
judge.judge({
    "context": "テストスイート完了。全テスト成功。CI/CDチェック合格。",
    "judgment_request": "本番環境にデプロイできるか？"
})
```

### 2. コンプライアンス判定
```python
judge.judge({
    "context": "新規機能は個人情報を処理しません。GDPR準拠チェック完了。",
    "judgment_request": "GDPR要件を満たしているか？"
})
```

### 3. 予算承認フロー
```python
judge.judge({
    "context": "予算要求: ¥5,000万。ROI見積: 年間¥10,000万。",
    "judgment_request": "この予算承認は合理的か？",
    "criteria": {"roi_threshold": "200%", "risk_level": "low"}
})
```

---

## ライセンス

MIT License

## 貢献

バグ報告や機能提案は GitHubのIssueで受け付けています。
