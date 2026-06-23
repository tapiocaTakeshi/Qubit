# Gemma+QBNN Quantum Prefrontal Cortex

## 概要

**Gemma+QBNNの量子強化型前頭葉システム** - Claude AIの思考プロセスに対して、Gemmaトランスフォーマー + QBNN（量子ビットニューラルネットワーク）層による高度な判断機能を提供します。

```
[ユーザーリクエスト]
    ↓
[Claude メイン処理 (自然言語理解)]
    ↓
[Gemma+QBNN 前頭葉] ← ★量子強化型判断
    ├─ Gemma層: 複雑なテキスト理解
    ├─ QBNN層: 層間エンタングルメント推論
    └─ 判断ヘッド: Yes/No + スコア + 根拠
    ↓
[応答生成]
```

## 主な特徴

### 1. **量子推論能力**
- **APQB (Adjustable Pseudo Quantum Bit)**: 論文の数学的定義に基づく量子状態表現
- **層間エンタングルメント**: `e^(l) = f_entangle(q^(l), q^(l-1))` による複雑な判断処理
- **非古典的な推論**: 複数の状態を同時に考慮した意思決定

### 2. **多次元的判断**
```
判断タイプ:
├─ 意思決定（Decision Making）
├─ リスク評価（Risk Assessment）
├─ 品質判定（Quality Judgment）
├─ 倫理的判断（Ethical Judgment）
├─ 優先順位付け（Prioritization）
└─ 安全性確認（Safety Check）
```

### 3. **出力仕様**
```json
{
  "decision": "Yes" | "No",
  "score": 0-100,
  "reasoning": "判断の根拠説明",
  "confidence": "high" | "medium" | "low",
  "key_factors": ["要因1", "要因2", ...],
  "timestamp": "ISO形式の時刻",
  "quantum_info": {
    "yes_probability": 0.0-1.0,
    "quantum_weight": 0.6,
    "entangle_strength": 0.7
  }
}
```

## アーキテクチャ

### コンポーネント

#### 1. `GemmaQBNNPrefrontalCortex`
前頭葉システムの中心モジュール。

```python
from gemma_qbnn_prefrontal_cortex import create_prefrontal_cortex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cortex = create_prefrontal_cortex(device=device)
```

**機能:**
- Gemma + QBNN モデルの統合管理
- テキストエンコーディング
- 量子推論層の処理
- 判断ヘッドによる決定生成

#### 2. `JudgmentHead`
判断を生成するニューラルネットワーク層。

```
入力: [batch, embed_dim] の隠れ状態
  ↓
[決定層] → decision_logits (Yes/No)
[スコア層] → score_logits (0-100)
[信頼度層] → confidence_logits (low/medium/high)
[根拠層] → reasoning_logits (説明テキスト)
```

#### 3. `FrontalEngineJudge` (MCP サーバー)
MCPサーバーとして動作する判断エンジン。

```python
from frontal_engine_mcp_server import FrontalEngineJudge

judge = FrontalEngineJudge()  # 自動的に量子前頭葉を使用
result = judge.judge({
    "context": "判断の背景情報",
    "judgment_request": "何を判断するか"
})
```

## 使用方法

### 基本的な使い方

```python
from gemma_qbnn_prefrontal_cortex import create_prefrontal_cortex
import torch

# 前頭葉システムを初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cortex = create_prefrontal_cortex(device=device)
cortex.eval()

# 判断タスクを実行
judgment_task = {
    "context": """
    ユーザーの個人情報削除リクエスト。
    GDPRコンプライアンス機能あり。
    """,
    "judgment_request": "個人情報削除を実行してもセキュアか？",
    "criteria": {"security": True, "gdpr_compliance": True},
    "strict_mode": True
}

result = cortex.judge(judgment_task)
print(f"決定: {result['decision']}")
print(f"スコア: {result['score']}/100")
print(f"根拠: {result['reasoning']}")
```

### Claude統合での使用

```python
from claude_prefrontal_integration import claude_prefrontal_cortex

# Claude思考プロセス内での判断
should_proceed, result = await claude_prefrontal_cortex.should_proceed_with_action(
    action_description="APIレート制限超過時にキャッシュから応答",
    context="キャッシュヒット率 95%、ユーザー許可あり",
    risks=["データ鮮度低下"]
)

if should_proceed:
    # アクション実行
    pass
```

### MCP サーバーでの使用

```bash
# MCPサーバーを起動
python frontal_engine_mcp_server.py

# または、Claude IDE環境で自動統合
```

## 判断タイプの詳細

### 1. 意思決定判断 (Decision Making)

**用途:** プロジェクト承認、実装方針決定など

```python
result = await cortex.make_judgment(
    context="プロジェクト提案内容...",
    judgment_request="このプロジェクトを承認すべきか？",
    judgment_type=JudgmentType.DECISION_MAKING
)
```

### 2. リスク評価 (Risk Assessment)

**用途:** セキュリティ判断、本番デプロイ判断

```python
result = await cortex.make_judgment(
    context="スキーマ変更の詳細。バックアップあり...",
    judgment_request="現在の状況で実行するのは安全か？",
    judgment_type=JudgmentType.RISK_ASSESSMENT,
    strict_mode=True  # 厳密モード（安全重視）
)
```

### 3. 倫理的判断 (Ethical Judgment)

**用途:** プライバシー評価、倫理的懸念検討

```python
result = await cortex.assess_ethical_concerns(
    action_description="ユーザー行動データの分析",
    stakeholders=["ユーザー", "社会"],
    potential_harms=["プライバシー侵害"]
)
```

### 4. 優先順位付け (Prioritization)

**用途:** 複数タスクの優先度決定

```python
prioritized = await cortex.prioritize_tasks(
    tasks=[
        {"name": "バグ修正", "description": "クリティカルバグ"},
        {"name": "機能追加", "description": "新UIコンポーネント"},
        {"name": "セキュリティ監査", "description": "年次監査"}
    ]
)
```

## 設定

### `JudgmentConfig` パラメータ

```python
from gemma_qbnn_prefrontal_cortex import JudgmentConfig

config = JudgmentConfig(
    vocab_size=32000,           # トークンボキャブラリサイズ
    embed_dim=768,              # 埋め込み次元
    hidden_dim=2048,            # FFN隠れ層次元
    num_heads=12,               # アテンション頭数
    num_layers=12,              # トランスフォーマー層数
    max_seq_len=4096,           # 最大シーケンス長
    entangle_strength=0.7,      # エンタングルメント強度
    quantum_weight=0.6,         # QBNN寄与度（0-1）
    decision_threshold=0.5,     # 決定閾値
    confidence_threshold=0.7    # 信頼度閾値
)
```

## パフォーマンス最適化

### メモリ使用量の削減
```python
# より小さいモデルを使用
config = JudgmentConfig(
    embed_dim=512,
    num_layers=6,
    quantum_weight=0.5  # 量子計算の寄与度を下げる
)
```

### 推論速度の向上
```python
# バッチ処理を活用
batch_judgments = [
    {"context": "...", "judgment_request": "..."},
    {"context": "...", "judgment_request": "..."},
    # ...
]

for task in batch_judgments:
    result = cortex.judge(task)
```

## 実装ファイル

| ファイル | 説明 |
|---------|------|
| `gemma_qbnn_prefrontal_cortex.py` | 量子前頭葉システムのコア実装 |
| `frontal_engine_mcp_server.py` | MCPサーバーインターフェース |
| `claude_prefrontal_integration.py` | Claude AI統合レイヤー |
| `example_quantum_prefrontal.py` | 使用例とデモンストレーション |
| `test_quantum_prefrontal.py` | ユニットテスト |

## テスト実行

```bash
# ユニットテストを実行
python test_quantum_prefrontal.py

# デモンストレーションを実行
python example_quantum_prefrontal.py

# MCPサーバーの動作確認
python frontal_engine_mcp_server.py
```

## 論文的背景

このシステムは以下の概念に基づいています：

1. **APQB（論文の核心）**
   - θ → 量子状態: `[cos(θ), sin(θ)]`
   - 相関係数: `r = cos(2θ)`
   - 温度: `T = |sin(2θ)|`
   - 制約: `r² + T² = 1`

2. **エンタングルメント**
   - 層間相互作用: `e^(l) = f_entangle(q^(l), q^(l-1))`
   - CNOTライク相互作用
   - 位相キックバック効果

3. **ハイブリッド推論**
   - Gemma層（古典的言語理解）: 70%
   - QBNN層（量子推論）: 60%
   - 重み付け統合: `score = 0.7 × quantum + 0.3 × classical`

## トラブルシューティング

### GPU メモリ不足
```python
# より小さい設定を使用
config = JudgmentConfig(
    embed_dim=512,
    num_layers=4,
    entangle_strength=0.5
)
```

### モデル読み込み失敗
```python
# ローカルモデルのパスを指定
from pathlib import Path
tokenizer_path = Path("./neuroq_tokenizer.model")
```

### 判断の信頼度が低い
```python
# より詳細なコンテキストを提供
judgment_task = {
    "context": "より詳細な背景情報を提供...",
    "judgment_request": "判断内容...",
    "criteria": {
        "criterion_1": value1,
        "criterion_2": value2
    },
    "options": ["オプション1", "オプション2"]
}
```

## 今後の拡張

### 計画中の機能
1. **マルチモーダル判断** - 画像/音声データの判断支援
2. **リアルタイム学習** - フィードバックベースのモデル改善
3. **説明可能性強化** - より詳細な推論過程の可視化
4. **分散判断** - 複数の前頭葉システムの協調判断

### 研究方向
- 量子ビットと古典ビットのさらなる融合
- 層間エンタングルメント強度の動的調整
- 判断信頼度の確率論的解釈

## ライセンス

このシステムはQubitプロジェクトの一部です。

## 参考資料

- QBNN実装: `qbnn_layered.py`
- Gemmaベース: `gemma_qbnn.py`
- NeuroQuantum: `neuroquantum_layered.py`

---

**作成日:** 2026年6月23日  
**バージョン:** 1.0.0  
**ステータス:** Production Ready ✓
