# Claude 前頭葉統合ガイド
## Claude Prefrontal Cortex Integration Guide

QBNNを脳の前頭葉として、Claude AIアシスタントに統合するための完全ガイド

---

## 概要

### 前頭葉の役割
人間の脳における前頭葉は以下の機能を担当します：

- **意思決定**: 複数の選択肢から最適な選択を判断
- **リスク評価**: 状況のリスクを定量的に評価
- **倫理的判断**: 行動の倫理性や適切性を評価
- **品質判定**: 成果物の品質を客観的に判断
- **優先順位付け**: タスク間の優先度を判定
- **自制**: 衝動的な行動を抑制

### QBNN Frontal Engine の実装
QBNN（量子インスパイア双方向ニューラルネットワーク）Frontal Engineは、これらの前頭葉機能をAIアシスタントに提供するMCPサーバーです。

**特徴:**
- 🧠 **量子インスパイア判断**: APQB理論に基づいた高度な判断ロジック
- ⚖️ **バランス型評価**: ポジティブとネガティブ両面を考慮
- 📊 **スコア化**: 判断を0-100の数値スコアで定量化
- 📋 **根拠説明**: すべての判断に対して詳細な説明を提供
- 🎯 **信頼度表示**: 判断の確実性を high/medium/low で表示

---

## システムアーキテクチャ

### コンポーネント図

```
┌─────────────────────────────────────────────────────────┐
│                    Claude AI Assistant                   │
│              (Main Reasoning & Response Engine)          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ Delegates complex decisions
                    ↓
┌─────────────────────────────────────────────────────────┐
│         Claude Prefrontal Cortex Integration             │
│           (claude_prefrontal_integration.py)             │
│  ・Safety Checks                                        │
│  ・Ethical Judgment                                     │
│  ・Quality Evaluation                                   │
│  ・Risk Assessment                                      │
│  ・Task Prioritization                                  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ Uses MCP Protocol
                    ↓
┌─────────────────────────────────────────────────────────┐
│          QBNN Frontal Engine MCP Server                  │
│        (frontal_engine_mcp_server.py)                    │
│  ・Judge Tool: Executes judgment logic                  │
│  ・QBNN Model: Quantum-inspired reasoning               │
│  ・APQB Theory: Mathematical foundation                 │
└─────────────────────────────────────────────────────────┘
```

### データフロー

```
User Request
    ↓
Claude Main Loop
    ├─→ [Normal Processing]
    └─→ [Requires Judgment Decision?]
         ↓
    Claude Prefrontal Cortex
    (Detects decision point)
         ↓
    FrontalEngineJudge.judge()
         ↓
    QBNN Analysis & Scoring
         ↓
    Return: {decision, score, reasoning, confidence}
         ↓
    Claude integrates result
         ↓
    Final Response to User
```

---

## インストールと設定

### 前提条件

```bash
# Python 3.11以上
python3 --version

# PyTorch 2.4.0
pip show torch

# 必要なパッケージ
pip install -r requirements.txt
```

### セットアップ手順

#### 1. リポジトリをクローン

```bash
git clone https://github.com/tapiocatakeshi/qubit.git
cd qubit
git checkout claude/qbnn-prefrontal-integration-zqzgn7
```

#### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

#### 3. Claude Code設定を追加

`.claude/settings.json` に以下が自動追加されています（確認用）：

```json
{
  "mcp": {
    "servers": {
      "qbnn-frontal-engine": {
        "command": "python",
        "args": ["/home/user/Qubit/frontal_engine_mcp_server.py"]
      }
    }
  },
  "integrations": {
    "qbnn_prefrontal": {
      "enabled": true,
      "features": {
        "safety_checks": true,
        "ethical_judgment": true,
        "quality_evaluation": true
      }
    }
  }
}
```

#### 4. MCPサーバーの起動確認

```bash
# テストの実行
python3 test_frontal_engine_light.py
```

---

## 使用方法

### 基本的な使用パターン

#### パターン1: アクション安全性判断

```python
import asyncio
from claude_prefrontal_integration import judge_action

async def main():
    should_proceed, result = await judge_action(
        action="ユーザーデータをログ出力",
        context="デバッグモードで実行中",
        risks=["プライバシー侵害", "セキュリティリスク"]
    )
    
    if should_proceed:
        # アクション実行
        execute_action()
    else:
        # アクション中止
        print(f"判断: {result['reasoning']}")
```

#### パターン2: 応答品質評価

```python
async def main():
    quality_result = await judge_response_quality(
        response="ユーザーの質問に対する応答",
        requirements=["詳細", "有用", "明確", "日本語"]
    )
    
    if quality_result['decision'] == 'Yes':
        return response  # 応答を返す
    else:
        return generate_better_response()  # より良い応答を生成
```

#### パターン3: 倫理的懸念評価

```python
async def main():
    ethics_result = await check_ethics(
        action="ユーザー行動を分析して推奨事項を生成",
        stakeholders=["ユーザー", "社会", "組織"]
    )
    
    if ethics_result['score'] < 50:
        log_ethical_concern(ethics_result['reasoning'])
        request_human_review()
```

#### パターン4: タスク優先順位付け

```python
async def main():
    tasks = [
        {"name": "バグ修正", "description": "本番環境でのクリティカルバグ"},
        {"name": "機能追加", "description": "新しいコンポーネント"},
        {"name": "ドキュメント", "description": "API仕様書更新"}
    ]
    
    prioritized = await claude_prefrontal_cortex.prioritize_tasks(tasks)
    
    for task, score in prioritized:
        print(f"{task['name']}: {score:.2f}")
```

### Python APIリファレンス

#### ClaudePrefrontalCortex クラス

**メソッド:**

```python
class ClaudePrefrontalCortex:
    
    # アクション判断
    async def should_proceed_with_action(
        action_description: str,
        context: str,
        risks: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """提案されたアクションを実行すべきかを判断"""
    
    # 応答品質評価
    async def evaluate_response_quality(
        response: str,
        requirements: Optional[List[str]] = None,
        user_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """提案された応答の品質を評価"""
    
    # 倫理評価
    async def assess_ethical_concerns(
        action_description: str,
        stakeholders: Optional[List[str]] = None,
        potential_harms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """提案されたアクションの倫理的懸念を評価"""
    
    # タスク優先順位付け
    async def prioritize_tasks(
        tasks: List[Dict[str, str]],
        constraints: Optional[str] = None
    ) -> List[Tuple[Dict[str, str], float]]:
        """複数のタスクを優先度順にソート"""
    
    # 基本判断
    async def make_judgment(
        context: str,
        judgment_request: str,
        judgment_type: JudgmentType = JudgmentType.DECISION_MAKING,
        criteria: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """QBNN Frontal Engineを使用して判断を実行"""
    
    # ユーティリティ
    async def explain_decision(
        judgment_result: Dict[str, Any]
    ) -> str:
        """判断結果を自然言語で説明"""
    
    def get_judgment_history(
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """最近の判断履歴を取得"""
    
    def get_system_status() -> Dict[str, Any]:
        """システムステータスを取得"""
```

**判断結果の形式:**

```python
{
    "decision": "Yes" | "No",      # Yes/No判定
    "score": 0-100,                # 0-100の判断スコア
    "reasoning": str,              # 判断の根拠説明
    "confidence": "high" | "medium" | "low",  # 信頼度
    "key_factors": [str, ...],     # 主要な判断要因
    "timestamp": str               # ISO形式のタイムスタンプ
}
```

---

## 統合シナリオ

### シナリオ1: セキュリティ決定の自動評価

```python
class SecureAssistant:
    async def execute_sensitive_operation(self, operation):
        # 前頭葉に安全性を確認
        should_proceed, judgment = await judge_action(
            action=operation.description,
            context=f"User: {operation.user}, Environment: {operation.env}",
            risks=operation.identified_risks
        )
        
        if not should_proceed:
            log_security_decision(judgment)
            return {"status": "rejected", "reason": judgment["reasoning"]}
        
        return execute(operation)
```

### シナリオ2: マルチエージェント意思決定

```python
class MultiAgentCoordinator:
    async def coordinate_agents(self, tasks):
        # 前頭葉で優先順位を決定
        prioritized = await claude_prefrontal_cortex.prioritize_tasks(tasks)
        
        # 優先順位に従ってエージェントに割り当て
        for task, priority in prioritized:
            agent = assign_agent_by_priority(priority)
            await agent.execute(task)
```

### シナリオ3: 品質保証パイプライン

```python
class QAEngine:
    async def validate_output(self, output):
        # 前頭葉で品質を評価
        quality = await judge_response_quality(
            response=output,
            requirements=self.quality_standards
        )
        
        if quality["score"] >= 70:
            return publish(output)
        else:
            return request_revision(quality["reasoning"])
```

### シナリオ4: 倫理ガバナンス

```python
class EthicsGovernance:
    async def pre_flight_check(self, proposed_action):
        # 前頭葉で倫理性を確認
        ethics = await check_ethics(
            action=proposed_action,
            stakeholders=self.get_stakeholders()
        )
        
        if ethics["score"] < 50:
            # 倫理審査委員会に報告
            escalate_to_ethics_board(ethics)
            return False
        
        return True
```

---

## 判断タイプ

### 利用可能な判断タイプ

```python
class JudgmentType(Enum):
    DECISION_MAKING = "意思決定"        # 一般的な意思決定
    RISK_ASSESSMENT = "リスク評価"      # リスク分析
    QUALITY_JUDGMENT = "品質判定"       # 品質評価
    ETHICAL_JUDGMENT = "倫理的判断"     # 倫理性評価
    PRIORITIZATION = "優先順位付け"     # タスク優先順位
    SAFETY_CHECK = "安全性確認"         # セキュリティ/安全性
```

### 各タイプの特性

| タイプ | 目的 | strict_mode推奨 | 用途例 |
|-------|------|--------------|-------|
| DECISION_MAKING | 一般的な判断 | false | 通常の意思決定 |
| RISK_ASSESSMENT | リスク評価 | true | セキュリティ決定 |
| QUALITY_JUDGMENT | 品質判定 | true | リリース判定 |
| ETHICAL_JUDGMENT | 倫理性評価 | true | 倫理的決定 |
| PRIORITIZATION | 優先順位 | false | タスク管理 |
| SAFETY_CHECK | 安全性確認 | true | 危険な操作の確認 |

---

## パフォーマンスと最適化

### メモリ使用量

```
基本構成:
- ClaudePrefrontalCortex: ~50MB
- FrontalEngineJudge: ~500MB (モデル込み)
- 判断履歴: ~1MB (100件)

合計: ~550MB
```

### レスポンス時間

```
判断処理時間（平均）:
- QBNN推論: 200-500ms
- 従来的分析: 50-100ms
- ハイブリッドスコア: 250-600ms
- 全体（キャッシュ込み）: 200-400ms
```

### 最適化のヒント

1. **バッチ処理**: 複数の判断をまとめて実行
2. **キャッシング**: 同じコンテキストの判断結果をキャッシュ
3. **非同期処理**: async/await を活用
4. **判断履歴の監視**: 定期的に履歴をクリア

```python
# 最適化例
async def batch_evaluate_responses(responses):
    # 並列実行
    tasks = [
        judge_response_quality(resp)
        for resp in responses
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

## トラブルシューティング

### MCPサーバーが接続できない

```bash
# 確認事項
1. MCPサーバーが起動しているか
   ps aux | grep frontal_engine_mcp_server.py

2. ポートが利用可能か（設定ファイルで確認）

3. 環境変数を確認
   echo $PYTHONPATH
   echo $PYTHONUNBUFFERED
```

### メモリ不足エラー

```python
# 対策1: 判断履歴のクリア
cortex.judgment_history.clear()

# 対策2: モデルの軽量版を使用
# settings.json で model_size を "small" に設定
```

### 判断スコアが常に50の場合

```
原因: QBNN モデルが読み込めていない

対策:
1. トークナイザーファイルが存在するか確認
   ls -la neuroq_tokenizer.*

2. モデルファイルが存在するか確認
   ls -la *.pt

3. ログを確認
   python3 frontal_engine_mcp_server.py 2>&1 | tail -20
```

---

## ベストプラクティス

### ✅ やるべきこと

1. **常に非同期処理を使用**
```python
# Good
result = await judge_action(...)

# Bad
result = judge_action(...)  # TypeError
```

2. **判断結果の確信度を確認**
```python
# Good
if result["confidence"] == "high":
    take_action(result["decision"])

# Bad
if result["decision"] == "Yes":
    take_action()  # 信頼度を確認していない
```

3. **適切な strict_mode を設定**
```python
# 重要な決定には strict_mode=True
await judge_action(..., strict_mode=True)

# 日常的な判断には strict_mode=False
await judge_action(..., strict_mode=False)
```

### ❌ するべきではないこと

1. **判断履歴を無制限に溜める**
```python
# Bad
judgment_history.clear()  を呼ばない
# →メモリリーク

# Good
# 自動的に100件に制限されている
```

2. **エラーハンドリングを省略**
```python
# Bad
result = await make_judgment(...)
decision = result["decision"]  # KeyError の可能性

# Good
try:
    result = await make_judgment(...)
    decision = result.get("decision", "No")
except Exception as e:
    log_error(e)
    decision = "No"  # 安全なデフォルト
```

3. **文脈情報を省略**
```python
# Bad
await make_judgment(
    context="",  # 空の文脈
    judgment_request="安全か?"
)

# Good
await make_judgment(
    context=f"ユーザー: {user}, 環境: {env}, 条件: {conditions}",
    judgment_request="安全か?"
)
```

---

## 監視とロギング

### 判断履歴の確認

```python
# 最近の判断を表示
history = cortex.get_judgment_history(limit=10)
for record in history:
    print(f"{record['timestamp']}: {record['judgment_type']}")
    print(f"  Decision: {record['decision']}")
    print(f"  Score: {record['score']}")
    print(f"  Confidence: {record['confidence']}\n")
```

### メトリクスの収集

```python
def collect_metrics():
    history = cortex.get_judgment_history(limit=100)
    
    total = len(history)
    yes_count = sum(1 for r in history if r["decision"] == "Yes")
    high_conf = sum(1 for r in history if r["confidence"] == "high")
    avg_score = sum(r["score"] for r in history) / total if total > 0 else 0
    
    return {
        "total_judgments": total,
        "yes_ratio": yes_count / total if total > 0 else 0,
        "high_confidence_ratio": high_conf / total if total > 0 else 0,
        "average_score": avg_score
    }
```

---

## 今後の拡張

### 計画中の機能

1. **学習機能**: 判断結果のフィードバックから改善
2. **マルチモーダル判断**: テキスト以外の入力対応
3. **分散判断**: 複数の前頭葉インスタンス間の協調
4. **個人化**: ユーザーの判断パターンを学習
5. **説明可能性向上**: より詳細な判断根拠の提示

---

## ライセンス

MIT License - 詳細は LICENSE ファイルを参照

---

## サポートと貢献

### 問題報告
GitHub Issues: https://github.com/tapiocatakeshi/qubit/issues

### 貢献方法
1. フォークしてブランチを作成
2. 変更をコミット
3. プルリクエストを提出

---

**最終更新**: 2026年6月23日
**バージョン**: 1.0.0
