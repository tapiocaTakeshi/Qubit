# Claude Prefrontal Cortex Integration - Quick Start Guide

クラウド前頭葉統合の5分クイックスタート

---

## 概要

**QBNN Frontal Engine** を **Claude AI** の前頭葉として統合し、複雑な意思決定、リスク評価、倫理的判断を自動化します。

```
Claude AI Assistant
        ↓
Prefrontal Cortex (Decision Maker)
        ↓
QBNN Frontal Engine (Quantum-inspired Judgment)
        ↓
Yes/No Decision + Score + Reasoning
```

---

## インストール（1分）

### 1. 依存パッケージをインストール

```bash
cd /home/user/Qubit
pip install -r requirements.txt
```

### 2. MCPサーバーが起動しているか確認

```bash
# MCPサーバーのテスト
python3 test_frontal_engine_light.py
```

### 3. 統合モジュールをインポート可能か確認

```bash
python3 -c "from claude_prefrontal_integration import ClaudePrefrontalCortex; print('✓ Integration loaded successfully')"
```

---

## 基本的な使用（1分）

### パターン1: アクション安全性判断

```python
import asyncio
from claude_prefrontal_integration import judge_action

async def main():
    # ユーザーデータをログに出力する前に前頭葉に確認
    should_proceed, result = await judge_action(
        action="ユーザーのメールをログに出力",
        context="デバッグモード。ログはサーバーに保存",
        risks=["プライバシー侵害", "GDPR違反"]
    )
    
    if should_proceed:
        print("✓ 安全です。実行します。")
    else:
        print(f"✗ 危険です。理由: {result['reasoning']}")
        print(f"  スコア: {result['score']}/100")

asyncio.run(main())
```

### パターン2: 応答品質評価

```python
import asyncio
from claude_prefrontal_integration import judge_response_quality

async def main():
    response = "Pythonでファイルを読むには open() を使います。"
    
    quality = await judge_response_quality(
        response=response,
        requirements=["詳細な例", "実用的", "わかりやすい"]
    )
    
    if quality['decision'] == 'Yes':
        print(f"✓ 高品質な応答（スコア: {quality['score']}/100）")
    else:
        print(f"✗ 改善が必要（理由: {quality['reasoning']}）")

asyncio.run(main())
```

### パターン3: 倫理的懸念評価

```python
import asyncio
from claude_prefrontal_integration import check_ethics

async def main():
    ethics = await check_ethics(
        action="ユーザー行動を分析して個人を特定する",
        stakeholders=["ユーザー", "社会", "企業"]
    )
    
    print(f"倫理的適切性: {ethics['decision']}")
    print(f"スコア: {ethics['score']}/100")
    
    if ethics['decision'] == 'No':
        print(f"懸念事項: {ethics['reasoning']}")

asyncio.run(main())
```

---

## 高度な使用（2分）

### パターン4: 統合的な意思決定パイプライン

```python
import asyncio
from claude_prefrontal_integration import ClaudePrefrontalCortex

async def main():
    cortex = ClaudePrefrontalCortex()
    
    # Step 1: セキュリティチェック
    safety, safety_result = await cortex.should_proceed_with_action(
        action_description="新しい機能を実装",
        context="本番環境",
        risks=["セキュリティ", "パフォーマンス"]
    )
    
    # Step 2: 倫理チェック
    ethics = await cortex.assess_ethical_concerns(
        action_description="新しい機能を実装",
        stakeholders=["ユーザー", "社会"]
    )
    
    # Step 3: 最終判断
    if safety and ethics['score'] >= 50:
        print("✅ 実装を進めてください")
    else:
        print("❌ 懸念事項を解決してから実装してください")
    
    # Step 4: 判断履歴を確認
    history = cortex.get_judgment_history(limit=5)
    print(f"\n最近の判断: {len(history)}件")

asyncio.run(main())
```

### パターン5: タスク優先順位付け

```python
import asyncio
from claude_prefrontal_integration import ClaudePrefrontalCortex

async def main():
    cortex = ClaudePrefrontalCortex()
    
    tasks = [
        {"name": "バグ修正", "description": "本番環境でのクリティカルバグ"},
        {"name": "機能追加", "description": "新しいUIコンポーネント"},
        {"name": "ドキュメント", "description": "APIドキュメント更新"}
    ]
    
    # 優先順位付け実行
    prioritized = await cortex.prioritize_tasks(tasks)
    
    print("優先順位（高い順）:")
    for rank, (task, score) in enumerate(prioritized, 1):
        print(f"  {rank}. {task['name']} (スコア: {score:.2f})")

asyncio.run(main())
```

---

## よくある使用シナリオ

### ✅ いつ前頭葉を使うべきか

| シナリオ | メソッド | 例 |
|--------|---------|-----|
| セキュリティ決定 | `should_proceed_with_action()` | データアクセス、API呼び出し |
| 品質チェック | `evaluate_response_quality()` | AIの応答を送信前に検証 |
| 倫理評価 | `assess_ethical_concerns()` | プライバシー関連の決定 |
| リスク分析 | `make_judgment()` | 新技術導入の是非 |
| タスク管理 | `prioritize_tasks()` | 複数プロジェクトの優先順位 |

### ❌ 使わなくても良い場面

- 単純なテキスト生成
- 定型的な応答
- リスクがない決定
- 実時間性が最重要な場合

---

## 出力形式リファレンス

### 判断結果

すべての判断メソッドは以下の形式で結果を返します：

```python
{
    "decision": "Yes",              # Yes または No
    "score": 75,                    # 0-100 (100に近いほどYes寄り)
    "reasoning": "説明...",         # なぜそう判断したか
    "confidence": "high",           # high / medium / low
    "key_factors": ["要因1", ...],  # 判断に影響した要因
    "timestamp": "2026-06-23T..."   # ISO形式の時刻
}
```

### 解釈ガイド

```
Score 70+:  強い肯定的判断（Yes推奨）
Score 50-70: 弱い肯定的判断（確認後にYes）
Score 30-50: 弱い否定的判断（懸念あり）
Score 0-30: 強い否定的判断（No推奨）

Confidence:
- high: この判断は信頼できる
- medium: いくつかの不確実性がある
- low: 判断が困難。人間レビューを推奨
```

---

## トラブルシューティング

### MCPサーバーが起動しない

```bash
# 確認1: Python バージョン
python3 --version  # 3.11+ 必須

# 確認2: 依存パッケージ
pip install -r requirements.txt

# 確認3: MCPサーバーを直接実行
python3 frontal_engine_mcp_server.py

# 確認4: ログを確認
python3 frontal_engine_mcp_server.py 2>&1 | head -20
```

### 判断スコアが常に50

```
理由: QBNN モデルが読み込めていない

解決策:
1. トークナイザーファイルを確認
   ls -la neuroq_tokenizer.*

2. モデルファイルを確認
   ls -la *.pt

3. ログを確認
   python3 -c "from claude_prefrontal_integration import ClaudePrefrontalCortex; c = ClaudePrefrontalCortex()" 2>&1
```

### パーミッション拒否エラー

```bash
# ファイルのパーミッションを確認
ls -la /home/user/Qubit/

# パーミッションを修正
chmod +x /home/user/Qubit/*.py
```

---

## パフォーマンスチューニング

### メモリ使用量を削減

```python
# 判断履歴をクリア
cortex.judgment_history.clear()

# または、古い履歴を削除
cortex.judgment_history = cortex.judgment_history[-50:]
```

### 判断処理を高速化

```python
import asyncio

# 複数の判断を並列実行
tasks = [
    judge_action("action1", "context1"),
    judge_action("action2", "context2"),
    judge_action("action3", "context3")
]
results = await asyncio.gather(*tasks)
```

---

## 次のステップ

### 1. 詳細ドキュメントを読む

```bash
cat CLAUDE_PREFRONTAL_INTEGRATION.md
```

### 2. 例を実行する

```bash
python3 examples_claude_prefrontal.py
```

### 3. テストを実行する

```bash
python3 test_claude_prefrontal_integration.py
```

### 4. Claude Code に統合

`.claude/settings.json` が自動的に設定されています。
Claude Code を起動するだけで前頭葉が利用可能になります。

---

## 主要な API

### 便利関数（最も簡単）

```python
# 1行で安全性判断
should_proceed, result = await judge_action(action, context)

# 1行で品質評価
quality = await judge_response_quality(response)

# 1行で倫理評価
ethics = await check_ethics(action)
```

### クラスメソッド（より詳細な制御）

```python
cortex = ClaudePrefrontalCortex()

# カスタム判断
result = await cortex.make_judgment(
    context="...",
    judgment_request="..?",
    judgment_type=JudgmentType.SAFETY_CHECK,
    strict_mode=True
)

# 履歴確認
history = cortex.get_judgment_history()

# 説明を取得
explanation = await cortex.explain_decision(result)
```

---

## チートシート

```python
# インポート
from claude_prefrontal_integration import (
    ClaudePrefrontalCortex,
    JudgmentType,
    judge_action,
    judge_response_quality,
    check_ethics
)

# インスタンス作成
cortex = ClaudePrefrontalCortex()

# 非同期で実行
async def my_decision():
    # アクション安全性
    ok, result = await judge_action(action, context, risks)
    
    # 応答品質
    quality = await judge_response_quality(response, requirements)
    
    # 倫理確認
    ethics = await check_ethics(action, stakeholders)
    
    # 優先順位付け
    prioritized = await cortex.prioritize_tasks(tasks)
    
    # 判断履歴
    history = cortex.get_judgment_history(limit=10)
    
    # ステータス
    status = cortex.get_system_status()

# 実行
import asyncio
asyncio.run(my_decision())
```

---

## 推奨設定

### 安全性重視

```python
# 厳密モードで実行
strict_mode = True

# リスク評価タイプを使用
judgment_type = JudgmentType.RISK_ASSESSMENT

# スコア70以上で進める
if result['score'] >= 70 and result['confidence'] == 'high':
    proceed()
```

### スピード重視

```python
# 通常モードで実行
strict_mode = False

# 複数判断を並列実行
results = await asyncio.gather(
    judge_action(...),
    judge_response_quality(...),
    check_ethics(...)
)
```

---

## 次のリソース

- 詳細ドキュメント: `CLAUDE_PREFRONTAL_INTEGRATION.md`
- 実装例: `examples_claude_prefrontal.py`
- テストスイート: `test_claude_prefrontal_integration.py`
- QBNN理論: `README.md`
- MCPサーバー: `FRONTAL_ENGINE_README.md`

---

**Happy Decision Making! 🧠⚡**

バージョン: 1.0.0  
最終更新: 2026年6月23日
