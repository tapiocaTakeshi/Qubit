# ChatGPT・Gemini統合ガイド

QBNN Frontal Engine を OpenAI ChatGPT および Google Gemini と統合するための完全ガイドです。

---

## 統合アーキテクチャ

```
┌─────────────────────────────────────────────┐
│   QBNN Frontal Engine Judge Core            │
│   (frontal_engine_mcp_server.py)            │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┐
    │          │          │          │
    v          v          v          v
┌──────┐  ┌────────┐ ┌──────────┐ ┌────────┐
│MCP   │  │REST    │ │ChatGPT   │ │Gemini  │
│      │  │API     │ │Function  │ │Tool    │
└──────┘  └────────┘ │Calling   │ │Calling │
   │         │       └──────────┘ └────────┘
   │         │            │            │
   v         v            v            v
┌──────────────────────────────────────────────┐
│  Usage: Claude Desktop, Cursor,              │
│         Web/Mobile, ChatGPT, Gemini         │
└──────────────────────────────────────────────┘
```

---

## 1. 前提条件

### 必須パッケージ

```bash
# REST API サーバー用
pip install fastapi uvicorn

# ChatGPT 統合用
pip install openai>=1.0.0

# Gemini 統合用
pip install google-generativeai
```

または requirements.txt を更新：

```bash
pip install -r requirements.txt
```

### APIキー

**ChatGPT:**
- OpenAI API キーが必要（https://platform.openai.com/account/api-keys）
- 環境変数 `OPENAI_API_KEY` に設定

**Gemini:**
- Google API キーが必要（https://makersuite.google.com/app/apikey）
- 環境変数 `GOOGLE_API_KEY` に設定

---

## 2. REST API サーバー（推奨）

すべてのプラットフォームから使用可能な REST API を提供します。

### 起動方法

```bash
python frontal_engine_api.py
```

サーバーは `http://localhost:8000` で起動します。

### API エンドポイント

#### ヘルスチェック
```bash
GET /health
```

**レスポンス:**
```json
{
  "status": "healthy",
  "service": "QBNN Frontal Engine",
  "timestamp": "2026-06-22T23:00:00.000000Z"
}
```

#### 単一判断
```bash
POST /judge
Content-Type: application/json

{
  "context": "プロジェクトは予定通り進行しており、品質基準をすべて満たしています。",
  "judgment_request": "このプロジェクトをリリースしても安全か？",
  "criteria": {"quality": "high", "risk": "low"},
  "options": null,
  "strict_mode": false
}
```

**レスポンス:**
```json
{
  "decision": "Yes",
  "score": 82,
  "reasoning": "指定された基準と文脈に基づいて、肯定的な判断が支持されています。",
  "confidence": "high",
  "key_factors": ["基準マッチ度: 81.0%"],
  "timestamp": "2026-06-22T23:00:00.000000Z"
}
```

#### バッチ判断
```bash
POST /judge/batch
Content-Type: application/json

[
  {"context": "...", "judgment_request": "..."},
  {"context": "...", "judgment_request": "..."}
]
```

### cURL 例

```bash
# 単一判断
curl -X POST http://localhost:8000/judge \
  -H "Content-Type: application/json" \
  -d '{
    "context": "プロジェクトは順調に進行しています。",
    "judgment_request": "リリースできるか？"
  }'

# ヘルスチェック
curl http://localhost:8000/health

# API ドキュメント
open http://localhost:8000/docs
```

### Docker 化（オプション）

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "frontal_engine_api.py"]
```

---

## 3. ChatGPT 統合

OpenAI Function Calling を使用した統合。

### セットアップ

```bash
# 環境変数を設定
export OPENAI_API_KEY="your-api-key-here"

# またはコードで指定
from chatgpt_integration import ChatGPTFrontalEngine
engine = ChatGPTFrontalEngine(api_key="your-api-key-here")
```

### Python コード例

```python
from chatgpt_integration import ChatGPTFrontalEngine

# エンジン初期化
engine = ChatGPTFrontalEngine()

# チャット実行
response = engine.chat(
    "新規プロジェクトの開始を検討しています。"
    "背景: 予算十分、チーム経験豊富、市場需要高い。"
    "開始すべきですか？"
)

print(response)
```

### 自動ツール呼び出し例

ユーザーが判断に関する質問をすると、ChatGPT が自動的に Judge ツールを呼び出します：

```
ユーザー: 「新規プロジェクトの開始を検討しています。背景: 予算十分、チーム経験豊富。」

ChatGPT: [Judge ツール自動呼び出し]
  - context: "新規プロジェクト、予算十分、チーム経験豊富"
  - judgment_request: "プロジェクト開始は適切か？"

Judge 結果:
  - decision: "Yes"
  - score: 78
  - reasoning: "ポジティブ要因が多い"

ChatGPT: 「判断結果に基づいて、このプロジェクトの開始はお勧めできます。
スコアは78点で、以下の理由があります：...」
```

### CLI デモ

```bash
python chatgpt_integration.py
```

---

## 4. Gemini 統合

Google Generative AI を使用した統合。

### セットアップ

```bash
# 環境変数を設定
export GOOGLE_API_KEY="your-api-key-here"

# またはコードで指定
from gemini_integration import GeminiFrontalEngine
engine = GeminiFrontalEngine(api_key="your-api-key-here")
```

### Python コード例

```python
from gemini_integration import GeminiFrontalEngine

# エンジン初期化
engine = GeminiFrontalEngine(model="gemini-2.0-flash")

# チャット実行
response = engine.chat(
    "3つのベンダー候補があります。\n"
    "ベンダーA: 安い、サポート弱い\n"
    "ベンダーB: 中程度、サポート強い\n"
    "ベンダーC: 高い、実績豊富\n"
    "どれが最適ですか？"
)

print(response)
```

### サポートモデル

- `gemini-2.0-flash` （推奨・高速）
- `gemini-2.0-flash-001`
- `gemini-1.5-pro`

### CLI デモ

```bash
python gemini_integration.py
```

---

## 5. 使用例

### シナリオ 1: 意思決定支援

```python
from chatgpt_integration import ChatGPTFrontalEngine

engine = ChatGPTFrontalEngine()

message = """
新規市場進出の判断をしてください。
背景:
- 市場成長率: 年30%
- 競合状況: 中程度
- 初期投資: ¥5000万
- 予想ROI: 年¥10000万
"""

response = engine.chat(message)
print(response)
```

### シナリオ 2: リスク評価

```python
from gemini_integration import GeminiFrontalEngine

engine = GeminiFrontalEngine()

message = """
新技術導入のリスクを評価してください。
リスク要因:
- 学習曲線が急
- 既存システムとの互換性問題
- 短期的な生産性低下

メリット:
- 長期的効率性30%向上
"""

response = engine.chat(message)
print(response)
```

### シナリオ 3: REST API から呼び出し

```bash
# Node.js / JavaScript
const response = await fetch('http://localhost:8000/judge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    context: "プロジェクト品質: 95%テストカバレッジ、ドキュメント完全",
    judgment_request: "本番環境への展開は適切か？",
    strict_mode: true
  })
});

const result = await response.json();
console.log(result);
```

---

## 6. トラブルシューティング

### ChatGPT 統合

**エラー: "openai module not found"**
```bash
pip install openai>=1.0.0
```

**エラー: "Invalid API key"**
```bash
export OPENAI_API_KEY="sk-..."
# または chatgpt_integration.py で api_key を指定
```

**ツールが呼び出されない**
- モデルが `gpt-4` 以上か確認
- Function calling は gpt-3.5-turbo では動作しません

### Gemini 統合

**エラー: "google.generativeai module not found"**
```bash
pip install google-generativeai
```

**エラー: "Invalid API key"**
```bash
export GOOGLE_API_KEY="..."
# または gemini_integration.py で api_key を指定
```

**ツール呼び出しエラー**
- Gemini モデルが Tool Use をサポートしているか確認
- `gemini-2.0-flash` 以上を推奨

### REST API

**ポート 8000 が既に使用中**
```bash
# 別のポートで起動
uvicorn frontal_engine_api:create_app --port 8001
```

**CORS エラー**
REST API に CORS ミドルウェアを追加:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

## 7. 本番環境への展開

### 環境変数管理

```bash
# .env ファイル
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
PYTHON_ENV=production
```

### Docker Compose

```yaml
version: '3.8'
services:
  frontal-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    restart: unless-stopped
```

### レート制限対策

```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

---

## 8. 参考リンク

- [OpenAI API ドキュメント](https://platform.openai.com/docs/api-reference)
- [Google Generative AI ドキュメント](https://ai.google.dev/docs)
- [FastAPI ドキュメント](https://fastapi.tiangolo.com/)
- FRONTAL_ENGINE_README.md
- MCP_INTEGRATION_GUIDE.md
