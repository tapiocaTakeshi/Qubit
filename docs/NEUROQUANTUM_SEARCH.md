# NeuroQuantum 検索機能 (Search / RAG)

NeuroQuantum に「文書検索」と「検索拡張生成 (RAG)」を追加する機能です。
登録した文書をキーワード (BM25) と NeuroQuantum モデルの埋め込み (Dense) の
ハイブリッドでランキングし、必要に応じて検索結果をプロンプトに埋め込んで
生成します。

| 構成要素 | 場所 | 説明 |
| --- | --- | --- |
| `NeuroQuantumSearchIndex` | `neuroquantum_search.py` | 検索インデックス本体 (BM25 + Dense) |
| `NeuroQuantumAI.search()` / `generate_with_search()` | `neuroquantum_layered.py` | Python からの検索 / RAG |
| `/search`, `/search/documents`, `/search/status`, `/inference` (`use_search`) | `api.py` (FastAPI) | 学習済みモデルを使うサーバー |
| `/api/v1/search`, `/api/v1/search/documents`, `/api/v1/search/status` | `neuroquantum_api_server.py` (Flask) | 判定エンジン用サーバー (BM25) |
| `NeuroQuantumClient.search()` など | `npm/src/client.ts` | TypeScript クライアント |

---

## 1. 仕組み

1. **BM25 (語彙一致)** - 純 Python 実装。Unicode NFKC 正規化 + 小文字化のうえ、
   英数字は単語、日本語などは 1 文字 + 文字 bigram でトークン化します。
   形態素解析器なしで日本語のキーワード検索ができ、`torch` も不要です。
2. **Dense (意味的類似)** - NeuroQuantum モデルの最終 LayerNorm 出力を
   forward hook で取り出し、パディングを除いて平均プーリング → L2 正規化した
   文ベクトルのコサイン類似度を使います。モデル本体は変更しません。
3. **ハイブリッド** - `score = alpha * BM25(正規化) + (1 - alpha) * Dense`
   (既定 `alpha = 0.5`)。モデルが無い場合は自動的に BM25 のみになります。

モード:

| mode | 説明 |
| --- | --- |
| `hybrid` | 既定 (モデルがある場合)。BM25 + Dense |
| `bm25` | キーワード一致のみ (モデル不要) |
| `dense` | 埋め込み類似度のみ (モデル必須) |

---

## 2. Python から使う

### インデックス単体

```python
from neuroquantum_search import NeuroQuantumSearchIndex, build_rag_prompt

index = NeuroQuantumSearchIndex()                  # BM25 のみ
# index = NeuroQuantumSearchIndex(model, tokenizer)  # ハイブリッド

index.add_documents(
    [
        "量子コンピュータは量子力学の原理を利用した計算機です。",
        "ニューラルネットワークは脳の神経細胞を模倣した計算モデルです。",
    ],
    doc_ids=["quantum", "nn"],
    metadatas=[{"topic": "quantum"}, {"topic": "ai"}],
)

for hit in index.search("量子力学", top_k=3):
    print(hit.rank, hit.score, hit.doc_id, hit.text)

# メタデータでの絞り込み・しきい値
index.search("計算", metadata_filter={"topic": "ai"}, min_score=0.1)

# 永続化 (埋め込みも保存されるので再計算不要)
index.save("search_index.json")
index = NeuroQuantumSearchIndex.from_file("search_index.json", model, tokenizer)

# RAG 用プロンプト
prompt = build_rag_prompt("量子コンピュータとは？", index.search("量子コンピュータとは？"))
```

`SearchResult` のフィールド: `doc_id`, `text`, `score` (0-1), `bm25_score`,
`dense_score`, `metadata`, `rank`。

### NeuroQuantumAI と組み合わせる

```python
from neuroquantum_layered import NeuroQuantumAI

ai = NeuroQuantumAI(embed_dim=64, hidden_dim=128, num_layers=2)
ai.train(texts, epochs=5)

ai.add_documents(texts)                       # 学習済みモデルの埋め込みで索引化
hits = ai.search("量子もつれ", top_k=3)

# 検索結果をプロンプトに埋め込んで生成 (RAG)
answer, hits = ai.generate_with_search(
    "量子もつれとは何ですか？", top_k=3, return_results=True, max_length=80
)
```

`train()` を実行するとモデルが差し替わるため、登録済み文書は新しいモデルの
埋め込みで自動的に索引し直されます。

### チャットモードのコマンド

`NeuroQuantumAI.chat()` に以下のコマンドが追加されています。

| コマンド | 説明 |
| --- | --- |
| `/search <語>` | 登録文書を検索して上位 5 件を表示 |
| `/index <file>` | `.txt` (空行区切り) / `.json` / `.jsonl` を文書として登録 |
| `/rag on\|off` | 生成時に検索結果をプロンプトへ含める |

---

## 3. REST API (`api.py`, FastAPI)

学習済みチェックポイントを読み込むサーバーです。モデルが読み込めれば
ハイブリッド検索、読み込めなければ BM25 にフォールバックします。

```bash
# 文書登録 (同じ id を再登録すると置き換え)
curl -X POST http://localhost:8000/search/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": [
        {"text": "量子コンピュータは量子力学の原理を利用した計算機です。", "id": "q1", "metadata": {"topic": "quantum"}},
        {"text": "ニューラルネットワークは脳の神経細胞を模倣した計算モデルです。"}
      ]}'

# 検索
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "量子力学", "top_k": 3, "mode": "hybrid"}'

# 状態
curl http://localhost:8000/search/status

# 削除 (doc_id 指定で 1 件、無指定で全件)
curl -X DELETE "http://localhost:8000/search/documents?doc_id=q1"
curl -X DELETE http://localhost:8000/search/documents

# 検索拡張生成 (RAG)
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "量子コンピュータとは？", "use_search": true, "search_top_k": 3, "max_new_tokens": 100}'
```

`/inference` に `use_search: true` を付けると、`search_results` に使用した
文書が含まれ、`generated_text` は検索結果を踏まえたプロンプトから生成されます。

### `/search` リクエスト

| フィールド | 型 | 既定 | 説明 |
| --- | --- | --- | --- |
| `query` | string | 必須 | 検索クエリ |
| `top_k` | int | 5 | 返す件数 |
| `mode` | `hybrid` / `bm25` / `dense` | インデックス既定 | スコアリング方式 |
| `min_score` | float | 0 | このスコア未満を除外 |
| `metadata_filter` | object | なし | 全キーが一致する文書のみ対象 |

---

## 4. REST API (`neuroquantum_api_server.py`, Flask)

判定エンジン用サーバーにも同じ操作を `/api/v1/search*` として追加しています
(モデルを読み込まないため BM25 のみ)。`include_prompt: true` を付けると
RAG 用プロンプト `rag_prompt` も返します。

```bash
curl -X POST http://localhost:5000/api/v1/search/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": ["量子コンピュータの説明", {"text": "AI の説明", "id": "ai"}]}'

curl -X POST http://localhost:5000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "量子", "top_k": 3, "include_prompt": true}'
```

---

## 5. TypeScript クライアント

```ts
import { NeuroQuantumClient } from "qubit_ai";

const client = new NeuroQuantumClient({
  endpointUrl: "http://localhost:8000/inference",
  searchEndpointUrl: "http://localhost:8000", // /search, /search/documents を付加
});

await client.addDocuments([
  "量子コンピュータは量子力学の原理を利用した計算機です。",
  { text: "ニューラルネットワークの説明", id: "nn", metadata: { topic: "ai" } },
]);

const result = await client.search("量子力学", { topK: 3, mode: "hybrid" });
for (const hit of result.results) {
  console.log(hit.rank, hit.score, hit.docId, hit.text);
}

await client.searchStatus();       // { documents, mode, alpha, denseAvailable, ngram }
await client.clearDocuments("nn"); // 1 件削除
await client.clearDocuments();     // 全件削除
```

---

## 6. テスト

```bash
python -m pytest -q tests/test_neuroquantum_search.py   # BM25 / ハイブリッド / RAG / 永続化
cd npm && npm test                                       # TypeScript クライアント
```
