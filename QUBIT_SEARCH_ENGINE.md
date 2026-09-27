# Qubit Search Engine

Algolia を使わずに riddle アプリの検索を提供する、Qubit 製の全文検索サーバーです。
`qubit_search_engine.py` 1 ファイル + `neuroquantum_search.py` (トークナイザー) だけで動き、
Python 標準ライブラリ以外の依存はありません。

## Algolia から置き換わる機能

| Algolia | Qubit Search Engine |
| --- | --- |
| Search (`HitsSearcher`) | `POST /1/indexes/{index}/query` |
| Recommend (Looking Similar) | `POST /1/indexes/{index}/recommend` (TF-IDF コサイン類似度) |
| Insights (click / conversion) | `POST /1/events` (人気度として順位に反映) |
| Firestore Algolia Extension | riddle の Cloud Functions `syncRiddlesToQubitSearch` / `syncChannelsToQubitSearch` |

## 検索の仕組み

- **トークン化**: NFKC 正規化 + 小文字化のうえ、英数字は単語、日本語などは
  「1 文字 + 文字 bigram」。形態素解析なしで「謎解き」が「謎解」「解き」どちらにも一致します。
- **採点**: `searchableAttributes` の先頭ほど重い属性重み付き BM25F。
  - クエリ語の被覆率が `minCoverage` (既定 0.5) 未満の文書は除外 (1 文字だけ一致するようなノイズを防止)
  - クエリ文字列がそのまま含まれる文書は 1.5 倍
  - 人気度 (イベント数) で `1 + popularityWeight * log(1 + 人気度)` 倍
  - 同点は `customRanking` (例: `desc(playCount)`) で並べ替え
- **入力途中の検索**: 最後の英数字の単語は前方一致で展開 (`quan` → `quantum`)。
- **空クエリ**: 全件を `customRanking` 順に返します。
- **増分更新**: 転置インデックスを 1 件単位で更新するので、Firestore のトリガーから都度反映できます。

## 起動

```bash
export QUBIT_SEARCH_ADMIN_KEY=$(openssl rand -hex 24)   # 書き込み用 (サーバー側だけに置く)
export QUBIT_SEARCH_API_KEY=$(openssl rand -hex 16)     # 検索用 (アプリに埋め込む)
python qubit_search_engine.py serve --port 8080 --data-dir ./search_data
```

Docker:

```bash
docker build -f docker/Dockerfile.search -t qubit-search .
docker run -p 8080:8080 -v qubit-search-data:/data \
  -e QUBIT_SEARCH_ADMIN_KEY=... -e QUBIT_SEARCH_API_KEY=... qubit-search
```

データは `--data-dir` にインデックスごとの JSON (`Riddles.json` など) として 2 秒ごとに
アトミックに書き出され、終了時 (SIGTERM を含む) にも保存されます。Cloud Run などで動かす場合は
永続ボリュームをマウントし、**インスタンス数を 1 に固定**してください (インデックスはプロセス内メモリにあります)。

| 環境変数 | 説明 |
| --- | --- |
| `QUBIT_SEARCH_ADMIN_KEY` | 書き込み API のキー。未設定だと誰でも書き込める開発モード |
| `QUBIT_SEARCH_API_KEY` | 検索 API のキー。未設定なら検索は誰でも可能 |
| `QUBIT_SEARCH_DATA_DIR` | 保存先 (既定 `./search_data`) |
| `PORT` / `QUBIT_SEARCH_PORT` | ポート (既定 8080) |
| `QUBIT_SEARCH_CORS_ORIGIN` | CORS の許可オリジン (既定 `*`) |

キーは `X-Qubit-Search-Key` ヘッダーか `Authorization: Bearer ...` で渡します。

## API

```bash
# 検索
curl -X POST $URL/1/indexes/Riddles/query -H "X-Qubit-Search-Key: $KEY" \
  -d '{"query": "謎解き", "hitsPerPage": 20, "page": 0, "filters": "NOT contentType:game"}'

# 似ている謎解き
curl -X POST $URL/1/indexes/Riddles/recommend -H "X-Qubit-Search-Key: $KEY" \
  -d '{"objectID": "abc", "maxRecommendations": 5}'

# クリック / コンバージョン
curl -X POST $URL/1/events -H "X-Qubit-Search-Key: $KEY" \
  -d '{"events": [{"eventType": "conversion", "index": "Riddles", "objectIDs": ["abc"]}]}'

# 登録・更新・削除 (admin)
curl -X PUT    $URL/1/indexes/Riddles/abc -H "X-Qubit-Search-Key: $ADMIN" -d '{"title": "量子の謎"}'
curl -X PATCH  $URL/1/indexes/Riddles/abc -H "X-Qubit-Search-Key: $ADMIN" -d '{"playCount": 10}'
curl -X DELETE $URL/1/indexes/Riddles/abc -H "X-Qubit-Search-Key: $ADMIN"
curl -X POST   $URL/1/indexes/Riddles/batch -H "X-Qubit-Search-Key: $ADMIN" \
  -d '{"requests": [{"action": "addObject", "body": {"objectID": "a", "title": "..."}}]}'

# 設定 (admin)
curl -X PUT $URL/1/indexes/Riddles/settings -H "X-Qubit-Search-Key: $ADMIN" \
  -d '{"searchableAttributes": ["title", "tags", "description"], "customRanking": ["desc(playCount)"]}'
```

検索レスポンスは Algolia と同じ形 (`hits` / `nbHits` / `page` / `nbPages` / `hitsPerPage` / `query`) です。
各 hit には `_rankingInfo` (`score`, `coverage`, `position`) が付きます。

### フィルタ

`filters` は `attr:value` / `attr=数値` / `<` `<=` `>` `>=` `!=` / `NOT` / `AND` / `OR` (括弧なし、AND 優先)。
`facetFilters` は `["category:なぞなぞ", ["tags:脱出", "tags:初心者"]]` (外側 AND、内側 OR)。

### 既定の設定

| インデックス | searchableAttributes | customRanking | 返さない属性 |
| --- | --- | --- | --- |
| `Riddles` | title, tags, category, authorName, description, text, content | playCount, answerCount, likes, createdAt (降順) | answer, answers, correctAnswers, hints |
| `Channels` | displayName, name, channelName, bio, description | followerCount, subscriberCount (降順) | email, fcmToken, fcmTokens |

答え系の属性は検索対象にも含まれず、公開の検索キーで取得されることもありません。

## CLI

```bash
python qubit_search_engine.py import riddles.json --index Riddles --replace
python qubit_search_engine.py search "謎解き" --index Riddles
```

## テスト

```bash
python -m pytest -q tests/test_qubit_search_engine.py
```
