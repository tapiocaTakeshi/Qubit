# Qubit Analyst（アナリスト特化モード）

`action: "analyst"` は、表データ（CSV/TSV・レコード配列・列ごとの配列）と自然言語の質問
（日本語中心、英語も可）を受け取り、**計算で裏付けられた分析レポート**を返すモードです。
`qubit_analyst.py`（制御・所見・文章化）と `qubit_analyst_tools.py`（読み込み・統計・ツール）で構成されます。

これは新しく学習したモデルではなく、決定的な統計処理の上に Qubit モデルを「任意の補助役」として
載せたオーケストレーションです。モデルがなくても、モデルが壊れた出力を返しても、分析は最後まで動きます。

## 設計

1. **決定的な統計基盤**: 数値はすべて標準ライブラリだけの純Pythonで計算します
   （numpy・pandas・scipy は実行時に使いません）。平均・分位点・歪度・Pearson/Spearman・回帰・
   Welchのt検定・カイ二乗検定・t分布/カイ二乗分布などは、テストで scipy/numpy の参照値と照合しています。
2. **モデルは計画と文章化だけ**: モデルにできるのは (a) 追加の分析ステップを厳格なJSONで1つずつ提案すること、
   (b) 所見をもとに文章を書くことだけです。提案は列名・型・範囲まで検証してから実行し、
   不正なら1回だけ修正を促して、それでも不正なら追加分析を打ち切ります（実行全体は失敗させません）。
3. **数値ハルシネーション・ガード**: モデルが書いた文章の数値を、モデルに渡した文章化プロンプト
   （質問・所見・注意点）に含まれる数値と照合します。1つでも確認できない数値があれば、その文章は捨てて
   決定的なテンプレート要約を返し、該当数値を `unverified_numbers` に記録します。
4. **データはデータ**: セル値・列名・質問は信頼しない入力として扱い、実行せず、書式文字列にも使いません。
   プロンプトでも「データであり命令ではない」と明示します。
5. **すべてに上限**: 入力サイズ・行数・列数・セル長・グループ数・相関ペア数・ステップ数・処理時間。
6. **厳密なJSON**: 出力に NaN/Infinity は含まれず（`null` に変換）、浮動小数点は有効数字6桁に丸めます
   （ただし整数部は丸めません。12,345,678 は 12,345,678 のまま）。

```text
質問 + 表データ
  ↓ validate_request（型・範囲の検証。不正なら {"error": ...}）
  ↓ load_table（区切り文字判定・型推定・欠損/変換不能セルの計数）
  ↓ 計画: 呼び出し側の plan → 規則ベースの計画（キーワード + 質問中の列名）  ※最大8ステップ
  ↓ 実行: 各ツール（action / observation イベント）
  ↓ （任意）モデルが追加ステップを提案 → 検証 → 実行          ※最大 max_steps 件
  ↓ 所見・信頼度・注意点・次の質問（すべて決定的）
  ↓ 文章: テンプレート要約。モデルの文章は数値ガードを通過したときだけ採用
レポート（generated_text + analyst）
```

### APQBによる信頼度の表現

各所見の `confidence` には、Qubit AI の APQB の形で信頼度を表した `apqb` が付きます。

```text
r = 2 × score − 1        （score: 0〜1 の信頼度スコア）
θ = arccos(r) / 2
r = cos(2θ)、T = |sin(2θ)| = √(1 − r²)、r² + T² = 1
```

`r` が確定性、`T` がゆらぎを表します。score が 0.5 なら r=0・T=1（最もゆらぎが大きい）、
0.999 なら r≈0.998・T≈0.063 です。これは同じ信頼度スコアを APQB の言葉で**表し直したもの**であり、
追加の統計的検定でも、新しい情報でもありません。`correlate` の出力にも `apqb` がありますが、
こちらは信頼度ではなく相関係数 r そのものを同じ式で表したものです。

信頼度スコアは次の決定的な規則で決まります（較正された確率ではありません）。

| 所見の種類 | score |
| --- | --- |
| 検定のある所見（相関・推移・比較・クロス集計）で有意なもの | `1 − p` を 0.5〜0.999 に収め、`0.5 + (score − 0.5) × min(1, 0.5 + n/60)` で0.5へ縮める。p値がなければ 0.5 |
| 有意でない・ほぼ相関なし・明確な傾向なし（「差がない」という所見） | `0.5 + 0.4 × min(1, n/100)`（最大0.9 = `medium`。大きな p ではなく大きな n を根拠にする） |
| 集計した事実（データ概要・要約統計・内訳・直近の変化・外れ値・ランキング・計算） | 0.95 |
| 平均・中央値・最小・最大の内訳で、最大／最小のグループが n<5 | 0.6 |
| 判定できなかった結果（z スコア法で n が小さい、四分位範囲が0、完全な期間が2つ未満） | 0.5 |
| 予測 | `0.5 + 0.45 × R² × min(1, n/30)` |

ラベルは score ≥ 0.95 で `high`、≥ 0.8 で `medium`、それ以外は `low` です。
p<0.001 でも n=16 なら score は約0.88（`medium`）、n=24 で約0.95 未満（`medium`）、n≥25 で `high` に届きます。
小さなデータで「強い結論」を出さないための設計です。
`high` の集計値は「データから直接数えた」という意味で、母集団への一般化を保証するものではありません。
レポートでの並び順は「質問への回答 → 補足（要約統計・直近の変化）→ データ概要」で、各段の中は計画の順
（規則ベースの計画は質問の意図の順に並びます）であり、信頼度の順ではありません。

## RunPodリクエスト

先に、`qubit_analyst.py` と `qubit_analyst_tools.py` を含むイメージを RunPod にデプロイしてください
（Dockerfile はファイルを個別に COPY するため、対象に含まれているか確認が必要です）。
複数回の推論を行う場合があるため、`/run` + `/status/{id}` での利用を推奨します（`/cancel/{id}` で取り消し）。

CSV文字列を渡す例:

```json
{"input":{"action":"analyst","prompt":"地域別の売上と、売上と広告費の相関を教えて","parameters":{
  "data":"月,地域,売上,広告費\n2025-01,東日本,1200,180\n2025-01,西日本,980,150\n2025-02,東日本,1250,200\n2025-02,西日本,1010,160\n2025-03,東日本,1310,210\n2025-03,西日本,990,140\n2025-04,東日本,1380,230\n2025-04,西日本,1050,170",
  "use_model":true,"max_steps":2,"max_seconds":120,"language":"ja","table_name":"売上データ"}}}
```

レコード配列を渡し、モデルを使わずに英語でレポートする例:

```json
{"input":{"action":"analyst","prompt":"Compare sales between East and West","parameters":{
  "records":[{"region":"East","sales":1200},{"region":"West","sales":980},
             {"region":"East","sales":1250},{"region":"West","sales":1010},
             {"region":"East","sales":1310},{"region":"West","sales":990}],
  "use_model":false,"language":"en"}}}
```

列ごとの配列と、先に実行させたい手順（`plan`）を渡す例:

```json
{"input":{"action":"analyst","prompt":"売上の今後3か月を予測して","parameters":{
  "data":{"月":["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06"],
          "売上":[2180,2260,2300,2430,2500,2620]},
  "use_model":false,
  "plan":[{"tool":"outliers","arguments":{"column":"売上","method":"zscore"}}]}}}
```

3つ目の例では、`outliers`（呼び出し側の手順）→ `profile` → `forecast`（質問の「3か月」から `periods=3`、
月初の日付なので `period="month"`）の順に実行されます。6件しかないので z スコア法では外れ値を判定できず、
その旨（信頼度 `low`）が所見になります（標本標準偏差では |z| が (n−1)/√n を超えないため、|z|>3 には n≥11 が必要です）。

| パラメータ | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `prompt`（または `inputs`） | 文字列 | 必須 | 質問。前後の空白を除いて1〜2,000文字 |
| `parameters.data` | 文字列 / 配列 / オブジェクト | — | CSV・TSV文字列、レコード配列 `[{"列名": 値}]`、または列ごとの配列 `{"列名": [値, ...]}` |
| `parameters.csv` | 文字列 | — | CSV/TSV文字列（`data` の代わり） |
| `parameters.records` | 配列 | — | レコード配列（`data` の代わり） |
| `use_model` | 真偽値 | `true` | `false` ならモデルを一切呼びません |
| `max_steps` | 整数 | `3` | 0〜6。モデルの提案で追加実行するステップ数の上限 |
| `max_seconds` | 数値 | `120` | 1〜240。ツール呼び出しとモデル呼び出しの前後で確認します |
| `language` | `"ja"` / `"en"` | `"ja"` | レポート・所見・注意点・警告の言語 |
| `table_name` | 文字列 | `"data"` | 80文字以内。`dataset.name` に入ります |
| `plan` | 配列 | `[]` | 最大8件の `{"tool": 名前, "arguments": {...}}`。規則ベースの計画より先に実行します |

- `data`・`csv`・`records` はちょうど1つ指定します。その他の未知のパラメータは無視します。
- `plan` の手順は実行前に検証します。不正な手順は実行せず、`source: "caller"`・`status: "failed"` の
  ステップとして記録し、警告を追加します。呼び出し側と規則ベースの手順は重複を除いて合計8件までで、呼び出し側が優先です。
- リクエストの形や型が不正な場合は、モデルを呼ばずに `{"error": "..."}`（英語のメッセージ）を返します。
  想定外の例外も `{"error": "analyst request failed"}` だけを返し、トレースバックは返しません。
- データを読み込めない場合（空・ヘッダーのみ・行数超過など）は `status: "failed"`、
  `stop_reason: "data_error"` のレポートを返し、`generated_text` に日本語の理由を入れます。このとき
  `dataset` は `null`、`findings`・`caveats`・`next_questions` は空配列です（レポートを作れなかった
  `stop_reason: "error"` の `failed` も同じです）。
- RunPod 上では `handler.py` がモデル推論を `generate` として渡します（temperature 0.2、反復抑制・重複除去は無効）。
  生成長は段階ごとで、計画の判断は96トークン、文章化は320トークンです（`qubit_analyst.generation_tokens`）。
  プロンプトと応答枠がモデルの文脈長（1,024トークン）に収まらない場合は推論せず、追加分析の提案を省略するか、
  文章をテンプレート要約に切り替えます（それぞれ「入力上限を超えた」という警告）。分析自体は続行します。
  プロンプトの文字数上限（計画1,300・文章1,000）は、リポジトリのトークナイザー（約0.6〜0.68トークン/文字）で
  この予算に収まるように決めています。

### 進捗の受け取り

`runpod_handler.py` と `handler.py` は、`agent` と同じアダプター（`neuroquantum_agent_progress.py`）で
進捗を送ります。`IN_PROGRESS` 中の `output` は次の形のJSON文字列で、毎回それまでの全イベントを含みます。

```json
{"analyst_event":{"sequence":1,"type":"started","label":"分析を開始しました","use_model":true,"max_steps":2,"language":"ja"},
 "analyst_events":[{"sequence":1,"type":"started","label":"分析を開始しました","use_model":true,"max_steps":2,"language":"ja"}]}
```

| イベント | 表示ラベル（ja） | 主なフィールド |
| --- | --- | --- |
| `started` | 分析を開始しました | `use_model`, `max_steps`, `language` |
| `decision` | 分析計画を作成しました／追加の分析を検討中 | 規則: `source: "rule"`, `plan`（ツール名）。モデル: `source: "model"`, `step` |
| `action` | ツールごとのラベル（下表） | `call_id`, `tool`, `arguments`, `source` |
| `observation` | 分析結果を受け取りました | `call_id`, `tool`, `status` |
| `retry` | 分析手順の指定を再確認中 | `step` |
| `narrative` | レポートを作成中 | — |
| `failed` | データを読み込めませんでした／レポートを作成できませんでした | — |
| `finished` | 分析完了／上限または停止で終了／分析失敗 | `status`, `stop_reason` |

`sequence` の増加で古い更新を捨て、`call_id` で `action` と `observation` を対応付けてください。
最終結果（`COMPLETED` の `output`）が正です。`on_event` などのコールバックは Python から渡すもので、
リクエストJSONからは指定できません。取り消し・時間切れの確認は協調的で、実行中の計算そのものは中断しません。

## ツール一覧

`TOOL_SPECS` の内容です。`*` は必須、列の型は 数値 / 日付 / 真偽値 / カテゴリ / テキスト。
未知の引数・型違い（真偽値を整数として渡すことも含む）・範囲外・存在しない列・型の合わない列はすべて拒否します。
列名は完全一致、なければ NFKC 正規化・大小文字無視・空白無視で一意に一致するものを使います。

| ツール | 進捗ラベル | 引数 | 主な出力 |
| --- | --- | --- | --- |
| `profile` | データ概要を作成中 | なし | `rows`, `n_columns`, `columns[{name, kind, missing, missing_share, unique, coerced, sample(≤3), unit?}]`, `kinds`, `missing_cells`, `coerced`, `coerced_columns` |
| `describe` | 要約統計を計算中 | `column`（全型。省略時は ID 的な列を除いて先頭から最大20列） | 数値: `count, missing, mean, std, min, q1, median, q3, max, skew, sum`。日付: `min, max, span_days, unique`。その他: `unique, top[≤10]{value, count, share}`。`truncated` |
| `correlate` | 相関を計算中 | `x`, `y`（数値列）、`method`: `pearson`（既定）/ `spearman` | x と y: `mode: "pair"`, `r, n, p_value, strength, strength_ja, direction, apqb{theta, r, T}`。片方だけ: `mode: "column"`（その列と他の数値列）。なし: `mode: "matrix"`（先頭20数値列の全組）。指定しない側からは ID 的な列と年の列を除きます。一覧は `pairs[≤10]`（\|r\| 降順）, `n_pairs`, `skipped`, `truncated` |
| `group_by` | グループ別に集計中 | `by*`（全型）、`value`（数値）、`agg`: `sum` / `mean` / `median` / `count` / `min` / `max`（既定は value ありで `sum`（%単位の列は `mean`）、なしで `count`。count 以外は value 必須） | `groups[≤50]{key, value, count, share}`（値の降順）, `n_groups`, `total`, `truncated`, `other`, `missing_keys`。`share` は sum / count で値がすべて非負、かつ %単位の列でないとき |
| `trend` | 推移を分析中 | `value*`（数値）、`time`（日付・数値。省略時は行順）、`agg`: `sum` / `mean`（既定は `sum`、%単位の列は `mean`）、`period`: `raw`（既定）/ `month` / `year`（日付の time が必要） | `n, rows_used, first, last, first_period, last_period, change, pct_change`（%）, `slope, slope_unit, slope_per_period, r2, p_value, direction`（`increasing` / `decreasing` / `flat` / `unknown`）, `direction_ja, cagr`（割合）, `cagr_pct`（%）, `peak, trough, last_periods[≤6]`, `partial_periods`（除外した不完全な期間）, `uneven_periods`（期間ごとの件数が20%超違う合計） |
| `outliers` | 外れ値を検出中 | `column*`（数値）、`method`: `iqr`（既定）/ `zscore`、`threshold`（iqr は 0.5〜5・既定1.5、zscore は 1.5〜6・既定3） | `bounds{lower, upper}, count, share, high, low, rows[≤20]{row, value, side, z?}, truncated`。iqr は `q1, q3, iqr`、zscore は `mean, std`（標本標準偏差）。四分位範囲が0、または z スコア法で (n−1)/√n ≤ threshold（k=3 なら n≤10）のときは判定できないので `count: null` と `reason` |
| `compare` | グループを比較中 | `value*`（数値）、`by*`（全型）、`a`, `b`（グループ名、1〜500文字。省略時は件数の多い順） | `a`, `b`: `{label, n, mean, std, median}`、`diff, pct_diff`（%）, `t, df, p_value`（Welch）, `cohen_d`（プールした標準偏差）, `effect, effect_ja, significant`（p<0.05）, `defaulted`（a・b とも省略して件数の多い2グループを選んだ） |
| `crosstab` | クロス集計中 | `row*`, `col*`（全型） | `table{rows, cols, counts, row_totals, col_totals, row_shares}`（各軸12水準まで、残りは「その他」に集約。実在の水準と同じ名前にならないよう「その他(集約)」「その他(集約2)」…を使う）, `n, missing_pairs, folded, truncated, chi2, dof, p_value, cramers_v, low_expected_share` |
| `top_n` | ランキングを作成中 | `column*`（数値・日付）、`n`: 1〜50（既定5）、`order`: `desc`（既定）/ `asc`、`label`（全型。省略時は最初のカテゴリ・テキスト列） | `rows[{row, value, label}], count, label_column, top_share`（数値で全値が非負、%単位でないとき）, `unit` |
| `forecast` | 将来値を予測中 | `value*`（数値）、`time`、`periods`: 1〜12（既定3）、`agg`: `sum` / `mean`（既定は trend と同じ）、`period`: `raw` / `month` / `year` | `method: "linear_trend", agg, level: 0.95, points[{step, period, estimate, lower, upper}], r2, slope, p_value, residual_std, last_period, last_value, caveat, partial_periods`。4時点未満は `reason`。月次以上の間隔の日付は暦の月で進め、月末・四半期末は月末に、15日などは同じ日にそろえます |
| `calculator` | 計算中 | `expression*`（最大200文字） | `{expression, value}`（`neuroquantum_agent.calculate`。数値・括弧・`+ - * / %` のみ、eval なし） |

補足:

- `row` は1始まりのデータ行番号（ヘッダーを除く）です。
- `correlate` の `strength` は \|r\| ≥ 0.7 で `strong`、≥ 0.4 で `moderate`、≥ 0.2 で `weak`、それ未満は `negligible`。
  欠損はペアごとに除外し、n<3 や値が一定の組は `skipped` に数えます。
- `compare` の `effect` は \|d\| ≥ 0.8 で `large`、≥ 0.5 で `medium`、≥ 0.2 で `small`、それ未満は `negligible`。
- `trend` は時間（日付なら日数）に対する最小二乗回帰で、p<0.05 のとき傾きの符号で増加/減少、それ以外は `flat` です。
  日付の場合 `slope` は1日あたりで、`slope_per_period` は時点間隔の中央値を掛けた値です。
  `cagr` は、日付で期間が365日以上、または年の列（`年`・`year` を含む名前の4桁の整数）で1年以上のときに、
  すべての時点の値が正で、%単位の列でない場合だけ計算します（途中で0以下になる系列の端点からは計算しません）。
  同じ時点の値は `agg` で集約します（`forecast` も同じ）。
- `period` が `month` / `year` のとき、最初と最後の月（年）に、中間の月（年）の80%未満の日数分しかデータがなければ、
  その期間は不完全として除外し `partial_periods` に入れます（月の途中で終わるデータの最終月を合計して
  「急減」と報告しないため）。完全な期間が2つ未満なら `reason` を返します。
- `crosstab` は Yates 補正なしの Pearson カイ二乗検定です。
- 計算できない統計量は `null` になり、`reason` に日本語の理由が入ります（例: 「データ数が不足しています」）。
  ゼロ除算や NaN で例外になることはありません。

### データの読み込みと型推定

- CSV/TSV: 区切り文字を `,` `\t` `;` `|` から判定。ヘッダー行が必須で、UTF-8 の BOM は除去します。
  バイト列（Python・CLIから）は UTF-8、だめなら CP932（Shift_JIS）で読みます。
- 欠損とみなす値: 空、`na`, `n/a`, `nan`, `null`, `none`, `-`, `—`, `欠損`, `不明`, `#n/a`（大小文字無視）。
- 数値: 全角数字、通貨記号（¥ ￥ $ € £ 円 ドル）、3桁区切りのカンマ、末尾 `%`（値は%のまま、列に `unit: "%"`）、
  `(123)` や `▲50` の負数、`千` `万` `億` `兆` の単位に対応。真偽値・inf・nan は数値として扱いません。
  列の単位は多数派の単位で、別の単位のセル（円の列に1つだけある `5%` など）は変換できないセルとして数えます。
- 日付: `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD`, `YYYY-MM`, `YYYY/MM`, `YYYY年M月D日`, `YYYY年M月`、
  ISO日時（日付部分のみ使用）、`YYYY.M`（列全体が年月の場合だけ。`2024.1` と `2024.10` が同じ値になる列か、
  列名に「月」「month」などを含む列）。`year` / `年` を含む名前で4桁の整数が入った列は数値のまま、時間軸として使います。
- 改行は LF・CRLF・CR のみ（古い Mac / Excel の書き出し）のいずれにも対応します。
- 真偽値: `true/false`, `yes/no`, `はい/いいえ`, `○/×`。
- 列の型は、欠損以外の値の90%以上が当てはまる型（数値 → 日付 → 真偽値の順に判定）。当てはまらないセルは欠損にして
  `coerced` に数え、注意点に出します。それ以外は、異なる値の数が max(20, 行数の5%) 以下ならカテゴリ、超えればテキストです。
- 重複・空の列名は `col`, `col_2`, `列3` のように一意にします。

### 規則ベースの計画

モデルを使わなくても、質問のキーワードと質問中に出てくる列名（NFKC・大小文字無視、長い名前を優先。
英数字の列名は単語境界でのみ一致。英語の複数形（`products`）や末尾の単位（`売上高(百万円)` の「売上高」）でも、
他の列と区別できるときは一致）から計画を立てます。先頭は必ず `profile` です。

| 意図 | 主なキーワード | ツール |
| --- | --- | --- |
| 要約 | 分布・要約・統計・概要 / describe, summary, distribution, overview | `describe` |
| 推移 | 推移・傾向・トレンド・成長・伸び・増加・減少・増減・時系列、月別・年ごと・毎月 / trend, growth, over time, monthly | `trend` |
| 内訳 | ◯◯別・ごと・毎・内訳・構成・割合・シェア / by, breakdown, share, per, each | `group_by` |
| 相関 | 相関・関係・関連・連動 / correlation, relationship | `correlate`（「スピアマン」「順位相関」で spearman）。カテゴリ列2つなら `crosstab`、カテゴリ列と数値列なら `group_by`（平均）+ `compare` |
| 比較 | 比較・違い・差 / compare, vs, versus, difference | `compare`（質問中のグループ名を a / b に。名前がなく3グループ以上なら、全グループの平均の `group_by` も） |
| クロス集計 | クロス・独立 / crosstab, independence, contingency | `crosstab` |
| ランキング | 上位・下位・ランキング・トップ・ワースト・ベスト / top, rank, bottom, highest, lowest | `top_n`（「上位10」などの数を n に）。質問中のラベル列に同じ値が繰り返し出る（「売上トップ3の店舗」）なら、行ではなく店舗ごとの `group_by` |
| 外れ値 | 外れ値・異常 / outlier, anomaly, unusual | `outliers`（「zスコア」「標準偏差」で zscore。n≤10 なら IQR法も追加） |
| 予測 | 予測・見通し・将来・今後・来月・来期・来年 / forecast, predict, projection, outlook | `forecast`（「3か月」などを periods に。「来月」は1期、月次データの「来年」は12期。日次データの「4週間」「10日」は月にまとめず観測間隔で数える） |

- 「最も・一番・最大 / highest, most」などは、グループ列が質問にあれば `group_by`、なければ `top_n` を加えます。
- 「◯◯別・ごと・毎」はどの列でもグループのキーにしますが、英語の「by / per / each ◯◯」は数値でない列（または年の列）
  だけをキーにします（「top 3 products by revenue」の revenue は集計する値）。
- `trend`・`forecast`・`group_by` の集計: 質問に「合計・総額 / total」があれば合計、「平均 / mean, average」があれば平均。
  どちらもなければ、%単位の列、0/1 の列、率・割合・満足度・スコア・単価・価格・気温・年齢 / rate, ratio, score, price,
  cvr, ctr などの名前の列は平均、それ以外（金額・件数など）は合計です。「中央値」で中央値。
- 日付列の `period` は、質問の「月次・毎月 / monthly」「年次・毎年 / yearly」で決まり、なければデータから決まります。
  日付がすべて月初なら、最も多い月の間隔が1か月で `month`、12か月の倍数で月がそろう（`YYYY-04-01` の年度も）なら
  `year`、四半期などはそのまま（`raw`）。それ以外は、60日分を超えるデータが180日以上にわたり、日付の間隔の中央値が
  3日以下（日次）のときだけ `month` にまとめます（週次データは週のまま）。列名が「月」「年」でも「毎月」「来月」
  「毎年」「来年」は時間のキーワードとして扱います。
- ID・番号・コードのような列や 1, 2, 3… の連番は、既定の分析対象から外します（`describe`・`correlate` を
  列を指定せずに実行したときも）。0/1 の列は外れ値の既定の対象にしません。
- 質問で名指しした列が、ほとんど数値なのに解釈できないセル（`#REF!` など）のせいで数値列にならなかった場合は、
  別の数値列で代用せず、その旨を注意点に出します。数値列を名指ししていない質問で既定の列を使ったときも注意点に出します。
- どの意図にも当たらない場合は概要計画（`describe`、数値列が2つ以上なら `correlate`、日付列があれば `trend`、
  カテゴリ列があれば `group_by`、0/1 でない数値列があれば `outliers`）を実行します。

### モデルによる追加ステップ

`use_model: true` かつモデルが渡されているとき、規則ベースの計画を実行した後で、モデルに次の1手を尋ねます。
プロンプトは1,300文字以内で、1行目が指示、残りが1つのJSON（`question`、最大30列の `columns[{name, kind}]`、
`tools`（ツール名と引数名、`*` は必須）、`done`（実行済みの手順。既定値の引数は省略、失敗は `"failed": true`）、
`remaining`）です。モデルは次のどちらか1つだけを返します（`thought` は任意で、捨てられます。```` ```json ```` の囲みは可）。

```json
{"status":"continue","action":"outliers","arguments":{"column":"売上"}}
```

```json
{"status":"complete"}
```

`planner_stop` には追加分析が終わった理由が入ります。

| 値 | 意味 |
| --- | --- |
| `disabled` | モデルを使わない（`use_model: false`、モデルなし、または `max_steps: 0`） |
| `not_started` | モデルに尋ねる前に終了した（データを読み込めなかった、または規則ベースの計画の途中で停止した） |
| `complete` | モデルが `complete` を返した |
| `max_steps` | `max_steps` 件を実行した |
| `invalid_decision` | 修正を1回促した後も、JSON・ツール名・引数が不正だった |
| `repeated_action` | 実行済みと同じツール・引数を提案した |
| `model_error` / `context_overflow` | 推論に失敗した／文脈長を超えた |
| `cancelled` / `timeout` / `error` | 停止・時間切れ・内部エラー |

### 数値ハルシネーション・ガード

モデルの文章は、処理が `completed` のときだけ、1,000文字以内のプロンプト（指示、質問、上位8件までの所見、
最大4件の注意点。収まらない行は省略）から作られます。次のいずれかに当たると採用せず、テンプレート要約を使います
（`narrative_source: "template"`、`warnings` に理由）。

- 空、3,000文字超、制御文字や置換文字（�）を含む、`{` `[` ```` ``` ```` で始まる、使われている文字が5種類未満
- **そのプロンプトの文面にない数値**を含む（最大20件を `unverified_numbers` に記録）。モデルが見ていない数値
  （信頼度スコア、ステップの生の出力、データ概要、省略された所見）とたまたま一致しても採用しません。

照合の規則:

- 表示桁での丸め、または相対0.5%以内の差は一致とみなします。
- `%`・`ポイント` 付きの値は、プロンプトの `%` 付きの値と同じ値、または単位のない値の ÷100（「26.29%」と 0.2629）と
  一致します。単位のない件数や値に `%` を付けた数値（n=24 に対する「24%」）や ×100 の取り違えは一致としません。
  「9割」は 90% として照合します。
- `倍` / `times` の数値は計算結果に倍率がないため、常に確認できない数値になります（「87.23%増」→「87.23倍」など）。
- `千` `万` `億` `兆`、`k` `M` `B`、`thousand` `million` などの倍率に対応します。
- 2桁以上の有効数字を持つ整数は、最後の0でない桁までの丸めを許容します（「2,900」は 2939.6 と一致）。
- カンマ・小数点・単位・符号のない 1900〜2100 の整数は年とみなし、ソースに同じ値がある場合だけ一致とします（丸めは許容しません）。
- 行頭の番号（`1.`、`(2)`、`①`）や、R²・χ² のような上付き・下付き数字は数値として扱いません。
- 符号: 文章の数値に明示的な符号（`-` `−` `▲` `+`）や直後の「減少・減・低下・増加・増・上昇・伸び」があるときは、
  同じ符号か符号のないソースとだけ一致します（「+87.23%」の結果を「87.23%減少」と書くと不一致）。
  符号のない数値はどちらの符号とも一致します。
- 漢数字（「三倍」「十二か月」）は数値として読みません。

採用されたモデルの文章はそのまま `generated_text` になり、注意点は自動では付け足しません（`caveats` は常にレポートにあります）。
このガードが保証するのは「文章中の数値がモデルに渡した結果の文面にある」ことだけで、数値と対象の対応や、
数値を伴わない増加/減少といった言葉の正しさまでは検証しません。

## 出力

次の16行を `sales.csv` として保存し、

```text
月,地域,売上,広告費
2025-01,東日本,1200,180
2025-01,西日本,980,150
2025-02,東日本,1250,200
2025-02,西日本,1010,160
2025-03,東日本,1310,210
2025-03,西日本,990,140
2025-04,東日本,1380,230
2025-04,西日本,1050,170
2025-05,東日本,1420,240
2025-05,西日本,1080,175
2025-06,東日本,1500,260
2025-06,西日本,1120,180
2025-07,東日本,1460,250
2025-07,西日本,1150,190
2025-08,東日本,1550,270
2025-08,西日本,1170,185
```

実際に `python qubit_analyst.py sales.csv "東日本と西日本で売上を比較して" --no-model --json` を実行した出力の抜粋です
（`…` は省略箇所。値は省略せずそのまま載せています）。

```text
{
  "generated_text": "16行×4列のデータを分析しました。\n\n主な結果:\n- 地域が「東日本」の売上平均（1,384、n=8）は「西日本」（1,069、n=8）より315（29.47%）高いです。この差は統計的に有意です（p<0.001、効果量大きい：d=3.118）。（信頼度: 中）\n- データは16行×4列です（日付1列・カテゴリ1列・数値2列）。欠損セルは0件です。（信頼度: 高）\n\n注意点:\n- データ数が少ない分析があります（最小でn=8）。結果は参考値として扱ってください。",
  "analyst": {
    "version": 1,
    "status": "completed",
    "stop_reason": "completed",
    "question": "東日本と西日本で売上を比較して",
    "language": "ja",
    "use_model": false,
    "dataset": {
      "name": "sales", "rows": 16, "n_columns": 4,
      "columns": [{"name": "月", "kind": "datetime", "missing": 0}, …],
      "kinds": {"datetime": 1, "categorical": 1, "numeric": 2},
      "missing_cells": 0, "coerced": 0
    },
    "plan": [
      {"tool": "profile", "arguments": {}, "source": "rule"},
      {"tool": "compare", "arguments": {"value": "売上", "by": "地域", "a": "東日本", "b": "西日本"}, "source": "rule"}
    ],
    "steps": [
      {"call_id": "call_1", "tool": "profile", "arguments": {}, "source": "rule", "status": "completed", "output": {…}},
      {
        "call_id": "call_2", "tool": "compare",
        "arguments": {"value": "売上", "by": "地域", "a": "東日本", "b": "西日本"},
        "source": "rule", "status": "completed",
        "output": {
          "value": "売上", "by": "地域", "n_groups": 2, "defaulted": false,
          "a": {"label": "東日本", "n": 8, "mean": 1383.75, "std": 122.7, "median": 1400.0},
          "b": {"label": "西日本", "n": 8, "mean": 1068.75, "std": 73.1803, "median": 1065.0},
          "diff": 315.0, "pct_diff": 29.4737, "t": 6.23629, "df": 11.4206, "p_value": 5.40855e-05,
          "cohen_d": 3.11815, "effect": "large", "effect_ja": "大きい", "significant": true
        }
      }
    ],
    "findings": [
      {"id": "F1", "tool": "profile", "kind": "overview", …},
      {
        "id": "F2", "tool": "compare", "kind": "comparison", "call_id": "call_2",
        "statement": "地域が「東日本」の売上平均（1,384、n=8）は「西日本」（1,069、n=8）より315（29.47%）高いです。この差は統計的に有意です（p<0.001、効果量大きい：d=3.118）。",
        "evidence": {"a": {…}, "b": {…}, "diff": 315.0, "pct_diff": 29.4737, "t": 6.23629, "df": 11.4206,
                     "p_value": 5.40855e-05, "cohen_d": 3.11815, "effect": "large", "significant": true},
        "confidence": {
          "label": "medium", "score": 0.882567, "basis": "p値と標本数に基づく（p<0.001、n=16）",
          "apqb": {"theta": 0.349774, "r": 0.765133, "T": 0.643872}
        }
      }
    ],
    "caveats": ["データ数が少ない分析があります（最小でn=8）。結果は参考値として扱ってください。"],
    "next_questions": ["売上の推移を確認しますか？", "地域別の売上の内訳を見ますか？",
                       "売上と広告費の関係を調べますか？", "売上に外れ値がないか確認しますか？"],
    "narrative_source": "template",
    "unverified_numbers": [],
    "planner_stop": "disabled",
    "decision_count": 0,
    "inference_count": 0,
    "events": [
      {"sequence": 1, "type": "started", "label": "分析を開始しました", "use_model": false, "max_steps": 3, "language": "ja"},
      {"sequence": 2, "type": "decision", "label": "分析計画を作成しました", "source": "rule", "plan": ["profile", "compare"]},
      {"sequence": 3, "type": "action", "label": "データ概要を作成中", "call_id": "call_1", "tool": "profile", "arguments": {}, "source": "rule"},
      …,
      {"sequence": 8, "type": "finished", "label": "分析完了", "status": "completed", "stop_reason": "completed"}
    ],
    "warnings": []
  }
}
```

このWelchのt検定（t=6.236、df=11.42、p=5.41e-05）と Cohen の d（3.118）は scipy の結果と一致します。
p<0.001 でも n=16 のため信頼度は `medium`（0.883）にとどまり、APQB では r=0.765・T=0.644 と、ゆらぎの残る状態として表されます。

| フィールド | 内容 |
| --- | --- |
| `generated_text` | 回答文。テンプレート要約（上位6件の所見と最大6件の注意点）か、ガードを通過したモデルの文章 |
| `analyst.version` | `1` |
| `analyst.status` | `completed`（処理が最後まで完了。内容の正しさの保証ではない）／`limited`（停止・時間切れ・内部エラーで途中まで）／`failed`（データを読めない、またはレポートを作れない） |
| `analyst.stop_reason` | `completed` / `cancelled` / `timeout` / `data_error` / `error` |
| `analyst.use_model` | 実際にモデルを使う設定だったか（`use_model` かつモデルあり） |
| `analyst.dataset` | `name, rows, n_columns, columns[{name, kind, missing, unit?}], kinds, missing_cells, coerced`（`status: "failed"` のときは `null`） |
| `analyst.plan` | 実行を計画した手順（`source: "caller"` / `"rule"`） |
| `analyst.steps` | 実行（または拒否）した各手順: `call_id, tool, arguments, source（rule/caller/model）, status（completed/failed）, output` |
| `analyst.findings` | 所見: `id, tool, kind, call_id, statement, evidence, confidence{label, score, basis, apqb{theta, r, T}}` |
| `analyst.caveats` | 注意点（数値として読めなかった名指しの列、質問から特定できず既定にした数値列、除外した不完全な期間、グループ別の推移は計算していない、期間ごとの件数の違い、データ数の少ないグループ、少ないn、欠損の多い列、変換できなかったセル、%単位、相関≠因果、多重検定、外れ値の影響、行順を時系列とみなした、予測の外挿、期待度数の小さいセル、省略・集約、失敗した手順）。`failed` のときは空配列 |
| `analyst.next_questions` | 次の質問候補（2〜4件。`status: "failed"` のときは空配列） |
| `analyst.narrative_source` | `template` / `model` |
| `analyst.unverified_numbers` | ガードで確認できなかったモデル文章中の数値（表記のまま、最大20件） |
| `analyst.planner_stop` | モデルによる追加分析が終わった理由（上表） |
| `analyst.decision_count` / `inference_count` | モデルへの判断要求の回数／実際に実行した推論の回数（文章化を含む。文脈長超過で推論前に拒否された呼び出しは数えない） |
| `analyst.events` | 進捗イベントの全履歴 |
| `analyst.warnings` | 警告（重複なし、`language` に従う） |

`status: "limited"` のときも、それまでに実行した手順の所見とテンプレート要約を返し、要約の先頭に停止の旨を書きます。

## 上限

| 項目 | 上限 |
| --- | --- |
| 入力テキスト | 2,000,000文字（バイト列は8,000,000バイト、レコード/列配列は文字列の合計で判定） |
| 行数 / 列数 | 20,000行 / 60列 |
| セルの長さ / 列名 | 500文字（超えるとエラー）/ 80文字（超える分は切り詰め） |
| 数値の大きさ | 絶対値 1e50 を超える値は数値として扱いません |
| 質問 / `table_name` | 2,000文字 / 80文字 |
| 計画 / モデルの追加手順 | 8ステップ / `max_steps` 0〜6 |
| 処理時間 | `max_seconds` 1〜240秒 |
| `describe` / `correlate` | 20列 / 数値列20列・上位10組 |
| `group_by` / `crosstab` | 50グループ（残りは `other` に要約）/ 各軸12水準（残りは「その他」） |
| `outliers` / `top_n` / `forecast` | 一覧20行 / n≤50 / 12期間 |
| プロンプト | 計画1,300文字（列は30まで）/ 文章1,000文字 |
| 生成長（RunPod） | 計画の判断96トークン / 文章320トークン（文脈長1,024トークンに収まらないプロンプトは推論しない） |
| モデル出力 | 判断4,000文字（`thought` 500文字）/ 文章3,000文字 |
| `calculator` / `compare` の a・b | 式200文字 / 500文字 |

## セキュリティ

- **データは信頼しない**: セル値・列名・質問は解析するだけで、実行も書式文字列としての使用もしません。
  列名（80文字まで）と、所見の文中に入るグループ名などのラベルは制御文字を除いて切り詰めます
  （ステップの `output` 内の値はセルの内容のまま、JSONとしてエスケープされます）。セル・質問・文字列の引数に含まれる
  単独のサロゲート（JSON の `\ud800` など。UTF-8 にできない）は `?` に置き換えます。
- **入力の解析は線形時間**: セルの数値解析は所有的量指定子（Python 3.11 以上が必要）で、質問の解析も長い入力で
  二乗時間にならないようにしています。読み込みは最初の時間確認より前に行われるため、ここが上限の前提です。
- **コード実行なし**: ツールは表データの読み取り専用の11種類だけです。`calculator` も eval を使わない
  `neuroquantum_agent.calculate` です。ファイルシステム・ネットワーク・シェルへのアクセスはありません
  （CLI が読むのは実行者が指定したローカルファイルだけです）。
- **モデルの提案は検証してから実行**: ツール名・引数名・型・範囲・列の存在と型を `validate_args` で確認します。
  質問やセル値によるプロンプトインジェクションがモデルに影響する可能性は残りますが、使えるツールの範囲は広がらず、
  文章中の数値はガードで照合されます。ただし言葉による主張（解釈・言い回し）はガードの対象外です。
- **エラーの無害化**: 返すエラーは定型メッセージだけです（データ・引数のエラーは日本語、リクエスト形式のエラーは英語）。
  例外の内容・パス・内部情報は返さず、想定外の例外は「分析を実行できませんでした。」などの汎用メッセージになります。
  `handler.py` でも想定外の例外は `{"error": "analyst request failed"}` にし、RunPod のトレースバックにはしません。
- **上限**: 上の表のとおり。時間の確認は各ツール・各推論の前後で行う協調的なもので、実行中の計算は中断しません。
- **レスポンスにはデータの一部が含まれます**: 列名、サンプル値（各列3件まで）、グループ名、行番号と値などです。
  機密データを扱う場合は、エンドポイントのアクセス制御とログの扱いを確認してください。

## CLI

モデルを読み込まずに、ローカルで決定的な分析だけを実行します（`--no-model` の有無にかかわらず常にモデルなしです）。

```bash
python qubit_analyst.py sales.csv "東日本と西日本で売上を比較して"
python qubit_analyst.py sales.csv "売上の推移と広告費との相関は？" --json
python qubit_analyst.py data.json "Which region has the highest sales?" --language en
```

```text
usage: qubit_analyst.py [-h] [--no-model] [--json] [--max-steps MAX_STEPS] [--language {ja,en}] file question
```

- `file`: `.csv` / `.tsv` などのテキスト（UTF-8 または Shift_JIS）、または `.json`（レコード配列か列ごとの配列）。
  ファイル名（拡張子なし）が `table_name` になります。読み込みはパイプやデバイスファイルでも8,000,000バイトまでです。
- 通常は要約、`所見:`（`[F1] high 0.95 …` の形式）、`次の質問候補:` を表示し、警告は標準エラーに出します。
  `--json` ではレポート全体を厳密なJSONで出力します。
- 終了コード: `0` 完了または途中まで（`completed` / `limited`）、`1` 失敗（例: データ行がない）、
  `2` 使い方の誤り・ファイルを読めない・大きすぎる・JSONを解析できない（深すぎる入れ子を含む）・パラメータが不正（例: `--max-steps 9`）。

Python から使う場合:

```python
from qubit_analyst import run_analyst

csv_text = open("sales.csv", encoding="utf-8").read()
result = run_analyst({"inputs": "地域別の売上は？", "parameters": {"data": csv_text, "use_model": False}})
print(result["generated_text"])
```

`run_analyst(data, generate=None, *, on_event=None, cancelled=None, clock=None)` の `generate` は
プロンプト文字列を受け取り文字列を返す関数です。`ValueError` は文脈長超過、その他の例外や文字列以外の戻り値は
推論失敗として扱い、どちらもその段階だけを省略して分析を続けます。`validate_request` が受け付けない入力では
`ValueError` を送出します（RunPod では `{"error": ...}` になります）。

## 学習（train_analyst.py）

`train_analyst.py` は、モデルに (a) 計画のJSON判断と (b) 計算結果に忠実な文章化を教えるための SFT 用カリキュラムを
作ります。`train_agent.py` と同じ方針で、既定は検証・書き出しのみです。**この変更では学習は一度も実行していません。**

- カリキュラムは合成データです。`random.Random(42)` で、傾向・相関・差・外れ値を意図的に仕込んだ小さな表
  （地域別月次売上（日本語・英語）、店舗の日次売上、商品、年次業績、A/B施策、アンケート、Webアクセス）を40個作り、
  1表あたり5問の質問を割り当てます。実際の分析者の対話ログではありません。
- 計画の正解: 規則ベースの計画がすでに必要な分析を含んでいれば `{"status":"complete"}`、足りなければ不足分を
  `continue` として順に1つずつ（既定値の引数は省略）。プロンプトは実行時と同じ `planner_prompt` で作ります。
- 文章化の正解: ツールを実際に実行して得た所見から作った `template_narrative` です（480文字を超えるもの、
  プロンプトに収まらなかった所見の数値を含むものは除外）。プロンプトは実行時と同じ `narrative_prompt` です。
- すべての正解は推論時と同じ検証を通します（計画は `parse_plan_decision` で解釈でき、実行済みの手順を繰り返さない。
  文章は、実行時と同じく文章化プロンプトだけをソースにした数値ガード `verify_numbers` を通過する）。
- 検証用データは (表, 質問) の単位で分け、同じ質問の計画・文章化の段階が学習側と検証側にまたがらないようにします。

検証・書き出しのみ（Torch・GPU 不要。出力ディレクトリは新規である必要があります）:

```bash
python train_analyst.py --output-dir /tmp/analyst-curriculum
```

`train.jsonl`、`validation.jsonl`、`report.json`（件数・段階ごとの内訳など）を書き出します。

別の学習用プロセス／Pod で、既存のチェックポイントと対応する SentencePiece トークナイザーを使う場合:

```bash
python train_analyst.py --train --model-dir /runpod-volume \
  --output-dir /runpod-volume/analyst-candidate-001 --epochs 1 --max-steps 20 --lr 0.00001
```

- `--model-dir` に指定したディレクトリだけを読み、語彙が一致しない場合は拒否し、重みは厳密に読み込みます
  （サイズ変更や乱数初期化はしません）。
- 損失は回答部分（と EOS/EOF）だけにかけます。プロンプトと回答が文脈長を超える例、プロンプトと実行時の生成長
  （計画96トークン・文章320トークン）が文脈長を超える例（配信側の `handler.py` が推論を拒否するもの）、回答がその
  生成長を超える例は、切り詰めずにエラーにします。
- `train_analyst.py` は RunPod イメージにも含まれるので、上のコマンドはこのイメージから作った Pod でも実行できます。
- 出力は `analyst_candidate.pt` と `report.json`（検証損失の学習前後、`processed_samples`、`target_tokens`、
  `optimizer_steps`、トークナイザーのハッシュ）で、配信中のチェックポイントは上書きしません。
- 上限: エポック1〜3、最適化ステップ1〜200、学習率は 0 より大きく 1e-4 以下、バッチサイズ1。
  `--dataset path.jsonl` には同じ形式（`table`, `question`, `language`, `data`, `stage`（`plan` / `narrate`）,
  `steps`, `remaining`（plan のみ）, `target`）の行を最大10,000件・20MBまで渡せ、各行を同じ検証にかけます。

トークン損失は分析能力のベンチマークではありません。候補を使う前に、実際の表と質問で、計画JSONの妥当性・
引数の正確さ・ガードの通過率・文章の正しさを評価してください。

## テスト

```bash
python -m pytest -q tests/test_analyst_tools.py tests/test_analyst.py tests/test_analyst_integration.py tests/test_train_analyst.py
```

統計関数は、テストに埋め込んだ scipy/numpy の参照値と照合します（scipy がある環境では乱数データでの比較も追加で実行）。
モデルはスクリプト化した偽の `generate` で置き換え、RunPod の入口・進捗アダプター・`handler.py` の経路も
モックで確認します。GPU・チェックポイント・外部APIは使いません。
CI（`.github/workflows/test-agent.yml`、Python 3.12）でもエージェントのテストと一緒に実行します。

## 制約と注意

- **予測は線形トレンドのみ**: 季節性・構造変化・外部要因は考慮しません。予測区間は残差が独立・正規・等分散であることを
  前提にした OLS の95%予測区間で、外挿です。
- **因果は推論しません**: 相関・比較・クロス集計はすべて観察データの関連です。交絡や選択バイアスは扱いません。
- **検定は観測の独立性を前提**にしています。時系列は自己相関があることが多く、`trend` の p値は楽観的になりがちです。
  Welchのt検定は平均の近似的な正規性を、カイ二乗検定は十分な期待度数を前提にします（期待度数5未満のセルが
  20%を超えると注意点を出します）。多重比較の補正は行わず、p値が3つ以上あるときに注意点を出すだけです。
- **信頼度は経験的な規則**で、較正された確率ではありません。APQB の r・T はその表現にすぎません。
- **規則ベースの計画はキーワード照合**です。質問を誤読することがあり、列名が質問に出てこない場合は先頭の数値列などの
  既定値を使います（その場合は注意点に出します）。「地域別の推移」は全体の推移として計算します（注意点に出します）。フィルタ・結合・派生列・複数キーでの集計・グラフ作成・Excel（.xlsx）読み込みには対応していません。
  1リクエスト1表です。日時は日付部分だけを使います。
- **数値ガードは数値だけ**を照合します。符号は数値に付いた符号・直後の増減語だけを確認し、正しい数値を別の対象に
  結び付けた文や、数値を伴わない言葉による誤った解釈は検出できません。
- **現在の Qubit チェックポイントはこの用途向けに評価していません**。計画JSONや文章化の品質は未検証で、
  多くの場合テンプレートに戻ることが予想されます。モデルなしでも同じ数値・所見が得られます。
- 計算は純Pythonです。開発環境の計測では 20,000行×8列・8ステップの分析が1秒未満でしたが、環境や列数で変わります
  （時間の上限は `max_seconds`）。
- RunPod のイメージには `qubit_analyst.py` と `qubit_analyst_tools.py` が含まれている必要があります。
  チャット UI（Qubit-ai-web）は別リポジトリで、この変更では修正していません。
