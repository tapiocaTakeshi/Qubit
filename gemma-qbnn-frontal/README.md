# Gemma + QBNN Hybrid Reasoning System

高度なハイブリッド推論システムで、Gemmaの言語処理能力とQBNNの量子判断層を統合しています。

## 概要

```
入力テキスト
    ↓
Gemma: 言語理解 → 質問/感情判定、要求の理解
    ↓
Gemma: 課題発見 → テキストから複数課題を抽出
    ↓
QBNN: 課題判断 → APQB量子状態計算、Yes/No判定
    ↓
Gemma: 言語生成 → 判断に基づいた動的応答生成
    ↓
出力
```

## 特徴

- **言語理解**: ユーザー入力の意図、感情、要求を理解
- **課題発見**: テキストから複数の課題をキーワードベースで抽出
- **量子判断**: APQB（Adjustable Pseudo Quantum Bit）を使用した判断処理
- **動的応答**: テンプレートなしの完全に動的な応答生成
- **スコアベース**: 0-100のスコアに基づいた段階的な表現
- **マルチコンテキスト**: キャリア、学習、感情など、複数のコンテキストに対応

## インストール

```bash
npm install gemma-qbnn-frontal
```

## 使用方法

### 基本的な使い方

```typescript
import { GemmaQBNNEngine } from "gemma-qbnn-frontal";

// エンジンを初期化
const engine = new GemmaQBNNEngine();

// 応答を生成
const response = await engine.generate(
  "プログラミングを学ぶコツは何ですか？"
);

console.log(response.response);
// Output: 学習への関心度が高く、強く推奨される状況です。
//        学習曲線を考慮した計画を立てることが成功の鍵になります...
```

### 複数の応答を生成

```typescript
const responses = await engine.generateBatch(
  "転職すべきですか？",
  3 // 3回の実行
);

responses.forEach((r, i) => {
  console.log(`実行 ${i + 1}:`, r.response);
  console.log(`判定: ${r.qbnn_decision}, スコア: ${r.qbnn_score}`);
});
```

### カスタム設定

```typescript
const engine = new GemmaQBNNEngine({
  entangle_strength: 0.8, // QBNN結合強度（デフォルト: 0.7）
  seed: 42, // ランダムシード（再現性が必要な場合）
});
```

## API リファレンス

### GemmaQBNNEngine

#### `constructor(config?: EngineConfig)`

エンジンを初期化します。

**パラメータ:**
- `config.entangle_strength` (number, optional): QBNN結合強度 (デフォルト: 0.7)
- `config.seed` (number, optional): ランダムシード

#### `async generate(userInput: string): Promise<HybridResponse>`

ユーザー入力に対して単一の応答を生成します。

**返却値:**
```typescript
{
  input: string;                    // ユーザー入力
  response: string;                 // 生成された応答
  issues_discovered: string[];      // 発見された課題
  qbnn_decision: "Yes" | "No";     // QBNN判定
  qbnn_score: number;              // 判定スコア (0-100)
  qbnn_tendency: string;           // 判定傾向 (positive/negative)
  confidence: number;              // 信頼度
  model: string;                   // モデル名
  processing_pipeline: string[];   // 処理パイプライン
  timestamp: string;               // タイムスタンプ
}
```

#### `async generateBatch(userInput: string, numVariations?: number): Promise<HybridResponse[]>`

複数の応答を生成します。

**パラメータ:**
- `userInput`: ユーザー入力
- `numVariations`: 生成する応答数 (デフォルト: 3)

#### `getInfo(): Object`

エンジンの情報を取得します。

## 応答タイプ

### キャリア変更関連
キーワード: "転職"
```
転職の検討は強く推奨される状況のようです。
市場ニーズも高く、スキルセットも合致する可能性が高いでしょう...
```

### 問題解決関連
キーワード: "困"、"悩"
```
「問題解決」について考えるのであれば、
まずは状況を客観的に整理することが重要です...
```

### 学習関連
キーワード: "学"
```
学習への関心度が高く、強く推奨される状況です。
学習曲線を考慮した計画を立てることが成功の鍵になります...
```

### 感情サポート関連
キーワード: "気分"
```
現在の状況は強く推奨される状況ですが、
ポジティブな側面もあります...
```

## スコア表現

| スコア範囲 | 表現 |
|----------|------|
| 85+ | 強く推奨される状況 |
| 70-84 | かなり良い状況 |
| 60-69 | 中程度の判断 |
| 50-59 | 検討の余地がある |
| <50 | 慎重な検討が必要 |

## 内部構造

### GemmaLanguageProcessor
言語理解、課題発見、応答生成を担当します。

### QBNNJudgment
APQB量子状態計算に基づいた判断を実行します。

### GemmaQBNNEngine
両コンポーネントを統合し、パイプラインを制御します。

## 実装の詳細

### APQB（Adjustable Pseudo Quantum Bit）計算

```
θ（シータ） → APQB量子状態
  ↓
r = cos(2θ)  相関係数
T = |sin(2θ)| 温度

量子補正 = (r × 0.3 + T × 0.2) × エンタングル強度
```

### パイプライン

1. **言語理解**: 質問/陳述、感情、要求の判別
2. **課題発見**: キーワードベースの課題抽出
3. **量子判断**: APQB計算によるスコアリング
4. **動的生成**: スコアと課題に基づいた応答生成

## 例

### 例1: プログラミング学習相談

```typescript
const engine = new GemmaQBNNEngine();

const response = await engine.generate(
  "プログラミングを学ぶコツは何ですか？"
);

console.log("課題:", response.issues_discovered); // ["スキル習得"]
console.log("判定:", response.qbnn_decision);      // "Yes"
console.log("スコア:", response.qbnn_score);        // 87.5
console.log("応答:");
console.log(response.response);
```

### 例2: キャリア決定サポート

```typescript
const response = await engine.generate(
  "転職すべきですか？給与は上がるけど、安定性が不安です。"
);

console.log("課題:", response.issues_discovered); // ["キャリア変更"]
console.log("判定:", response.qbnn_decision);      // "Yes"
console.log("傾向:", response.qbnn_tendency);      // "positive"
console.log(response.response);
```

### 例3: 複数応答の比較

```typescript
const responses = await engine.generateBatch(
  "AIの今後について、どう思いますか？",
  5
);

console.log("QBNN判断の一貫性:");
responses.forEach((r, i) => {
  console.log(
    `実行${i + 1}: スコア=${r.qbnn_score.toFixed(1)}, 判定=${r.qbnn_decision}`
  );
});
```

## ライセンス

MIT

## サポート

問題が発生した場合は、GitHubのIssueを作成してください。

## 関連プロジェクト

- [Qubit](https://github.com/tapiocatakeshi/Qubit) - メインリポジトリ
- [run_gemma_qbnn_frontal_random_responses.py](https://github.com/tapiocatakeshi/Qubit/blob/claude/gemma-qbnn-random-responses-oeim2f/run_gemma_qbnn_frontal_random_responses.py) - Python実装版
