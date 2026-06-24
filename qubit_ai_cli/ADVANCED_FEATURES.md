# Advanced Features Guide - Qubit AI CLI

高度な推論、長文生成、コード生成を行うための完全ガイド。

## 概要

Qubit AI CLIは単なるチャットツール以上の機能を持っています。NeuroQuantumとGemmaのハイブリッド推論により、以下の高度なタスクが可能です：

- **Long-form Writing**: 500-1000トークンの記事・エッセイ生成
- **Code Generation**: 完全なアプリケーション・API生成
- **Complex Reasoning**: 数学や論理的問題の段階的解決
- **Deep Analysis**: 多角的で深い分析と洞察

## 生成モード

### 1. LONGFORM - 長文生成モード

500-1000トークンの詳細で構造化された内容を生成します。

**特徴:**
- 複数段落の構造化
- 導入・本文・結論
- 実例と詳細な説明
- エッセイや記事に最適

**温度設定:** 0.7（バランス型創造性）

**使用例:**

```typescript
const writer = new ContentWriter();

// 記事執筆
const article = await writer.writeArticle(
  "人工知能の将来",
  1000  // 約1000語
);

// ブログ投稿
const blogPost = await writer.writeBlogPost(
  "なぜプログラミングスキルは重要なのか"
);

// 深い分析
const analysis = await writer.analyzeTopicDeeply(
  "リモートワークの長期的影響"
);
```

**出力例:**

```
【序論】
人工知能（AI）は現代で最も重要な技術です...

【主要な発展】
過去5年間でAI技術は驚異的な進歩を遂げました...

【実践的応用】
- 医療：診断精度の向上...
- 金融：リスク分析...

【倫理的考察】
AIの発展と同時に、倫理的な課題も...

【未来の展望】
今後10年間、AIはさらに高度になり...
```

---

### 2. CODE - コード生成モード

完全で実行可能なアプリケーションを生成します。

**特徴:**
- 400-800トークンの完全なコード
- 複数言語対応（Python、TypeScript、JavaScript）
- エラーハンドリング込み
- 本番対応品質

**温度設定:** 0.3（低い - 正確性重視）

**使用例:**

```typescript
const generator = new CodeGenerator();

// REST API生成
const api = await generator.generateCode(
  "User management REST API with full CRUD operations",
  "TypeScript"
);

// コード説明
const explanation = await generator.explainCode(code);

// コード最適化
const optimized = await generator.optimizeCode(code);
```

**出力例:**

```typescript
import express, { Request, Response } from 'express';

interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
}

const app = express();
app.use(express.json());

const users: User[] = [];
let nextId = 1;

// Get all users
app.get('/api/users', (req: Request, res: Response) => {
  res.json({ success: true, data: users });
});

// Create user
app.post('/api/users', (req: Request, res: Response) => {
  const user: User = {
    id: nextId++,
    ...req.body,
    createdAt: new Date()
  };
  users.push(user);
  res.status(201).json({ success: true, data: user });
});

// ... (その他の操作)
```

**サポート言語:**
- Python
- TypeScript
- JavaScript
- SQL
- HTML/CSS

---

### 3. REASONING - 推論モード

複雑な問題をステップバイステップで解決します。

**特徴:**
- 段階的な論理展開
- 数学的問題解決
- 検証と確認
- 明確な結論

**温度設定:** 0.4（低い - 論理性重視）

**使用例:**

```typescript
const reasoner = new ReasoningEngine();

// 数学問題解決
const mathSolution = await reasoner.solveMath(
  "f(x) = -x² + 4x + 5 の最大値と最大値を取るxを求めよ"
);

// 論理問題分析
const logicAnalysis = await reasoner.analyzeLogic(
  "3人がいて、1人が嘘をついている。以下の発言から真実を推測せよ..."
);

// トピック説明
const explanation = await reasoner.explain(
  "相対性理論の基本概念"
);
```

**出力例:**

```
【Step 1】関数の形を確認
f(x) = -x² + 4x + 5
最高次の係数が負なので下に開く放物線

【Step 2】頂点の公式を使用
x = -b/(2a) = -4/(-2) = 2

【Step 3】最大値を計算
f(2) = -(2)² + 4(2) + 5 = 9

【Step 4】検証
f'(x) = -2x + 4 = 0 → x = 2 ✓

【結論】
最大値は 9 （x = 2 のとき）
```

---

### 4. ANALYSIS - 深い分析モード

多角的で深い分析と洞察を提供します。

**特徴:**
- 複数の視点
- 批判的思考
- 根拠に基づく推論
- 実装可能な提案

**温度設定:** 0.7（バランス - 創造性と論理性）

**使用例:**

```typescript
const writer = new ContentWriter();

// トピック深掘り
const deepAnalysis = await writer.analyzeTopicDeeply(
  "2030年に最も価値のあるスキル"
);

// ビジネス分析
const businessAnalysis = await writer.analyzeTopicDeeply(
  "AIが雇用市場に与える影響"
);

// 戦略的洞察
const strategicInsight = await writer.analyzeTopicDeeply(
  "クラウドコンピューティングの将来展望"
);
```

**出力例:**

```
【1】技術スキル
• AI/機械学習：導入企業が急速に増加
• 量子コンピューティング：新しい問題解決
• サイバーセキュリティ：デジタル脅威に対応

【2】ビジネススキル
• データ駆動意思決定
• クロス機能的協働
• 変化への適応力

【3】人間特有のスキル
• クリエイティビティ
• 感情知能
• 倫理的判断

【4】複合的スキルセット
最も価値があるのは技術 + ビジネス + 人間スキル

【5】将来への示唆
「学習能力」が最大の競争力に
```

---

## 温度設定ガイド

各モードに最適な温度設定：

| モード | 温度 | 説明 | 用途 |
|--------|------|------|------|
| **CODE** | 0.2-0.3 | 確実で予測可能 | コード生成、算数 |
| **REASONING** | 0.3-0.4 | 論理的で正確 | 複雑な問題、証明 |
| **ANALYSIS** | 0.6-0.7 | バランス型 | 分析、説明 |
| **LONGFORM** | 0.7-0.8 | やや創造的 | 記事、エッセイ |
| **CREATIVE** | 0.9-1.2 | 創造的で多様 | 物語、詩、アイデア |

---

## トークン消費パターン

各モードの典型的なトークン使用量：

```
単語数 → トークン数（概算）

短い回答（1-2文）
  10-50 words → 15-75 tokens

標準会話（段落）
  50-150 words → 75-225 tokens

コード生成
  100-300 lines → 400-800 tokens

長文記事
  300-800 words → 450-1200 tokens

深い分析
  250-600 words → 375-900 tokens
```

---

## 実践的な使用例

### 例1：API設計から実装まで

```typescript
const generator = new CodeGenerator();

// Step 1: 要件を説明
const description = `
REST API for task management:
- Create, read, update, delete tasks
- User authentication with JWT
- Task prioritization
- Due date tracking
`;

// Step 2: コード生成
const code = await generator.generateCode(description, "TypeScript");

// Step 3: コード説明
const explanation = await generator.explainCode(code);

// Step 4: 最適化
const optimized = await generator.optimizeCode(code);
```

### 例2：複雑な問題解決

```typescript
const reasoner = new ReasoningEngine();

// 複合的な問題
const problem = `
A company has 500 employees.
30% work in sales, 20% in engineering, rest in support.
Sales employees earn $60k, engineers $90k, support $40k.
Calculate total payroll and average salary.
`;

// Step-by-stepで解決
const solution = await reasoner.solveMath(problem);
```

### 例3：コンテンツ制作

```typescript
const writer = new ContentWriter();

// 記事作成
const article = await writer.writeArticle(
  "The Future of Remote Work",
  1500
);

// ブログポスト
const blogPost = await writer.writeBlogPost(
  "5 Tips for Effective Remote Collaboration"
);

// 深い分析
const analysis = await writer.analyzeTopicDeeply(
  "Psychological effects of remote work on mental health"
);
```

---

## パフォーマンス最適化

### 推奨される使用方法

1. **バッチ処理**: 複数の関連タスクをまとめて実行
```typescript
const tasks = [
  "説明してください: クラウドコンピューティング",
  "説明してください: エッジコンピューティング",
  "説明してください: フォグコンピューティング"
];

for (const task of tasks) {
  const result = await reasoner.explain(task);
}
```

2. **コンテキスト再利用**: 会話履歴を活かす
```typescript
// 最初のタスク
const initial = await chat.generateWithMode(
  "What is machine learning?",
  GENERATION_MODES.longform
);

// フォローアップ（コンテキストを利用）
const followUp = await chat.generateWithMode(
  "How is it different from deep learning?",
  GENERATION_MODES.longform
);
```

3. **温度の段階的調整**: 必要に応じて調整
```typescript
// 詳細が必要な場合は長めに
chat.updateConfig({ maxTokens: 800 });

// 要約が必要な場合は短めに
chat.updateConfig({ maxTokens: 200 });
```

---

## トラブルシューティング

### 応答が短すぎる
- `maxTokens` を増やす
- 温度を0.5-0.7に調整
- より詳細なプロンプトを使用

### 応答が予測不可能
- 温度を下げる（0.3-0.5）
- Few-shot例を追加
- より具体的な指示を含める

### タイムアウト
- トークン数を減らす
- `timeoutMs` を増やす（最大120000ms）
- リトライを設定

### 記憶不足
- 会話履歴をクリア：`clearHistory()`
- 古いメッセージを削除
- 新しいセッションを開始

---

## 拡張性

### カスタムモード追加

```typescript
class CustomSpecialist extends AdvancedChat {
  async generateInCustomMode(input: string): Promise<string> {
    const customMode: GenerationMode = {
      type: "custom",
      description: "My custom mode"
    };
    return await this.generateWithMode(input, customMode);
  }
}
```

### カスタム Few-shot例

```typescript
const customExamples = [
  { prompt: "Your example 1", completion: "Your response 1" },
  { prompt: "Your example 2", completion: "Your response 2" }
];

const result = await client.generateWithExamples(
  prompt,
  customExamples,
  options
);
```

---

## ベストプラクティス

✅ **すべき:**
- 明確で詳細なプロンプト
- 適切なモード選択
- 温度を意図的に設定
- コンテキストを活用
- 結果を検証

❌ **すべきではない:**
- 曖昧な指示
- すべてのタスクに同じ設定
- トークン制限を無視
- 履歴を無制限に蓄積
- 結果を検証なしで使用

---

## 次のステップ

1. **実験**: 各モードを試す
2. **パラメータチューニング**: 最適な設定を見つける
3. **ワークフロー開発**: 定期的なタスクを自動化
4. **統合**: 他のシステムとの連携

---

**Advanced Features Demo実行:**
```bash
node qubit_ai_cli/demo-advanced-features.js
```

**詳細なドキュメント:**
- `README.md` - 基本情報
- `USAGE.md` - 基本的な使い方
- `src/advanced-chat.ts` - 実装詳細
