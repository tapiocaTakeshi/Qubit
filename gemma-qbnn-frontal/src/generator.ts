/**
 * 量子補助テキスト生成エンジン
 * Python の QuantumTextGenerator と同等の機能を提供
 */

interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

type IntentType =
  | "explanation_question"
  | "judgment_question"
  | "general_question"
  | "judgment_request"
  | "explanation_request"
  | "consultation"
  | "agreement"
  | "casual_conversation";

interface SentimentAnalysis {
  positive: number;
  negative: number;
  uncertain: number;
  overall: number;
}

export class QuantumTextGenerator {
  private theta: number = 0.3;
  private conversationHistory: ConversationMessage[] = [];
  private knowledgeBase: Record<string, string[]>;
  private readonly MAX_HISTORY_SIZE = 1000;

  constructor() {
    this.knowledgeBase = this._buildKnowledgeBase();
  }

  /**
   * 知識ベース構築
   */
  private _buildKnowledgeBase(): Record<string, string[]> {
    return {
      greeting: [
        "こんにちは。何かお手伝いできることはありますか？",
        "こんばんは。今日はどのようなことについて話したいですか？",
        "ご質問やご相談があればお聞きします。",
      ],
      explain: [
        "説明させていただきます。",
        "詳しく解説いたします。",
        "わかりやすくお答えします。",
      ],
      advice: [
        "アドバイスとしては以下のようなことが考えられます。",
        "参考になるかもしれない視点をいくつかご紹介します。",
        "様々な観点からお考えになるといいでしょう。",
      ],
      agreement: [
        "そうですね。確かに。",
        "その通りです。",
        "ご指摘の通りです。",
      ],
      question: [
        "それについてもう少し詳しく教えていただけますか？",
        "もう少し詳しくお聞きしたいのですが。",
        "その背景はどのようなことですか？",
      ],
    };
  }

  /**
   * 量子因子を計算
   */
  private _getQuantumFactor(): number {
    const r = Math.cos(2 * this.theta);
    const T = Math.abs(Math.sin(2 * this.theta));
    return r * 0.3 + T * 0.2;
  }

  /**
   * ランダムアイテムを選択
   */
  private _randomChoice<T>(items: T[]): T {
    return items[Math.floor(Math.random() * items.length)];
  }

  /**
   * ユーザーの意図を検出
   */
  private _detectIntent(userInput: string): IntentType {
    const inputLower = userInput.toLowerCase();

    // 質問検出
    if (
      ["か？", "？", "?", "ですか", "ますか"].some((q) =>
        inputLower.includes(q)
      )
    ) {
      if (
        ["なぜ", "どう", "どこ", "だれ", "なに", "いつ"].some((w) =>
          inputLower.includes(w)
        )
      ) {
        return "explanation_question";
      }
      if (
        ["すべき", "した方が", "いい", "ない"].some((w) =>
          inputLower.includes(w)
        )
      ) {
        return "judgment_question";
      }
      return "general_question";
    }

    // 判定要求
    if (
      ["すべきか", "判断", "意見", "考え"].some((w) =>
        inputLower.includes(w)
      )
    ) {
      return "judgment_request";
    }

    // 説明要求
    if (
      ["説明", "教えて", "知りたい", "わかりません"].some((w) =>
        inputLower.includes(w)
      )
    ) {
      return "explanation_request";
    }

    // 相談
    if (
      ["相談", "困っ", "悩ん", "助けて"].some((w) => inputLower.includes(w))
    ) {
      return "consultation";
    }

    // 同意・反論
    if (
      ["そう", "賛成", "反対", "違う"].some((w) => inputLower.includes(w))
    ) {
      return "agreement";
    }

    // デフォルト
    return "casual_conversation";
  }

  /**
   * 感情分析
   */
  private _analyzeSentiment(text: string): SentimentAnalysis {
    const textLower = text.toLowerCase();

    const positiveWords = [
      "良い",
      "好き",
      "素晴らしい",
      "優秀",
      "成功",
      "利益",
      "楽しい",
      "嬉しい",
    ];
    const negativeWords = [
      "悪い",
      "嫌い",
      "困っ",
      "失敗",
      "リスク",
      "危険",
      "つらい",
      "悔しい",
    ];
    const uncertainWords = [
      "かもしれない",
      "おそらく",
      "可能性",
      "不確実",
      "わかりません",
    ];

    const positiveScore = positiveWords.filter((w) =>
      textLower.includes(w)
    ).length;
    const negativeScore = negativeWords.filter((w) =>
      textLower.includes(w)
    ).length;
    const uncertainScore = uncertainWords.filter((w) =>
      textLower.includes(w)
    ).length;

    return {
      positive: positiveScore,
      negative: negativeScore,
      uncertain: uncertainScore,
      overall:
        (positiveScore - negativeScore) / Math.max(1, textLower.length / 10),
    };
  }

  /**
   * 判定質問への応答生成
   */
  private _generateJudgmentResponse(userInput: string): string {
    const sentiment = this._analyzeSentiment(userInput);
    const quantumFactor = this._getQuantumFactor();

    // スコア計算
    let score =
      50 +
      sentiment.positive * 10 -
      sentiment.negative * 8 +
      quantumFactor * 5;
    score = Math.max(0, Math.min(100, score));

    // 応答の構築
    const responseParts: string[] = [];

    // イントロ
    responseParts.push(this._randomChoice(this.knowledgeBase["advice"]));

    // 分析
    if (sentiment.positive > sentiment.negative) {
      responseParts.push("全体的には肯定的な側面が多く見られます。");
    } else if (sentiment.negative > sentiment.positive) {
      responseParts.push("いくつかの懸念点や課題が指摘されています。");
    } else {
      responseParts.push("メリットとデメリットの両方があるようです。");
    }

    // スコアベースの判定
    if (score >= 70) {
      responseParts.push("結論として、推奨できる判断と言えます。");
    } else if (score >= 50) {
      responseParts.push("全体的にはバランスの取れた判断と考えられます。");
    } else {
      responseParts.push("慎重な検討が必要かもしれません。");
    }

    // 次のステップ
    responseParts.push("より詳しい情報があれば、さらに精密な判断ができます。");

    return responseParts.join(" ");
  }

  /**
   * 説明質問への応答生成
   */
  private _generateExplanationResponse(userInput: string): string {
    const responseParts: string[] = [];

    responseParts.push(this._randomChoice(this.knowledgeBase["explain"]));

    // 質問内容に応じた説明
    if (
      ["なぜ", "理由", "原因"].some((w) => userInput.includes(w))
    ) {
      responseParts.push("その背景には複数の要因があります。");
      responseParts.push(
        "1つには、市場の変化や社会的ニーズが挙げられます。"
      );
      responseParts.push("2つには、個人的な状況や価値観も大きく影響します。");
    } else if (
      ["どう", "方法", "やり方"].some((w) => userInput.includes(w))
    ) {
      responseParts.push("いくつかのアプローチが考えられます。");
      responseParts.push("まずは現状を正確に把握することが重要です。");
      responseParts.push("次に、複数の選択肢を検討することをお勧めします。");
    } else if (
      ["どこ", "どの", "どれ"].some((w) => userInput.includes(w))
    ) {
      responseParts.push("複数の観点から比較検討する必要があります。");
      responseParts.push(
        "各選択肢の長所と短所を整理してみてください。"
      );
    } else {
      responseParts.push("これは複雑な質問ですね。");
      responseParts.push("様々な視点から考えることが大切です。");
    }

    return responseParts.join(" ");
  }

  /**
   * 相談への応答生成
   */
  private _generateConsultationResponse(userInput: string): string {
    const sentiment = this._analyzeSentiment(userInput);
    const responseParts: string[] = [];

    // 共感
    if (sentiment.negative > 0) {
      responseParts.push("そのようなご状況なのですね。");
      responseParts.push("そういった課題は多くの方が経験されています。");
    } else {
      responseParts.push("そのようなご相談ですね。");
    }

    // アドバイス
    responseParts.push(this._randomChoice(this.knowledgeBase["advice"]));

    responseParts.push("まずは冷静に状況を整理することをお勧めします。");
    responseParts.push("その上で、信頼できる方に相談するのも良いでしょう。");
    responseParts.push(
      "一つの視点にとらわれず、複数の角度から考えることが大切です。"
    );

    return responseParts.join(" ");
  }

  /**
   * 通常の会話への応答生成
   */
  private _generateCasualResponse(userInput: string): string {
    const responseParts: string[] = [];

    // キーワードに基づいた応答
    if (
      ["面白い", "興味深い", "素晴らしい"].some((w) =>
        userInput.includes(w)
      )
    ) {
      responseParts.push("そうですね。確かに興味深い観点です。");
    } else if (
      ["難しい", "複雑", "難しい"].some((w) => userInput.includes(w))
    ) {
      responseParts.push("そうですね。複雑な問題ですね。");
    } else {
      responseParts.push("そのようなことですね。");
    }

    responseParts.push("もう少し詳しくお聞かせいただけますか？");
    responseParts.push("より具体的な情報があると、さらに有用なお答えができます。");

    return responseParts.join(" ");
  }

  /**
   * ユーザー入力に対して自由な応答を生成
   */
  generate(userInput: string): string {
    // 入力を履歴に追加
    this.conversationHistory.push({ role: "user", content: userInput });

    // 意図検出
    const intent = this._detectIntent(userInput);

    // 意図に応じた応答生成
    let response: string;
    if (intent === "judgment_question" || intent === "judgment_request") {
      response = this._generateJudgmentResponse(userInput);
    } else if (
      intent === "explanation_question" ||
      intent === "explanation_request"
    ) {
      response = this._generateExplanationResponse(userInput);
    } else if (intent === "consultation") {
      response = this._generateConsultationResponse(userInput);
    } else if (intent === "general_question") {
      response = this._generateExplanationResponse(userInput);
    } else {
      response = this._generateCasualResponse(userInput);
    }

    // 量子要素の付加
    const quantumBonus = this._getQuantumFactor();
    if (Math.random() < quantumBonus) {
      response +=
        " 量子推論の観点からも、これは興味深い質問です。";
    }

    // 対話性の追加
    if (!userInput.endsWith("？") && !userInput.endsWith("?")) {
      response += " いかがでしょうか？";
    }

    // 応答を履歴に追加
    this.conversationHistory.push({ role: "assistant", content: response });

    // 履歴サイズ制限を適用
    if (this.conversationHistory.length > this.MAX_HISTORY_SIZE) {
      this.conversationHistory = this.conversationHistory.slice(-this.MAX_HISTORY_SIZE);
    }

    return response;
  }

  /**
   * 会話履歴を取得
   */
  getConversationHistory(): ConversationMessage[] {
    return this.conversationHistory;
  }

  /**
   * 会話履歴をクリア
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }
}
