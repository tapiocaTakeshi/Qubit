/**
 * Gemma言語処理エンジン
 * 言語理解、課題発見、言語生成
 */

import { LanguageUnderstanding, DiscoveredIssue, QBNNJudgmentResult, ScoreExpressionRange } from "./types";

export class GemmaLanguageProcessor {
  private issue_keywords: Map<string, string> = new Map([
    ["転職", "キャリア変更"],
    ["困", "問題解決"],
    ["悩", "意思決定"],
    ["判断", "判断・意思決定"],
    ["アドバイス", "指導・支援"],
    ["学", "スキル習得"],
    ["改善", "プロセス改善"],
    ["リスク", "リスク評価"],
    ["成長", "成長"],
    ["気分", "感情"],
    ["手伝", "支援"],
  ]);

  /**
   * 言語を理解
   */
  understandLanguage(userInput: string): LanguageUnderstanding {
    const input_lower = userInput.toLowerCase();

    return {
      raw_text: userInput,
      is_question: userInput.includes("？") || userInput.includes("?"),
      is_request: ["教えて", "知りたい", "わかりません"].some((w) =>
        input_lower.includes(w)
      ),
      is_decision: ["すべき", "判断", "選ぶ"].some((w) =>
        input_lower.includes(w)
      ),
      is_emotional: ["困っ", "悩ん", "つらい", "嬉しい"].some((w) =>
        input_lower.includes(w)
      ),
      keywords: userInput.split(/[\s、。]/g).filter((w) => w.length > 2),
    };
  }

  /**
   * 課題を発見
   */
  discoverIssues(understanding: LanguageUnderstanding): string[] {
    const { raw_text } = understanding;
    const input_lower = raw_text.toLowerCase();
    const issues: string[] = [];

    for (const [keyword, issue] of this.issue_keywords) {
      if (input_lower.includes(keyword)) {
        issues.push(issue);
      }
    }

    return issues.length > 0 ? issues : ["一般的な対話"];
  }

  /**
   * スコアを表現に変換
   */
  private scoreToExpression(score: number): string {
    if (score >= 85) {
      return "強く推奨される状況";
    } else if (score >= 70) {
      return "かなり良い状況";
    } else if (score >= 60) {
      return "中程度の判断";
    } else if (score >= 50) {
      return "検討の余地がある";
    } else {
      return "慎重な検討が必要";
    }
  }

  /**
   * キャリア変更に関する応答
   */
  private respondToCareerChange(
    userInput: string,
    score: number,
    decision: string,
    expr: string
  ): string {
    if (decision === "Yes") {
      if (score > 80) {
        return `転職の検討は${expr}のようです。市場ニーズも高く、スキルセットも合致する可能性が高いでしょう。新しい環境での経験が大きな成長につながるかもしれません。トランジション計画を立てることをお勧めします。`;
      } else if (score > 60) {
        return `転職の検討は${expr}のようです。ポジティブな側面が多く見られますが、転職先の企業文化や待遇面をしっかり確認してから判断することが大切です。`;
      } else {
        return `転職の検討は${expr}のようです。事前に十分なリサーチと準備が必要そうです。複数の企業を比較検討してみてはいかがでしょうか。`;
      }
    } else {
      return `現在の職場でのキャリア継続がより安定的な選択肢かもしれません。スキル向上や昇進の可能性を探ってみるのも良いでしょう。`;
    }
  }

  /**
   * 困りごと・悩みに関する応答
   */
  private respondToTrouble(
    userInput: string,
    issues: string[],
    expr: string
  ): string {
    const issue_str = issues.length > 0 ? issues[0] : "その課題";
    return `「${issue_str}」について考えるのであれば、まずは状況を客観的に整理することが重要です。${expr}であることが分かりました。段階的にアプローチしてみることをお勧めします。周囲の視点や専門家の意見も参考にすると、より良い解決策が見つかるかもしれません。`;
  }

  /**
   * 学習に関する応答
   */
  private respondToLearning(
    userInput: string,
    score: number,
    expr: string
  ): string {
    if (score > 75) {
      return `学習への関心度が高く、${expr}です。学習曲線を考慮した計画を立てることが成功の鍵になります。小さな目標から始めて、段階的に難度を上げていくアプローチが効果的でしょう。継続性と実践が大切です。`;
    } else {
      return `学習には時間と根気が必要ですが、${expr}のようです。基礎からしっかり学ぶことで、長期的な成長が期待できます。同じ目標を持つ仲間との学習環境も検討してみてください。`;
    }
  }

  /**
   * 感情に関する応答
   */
  private respondToEmotion(
    userInput: string,
    tendency: string,
    expr: string
  ): string {
    if (tendency === "positive") {
      return `現在の状況は${expr}ですが、ポジティブな側面もあります。気分の変化は自然なことです。小さなことから始めて、段階的に気分を盛り上げていくことが効果的です。信頼できる人に話を聞いてもらうのも良いでしょう。`;
    } else {
      return `今は難しい時期かもしれませんが、${expr}。このような時期も成長の機会になることが多くあります。焦らず、今できることに集中することをお勧めします。専門的なサポートも含めて、利用できるリソースを探ってみてください。`;
    }
  }

  /**
   * 一般的な応答
   */
  private respondGeneric(
    userInput: string,
    score: number,
    decision: string,
    tendency: string
  ): string {
    if (tendency === "positive") {
      if (score > 80) {
        return `その判断は${Math.round(score)}点という高いスコアが出ています。前向きに進めて問題ないでしょう。ただし、詳細な計画や実行方法については、さらに深掘りして検討することが大切です。`;
      } else {
        return `肯定的な方向性が見えています。ただし、実行には計画性が必要です。段階的にアプローチして、各段階での結果を検証しながら進めることをお勧めします。`;
      }
    } else {
      if (score < 40) {
        return `現在のところ、慎重な姿勢が必要な状況のようです。急ぐ必要はありません。十分な情報収集と検討を重ねた上で、判断することが大切です。`;
      } else {
        return `判断が分かれるところですが、状況によっては検討の余地があります。メリットとデメリット、リスクとリターンをしっかり比較検討してみてください。`;
      }
    }
  }

  /**
   * 動的応答を生成
   */
  generateDynamicResponse(
    understanding: LanguageUnderstanding,
    judgment: QBNNJudgmentResult
  ): string {
    const { raw_text } = understanding;
    const { score, decision, tendency, issues } = judgment;

    // スコア表現を取得
    const expr = this.scoreToExpression(score);

    // キーワード検出
    const keywords = new Set<string>();
    for (const word of ["転職", "困", "悩", "判断", "学", "成長", "改善", "気分", "助け"]) {
      if (raw_text.includes(word)) {
        keywords.add(word);
      }
    }

    // キーワードに基づいてルーティング
    if (keywords.has("転職")) {
      return this.respondToCareerChange(raw_text, score, decision, expr);
    } else if (keywords.has("困") || keywords.has("悩")) {
      return this.respondToTrouble(raw_text, issues, expr);
    } else if (keywords.has("学")) {
      return this.respondToLearning(raw_text, score, expr);
    } else if (keywords.has("気分")) {
      return this.respondToEmotion(raw_text, tendency, expr);
    } else {
      return this.respondGeneric(raw_text, score, decision, tendency);
    }
  }
}
