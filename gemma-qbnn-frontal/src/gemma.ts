/**
 * Gemma言語処理エンジン
 * 言語理解、課題発見、言語生成
 */

import { LanguageUnderstanding, DiscoveredIssue, QBNNJudgmentResult, ScoreExpressionRange } from "./types";

type SafetyCategory = "self_harm" | "drugs" | "violence" | "cyber" | "financial_fraud";

const SAFETY_KEYWORDS: Record<SafetyCategory, string[]> = {
  self_harm: ["死にたい", "自殺", "自傷", "消えたい"],
  drugs: ["覚醒剤", "麻薬", "大麻", "違法薬物"],
  violence: ["殺す", "爆弾", "脅迫"],
  cyber: ["ハッキング", "不正アクセス", "侵入", "乗っ取り", "マルウェア", "フィッシング"],
  financial_fraud: ["怪しい", "元本保証", "絶対儲", "投資詐欺", "高配当", "投資話"],
};

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
    ["怪しい", "安全性・リスク評価"],
    ["投資詐欺", "金融詐欺"],
    ["元本保証", "金融詐欺"],
    ["絶対儲", "金融詐欺"],
    ["高配当", "金融詐欺"],
    ["ハッキング", "サイバー安全性"],
    ["不正アクセス", "サイバー安全性"],
    ["侵入", "サイバー安全性"],
    ["乗っ取り", "サイバー安全性"],
    ["マルウェア", "サイバー安全性"],
    ["フィッシング", "サイバー安全性"],
    ["死にたい", "メンタル危機"],
    ["自殺", "メンタル危機"],
    ["自傷", "メンタル危機"],
    ["覚醒剤", "薬物・違法行為"],
    ["麻薬", "薬物・違法行為"],
    ["大麻", "薬物・違法行為"],
    ["殺す", "暴力・脅迫"],
    ["脅迫", "暴力・脅迫"],
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
      is_decision: ["すべき", "判断", "選ぶ", "乗るべき", "やるべき"].some((w) =>
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
   * 危険カテゴリを検知して返す。安全な入力なら null。
   * 検査順序: 緊急度の高いものを先に。
   */
  detectSafetyCategory(text: string): SafetyCategory | null {
    for (const category of ["self_harm", "drugs", "violence", "cyber", "financial_fraud"] as SafetyCategory[]) {
      if (SAFETY_KEYWORDS[category].some((w) => text.includes(w))) {
        return category;
      }
    }
    return null;
  }

  private respondToMentalCrisis(): string {
    return "今とてもつらい気持ちでいるようです。一人で抱え込まないでください。いのちの電話（0120-783-556、毎日16時〜21時・毎月10日は8時〜翌8時）やよりそいホットライン（0120-279-338、24時間）に無料で相談できます。";
  }

  private respondToDrugs(): string {
    return "違法薬物に関する情報の提供はできません。覚醒剤・麻薬・大麻等の所持・使用・売買は重大な犯罪であり、心身に深刻な影響を与えます。薬物問題で困っている場合は、よりそいホットライン（0120-279-338）や地域の保健センターへご相談ください。";
  }

  private respondToViolence(): string {
    return "暴力・脅迫・危険物の製造につながる内容にはお答えできません。他者への危害や脅迫は犯罪です。緊急の場合は110番へ通報してください。";
  }

  /**
   * 危険・不正・詐欺リスクがある相談への応答
   */
  private respondToSafetyRisk(category: SafetyCategory): string {
    if (category === "self_harm") return this.respondToMentalCrisis();
    if (category === "drugs") return this.respondToDrugs();
    if (category === "violence") return this.respondToViolence();
    if (category === "cyber") {
      return "不正アクセスや他者への攻撃につながるハッキングは進めるべきではありません。学習目的であれば、CTF、検証用ラボ、自分が管理する環境など、明確に許可された範囲だけで防御・診断の観点から取り組んでください。";
    }
    // financial_fraud
    return "その投資話には強い警戒が必要です。詳細説明がなく「確実」などと断言する勧誘は詐欺や高リスク商品の典型的なサインなので、すぐに送金・契約せず、相手の登録状況、契約書、手数料、解約条件、第三者の確認を取ってください。";
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
    const expr = this.scoreToExpression(score);

    // 安全性・倫理リスクはスコアより優先して専用応答を返す
    const safetyCategory = this.detectSafetyCategory(raw_text);
    if (safetyCategory !== null) {
      return this.respondToSafetyRisk(safetyCategory);
    }

    if (raw_text.includes("転職")) {
      return this.respondToCareerChange(raw_text, score, decision, expr);
    } else if (raw_text.includes("困") || raw_text.includes("悩")) {
      return this.respondToTrouble(raw_text, issues, expr);
    } else if (raw_text.includes("学")) {
      return this.respondToLearning(raw_text, score, expr);
    } else if (raw_text.includes("気分")) {
      return this.respondToEmotion(raw_text, tendency, expr);
    } else {
      return this.respondGeneric(raw_text, score, decision, tendency);
    }
  }
}
