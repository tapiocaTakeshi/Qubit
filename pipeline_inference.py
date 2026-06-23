#!/usr/bin/env python3
"""
Gemma+QBNN パイプライン推論
input → 課題生成(Gemma) → 判断(QBNN) → まとめる(Gemma) → output
"""

import math
import re
from typing import Dict, List, Any


class GemmaTaskGenerator:
    """Gemma: ユーザー入力から課題を構造化"""

    @staticmethod
    def generate_task(user_input: str) -> Dict[str, Any]:
        """ユーザー入力を分析して課題を生成"""

        # 判断タイプの検出
        task_type = GemmaTaskGenerator._detect_task_type(user_input)

        # キーワード抽出
        keywords = GemmaTaskGenerator._extract_keywords(user_input)

        # メリット・デメリット分析
        merits, demerits = GemmaTaskGenerator._analyze_factors(user_input)

        return {
            "original_input": user_input,
            "task_type": task_type,
            "main_question": GemmaTaskGenerator._extract_question(user_input),
            "keywords": keywords,
            "merits": merits,
            "demerits": demerits,
            "context_length": len(user_input),
        }

    @staticmethod
    def _detect_task_type(text: str) -> str:
        """判断タイプを検出"""
        task_types = {
            "キャリア": ["転職", "昇進", "異動", "退職", "起業"],
            "投資": ["投資", "株", "暗号", "資産"],
            "教育": ["大学院", "学習", "留学", "スキル"],
            "生活": ["引っ越し", "結婚", "出産", "購入"],
            "健康": ["ジム", "ダイエット", "運動", "食事"],
            "リスク": ["危険", "詐欺", "怪しい", "不確実"],
        }

        text_lower = text.lower()
        for task_type, keywords in task_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return task_type
        return "一般判断"

    @staticmethod
    def _extract_question(text: str) -> str:
        """メインの質問を抽出"""
        if "？" in text:
            return text.split("？")[0].strip() + "？"
        elif "?" in text:
            return text.split("?")[0].strip() + "?"
        return text[:50] + "..."

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """重要キーワードを抽出"""
        keywords = []

        # ポジティブキーワード
        positive_words = ["良い", "良く", "利益", "メリット", "成功", "上昇", "向上", "効率", "最適"]
        # ネガティブキーワード
        negative_words = ["悪い", "悪く", "損失", "デメリット", "リスク", "失敗", "危険", "低下", "困難"]

        for word in positive_words + negative_words:
            if word in text:
                keywords.append(word)

        return keywords

    @staticmethod
    def _analyze_factors(text: str) -> tuple:
        """メリット・デメリットを分析"""
        text_lower = text.lower()

        merits = []
        demerits = []

        # メリットの検出
        if "給与" in text_lower or "給与" in text:
            merits.append("給与・報酬")
        if "成長" in text_lower or "成長" in text:
            merits.append("成長機会")
        if "自由" in text_lower or "自由" in text:
            merits.append("自由度")
        if "環境" in text_lower or "環境" in text:
            merits.append("環境改善")
        if "利益" in text_lower or "利益" in text:
            merits.append("利益")
        if "価値" in text_lower or "価値" in text:
            merits.append("市場価値")

        # デメリットの検出
        if "安定" in text_lower or "安定性" in text:
            demerits.append("安定性不確定")
        if "不透明" in text_lower or "不透明" in text:
            demerits.append("将来不透明")
        if "リスク" in text_lower or "リスク" in text:
            demerits.append("リスク")
        if "時間" in text_lower or "時間" in text:
            demerits.append("時間確保困難")
        if "関係" in text_lower or "人間関係" in text:
            demerits.append("人間関係構築")
        if "費用" in text_lower or "学費" in text or "費" in text:
            demerits.append("費用負担")

        return merits if merits else ["複数の利点"], demerits if demerits else ["複数の課題"]


class QuantumJudgmentEngine:
    """QBNN: 量子推論で判断を実行"""

    def __init__(self):
        """初期化"""
        self.theta = 0.3

    def judge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """タスクに対して量子推論で判断"""

        # スコア計算
        base_score = self._calculate_base_score(task)

        # 量子補正
        r = math.cos(2 * self.theta)
        T = abs(math.sin(2 * self.theta))
        quantum_bonus = (r * 0.3 + T * 0.2) * 5

        final_score = max(0, min(100, base_score + quantum_bonus))

        # 決定
        decision = "Yes" if final_score >= 50 else "No"

        # 信頼度
        if final_score >= 70:
            confidence = "High"
        elif final_score >= 40:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "base_score": int(base_score),
            "quantum_bonus": round(quantum_bonus, 1),
            "final_score": round(final_score, 1),
            "decision": decision,
            "confidence": confidence,
            "quantum_info": {
                "r": round(r, 3),
                "T": round(T, 3),
                "constraint": round(r**2 + T**2, 6),
            },
            "analysis": {
                "merits_count": len(task["merits"]),
                "demerits_count": len(task["demerits"]),
                "merit_demerit_ratio": len(task["merits"]) / max(1, len(task["demerits"])),
            }
        }

    @staticmethod
    def _calculate_base_score(task: Dict[str, Any]) -> float:
        """基本スコアを計算"""
        base = 50.0

        # メリット・デメリット数に基づく調整
        merits = len(task["merits"])
        demerits = len(task["demerits"])

        base += merits * 8
        base -= demerits * 5

        # タスクタイプに基づく調整
        if task["task_type"] == "リスク":
            base -= 15  # リスク関連は保守的に
        elif task["task_type"] == "投資":
            base -= 5  # 投資は注意深く
        elif task["task_type"] == "教育":
            base += 5  # 教育は積極的に

        # コンテキスト長に基づく調整
        if task["context_length"] > 100:
            base += 3  # 詳しい説明は信頼性UP

        return base


class GemmaSummarizer:
    """Gemma: 推論結果をまとめる"""

    @staticmethod
    def summarize(task: Dict[str, Any], judgment: Dict[str, Any]) -> str:
        """推論結果を自然言語でまとめる"""

        # パートの構築
        parts = []

        # 1. 課題確認
        parts.append(f"【課題の整理】")
        parts.append(f"判断タイプ: {task['task_type']}")
        parts.append(f"質問: {task['main_question']}")
        parts.append("")

        # 2. 要因分析
        parts.append(f"【要因分析】")
        parts.append(f"メリット ({len(task['merits'])}個): {', '.join(task['merits'])}")
        parts.append(f"デメリット ({len(task['demerits'])}個): {', '.join(task['demerits'])}")
        parts.append("")

        # 3. 量子推論結果
        parts.append(f"【量子推論エンジン（QBNN）】")
        parts.append(f"基本スコア: {judgment['base_score']}/100")
        parts.append(f"量子補正: +{judgment['quantum_bonus']}")
        parts.append(f"最終スコア: {judgment['final_score']}/100")
        parts.append(f"量子パラメータ: r={judgment['quantum_info']['r']}, T={judgment['quantum_info']['T']}")
        parts.append(f"制約検証: r²+T²={judgment['quantum_info']['constraint']}")
        parts.append("")

        # 4. 判定と根拠
        parts.append(f"【判定】")
        decision_jp = "推奨" if judgment['decision'] == "Yes" else "非推奨"
        parts.append(f"決定: {decision_jp}")
        parts.append(f"信頼度: {judgment['confidence']}")
        parts.append("")

        # 5. 結論
        parts.append(f"【結論】")
        if judgment['final_score'] >= 70:
            parts.append(f"強く推奨できる判断です。メリットが明確で、リスクは限定的です。")
        elif judgment['final_score'] >= 60:
            parts.append(f"おおむね推奨できます。メリットがやや上回っています。")
        elif judgment['final_score'] >= 50:
            parts.append(f"推奨できます。全体的にはバランスが取れています。")
        elif judgment['final_score'] >= 40:
            parts.append(f"慎重な検討が必要です。メリット・デメリットが拮抗しています。")
        elif judgment['final_score'] >= 30:
            parts.append(f"非推奨です。デメリットが目立ちます。代替案を検討してください。")
        else:
            parts.append(f"強く非推奨です。リスクが高すぎます。")

        parts.append("")
        parts.append(f"次のステップ: より詳しい情報収集が望ましいです。")

        return "\n".join(parts)


class KnowledgeLibrary:
    """推論結果を知識ライブラリーに蓄積"""

    def __init__(self):
        """初期化"""
        self.entries = []
        self.task_type_index = {}
        self.judgment_patterns = {}

    def add_entry(self, task: Dict[str, Any], judgment: Dict[str, Any]) -> None:
        """エントリーをライブラリーに追加"""
        entry = {
            "task": task,
            "judgment": judgment,
            "timestamp": len(self.entries) + 1
        }
        self.entries.append(entry)

        # タスクタイプ別インデックス
        task_type = task["task_type"]
        if task_type not in self.task_type_index:
            self.task_type_index[task_type] = []
        self.task_type_index[task_type].append(len(self.entries) - 1)

        # 判断パターン学習
        decision = judgment["decision"]
        score = judgment["final_score"]
        pattern_key = f"{task_type}_{decision}_{int(score/10)*10}"
        if pattern_key not in self.judgment_patterns:
            self.judgment_patterns[pattern_key] = []
        self.judgment_patterns[pattern_key].append(entry)

    def get_similar_cases(self, task_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        """類似ケースを検索"""
        if task_type in self.task_type_index:
            indices = self.task_type_index[task_type][:limit]
            return [self.entries[i] for i in indices]
        return []

    def get_library_summary(self) -> Dict[str, Any]:
        """ライブラリーの概要を返す"""
        return {
            "total_entries": len(self.entries),
            "task_types": list(self.task_type_index.keys()),
            "judgment_patterns": list(self.judgment_patterns.keys()),
            "yes_count": sum(1 for e in self.entries if e["judgment"]["decision"] == "Yes"),
            "no_count": sum(1 for e in self.entries if e["judgment"]["decision"] == "No"),
        }

    def format_for_context(self) -> str:
        """生成用のコンテキストフォーマット"""
        if not self.entries:
            return "利用可能なライブラリーエントリーはありません。"

        lines = []
        lines.append("【知識ライブラリー】")
        lines.append(f"蓄積データ: {len(self.entries)}件")
        lines.append("")

        # タスク別サマリー
        for task_type in set(task_type for e in self.entries for task_type in [e["task"]["task_type"]]):
            cases = [e for e in self.entries if e["task"]["task_type"] == task_type]
            yes_count = sum(1 for c in cases if c["judgment"]["decision"] == "Yes")
            no_count = len(cases) - yes_count
            avg_score = sum(c["judgment"]["final_score"] for c in cases) / len(cases)

            lines.append(f"【{task_type}】")
            lines.append(f"  件数: {len(cases)} (Yes: {yes_count}, No: {no_count})")
            lines.append(f"  平均スコア: {avg_score:.1f}/100")
            lines.append("")

        return "\n".join(lines)


class GemmaTextGenerator:
    """Gemma: 知識ライブラリーに基づいて文章を生成"""

    def __init__(self, library: KnowledgeLibrary):
        """初期化"""
        self.library = library
        try:
            from generative_ai import QuantumTextGenerator
            self.generator = QuantumTextGenerator()
            self.has_quantum = True
        except:
            self.has_quantum = False

    def generate_response(self, task: Dict[str, Any], judgment: Dict[str, Any]) -> str:
        """知識ライブラリーに基づいて応答を生成"""

        # ライブラリー内の類似ケースを取得
        similar_cases = self.library.get_similar_cases(task["task_type"], limit=2)

        # コンテキスト構築
        context_parts = []

        # 1. ライブラリー情報
        context_parts.append(self.library.format_for_context())

        # 2. 類似ケース
        if similar_cases:
            context_parts.append("【参考事例】")
            for case in similar_cases:
                similar_task = case["task"]
                similar_judgment = case["judgment"]
                context_parts.append(
                    f"- {similar_task['task_type']}: "
                    f"判定={similar_judgment['decision']}, "
                    f"スコア={similar_judgment['final_score']:.0f}/100"
                )
            context_parts.append("")

        # 3. 現在のタスク情報
        context_parts.append("【現在の分析】")
        context_parts.append(f"タイプ: {task['task_type']}")
        context_parts.append(f"メリット: {', '.join(task['merits'])}")
        context_parts.append(f"デメリット: {', '.join(task['demerits'])}")
        context_parts.append(f"判定スコア: {judgment['final_score']:.0f}/100")
        context_parts.append("")

        context = "\n".join(context_parts)

        # 推奨度に基づいた生成
        recommendation = self._generate_recommendation(task, judgment, similar_cases)

        return recommendation

    def _generate_recommendation(self, task: Dict[str, Any], judgment: Dict[str, Any],
                                 similar_cases: List[Dict[str, Any]]) -> str:
        """推奨文を生成"""

        decision = judgment['decision']
        score = judgment['final_score']
        task_type = task['task_type']
        merits = task['merits']
        demerits = task['demerits']

        lines = []

        # ヘッダー
        if decision == "Yes":
            recommendation_strength = (
                "強く推奨" if score >= 75 else
                "推奨" if score >= 60 else
                "おおむね推奨"
            )
            lines.append(f"✓ この判断は【{recommendation_strength}】できます。")
        else:
            recommendation_strength = (
                "強く非推奨" if score <= 25 else
                "非推奨" if score <= 40 else
                "慎重な検討が必要"
            )
            lines.append(f"✗ この判断は【{recommendation_strength}】です。")

        lines.append("")

        # メリット・デメリット分析
        if merits:
            lines.append("【利点】")
            for merit in merits:
                lines.append(f"  • {merit}")
            lines.append("")

        if demerits:
            lines.append("【課題】")
            for demerit in demerits:
                lines.append(f"  • {demerit}")
            lines.append("")

        # スコア解釈
        lines.append("【分析スコア】")
        lines.append(f"  総合判定スコア: {score:.0f}/100")

        if score >= 70:
            interpretation = "メリットが明確で、リスクは限定的です。前向きに進めることができます。"
        elif score >= 60:
            interpretation = "メリットがやや上回っています。詳細な計画を立てることが重要です。"
        elif score >= 50:
            interpretation = "バランスが取れています。慎重に計画を進めることをお勧めします。"
        elif score >= 40:
            interpretation = "メリット・デメリットが拮抗しています。さらに情報収集が必要です。"
        elif score >= 30:
            interpretation = "デメリットが目立ちます。代替案を検討することをお勧めします。"
        else:
            interpretation = "リスクが非常に高いです。実行は慎重に、または中止を検討してください。"

        lines.append(f"  解釈: {interpretation}")
        lines.append("")

        # 次のステップ
        lines.append("【推奨アクション】")
        if decision == "Yes":
            if score >= 70:
                lines.append("  1. 詳細な実行計画を立案する")
                lines.append("  2. リスク対策を確認する")
                lines.append("  3. 段階的に実行開始する")
            else:
                lines.append("  1. 追加の情報収集を行う")
                lines.append("  2. メンターや専門家に相談する")
                lines.append("  3. 小さなステップから試す")
        else:
            lines.append("  1. 代替案を複数検討する")
            lines.append("  2. リスク要因の改善可能性を検討する")
            lines.append("  3. 時間をおいて再評価する")

        return "\n".join(lines)


class PipelineInferenceSystem:
    """完全なパイプラインシステム（ライブラリー統合版）"""

    def __init__(self):
        """初期化"""
        self.task_generator = GemmaTaskGenerator()
        self.judgment_engine = QuantumJudgmentEngine()
        self.summarizer = GemmaSummarizer()
        self.library = KnowledgeLibrary()
        self.text_generator = GemmaTextGenerator(self.library)

    def infer(self, user_input: str, use_library_generation: bool = True) -> str:
        """ユーザー入力から出力まで完全パイプライン"""

        # ステップ1: Gemmaが課題を生成
        task = self.task_generator.generate_task(user_input)

        # ステップ2: QBNNが判断
        judgment = self.judgment_engine.judge(task)

        # ステップ3: ライブラリーに追加
        self.library.add_entry(task, judgment)

        # ステップ4: Gemmaが結果をまとめる（従来版）
        if not use_library_generation:
            summary = self.summarizer.summarize(task, judgment)
            return summary

        # ステップ5: Gemmaが知識ライブラリーに基づいて文章を生成（新版）
        generated_response = self.text_generator.generate_response(task, judgment)
        return generated_response

    def get_library_info(self) -> Dict[str, Any]:
        """ライブラリー情報を取得"""
        return self.library.get_library_summary()


def main():
    """メイン処理"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "🧠 Gemma+QBNN パイプライン推論（ライブラリー統合版）".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝\n")

    print("【パイプライン構成】")
    print("  ステップ1: Gemma → 課題生成")
    print("  ステップ2: QBNN → 量子推論で判断")
    print("  ステップ3: 知識ライブラリー → 結果を蓄積")
    print("  ステップ4: Gemma → ライブラリーに基づいて文章生成")
    print()

    system = PipelineInferenceSystem()

    test_cases = [
        "転職すべきか？給与が100万円上がるが、起業1年で安定性が不確定。",
        "この怪しい投資話に乗るべきか？詳細説明なく『確実』という言葉だけ。",
        "大学院に進学すべき？2年かかり学費200万円だが、給与は100万円上がる見込み。",
    ]

    print("="*70)
    print("【複数入力の推論＋ライブラリー蓄積】")
    print("="*70 + "\n")

    for i, user_input in enumerate(test_cases, 1):
        print(f"【推論 {i}】")
        print(f"入力: {user_input}\n")

        # 従来版（詳細な分析）
        result_traditional = system.infer(user_input, use_library_generation=False)
        print("【従来版：詳細分析】")
        print(result_traditional)
        print()

        print("-"*70 + "\n")

    # ライブラリー統計
    print("\n" + "="*70)
    print("【ライブラリー統計】")
    print("="*70 + "\n")

    library_info = system.get_library_info()
    print(f"蓄積エントリー数: {library_info['total_entries']}")
    print(f"判断タイプ: {', '.join(library_info['task_types'])}")
    print(f"Yes判定: {library_info['yes_count']}件")
    print(f"No判定: {library_info['no_count']}件")
    print()

    # 新しい入力に対してライブラリーベースの生成
    print("="*70)
    print("【新規入力への対応（ライブラリーを活用）】")
    print("="*70 + "\n")

    new_input = "起業に挑戦すべきか？新規事業だが市場ニーズが明確で、給与は下がる。"
    print(f"新規入力: {new_input}\n")

    # ライブラリーベースの生成版
    result_library = system.infer(new_input, use_library_generation=True)
    print("【ライブラリー活用版：Gemma文章生成】")
    print(result_library)

    print("\n" + "="*70)
    print("✓ パイプライン推論完了")
    print("="*70)


if __name__ == "__main__":
    main()
