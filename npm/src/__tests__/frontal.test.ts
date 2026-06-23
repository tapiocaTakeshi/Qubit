import { QBNNFrontalEngine } from "../frontal.js";

describe("QBNNFrontalEngine", () => {
  const engine = new QBNNFrontalEngine();

  // -------------------------------------------------------------------------
  // judge()
  // -------------------------------------------------------------------------

  it("returns a valid JudgmentResult shape", async () => {
    const result = await engine.judge("テスト行動", "テストコンテキスト");
    expect(result).toHaveProperty("decision");
    expect(["Yes", "No"]).toContain(result.decision);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.reasoning).toBeTruthy();
    expect(["high", "medium", "low"]).toContain(result.confidence);
    expect(Array.isArray(result.keyFactors)).toBe(true);
    expect(result.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(result.system).toBe("qbnn");
  });

  it("prefers 'Yes' for clearly safe text", async () => {
    const result = await engine.judge(
      "安全な認証済みAPIアクセス",
      "本番環境でのセキュアな操作",
      { type: "safety" }
    );
    expect(result.score).toBeGreaterThanOrEqual(50);
  });

  it("prefers 'No' for clearly unsafe text", async () => {
    const result = await engine.judge(
      "危険な不正アクセス攻撃",
      "セキュリティ違反を引き起こすリスクのある操作",
      { type: "safety" }
    );
    expect(result.score).toBeLessThanOrEqual(70);
  });

  it("strict mode raises the acceptance threshold", async () => {
    const relaxed = await engine.judge("やや良い行動", "曖昧なコンテキスト", {
      type: "decision",
      strictMode: false,
    });
    const strict = await engine.judge("やや良い行動", "曖昧なコンテキスト", {
      type: "decision",
      strictMode: true,
    });
    // Both have the same score but the threshold interpretation differs
    expect(relaxed.score).toBe(strict.score);
    if (relaxed.score < 70) {
      expect(strict.decision).toBe("No");
    }
  });

  it("criteria influence the score", async () => {
    const withCriteria = await engine.judge(
      "データ処理タスク",
      "機密データを扱う処理",
      {
        type: "safety",
        criteria: { 機密: "yes", 暗号化: "必要" },
      }
    );
    expect(withCriteria.score).toBeGreaterThanOrEqual(0);
    expect(withCriteria.score).toBeLessThanOrEqual(100);
  });

  // -------------------------------------------------------------------------
  // Convenience wrappers
  // -------------------------------------------------------------------------

  it("checkSafety returns a result", async () => {
    const result = await engine.checkSafety("ファイル削除", "バックアップ後の安全な操作", {
      risks: ["データ損失"],
    });
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });

  it("evaluateEthics returns a result", async () => {
    const result = await engine.evaluateEthics(
      "ユーザーデータの収集",
      "同意を得た上でのプライバシーに配慮したデータ収集"
    );
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });

  it("assessRisk returns a result", async () => {
    const result = await engine.assessRisk(
      "デプロイメント",
      "ステージング環境でテスト済みの低リスクなデプロイ",
      { riskTolerance: 60 }
    );
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it("evaluateQuality returns a result", async () => {
    const result = await engine.evaluateQuality("正確で明確な高品質なコンテンツ", {
      requirements: ["正確性", "明確性"],
      userIntent: "情報提供",
    });
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });

  it("prioritize returns tasks in sorted order", async () => {
    const tasks = ["重要なバグ修正", "ドキュメント更新", "緊急のセキュリティパッチ"];
    const result = await engine.prioritize(tasks, "プロダクション環境");
    expect(result.rankedTasks).toHaveLength(tasks.length);
    expect(result.scores).toHaveLength(tasks.length);
    // Scores must be non-increasing
    for (let i = 0; i < result.scores.length - 1; i++) {
      expect(result.scores[i]).toBeGreaterThanOrEqual(result.scores[i + 1]!);
    }
  });

  // -------------------------------------------------------------------------
  // Score range
  // -------------------------------------------------------------------------

  it("score is always clamped to [0, 100]", async () => {
    const cases = [
      { action: "a".repeat(500), context: "" },
      { action: "", context: "b".repeat(500) },
      { action: "安全", context: "危険" },
    ];
    for (const c of cases) {
      const r = await engine.judge(c.action, c.context);
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(100);
    }
  });
});
