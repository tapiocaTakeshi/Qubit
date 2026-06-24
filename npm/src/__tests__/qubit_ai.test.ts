import {
  QubitAI,
  getQubitAI,
  resetQubitAI,
  judge,
  safetyCheck,
  evaluateQuality,
  ethicsCheck,
} from "../qubit_ai.js";

describe("QubitAI", () => {
  beforeEach(() => resetQubitAI());

  // ---------------------------------------------------------------------------
  // Construction & configuration
  // ---------------------------------------------------------------------------

  it("constructs with default config", () => {
    const qubit = new QubitAI();
    expect(qubit).toBeDefined();
  });

  it("constructs with custom config", () => {
    const qubit = new QubitAI({ productName: "TestAI", strictMode: true });
    const info = qubit.getInfo();
    expect(info.product).toBe("TestAI");
  });

  // ---------------------------------------------------------------------------
  // getInfo / getStatus
  // ---------------------------------------------------------------------------

  it("getInfo() returns expected shape", () => {
    const qubit = new QubitAI();
    const info = qubit.getInfo();
    expect(info.product).toBe("Qubit.ai");
    expect(info.version).toBe("1.2.2");
    expect(info.status).toBe("operational");
    expect(info.sessionId).toMatch(/^qubit-ai-/);
    expect(info.description).toBeTruthy();
  });

  it("getStatus() returns expected shape", () => {
    const qubit = new QubitAI();
    const status = qubit.getStatus();
    expect(status.frontalEngineAvailable).toBe(true);
    expect(status.judgmentHistorySize).toBe(0);
    expect(status.status).toBe("operational");
    expect(typeof status.maxHistory).toBe("number");
  });

  // ---------------------------------------------------------------------------
  // judge()
  // ---------------------------------------------------------------------------

  it("judge() returns QubitAIResult shape", async () => {
    const qubit = new QubitAI();
    const result = await qubit.judge("テスト行動", "テストコンテキスト");
    expect(["Yes", "No"]).toContain(result.decision);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(Array.isArray(result.factors)).toBe(true);
    expect(result.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(result.reasoning).toBeTruthy();
    expect(["high", "medium", "low"]).toContain(result.confidence);
  });

  it("judge() respects judgment type", async () => {
    const qubit = new QubitAI();
    const r = await qubit.judge("安全な操作", "認証済み環境", "safety");
    expect(["Yes", "No"]).toContain(r.decision);
  });

  // ---------------------------------------------------------------------------
  // safetyCheck()
  // ---------------------------------------------------------------------------

  it("safetyCheck() returns [boolean, QubitAIResult]", async () => {
    const qubit = new QubitAI();
    const [safe, result] = await qubit.safetyCheck(
      "安全なAPIアクセス",
      "本番環境での認証済み操作"
    );
    expect(typeof safe).toBe("boolean");
    expect(["Yes", "No"]).toContain(result.decision);
    expect(safe).toBe(result.decision === "Yes");
  });

  it("safetyCheck() accepts risks option", async () => {
    const qubit = new QubitAI();
    const [, result] = await qubit.safetyCheck(
      "APIキーをログに出力",
      "本番環境",
      { risks: ["情報漏洩", "セキュリティ違反"] }
    );
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  // ---------------------------------------------------------------------------
  // evaluateQuality()
  // ---------------------------------------------------------------------------

  it("evaluateQuality() returns QubitAIResult", async () => {
    const qubit = new QubitAI();
    const result = await qubit.evaluateQuality("正確で明確なコンテンツ", {
      requirements: ["正確性", "明確性"],
    });
    expect(result.decision).toMatch(/^(Yes|No)$/);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  // ---------------------------------------------------------------------------
  // ethicsCheck()
  // ---------------------------------------------------------------------------

  it("ethicsCheck() returns QubitAIResult", async () => {
    const qubit = new QubitAI();
    const result = await qubit.ethicsCheck(
      "ユーザーデータの分析",
      ["ユーザー", "社会"]
    );
    expect(result.decision).toMatch(/^(Yes|No)$/);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it("ethicsCheck() works without optional params", async () => {
    const qubit = new QubitAI();
    const result = await qubit.ethicsCheck("倫理的なアクション");
    expect(["Yes", "No"]).toContain(result.decision);
  });

  // ---------------------------------------------------------------------------
  // prioritize()
  // ---------------------------------------------------------------------------

  it("prioritize() returns items sorted by score desc", async () => {
    const qubit = new QubitAI();
    const items = [
      { name: "バグ修正", description: "本番環境でのクリティカルバグ" },
      { name: "機能追加", description: "新しいUIコンポーネント" },
      { name: "ドキュメント", description: "APIドキュメントの更新" },
    ];
    const results = await qubit.prioritize(items);
    expect(results).toHaveLength(items.length);
    for (const [item, score] of results) {
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      expect(item.name).toBeTruthy();
    }
    // Scores must be non-increasing
    for (let i = 0; i < results.length - 1; i++) {
      expect(results[i]![1]).toBeGreaterThanOrEqual(results[i + 1]![1]);
    }
  });

  it("prioritize() accepts constraints", async () => {
    const qubit = new QubitAI();
    const items = [{ name: "緊急タスク", description: "即時対応が必要" }];
    const results = await qubit.prioritize(items, "リソース制限あり");
    expect(results).toHaveLength(1);
  });

  // ---------------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------------

  it("records judgment history after each call", async () => {
    const qubit = new QubitAI();
    await qubit.judge("action1", "context1");
    await qubit.judge("action2", "context2");
    const history = qubit.getHistory();
    expect(history.length).toBe(2);
    expect(history[0]!.judgmentType).toBe("safety");
  });

  it("getHistory(limit) limits results to most recent", async () => {
    const qubit = new QubitAI();
    for (let i = 0; i < 5; i++) {
      await qubit.judge(`action${i}`, `context${i}`);
    }
    const history = qubit.getHistory(3);
    expect(history.length).toBe(3);
  });

  it("clearHistory() empties the history", async () => {
    const qubit = new QubitAI();
    await qubit.judge("action", "context");
    qubit.clearHistory();
    expect(qubit.getHistory().length).toBe(0);
  });

  it("getStatus() reflects current history size", async () => {
    const qubit = new QubitAI();
    await qubit.judge("action", "context");
    expect(qubit.getStatus().judgmentHistorySize).toBe(1);
  });

  // ---------------------------------------------------------------------------
  // explain()
  // ---------------------------------------------------------------------------

  it("explain() returns a non-empty formatted string", async () => {
    const qubit = new QubitAI();
    const result = await qubit.judge("テスト", "コンテキスト");
    const explanation = qubit.explain(result);
    expect(typeof explanation).toBe("string");
    expect(explanation.length).toBeGreaterThan(0);
    expect(explanation).toContain("判断結果");
    expect(explanation).toContain("根拠");
    expect(explanation).toContain("主要要因");
  });

  // ---------------------------------------------------------------------------
  // Singleton
  // ---------------------------------------------------------------------------

  it("getQubitAI() returns the same instance", () => {
    const a = getQubitAI();
    const b = getQubitAI();
    expect(a).toBe(b);
  });

  it("resetQubitAI() creates a fresh instance on next call", () => {
    const a = getQubitAI();
    resetQubitAI();
    const b = getQubitAI();
    expect(a).not.toBe(b);
  });

  // ---------------------------------------------------------------------------
  // Module-level convenience functions
  // ---------------------------------------------------------------------------

  it("module-level judge() works", async () => {
    const result = await judge("テスト行動", "テストコンテキスト");
    expect(["Yes", "No"]).toContain(result.decision);
  });

  it("module-level safetyCheck() works", async () => {
    const [safe, result] = await safetyCheck("安全なアクション", "安全な状況");
    expect(typeof safe).toBe("boolean");
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });

  it("module-level evaluateQuality() works", async () => {
    const result = await evaluateQuality("高品質なコンテンツ");
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });

  it("module-level ethicsCheck() works", async () => {
    const result = await ethicsCheck("倫理的なアクション");
    expect(result.decision).toMatch(/^(Yes|No)$/);
  });
});
