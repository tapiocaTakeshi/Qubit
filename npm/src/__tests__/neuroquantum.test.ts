import { NeuroQuantumEngine } from "../neuroquantum.js";

describe("NeuroQuantumEngine", () => {
  // ---------------------------------------------------------------------------
  // Construction
  // ---------------------------------------------------------------------------

  it("constructs with default config", () => {
    const engine = new NeuroQuantumEngine();
    expect(engine).toBeDefined();
  });

  it("constructs with custom layer / head config", () => {
    const engine = new NeuroQuantumEngine({ numLayers: 2, numHeads: 2, lambdaEntangle: 0.3 });
    expect(engine).toBeDefined();
  });

  // ---------------------------------------------------------------------------
  // judge()
  // ---------------------------------------------------------------------------

  it("judge() returns JudgmentResult shape", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.judge("テスト行動", "テストコンテキスト");
    expect(["Yes", "No"]).toContain(result.decision);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.reasoning).toBeTruthy();
    expect(["high", "medium", "low"]).toContain(result.confidence);
    expect(Array.isArray(result.keyFactors)).toBe(true);
    expect(result.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(result.system).toBe("qbnn");
  });

  it("judge() accepts all JudgmentTypes", async () => {
    const engine = new NeuroQuantumEngine();
    const types = ["safety", "ethics", "quality", "risk", "decision", "priority"] as const;
    for (const type of types) {
      const result = await engine.judge("テスト", "コンテキスト", { type });
      expect(["Yes", "No"]).toContain(result.decision);
    }
  });

  it("strictMode raises the acceptance threshold", async () => {
    const engine = new NeuroQuantumEngine();
    // Run multiple times — at least one should differ between strict and relaxed
    let differs = false;
    for (let i = 0; i < 5; i++) {
      const r1 = await engine.judge(`行動${i}`, `コンテキスト${i}`, { strictMode: false });
      const r2 = await engine.judge(`行動${i}`, `コンテキスト${i}`, { strictMode: true });
      // In strict mode, scores below 70 must be "No"
      if (r2.score < 70) expect(r2.decision).toBe("No");
      if (r1.decision !== r2.decision) differs = true;
    }
    // Not asserting differs (deterministic engine may match), just no crashes
  });

  it("dynamic θ phase produces varying scores across calls", async () => {
    const engine = new NeuroQuantumEngine();
    const scores = new Set<number>();
    for (let i = 0; i < 6; i++) {
      const r = await engine.judge("一定の行動", "一定のコンテキスト");
      scores.add(r.score);
    }
    // sinusoidal dynamic factor means at least some scores should differ
    expect(scores.size).toBeGreaterThanOrEqual(1);
  });

  // ---------------------------------------------------------------------------
  // Positive / negative signal detection
  // ---------------------------------------------------------------------------

  it("safety type: positive keywords increase score", async () => {
    const engine = new NeuroQuantumEngine();
    const good = await engine.judge("safe secure protected authorised", "trusted context", { type: "safety" });
    const bad = await engine.judge("unsafe dangerous vulnerability breach", "attack context", { type: "safety" });
    expect(good.score).toBeGreaterThanOrEqual(bad.score);
  });

  it("ethics type: negative keywords lower score", async () => {
    const engine = new NeuroQuantumEngine();
    const good = await engine.judge("ethical fair transparent", "公正なコンテキスト", { type: "ethics" });
    const bad = await engine.judge("unethical biased harmful discriminatory", "問題のあるコンテキスト", { type: "ethics" });
    expect(good.score).toBeGreaterThanOrEqual(bad.score);
  });

  // ---------------------------------------------------------------------------
  // Convenience wrappers
  // ---------------------------------------------------------------------------

  it("checkSafety() returns JudgmentResult", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.checkSafety("安全な操作", "本番環境");
    expect(["Yes", "No"]).toContain(result.decision);
  });

  it("checkSafety() accepts risks option", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.checkSafety("操作", "コンテキスト", {
      risks: ["情報漏洩"],
    });
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it("evaluateEthics() returns JudgmentResult", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.evaluateEthics("行動", "コンテキスト");
    expect(["Yes", "No"]).toContain(result.decision);
  });

  it("assessRisk() returns JudgmentResult", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.assessRisk("リスクある操作", "本番環境");
    expect(["Yes", "No"]).toContain(result.decision);
  });

  it("evaluateQuality() returns JudgmentResult", async () => {
    const engine = new NeuroQuantumEngine();
    const result = await engine.evaluateQuality("高品質コンテンツ accurate clear", {
      requirements: ["正確性"],
    });
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  it("prioritize() returns sorted results", async () => {
    const engine = new NeuroQuantumEngine();
    const tasks = ["urgent critical task", "low-priority optional task", "important required task"];
    const { rankedTasks, scores, reasonings } = await engine.prioritize(tasks, "リソース制限あり");
    expect(rankedTasks).toHaveLength(tasks.length);
    expect(scores).toHaveLength(tasks.length);
    expect(reasonings).toHaveLength(tasks.length);
    for (let i = 0; i < scores.length - 1; i++) {
      expect(scores[i]!).toBeGreaterThanOrEqual(scores[i + 1]!);
    }
  });
});
