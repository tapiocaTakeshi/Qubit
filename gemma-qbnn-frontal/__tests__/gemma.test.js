const { GemmaLanguageProcessor } = require('../dist/gemma');

const createJudgment = (score = 90) => ({
  score,
  decision: 'Yes',
  tendency: 'positive',
  confidence: 0.1,
  issues: ['一般的な対話'],
  quantum_info: {
    raw_score: 0.2,
    quantum_correction_magnitude: 0.2,
    entangle_strength: 0.7,
  },
});

describe('GemmaLanguageProcessor safety routing', () => {
  it('does not positively recommend hacking requests even with a high score', () => {
    const processor = new GemmaLanguageProcessor();
    const understanding = processor.understandLanguage('ChatGPTでハッキング');

    const response = processor.generateDynamicResponse(
      understanding,
      createJudgment()
    );

    expect(response).toContain('進めるべきではありません');
    expect(response).toContain('許可された範囲');
  });

  it('warns about suspicious investment pitches before generic score guidance', () => {
    const processor = new GemmaLanguageProcessor();
    const understanding = processor.understandLanguage('怪しい投資話');

    const response = processor.generateDynamicResponse(
      understanding,
      createJudgment()
    );

    expect(response).toContain('強い警戒が必要');
    expect(response).toContain('すぐに送金・契約せず');
  });
});
