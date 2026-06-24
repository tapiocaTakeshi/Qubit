/**
 * Qubit AI v4.0.0 - Pyodide + NeuroQuantum Generation Examples
 *
 * Demonstrates quantum-inspired text generation without external APIs
 */

import {
  getQubitAIGenerative,
  generate,
  generateWithExamples,
  generateBatch,
  trainOnData,
} from '../npm/src/qubit_ai_generative_pyodide.js';

// ============================================
// Example 1: Basic Text Generation (Japanese)
// ============================================
async function example1_BasicGeneration() {
  console.log('\n🔷 Example 1: Basic Text Generation (Japanese)');

  const qubit = getQubitAIGenerative();

  const result = await qubit.generate('こんにちは、今日は');

  console.log('Generated:', result.text);
  console.log('Tokens:', result.tokensUsed);
  console.log('Time:', result.timeMs, 'ms');
}

// ============================================
// Example 2: Few-Shot Learning (Translation)
// ============================================
async function example2_FewShot() {
  console.log('\n🔷 Example 2: Few-Shot Learning (Translation)');

  const qubit = getQubitAIGenerative();

  const result = await qubit.generateWithExamples(
    'Translate to English: さようなら',
    [
      'Translate to English: こんにちは',
      'Hello',
      'Translate to English: ありがとう',
      'Thank you',
    ],
    { temperature: 0.5, maxTokens: 50 }
  );

  console.log('Generated:', result.text);
}

// ============================================
// Example 3: Batch Generation
// ============================================
async function example3_BatchGeneration() {
  console.log('\n🔷 Example 3: Batch Generation');

  const qubit = getQubitAIGenerative();

  const prompts = [
    '今日の天気は',
    '将来のテクノロジーは',
    '量子コンピュータの応用は',
  ];

  const results = await qubit.generateBatch(prompts, {
    temperature: 0.7,
    maxTokens: 100,
  });

  results.forEach((result, i) => {
    console.log(`\nPrompt ${i + 1}: ${prompts[i]}`);
    console.log('Generated:', result.text);
    console.log('Tokens:', result.tokensUsed);
  });
}

// ============================================
// Example 4: Custom Training
// ============================================
async function example4_CustomTraining() {
  console.log('\n🔷 Example 4: Custom Training');

  const qubit = getQubitAIGenerative({
    sessionKey: 'quantum-session',
  });

  // Train on quantum-related texts
  const quantumTexts = [
    'Quantum computing leverages quantum mechanics principles.',
    'Qubits can exist in superposition of 0 and 1.',
    'Quantum entanglement enables qubit correlation.',
    'Quantum gates manipulate qubit states and phases.',
    'Quantum algorithms solve problems exponentially faster.',
  ];

  console.log('Training on quantum domain texts...');
  await qubit.train(quantumTexts);

  const result = await qubit.generate(
    'Quantum technology is',
    { temperature: 0.6, maxTokens: 150 }
  );

  console.log('Generated:', result.text);
}

// ============================================
// Example 5: Temperature Effects
// ============================================
async function example5_TemperatureEffects() {
  console.log('\n🔷 Example 5: Temperature Effects (Creativity)');

  const prompt = 'The future of AI is';

  const temperatures = [0.3, 0.7, 1.2];

  for (const temp of temperatures) {
    const result = await generate(prompt, {
      temperature: temp,
      maxTokens: 100,
    });

    console.log(`\nTemperature ${temp}:`);
    console.log('Generated:', result.text);
  }
}

// ============================================
// Example 6: Top-K and Top-P Sampling
// ============================================
async function example6_SamplingStrategies() {
  console.log('\n🔷 Example 6: Sampling Strategies');

  const prompt = 'In quantum computing';

  // Conservative (high quality)
  const conservative = await generate(prompt, {
    topK: 10,
    topP: 0.85,
    temperature: 0.5,
  });

  // Balanced
  const balanced = await generate(prompt, {
    topK: 40,
    topP: 0.9,
    temperature: 0.7,
  });

  // Creative
  const creative = await generate(prompt, {
    topK: 100,
    topP: 0.95,
    temperature: 0.9,
  });

  console.log('\nConservative (top-k=10, top-p=0.85):');
  console.log(conservative.text);

  console.log('\nBalanced (top-k=40, top-p=0.9):');
  console.log(balanced.text);

  console.log('\nCreative (top-k=100, top-p=0.95):');
  console.log(creative.text);
}

// ============================================
// Example 7: Repetition Penalty
// ============================================
async function example7_RepetitionPenalty() {
  console.log('\n🔷 Example 7: Repetition Penalty');

  const prompt = 'The word is';

  // Low penalty (may repeat)
  const lowPenalty = await generate(prompt, {
    repetitionPenalty: 1.0,
    maxTokens: 100,
  });

  // High penalty (diverse tokens)
  const highPenalty = await generate(prompt, {
    repetitionPenalty: 1.5,
    maxTokens: 100,
  });

  console.log('\nLow penalty (1.0):');
  console.log(lowPenalty.text);

  console.log('\nHigh penalty (1.5):');
  console.log(highPenalty.text);
}

// ============================================
// Example 8: Multi-Session Management
// ============================================
async function example8_MultiSession() {
  console.log('\n🔷 Example 8: Multi-Session Management');

  // Create two independent sessions
  const chatSession = getQubitAIGenerative({ sessionKey: 'chat' });
  const translatorSession = getQubitAIGenerative({ sessionKey: 'translator' });

  // Train translator on translation-specific texts
  await translatorSession.train([
    'Translate to English: こんにちは',
    'Translate to English: さようなら',
  ]);

  // Use chat session
  const chatResult = await chatSession.generate('How are you');

  // Use translator session
  const translatorResult = await translatorSession.generateWithExamples(
    'Translate to English: おはよう',
    ['Translate to English: こんばんは', 'Good evening']
  );

  console.log('Chat:', chatResult.text);
  console.log('Translator:', translatorResult.text);

  // Check configs
  console.log('\nChat config:', chatSession.getConfig());
  console.log('Translator config:', translatorSession.getConfig());
}

// ============================================
// Example 9: Quantum-Inspired Features
// ============================================
async function example9_QuantumFeatures() {
  console.log('\n🔷 Example 9: Quantum-Inspired Features');

  const qubit = getQubitAIGenerative({
    seed: 12345, // For reproducibility
  });

  const result = await qubit.generate(
    'Quantum entanglement',
    {
      temperature: 0.7,
      topK: 40,
      topP: 0.9,
      repetitionPenalty: 1.2,
      // These parameters combine to create quantum-inspired sampling:
      // - Dynamic temperature (phase evolution)
      // - Top-K/Top-P filtering (state restriction)
      // - Repetition penalty (entanglement effects)
    }
  );

  console.log('Generated:', result.text);
  console.log('Seed ensures reproducible results');
}

// ============================================
// Example 10: Status and Configuration
// ============================================
async function example10_StatusCheck() {
  console.log('\n🔷 Example 10: Status and Configuration');

  const qubit = getQubitAIGenerative();

  // Get status
  const status = await qubit.getStatus();
  console.log('Status:', status);

  // Get config
  const config = qubit.getConfig();
  console.log('Config:', config);

  // Generate something to update training state
  await qubit.generate('Test');

  // Check status again
  const updatedStatus = await qubit.getStatus();
  console.log('Updated status:', updatedStatus);
}

// ============================================
// Main: Run all examples
// ============================================
async function runAllExamples() {
  console.log('╔════════════════════════════════════════════════╗');
  console.log('║  Qubit AI v4.0.0 - Pyodide Generation Examples ║');
  console.log('╚════════════════════════════════════════════════╝');

  try {
    await example1_BasicGeneration();
    await example2_FewShot();
    await example3_BatchGeneration();
    await example4_CustomTraining();
    await example5_TemperatureEffects();
    await example6_SamplingStrategies();
    await example7_RepetitionPenalty();
    await example8_MultiSession();
    await example9_QuantumFeatures();
    await example10_StatusCheck();

    console.log('\n✅ All examples completed successfully!');
  } catch (error) {
    console.error('❌ Error running examples:', error);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples();
}

export {
  example1_BasicGeneration,
  example2_FewShot,
  example3_BatchGeneration,
  example4_CustomTraining,
  example5_TemperatureEffects,
  example6_SamplingStrategies,
  example7_RepetitionPenalty,
  example8_MultiSession,
  example9_QuantumFeatures,
  example10_StatusCheck,
};
