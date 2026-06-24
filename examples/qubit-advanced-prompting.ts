/**
 * Example: Advanced Prompt Engineering with Qubit AI
 *
 * Demonstrates sophisticated prompting techniques:
 * - Chain-of-thought prompting
 * - Multi-step reasoning
 * - Context-aware generation
 * - Custom prompt templates
 *
 * Prerequisites:
 * 1. Set HF_TOKEN environment variable:
 *    export HF_TOKEN="hf_..."
 *
 * 2. Install dependencies:
 *    npm install qubit_ai
 *
 * 3. Run:
 *    npx ts-node examples/qubit-advanced-prompting.ts
 */

import { NeuroQuantumClient } from "qubit_ai";

interface Example {
  prompt: string;
  completion: string;
}

/**
 * Chain-of-Thought Prompting: Get detailed reasoning steps
 */
async function chainOfThoughtReasoning() {
  console.log("🧠 Chain-of-Thought Reasoning\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Examples showing step-by-step reasoning
  const cotExamples: Example[] = [
    {
      prompt: "If I have 3 apples and get 2 more, how many do I have?",
      completion: "Let me think step by step:\n1. I start with 3 apples\n2. I get 2 more apples\n3. Total: 3 + 2 = 5 apples",
    },
    {
      prompt: "A book costs $5 and a pen costs $2. How much do 3 books and 2 pens cost?",
      completion: "Let me think step by step:\n1. 3 books cost: 3 × $5 = $15\n2. 2 pens cost: 2 × $2 = $4\n3. Total cost: $15 + $4 = $19",
    },
  ];

  const problem = "If a car travels 60km per hour, how far will it travel in 3.5 hours?";

  console.log(`📝 Problem: "${problem}"\n`);
  console.log("📚 Expecting step-by-step reasoning...\n");

  try {
    const result = await client.generateWithExamples(
      problem,
      cotExamples,
      {
        numExamples: 2,
        exampleTemplate: "Q: {prompt}\n{completion}",
        queryTemplate: "Q: {prompt}\nA: Let me think step by step:",
        maxNewTokens: 120,
        temperature: 0.5,
      }
    );

    console.log("✅ Reasoning:\n");
    console.log(`A: Let me think step by step:${result.generatedText}\n`);
  } catch (error) {
    console.error(
      "❌ Failed:",
      error instanceof Error ? error.message : error
    );
  }
}

/**
 * Context-Aware Generation: Use domain-specific context
 */
async function contextAwareGeneration() {
  console.log("📚 Context-Aware Generation\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Medical domain examples
  const medicalExamples: Example[] = [
    {
      prompt: "What are common symptoms of influenza?",
      completion: "Common symptoms of influenza include fever, cough, sore throat, body aches, and fatigue. Some patients may also experience headache and chills.",
    },
    {
      prompt: "What is the recommended treatment for a sprained ankle?",
      completion: "The recommended treatment (RICE protocol) includes: Rest to avoid further injury, Ice to reduce inflammation, Compression with bandage, and Elevation to minimize swelling.",
    },
  ];

  const medicalQuestion = "What precautions should be taken for patients with hypertension?";

  console.log(`🏥 Medical Context Question: "${medicalQuestion}"\n`);

  try {
    const result = await client.generateWithExamples(
      medicalQuestion,
      medicalExamples,
      {
        numExamples: 2,
        exampleTemplate: "Q: {prompt}\nA: {completion}",
        queryTemplate: "Q: {prompt}\nA:",
        maxNewTokens: 100,
        temperature: 0.4,
      }
    );

    console.log("✅ Medical Advice:\n");
    console.log(result.generatedText);
    console.log();
  } catch (error) {
    console.error(
      "❌ Failed:",
      error instanceof Error ? error.message : error
    );
  }

  // Technical domain examples
  console.log("━".repeat(60) + "\n");

  const technicalExamples: Example[] = [
    {
      prompt: "What is a REST API?",
      completion: "A REST API is a web service that uses HTTP requests to perform operations. It follows REST principles: using standard HTTP methods (GET, POST, PUT, DELETE), resource-based URLs, and stateless communication.",
    },
    {
      prompt: "How do you handle CORS issues?",
      completion: "CORS (Cross-Origin Resource Sharing) issues are handled by: configuring proper CORS headers on the server, using preflight requests, setting Access-Control-Allow-Origin headers, or using a CORS proxy.",
    },
  ];

  const technicalQuestion = "What are best practices for API versioning?";

  console.log(`💻 Technical Context Question: "${technicalQuestion}"\n`);

  try {
    const result = await client.generateWithExamples(
      technicalQuestion,
      technicalExamples,
      {
        numExamples: 2,
        exampleTemplate: "Q: {prompt}\nA: {completion}",
        queryTemplate: "Q: {prompt}\nA:",
        maxNewTokens: 100,
        temperature: 0.4,
      }
    );

    console.log("✅ Technical Answer:\n");
    console.log(result.generatedText);
    console.log();
  } catch (error) {
    console.error(
      "❌ Failed:",
      error instanceof Error ? error.message : error
    );
  }
}

/**
 * Multi-Turn Conversation Simulation
 */
async function conversationalInference() {
  console.log("💬 Conversational Inference (Multi-turn)\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  const conversationExamples: Example[] = [
    {
      prompt: "User: What's your name?\nAssistant:",
      completion: "I'm Claude, an AI assistant created by Anthropic.",
    },
    {
      prompt: "User: How can you help me?\nAssistant:",
      completion: "I can help you with writing, analysis, math, coding, creative tasks, and answering questions.",
    },
  ];

  const userMessages = [
    "What are your main capabilities?",
    "Can you write code?",
    "Do you learn from conversations?",
  ];

  console.log("🤖 Simulating conversational AI responses...\n");

  for (const message of userMessages) {
    const prompt = `User: ${message}\nAssistant:`;

    console.log(`👤 User: "${message}"`);

    try {
      const result = await client.generateWithExamples(
        prompt,
        conversationExamples,
        {
          numExamples: 2,
          exampleTemplate: "{prompt} {completion}",
          queryTemplate: "{prompt}",
          maxNewTokens: 80,
          temperature: 0.6,
        }
      );

      console.log(`🤖 Assistant: ${result.generatedText}\n`);
    } catch (error) {
      console.error(
        `❌ Failed: ${error instanceof Error ? error.message : error}\n`
      );
    }
  }
}

/**
 * Structured Output Generation
 */
async function structuredOutputGeneration() {
  console.log("📋 Structured Output Generation (JSON/Formatted)\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  // Examples showing structured outputs
  const structuredExamples: Example[] = [
    {
      prompt: "Extract information from: 'John Smith, age 28, works in software engineering'",
      completion: JSON.stringify({
        name: "John Smith",
        age: 28,
        profession: "Software Engineer",
      }, null, 2),
    },
  ];

  const inputText = "Sarah Johnson, age 35, is a professional data scientist";

  console.log(`📝 Input: "${inputText}"\n`);
  console.log("📚 Expected structured output (JSON)...\n");

  try {
    const result = await client.generateWithExamples(
      inputText,
      structuredExamples,
      {
        numExamples: 1,
        exampleTemplate: "Text: {prompt}\nJSON:\n{completion}",
        queryTemplate: "Text: {prompt}\nJSON:",
        maxNewTokens: 100,
        temperature: 0.2,
      }
    );

    console.log("✅ Generated JSON:");
    console.log(result.generatedText);
    console.log();
  } catch (error) {
    console.error(
      "❌ Failed:",
      error instanceof Error ? error.message : error
    );
  }
}

/**
 * Temperature Comparison: Deterministic vs Creative
 */
async function temperatureComparison() {
  console.log("🌡️  Temperature Impact on Output Creativity\n");

  const client = new NeuroQuantumClient({
    hfToken: process.env.HF_TOKEN,
  });

  const examples: Example[] = [
    {
      prompt: "Complete this story: 'Once upon a time, there was a magical forest'",
      completion: "where ancient trees whispered secrets to the wind.",
    },
  ];

  const prompt = "Complete this story: 'In a quiet town, there lived a mysterious stranger'";
  const temperatures = [0.2, 0.7, 1.2];

  console.log(`📖 Prompt: "${prompt}"\n`);
  console.log("Comparing different temperature settings:\n");

  for (const temp of temperatures) {
    const setting = temp < 0.5 ? "Deterministic" : temp < 0.9 ? "Balanced" : "Creative";

    console.log(`🌡️  Temperature: ${temp} (${setting})`);

    try {
      const result = await client.generateWithExamples(
        prompt,
        examples,
        {
          numExamples: 1,
          exampleTemplate: "{prompt}: {completion}",
          queryTemplate: "{prompt}:",
          maxNewTokens: 50,
          temperature: temp,
        }
      );

      console.log(`   ${result.generatedText}\n`);
    } catch (error) {
      console.error(`   ❌ Failed: ${error instanceof Error ? error.message : error}\n`);
    }
  }
}

async function main() {
  try {
    if (!process.env.HF_TOKEN) {
      console.error("❌ Error: HF_TOKEN environment variable is not set");
      console.error("Please set your HuggingFace token:");
      console.error("  export HF_TOKEN='hf_...'");
      process.exit(1);
    }

    await chainOfThoughtReasoning();
    console.log("━".repeat(60) + "\n");

    await contextAwareGeneration();
    console.log("━".repeat(60) + "\n");

    await conversationalInference();
    console.log("━".repeat(60) + "\n");

    await structuredOutputGeneration();
    console.log("━".repeat(60) + "\n");

    await temperatureComparison();

    console.log("✨ All advanced prompting examples completed!");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main();
