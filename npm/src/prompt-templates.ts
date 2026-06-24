/**
 * PromptTemplates — Judgment-specific LLM prompt engineering
 *
 * Each judgment type has optimized system prompts, user prompt templates,
 * and few-shot examples to guide LLM reasoning.
 */

import type { JudgmentType } from "./types.js";

/**
 * Template for a single judgment type
 */
export interface PromptTemplate {
  type: JudgmentType;
  /** System prompt to set LLM behavior */
  systemPrompt: string;
  /** User prompt template with {action}, {context}, {criteria} placeholders */
  userPromptTemplate: string;
  /** Few-shot examples in conversation format */
  examples: Array<{ prompt: string; completion: string }>;
  /** Instructions for JSON response format */
  responseFormatInstructions: string;
}

/**
 * PromptTemplates manager — Provides judgment-specific prompts
 */
export class PromptTemplates {
  private static readonly templates: Record<JudgmentType, PromptTemplate> = {
    safety: {
      type: "safety",
      systemPrompt: `You are a security judgment engine specialized in evaluating whether proposed actions are safe in their intended context.

Your role is to:
1. Identify potential security risks and vulnerabilities
2. Assess mitigation measures and safeguards
3. Consider compliance requirements and restrictions
4. Evaluate access controls and data protection
5. Provide a binary safe/unsafe decision with reasoning

Always be thorough but practical. Consider both the action itself and the context in which it occurs.
Focus on real, actionable security concerns rather than theoretical edge cases.

Respond with valid JSON containing: decision, score (0-100), reasoning, confidence, factors.`,

      userPromptTemplate: `Evaluate the security of this action:

Action: {action}
Context: {context}

{criteria}

Please analyze:
- Security risks in this context
- Any mitigating factors or safeguards
- Compliance or policy implications
- Your confidence in this assessment

Respond ONLY with JSON in this format (no markdown, no explanations):
{"decision":"Yes"|"No","score":0-100,"reasoning":"1-2 sentences","confidence":"high"|"medium"|"low","factors":["factor1","factor2","factor3"]}`,

      examples: [
        {
          prompt: "Action: Enable debug logging\nContext: Production database server\n\nConsider: Data exposure risk, performance impact, unauthorized access",
          completion: `{"decision":"No","score":25,"reasoning":"Debug logging exposes sensitive database queries and credentials to logs, creating data breach risk in production.","confidence":"high","factors":["PII exposure","Log accessibility","Production environment"]}`,
        },
        {
          prompt: "Action: Require API authentication\nContext: Public health data endpoint\n\nConsider: Data sensitivity, access control, rate limiting",
          completion: `{"decision":"Yes","score":85,"reasoning":"API authentication provides essential access control for sensitive health data and enables rate limiting against abuse.","confidence":"high","factors":["Access control","Rate limiting","Data sensitivity"]}`,,
        },
      ],

      responseFormatInstructions: `Return JSON with:
- decision: "Yes" or "No"
- score: 0-100 (0=unsafe, 100=safe)
- reasoning: 1-2 sentence explanation
- confidence: "high", "medium", or "low"
- factors: array of 3-5 key factors`,
    },

    ethics: {
      type: "ethics",
      systemPrompt: `You are an ethics evaluation engine specialized in assessing the ethical implications of proposed actions.

Your role is to:
1. Identify stakeholders affected by the action
2. Assess impacts on fairness, transparency, and consent
3. Consider potential harms and privacy implications
4. Evaluate alignment with ethical principles
5. Provide a nuanced ethical assessment

Balance competing values and acknowledge ethical tensions.
Consider both immediate and long-term consequences.
Be sensitive to cultural and contextual factors.`,

      userPromptTemplate: `Evaluate the ethical implications of this action:

Action: {action}
Stakeholders: {stakeholders}
Potential harms: {potentialHarms}

{criteria}

Please analyze:
- Who is affected and how
- Fairness and equity implications
- Privacy and consent considerations
- Long-term societal impacts
- Alignment with ethical principles

Respond ONLY with JSON:
{"decision":"Yes"|"No","score":0-100,"reasoning":"1-2 sentences","confidence":"high"|"medium"|"low","factors":["factor1","factor2","factor3"]}`,

      examples: [
        {
          prompt: "Action: Use user location data for personalized ads\nStakeholders: Users, advertisers, platform\nHarms: Privacy invasion, manipulation",
          completion: `{"decision":"No","score":30,"reasoning":"Using location data for targeted ads raises consent and manipulation concerns without clear user benefit or transparency.","confidence":"high","factors":["Privacy","Consent","User autonomy"]}`,
        },
        {
          prompt: "Action: Provide free mental health resources\nStakeholders: Low-income individuals, mental health providers\nHarms: Potential quality concerns",
          completion: `{"decision":"Yes","score":80,"reasoning":"Providing free mental health resources promotes equity and access while benefiting vulnerable populations, with manageable quality risks.","confidence":"high","factors":["Equity","Health access","Social benefit"]}`,
        },
      ],

      responseFormatInstructions: `Return JSON with:
- decision: "Yes" or "No"
- score: 0-100 (0=unethical, 100=ethical)
- reasoning: 1-2 sentence explanation
- confidence: "high", "medium", or "low"
- factors: array of 3-5 key ethical factors`,
    },

    quality: {
      type: "quality",
      systemPrompt: `You are a quality evaluation engine specialized in assessing whether content or outputs meet specified standards.

Your role is to:
1. Evaluate accuracy and factual correctness
2. Assess completeness and thoroughness
3. Check clarity and understandability
4. Verify relevance to stated intent
5. Identify strengths and weaknesses

Provide constructive assessment focused on practical improvement.
Consider the specific use case and requirements.
Be fair in evaluating against realistic standards.`,

      userPromptTemplate: `Evaluate the quality of this content:

Content: {action}
Requirements: {criteria}
User intent: {potentialHarms}

{stakeholders}

Please assess:
- Accuracy and factual correctness
- Completeness relative to requirements
- Clarity and readability
- Relevance to user intent
- Key quality strengths and gaps

Respond ONLY with JSON:
{"decision":"Yes"|"No","score":0-100,"reasoning":"1-2 sentences","confidence":"high"|"medium"|"low","factors":["factor1","factor2","factor3"]}`,

      examples: [
        {
          prompt: "Content: Brief API documentation with examples\nRequirements: Complete reference for all endpoints, clear error handling\nIntent: Developers can integrate without support",
          completion: `{"decision":"No","score":45,"reasoning":"Documentation provides helpful examples but lacks error handling details and complete endpoint reference, requiring developer support.","confidence":"high","factors":["Incomplete coverage","Missing error docs","Good examples"]}`,
        },
        {
          prompt: "Content: Comprehensive guide with examples and troubleshooting\nRequirements: Clear, complete, actionable\nIntent: Users can self-serve",
          completion: `{"decision":"Yes","score":82,"reasoning":"Guide is clear, comprehensive, and actionable with good examples and troubleshooting section supporting self-service use.","confidence":"high","factors":["Complete coverage","Clear examples","Good structure"]}`,
        },
      ],

      responseFormatInstructions: `Return JSON with:
- decision: "Yes" or "No"
- score: 0-100 (0=poor quality, 100=excellent)
- reasoning: 1-2 sentence explanation
- confidence: "high", "medium", or "low"
- factors: array of 3-5 quality dimensions`,
    },

    risk: {
      type: "risk",
      systemPrompt: `You are a risk assessment engine specialized in evaluating the risk level and manageability of proposed actions.

Your role is to:
1. Identify potential failure modes and risks
2. Assess likelihood and potential impact
3. Evaluate risk mitigation strategies
4. Determine if risks are reversible and controllable
5. Provide a risk level assessment

Balance catastrophic risks heavily while maintaining proportionality.
Consider both probability and impact.
Evaluate mitigation effectiveness realistically.`,

      userPromptTemplate: `Assess the risk level of this action:

Action: {action}
Context: {context}
Risk tolerance: {criteria}

{potentialHarms}

Please evaluate:
- Key risks and failure modes
- Likelihood and impact assessment
- Available mitigations and controls
- Reversibility of the action
- Overall risk level vs tolerance

Respond ONLY with JSON:
{"decision":"Yes"|"No","score":0-100,"reasoning":"1-2 sentences","confidence":"high"|"medium"|"low","factors":["factor1","factor2","factor3"]}`,

      examples: [
        {
          prompt: "Action: Deploy new code to production without automated rollback\nContext: E-commerce platform during peak season\nTolerance: Low\nRisks: Outage, revenue loss, customer data exposure",
          completion: `{"decision":"No","score":15,"reasoning":"Deploying without rollback capability in high-traffic environment creates unacceptable outage and financial risk.","confidence":"high","factors":["No rollback plan","Peak traffic period","High revenue impact"]}`,
        },
        {
          prompt: "Action: Add A/B test variant with gradual rollout\nContext: Internal tools\nTolerance: Medium\nRisks: Minor UI issues, user confusion",
          completion: `{"decision":"Yes","score":72,"reasoning":"Gradual A/B test rollout on internal tools allows monitoring and quick rollback if issues occur.","confidence":"high","factors":["Gradual rollout","Reversible","Low impact scope"]}`,
        },
      ],

      responseFormatInstructions: `Return JSON with:
- decision: "Yes" or "No"
- score: 0-100 (0=extreme risk, 100=negligible risk)
- reasoning: 1-2 sentence explanation
- confidence: "high", "medium", or "low"
- factors: array of 3-5 risk factors`,
    },

    decision: {
      type: "decision",
      systemPrompt: `You are a decision-making engine specialized in evaluating whether proposed courses of action are recommended.

Your role is to:
1. Analyze multiple dimensions of the decision
2. Weigh pros and cons
3. Consider alternatives and trade-offs
4. Evaluate alignment with stated goals
5. Provide a clear recommendation

Be practical and decisive while acknowledging trade-offs.
Consider both quantitative and qualitative factors.
Provide clear reasoning for your recommendation.`,

      userPromptTemplate: `Evaluate this decision:

Proposed action: {action}
Decision context: {context}

{criteria}

Please consider:
- Benefits and drawbacks
- Alignment with goals
- Feasibility and resource requirements
- Alternative options
- Key decision factors

Respond ONLY with JSON:
{"decision":"Yes"|"No","score":0-100,"reasoning":"1-2 sentences","confidence":"high"|"medium"|"low","factors":["factor1","factor2","factor3"]}`,

      examples: [
        {
          prompt: "Action: Invest in new ML infrastructure\nContext: Startup with limited runway\nGoals: Rapid feature development, cost efficiency",
          completion: `{"decision":"No","score":35,"reasoning":"ML infrastructure investment conflicts with startup's need for rapid iteration and cost efficiency on limited runway.","confidence":"high","factors":["High upfront cost","Long payoff period","Better alternatives exist"]}`,
        },
        {
          prompt: "Action: Implement automated testing framework\nContext: Growing team, frequent deployments\nGoals: Reliability, development speed",
          completion: `{"decision":"Yes","score":88,"reasoning":"Automated testing reduces bugs, enables faster deployment, and pays for itself in team productivity and reliability.","confidence":"high","factors":["Deployment frequency","Team growth","Reliability requirement"]}`,
        },
      ],

      responseFormatInstructions: `Return JSON with:
- decision: "Yes" or "No"
- score: 0-100 (0=not recommended, 100=strongly recommended)
- reasoning: 1-2 sentence explanation
- confidence: "high", "medium", or "low"
- factors: array of 3-5 decision factors`,
    },

    priority: {
      type: "priority",
      systemPrompt: `You are a prioritization engine specialized in ranking tasks and initiatives by importance and urgency.

Your role is to:
1. Evaluate urgency and importance of each task
2. Assess impact and effort trade-offs
3. Consider dependencies and sequencing
4. Account for constraints and resources
5. Provide ranked priority scores

Balance urgent crises with important long-term work.
Consider team capacity and dependencies.
Provide clear reasoning for relative priorities.`,

      userPromptTemplate: `Prioritize these tasks:

Tasks: {action}
Constraints: {context}

{criteria}

Consider:
- Urgency and time sensitivity
- Impact on users and business
- Effort and resource requirements
- Dependencies and sequencing
- Team capacity

Provide a JSON array ranking tasks from highest to lowest priority:
[{"task":"Task 1","score":95,"reasoning":"..."},...]`,

      examples: [
        {
          prompt: "Tasks: 1) Fix security vulnerability, 2) UI redesign, 3) Performance optimization\nConstraints: 2 engineers, 1 week\nTimeline: Urgent security patch expected",
          completion: `[{"task":"Fix security vulnerability","score":100,"reasoning":"Critical security issue must be patched immediately before any other work.","factors":["Security risk","Time-sensitive","Required fix"]},{"task":"Performance optimization","score":65,"reasoning":"Can improve user experience but less urgent than security.","factors":["User impact","Medium effort"]},{"task":"UI redesign","score":40,"reasoning":"Nice-to-have improvement that can wait until security patch is complete.","factors":["Lower priority","High effort","Deferrable"]}]`,
        },
      ],

      responseFormatInstructions: `Return JSON array of tasks sorted by priority (descending):
[{"task":"Task name","score":0-100,"reasoning":"...","factors":["factor1","factor2"]},...]`,
    },
  };

  /**
   * Get template for a judgment type
   */
  static getTemplate(type: JudgmentType): PromptTemplate {
    const template = this.templates[type];
    if (!template) {
      throw new Error(`Unknown judgment type: ${type}`);
    }
    return template;
  }

  /**
   * Build a complete prompt for judgment
   *
   * @param type - Judgment type
   * @param action - Action or content to judge
   * @param context - Situational context
   * @param criteria - Additional criteria/constraints
   * @returns System and user prompts
   */
  static buildPrompt(
    type: JudgmentType,
    action: string,
    context: string,
    criteria?: string | Record<string, unknown>
  ): { system: string; user: string } {
    const template = this.getTemplate(type);

    // Format criteria
    let criteriaText = "";
    if (criteria) {
      if (typeof criteria === "string") {
        criteriaText = criteria;
      } else {
        criteriaText = Object.entries(criteria)
          .map(([key, val]) => `${key}: ${val}`)
          .join("\n");
      }
    }

    // Build user prompt from template
    let user = template.userPromptTemplate
      .replace("{action}", sanitizeInput(action))
      .replace("{context}", sanitizeInput(context))
      .replace("{criteria}", criteriaText);

    // Remove placeholder lines that weren't filled
    user = user
      .split("\n")
      .filter((line) => !line.includes("{") && !line.includes("}"))
      .join("\n");

    return {
      system: template.systemPrompt,
      user: user.trim(),
    };
  }

  /**
   * Get all available judgment types
   */
  static getAvailableTypes(): JudgmentType[] {
    return Object.keys(this.templates) as JudgmentType[];
  }
}

/**
 * Sanitize user input to prevent prompt injection
 */
function sanitizeInput(input: string): string {
  // Remove any {placeholder} patterns that might interfere
  return input
    .replace(/\{[^}]*\}/g, "")
    .replace(/["'`]/g, "")
    .trim();
}
