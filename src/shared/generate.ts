import { Agent } from "@mastra/core/agent";
import {
  ANSWER_SYSTEM_PROMPT,
  DIRECT_ANSWER_SYSTEM_PROMPT,
  buildAnswerPrompt,
} from "./answer-prompt.js";
import { GENERATION_MODEL } from "./llm-client.js";
import type { FetchResult } from "./types.js";

/**
 * 3パターン共通の回答生成。
 *
 * モデル・システムプロンプト・プロンプト組み立てを1箇所に閉じ込め、
 * どのパターンからも**同じ経路**で生成が走ることを保証する
 * （CLAUDE.md の不変条件2）。
 */

export interface GenerationOutcome {
  text: string;
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
}

/** 回答生成専用エージェント。ツールを持たない＝検索結果をそのまま使う */
const answerAgent = new Agent({
  id: "answer-agent",
  name: "Answer Agent",
  instructions: ANSWER_SYSTEM_PROMPT,
  model: GENERATION_MODEL,
});

/** FullOutput.usage / totalUsage は provider によって欠けうるので安全に取り出す */
export function readUsage(usage: unknown): { inputTokens: number; outputTokens: number } {
  const u = usage as { inputTokens?: number; outputTokens?: number } | undefined;
  return {
    inputTokens: u?.inputTokens ?? 0,
    outputTokens: u?.outputTokens ?? 0,
  };
}

export async function generateAnswer(
  question: string,
  docs: FetchResult[],
): Promise<GenerationOutcome> {
  const result = await answerAgent.generate(buildAnswerPrompt(question, docs));
  // ループを含むパターンとの比較のため、常に totalUsage（全ステップ合計）を見る
  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);

  return {
    text: result.text,
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}

/**
 * 直接回答スキル専用の生成（パターン3の direct 分岐のみ）。
 * 文書接地プロンプトを使わない理由は answer-prompt.ts の
 * DIRECT_ANSWER_SYSTEM_PROMPT のコメントを参照。
 */
const directAgent = new Agent({
  id: "direct-answer-agent",
  name: "Direct Answer Agent",
  instructions: DIRECT_ANSWER_SYSTEM_PROMPT,
  model: GENERATION_MODEL,
});

export async function generateDirectAnswer(question: string): Promise<GenerationOutcome> {
  const result = await directAgent.generate(question);
  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);

  return {
    text: result.text,
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}
