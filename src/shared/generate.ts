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

/**
 * 思考タグを本文から取り除く。
 *
 * ローカルの推論モデル（Qwen3.8 等）は、reasoning_effort を切っていても
 * `<think>…</think>` を本文に混ぜて返すことがある。素通しすると
 * 回答文に思考過程が紛れ込み、faithfulness の目視評価が破綻する。
 *
 * **3パターン共通のこのファイルに置くこと。** どれか1つのワークフローだけに入れると
 * 「回答の後処理がパターンによって違う」状態になり、比較実験が無効になる
 * （CLAUDE.md の不変条件2）。クラウドモデルの出力には該当タグが無いので実害はない。
 *
 * 推論モデルは「思考 → 回答」の順に吐くため、閉じタグがあれば**最後の閉じタグ以降**が回答本文。
 * 開きタグの有無で判定しないのは、開きタグが無いまま `</think>` だけが出てくる
 * テンプレート実装があるため（その場合も後半だけを採れば正しく回答が残る）。
 */
export function stripReasoningTags(text: string): string {
  const lastClose = text.toLowerCase().lastIndexOf("</think>");
  if (lastClose !== -1) return text.slice(lastClose + "</think>".length).trim();

  // 閉じタグが無い＝思考の途中で出力が打ち切られた。開きタグより前だけが使える本文
  const open = text.toLowerCase().indexOf("<think>");
  if (open !== -1) return text.slice(0, open).trim();

  return text;
}

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
    text: stripReasoningTags(result.text),
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
    text: stripReasoningTags(result.text),
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}
