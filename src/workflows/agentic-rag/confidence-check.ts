import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { readUsage } from "../../shared/generate.js";
import { GENERATION_MODEL } from "../../shared/llm-client.js";
import type { FetchResult } from "../../shared/types.js";

/**
 * 確信度チェック（FLARE的な事後補正）。
 *
 * 検索要否の事前判定は原理的に完璧にはできない（LLMは自身の知識境界を正確に
 * 自己申告できない = calibration 問題）。完璧な事前判定を目指す代わりに、
 * 生成後に不確実さを検出して追加検索に戻す保険を後段に置く
 * （docs/architecture.md「検索要否は完璧に判定できるか」）。
 *
 * search 内部の turn 上限とは**独立した安全装置**。両者を併用する。
 */

/**
 * 追加検索への再突入回数の上限。turn 上限とは別軸のガード。
 *
 * 根拠の充足チェック（`evidence-check.ts`）を収集段階に置いたことで、
 * このチェックの役割は「根拠は足りているのに回答がそれを正しく使えていない」
 * ケースに絞られた。守備範囲が狭くなったぶん 2 → 1 に減らしている。
 */
export const MAX_RECHECKS = 1;

const ConfidenceVerdict = z.object({
  sufficient: z
    .boolean()
    .describe("提供された根拠だけで質問に十分答えられているなら true"),
  missing: z
    .string()
    .describe("不足している情報。sufficient が true の場合は空文字"),
  followup_query: z
    .string()
    .describe(
      "追加検索すべきクエリ。sufficient が true の場合は空文字",
    ),
});

const checkerAgent = new Agent({
  id: "confidence-checker",
  name: "Confidence Checker",
  instructions: `あなたは、生成された回答が提供された根拠で十分に裏付けられているかを判定します。

以下のいずれかに該当する場合は sufficient=false としてください:
- 回答が根拠に含まれない情報を含んでいる
- 回答が「判断できません」「情報がありません」と述べている
- 質問の一部にしか答えられていない
- 根拠が質問の主題と食い違っている

十分に裏付けられている場合のみ sufficient=true としてください。
sufficient=false の場合は、不足を埋めるための具体的な検索クエリを followup_query に書いてください。`,
  model: GENERATION_MODEL,
});

export interface ConfidenceResult {
  sufficient: boolean;
  missing: string;
  followupQuery: string;
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
}

export async function checkConfidence(
  question: string,
  answer: string,
  docs: FetchResult[],
): Promise<ConfidenceResult> {
  const context = docs
    .map((d, i) => `[${i + 1}] ${d.heading_path}\n${d.text}`)
    .join("\n\n---\n\n");

  const result = await checkerAgent.generate(
    `質問: ${question}\n\n生成された回答:\n${answer}\n\n根拠として与えた文書:\n${context || "(なし)"}`,
    { structuredOutput: { schema: ConfidenceVerdict } },
  );

  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);
  const verdict = result.object;

  return {
    // 判定が取れなかった場合は「十分」とみなしてループを止める。
    // ここでフォールバックを false にすると、判定失敗時に無限に追加検索が走る
    sufficient: verdict?.sufficient ?? true,
    missing: verdict?.missing ?? "",
    followupQuery: verdict?.followup_query ?? "",
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}
