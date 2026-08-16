import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { GENERATION_MODEL } from "../../shared/llm-client.js";
import { readUsage } from "../../shared/generate.js";
import {
  DIRECT_ANSWER_SKILL_DESCRIPTION,
  SEARCH_SKILL_DESCRIPTION,
  SKILL_SELECTION_POLICY,
} from "../../skills/descriptions.js";

/**
 * スキル選択。検索要否の判定を独立ステップとして持たず、スキル選択に兼ねさせる
 * （docs/decisions/0001-skill-based-routing.md）。
 *
 * 判定ロジックはコードではなく2つのdescriptionに集約されている。
 * ここで structured output を使い、どちらが選ばれたかを確実に構造化して記録する
 * （skill-selection-accuracy の採点対象になるため、自由文で返させてはいけない）。
 */

const SkillChoice = z.object({
  skill: z
    .enum(["search", "direct"])
    .describe("使用するスキル。search=検索スキル, direct=直接回答スキル"),
  reason: z.string().describe("なぜそのスキルを選んだかの短い理由"),
});

const selectorAgent = new Agent({
  id: "skill-selector",
  name: "Skill Selector",
  instructions: `あなたはユーザーの質問に対して、使用するスキルを1つ選ぶルーターです。

## 利用可能なスキル

### search（検索スキル）
${SEARCH_SKILL_DESCRIPTION}

### direct（直接回答スキル）
${DIRECT_ANSWER_SKILL_DESCRIPTION}

${SKILL_SELECTION_POLICY}

どちらか一方を必ず選んでください。`,
  model: GENERATION_MODEL,
});

export interface SkillSelection {
  skill: "search" | "direct";
  reason: string;
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
  /**
   * 構造化出力が取れずフォールバックしたか。
   * フォールバック先が "search"（＝境界ケース以外では正解）なので、
   * これを返さないと**判定できなかった実行がスキル選択精度に正解として混ざる**
   */
  structuredOutputFailed: boolean;
}

export async function selectSkill(question: string): Promise<SkillSelection> {
  const result = await selectorAgent.generate(question, {
    structuredOutput: { schema: SkillChoice },
  });

  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);
  const choice = result.object;

  return {
    // 構造化出力が取れなかった場合は検索側に倒す。
    // 非対称コスト設計（迷ったら検索）をフォールバックにも一貫させる
    skill: choice?.skill ?? "search",
    reason: choice?.reason ?? "(構造化出力の取得に失敗したため検索にフォールバック)",
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
    structuredOutputFailed: choice === undefined,
  };
}
