import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { readUsage } from "../../shared/generate.js";
import { GENERATION_MODEL } from "../../shared/llm-client.js";

/**
 * 質問分解。複合質問（独立した複数の対象・論点を含む質問）を、
 * 対象ごとの単一クエリに分解する。
 *
 * ## 狙い
 *
 * `docs/decisions/0009-agentic-loss-decomposition.md` の診断で、単一の言い換えクエリは
 * 元の質問より検索性能が下がることが分かっている（複数対象を1文に混ぜると、
 * BM25は一致度が薄まり、ベクトルは意味が曖昧になる）。
 *
 * 一方 `experiments/2026-08-09f-snippet-500.md` で単発ハイブリッドに負けた質問は、
 * 「AではX、BではYですか」のように**独立した2つの対象**を1問に混ぜたものに偏っていた。
 * これは言い換え（同じ対象を別の言葉で言う）ではなく分解（対象を分ける）でしか解けない。
 *
 * 「言い換えは足すな、置き換えろ」という search-fetch-loop.ts の指示と役割が違うため、
 * 独立したステップとして切り出す。
 */

const DecomposeResult = z.object({
  is_compound: z
    .boolean()
    .describe(
      "質問が独立した複数の対象・論点を含み、それぞれ別々に検索した方が両方を見つけやすいなら true。" +
        "単一の対象・論点についての質問（言い換えでは対応できるが分解する意味がない質問）は false",
    ),
  sub_queries: z
    .array(z.string())
    .max(2)
    .describe(
      "分解した検索クエリ。対象ごとに1つ、最大2件。それぞれ元の質問と同じ長さかそれ以下にすること。" +
        "is_compound が false なら空配列",
    ),
});

const decomposerAgent = new Agent({
  id: "query-decomposer",
  name: "Query Decomposer",
  instructions: `あなたは質問を検索クエリに分解する担当です。

複合質問（例:「AではX、BではYですか」「AとBの違いは」「AとBに共通する〜は」）を、
対象ごとの独立した検索クエリに分解してください。

分解のルール:
- **分解は対象を分けるためだけに行う。** 語を継ぎ足して長くしないこと。
  各クエリは元の質問と同じ長さかそれ以下の、その対象だけについての短い文にする
- 対象は最大2つまで。3つ以上ある場合は最も重要な2つに絞る
- 単一の対象・論点しか含まない質問は分解しない（is_compound=false, sub_queries=[]）。
  「〜の理由は」「〜する方法は」のような単純な質問を無理に分解しないこと`,
  model: GENERATION_MODEL,
});

export interface DecomposeOutcome {
  subQueries: string[];
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
}

export async function decomposeQuery(question: string): Promise<DecomposeOutcome> {
  const result = await decomposerAgent.generate(question, {
    structuredOutput: { schema: DecomposeResult },
  });

  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);
  const verdict = result.object;

  return {
    // 構造化出力が取れなかった場合は分解しない（元の質問1本の従来動作にフォールバック）
    subQueries: verdict?.is_compound ? (verdict.sub_queries ?? []).slice(0, 2) : [],
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}
