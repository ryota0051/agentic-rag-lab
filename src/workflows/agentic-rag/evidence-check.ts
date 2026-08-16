import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { readUsage } from "../../shared/generate.js";
import { FINAL_CONTEXT_K, GENERATION_MODEL } from "../../shared/llm-client.js";
import type { FetchResult } from "../../shared/types.js";

/**
 * 根拠の充足チェック。**回答生成の前**に、集めた根拠だけで質問に答えられるかを判定する。
 *
 * ## なぜ回答後ではなく収集段階なのか
 *
 * 初回実験では、確信度チェックを回答生成の**後**に置いていた。結果、
 * ターン数の中央値は 1 で、46問中25問が1回の検索で完結した。
 * つまり **search→評価→再検索のループがほとんど働いていなかった**。
 *
 * 原因は2つある。
 *
 * 1. エージェント自身が1回目の検索結果で「十分」と判断してループを抜ける
 * 2. 回答生成後のチェックは、**すでに書かれた回答**を見て判定するため、
 *    根拠が足りなくても文章としては尤もらしく仕上がっていると「十分」と判定しやすい
 *
 * 2 が効いている。一度回答文が生成されると、その流暢さが不足を覆い隠す。
 * 根拠そのものを見て「この質問に答えるには何が足りないか」を問う方が、
 * 不足を検出しやすい。
 *
 * multihop 質問（複数の文書を統合しないと答えられない）では特に効くはず。
 * 片方の文書しか取れていない状態を「足りない」と言えるのは、
 * 回答文ではなく根拠を見ているときだけである。
 */

const EvidenceVerdict = z.object({
  sufficient: z
    .boolean()
    .describe("提供された根拠だけで質問に完全に答えられるなら true"),
  missing: z
    .string()
    .describe(
      "答えるために足りていない情報を具体的に書く。sufficient が true なら空文字",
    ),
  followup_query: z
    .string()
    .describe(
      "不足を埋めるための検索クエリ。既に試したクエリとは別の語彙を使うこと。" +
        "sufficient が true なら空文字",
    ),
});

const checkerAgent = new Agent({
  id: "evidence-checker",
  name: "Evidence Checker",
  instructions: `あなたは、集められた根拠だけで質問に答えられるかを判定します。
**回答文は与えられません。根拠そのものを見て判断してください。**

質問を構成要素に分解し、それぞれに対応する記述が根拠の中にあるかを確認してください。

sufficient=false とすべき典型例:
- 質問が2つ以上の対象を比較・統合しているのに、根拠が片方の対象しかカバーしていない
- 質問が具体的な数値・設定値・手順を求めているのに、根拠にその値が書かれていない
- 根拠が質問の主題と関連はあるが、問われている論点そのものには触れていない
- 根拠が0件

sufficient=true とすべき場合:
- 質問のすべての構成要素に対応する記述が根拠の中にある

「だいたい関連しているから十分」と判断しないでください。
足りない場合は、何が足りないかを具体的に述べ、それを見つけるための
検索クエリを提案してください。既に試したクエリと同じ語彙を繰り返さないこと。`,
  model: GENERATION_MODEL,
});

export interface EvidenceVerdictResult {
  sufficient: boolean;
  missing: string;
  followupQuery: string;
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
  /**
   * 構造化出力が取れずフォールバックしたか。
   * フォールバック先が sufficient=true なので、これを返さないと
   * **判定できなかった実行が「1回で足りた」＝ループ不要として記録される**
   */
  structuredOutputFailed: boolean;
}

export async function checkEvidence(
  question: string,
  docs: FetchResult[],
  triedQueries: string[],
): Promise<EvidenceVerdictResult> {
  const context = docs.length
    ? docs.map((d, i) => `[${i + 1}] ${d.heading_path}\n${d.text}`).join("\n\n---\n\n")
    : "(根拠が1件も集まっていません)";

  const result = await checkerAgent.generate(
    `質問: ${question}\n\n` +
      `これまでに試した検索クエリ: ${triedQueries.map((q) => `「${q}」`).join(" ") || "(なし)"}\n\n` +
      `集まった根拠（${docs.length}/${FINAL_CONTEXT_K} 件）:\n${context}`,
    { structuredOutput: { schema: EvidenceVerdict } },
  );

  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);
  const verdict = result.object;

  return {
    // 判定が取れなかった場合は「十分」に倒してループを止める。
    // false に倒すと判定失敗時に追加検索が延々と走る
    sufficient: verdict?.sufficient ?? true,
    missing: verdict?.missing ?? "",
    followupQuery: verdict?.followup_query ?? "",
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
    structuredOutputFailed: verdict === undefined,
  };
}
