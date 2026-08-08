import { createStep, createWorkflow } from "@mastra/core/workflows";
import { z } from "zod";
import type { RagPattern, RagRunResult } from "../shared/types.js";
import { runAgenticRag } from "./agentic-rag/index.js";
import { runHybridRag } from "./hybrid-rag.js";
import { runNaiveRag } from "./naive-rag.js";

/**
 * 3パターンを Mastra ワークフローとして公開する薄いラッパ。
 *
 * 実装本体は各 `run*Rag()` 関数側に置き、ここではワークフロー化だけを行う。
 * ワークフローにする理由は2つ:
 *  1. `runEvals({ target })` が Agent か Workflow を要求するため（Phase 5 の評価で必要）
 *  2. Mastra のトレースに実行が記録され、turn 数やツール呼び出しを後から追えるため
 *
 * ロジックを関数側に残しておくことで、スクリプトから直接呼んで手動確認することもできる。
 */

const inputSchema = z.object({ question: z.string() });

/** RagRunResult をそのまま出力スキーマにする。スコアラーがこの形状に依存する */
const outputSchema = z.object({
  pattern: z.enum(["naive", "hybrid", "agentic"]),
  question: z.string(),
  answer: z.string(),
  citations: z.array(
    z.object({
      chunk_id: z.string(),
      title: z.string(),
      url: z.string(),
      heading_path: z.string(),
    }),
  ),
  retrieved_chunk_ids: z.array(z.string()),
  usage: z.object({
    inputTokens: z.number(),
    outputTokens: z.number(),
    llmCalls: z.number(),
  }),
  latencyMs: z.number(),
  turns: z.number(),
  selectedSkill: z.enum(["search", "direct"]).optional(),
  queryTrace: z.array(z.string()).optional(),
});

function buildRagWorkflow(
  pattern: RagPattern,
  run: (question: string) => Promise<RagRunResult>,
) {
  const step = createStep({
    id: `${pattern}-step`,
    inputSchema,
    outputSchema,
    execute: async ({ inputData }) => run(inputData.question),
  });

  return createWorkflow({
    id: `${pattern}-rag`,
    inputSchema,
    outputSchema,
  })
    .then(step)
    .commit();
}

export const naiveRagWorkflow = buildRagWorkflow("naive", runNaiveRag);
export const hybridRagWorkflow = buildRagWorkflow("hybrid", runHybridRag);
export const agenticRagWorkflow = buildRagWorkflow("agentic", runAgenticRag);

export const ragWorkflows = {
  naiveRagWorkflow,
  hybridRagWorkflow,
  agenticRagWorkflow,
};
