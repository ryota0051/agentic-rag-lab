import { FINAL_CONTEXT_K } from "../shared/llm-client.js";
import { fetchChunks } from "../search/fetch.js";
import { vectorSearch } from "../search/vector-search.js";
import { generateAnswer } from "../shared/generate.js";
import { toCitations } from "../shared/answer-prompt.js";
import { collectRetrievedChunkIds, countContextChars } from "../shared/evidence-stats.js";
import type { RagRunResult } from "../shared/types.js";

/**
 * パターン1: ナイーブRAG（ベースライン）。
 *
 * クエリを埋め込み → ベクトル top-k → そのまま生成に渡す。
 * ツール呼び出しなし、ループなし、クエリ整形なし。
 *
 * **ここに手を入れないこと。** ベースラインに小細工を入れると、
 * 「エージェント化で何が改善したか」の基準線が動いて比較が無意味になる。
 */
export async function runNaiveRag(question: string): Promise<RagRunResult> {
  const started = Date.now();

  const hits = await vectorSearch(question, FINAL_CONTEXT_K);
  // ナイーブRAGでも本文は必要なので fetch は通すが、範囲拡張はしない（chunk_only）。
  // これによりパターン間の差は「どのchunkを選んだか」と「ループの有無」に限定される
  const docs = await fetchChunks(
    hits.map((h) => h.chunk_id),
    "chunk_only",
  );

  const gen = await generateAnswer(question, docs);

  return {
    pattern: "naive",
    question,
    answer: gen.text,
    citations: toCitations(docs),
    retrieved_chunk_ids: collectRetrievedChunkIds(docs),
    contextChars: countContextChars(docs),
    usage: {
      inputTokens: gen.inputTokens,
      outputTokens: gen.outputTokens,
      llmCalls: gen.llmCalls,
    },
    latencyMs: Date.now() - started,
    turns: 0,
  };
}
