import { FINAL_CONTEXT_K } from "../shared/llm-client.js";
import { fetchChunks } from "../search/fetch.js";
import { hybridSearch } from "../search/hybrid-search.js";
import { generateAnswer } from "../shared/generate.js";
import { toCitations } from "../shared/answer-prompt.js";
import { collectRetrievedChunkIds, countContextChars } from "../shared/evidence-stats.js";
import type { RagRunResult } from "../shared/types.js";

/**
 * パターン2: ハイブリッド検索（BM25 + ベクトル、RRF融合）単発。
 *
 * パターン1との差分は**検索方法のみ**。ループはさせず1回で打ち切る。
 * これにより「検索精度の改善による効果」だけを切り出して観測できる。
 * パターン3との差分が「エージェント化（ループ）による効果」になる。
 *
 * Cross-Encoder によるリランクは行わない（docs/decisions/0005-no-cross-encoder-rerank.md）。
 */
export async function runHybridRag(question: string): Promise<RagRunResult> {
  const started = Date.now();

  const hits = await hybridSearch(question, FINAL_CONTEXT_K);
  const docs = await fetchChunks(
    hits.map((h) => h.chunk_id),
    "chunk_only",
  );

  const gen = await generateAnswer(question, docs);

  return {
    pattern: "hybrid",
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
