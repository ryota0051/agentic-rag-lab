import * as lancedb from "@lancedb/lancedb";
import { openChunksTable } from "../index/build-index.js";
import type { SearchHit } from "../shared/types.js";
import { embedQuery } from "./embed.js";
import { toSearchHit, type RawHit } from "./to-hit.js";

/**
 * パターン2・3が使うハイブリッド検索（BM25 + ベクトル）。
 *
 * LanceDB のハイブリッド検索は2つの結果集合を融合するステップを必ず要求し、
 * 既定は RRF（Reciprocal Rank Fusion）。これは**順位の統計的融合**であって、
 * Cross-Encoder による意味的な再スコアリングではない。本プロジェクトが見送ったのは後者
 * （docs/decisions/0005-no-cross-encoder-rerank.md）。
 *
 * 注意: TypeScript SDK には Python の `.vector()/.text()` パターンがなく、
 * 埋め込みを自前で計算して `.nearestTo()` に渡す必要がある。
 */

/** RRFReranker はステートレスなので使い回す */
let rerankerPromise: Promise<lancedb.rerankers.RRFReranker> | undefined;

function getReranker(): Promise<lancedb.rerankers.RRFReranker> {
  rerankerPromise ??= lancedb.rerankers.RRFReranker.create();
  return rerankerPromise;
}

export interface HybridSearchOptions {
  /**
   * 指定した article_id を検索対象から除外する。
   * パターン3の記事多様化（`search-fetch-loop.ts` の `diversifySeed`）専用。
   * パターン1・2は使わないので、渡さない限り従来と挙動は変わらない。
   */
  excludeArticleIds?: string[];
}

export async function hybridSearch(
  query: string,
  k: number,
  opts?: HybridSearchOptions,
): Promise<SearchHit[]> {
  const table = await openChunksTable();
  const [vector, reranker] = await Promise.all([embedQuery(query), getReranker()]);

  let q = table.query().fullTextSearch(query).nearestTo(vector).rerank(reranker);
  if (opts?.excludeArticleIds?.length) {
    const list = opts.excludeArticleIds
      .map((id) => `'${id.replace(/'/g, "''")}'`)
      .join(",");
    q = q.where(`article_id NOT IN (${list})`);
  }

  const rows = (await q.limit(k).toArray()) as RawHit[];

  return rows.map(toSearchHit);
}
