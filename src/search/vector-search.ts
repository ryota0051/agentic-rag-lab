import type { SearchHit } from "../shared/types.js";
import { openChunksTable } from "../index/build-index.js";
import { embedQuery } from "./embed.js";
import { SNIPPET_CHARS, toSearchHit, type RawHit } from "./to-hit.js";

/**
 * パターン1（ナイーブRAG）用のベクトル検索。
 *
 * クエリをそのまま埋め込んで top-k を引くだけ。クエリ整形も言い換えもしない
 * ——これがベースラインの定義であり、ここに小細工を入れると比較の意味が消える。
 */
export async function vectorSearch(query: string, k: number): Promise<SearchHit[]> {
  const table = await openChunksTable();
  const vector = await embedQuery(query);

  const rows = (await table
    .query()
    .nearestTo(vector)
    .limit(k)
    .toArray()) as RawHit[];

  return rows.map(toSearchHit);
}

export { SNIPPET_CHARS };
