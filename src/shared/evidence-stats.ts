import type { FetchResult } from "./types.js";

/**
 * fetch 結果から採点用の統計を取る。3パターンすべてがこれを通ることで、
 * 「何をもって取得済みとみなすか」の定義がズレないようにする。
 */

/**
 * プロンプトに入った chunk_id をすべて集める。
 *
 * `FetchResult.chunk_id` は範囲拡張の**起点**でしかない。
 * `with_neighbors` や `whole_section` を使うと、実際には
 * `included_chunk_ids` にある複数のチャンクが本文として結合されている。
 * 採点はモデルが実際に読んだものを対象にすべきなので、こちらを使う。
 */
export function collectRetrievedChunkIds(docs: FetchResult[]): string[] {
  const ids = new Set<string>();
  for (const doc of docs) {
    for (const id of doc.included_chunk_ids) ids.add(id);
    // included_chunk_ids が空になる経路は無いはずだが、保険として seed も入れる
    ids.add(doc.chunk_id);
  }
  return [...ids];
}

/** プロンプトに入った本文の総文字数 */
export function countContextChars(docs: FetchResult[]): number {
  return docs.reduce((sum, d) => sum + d.text.length, 0);
}
