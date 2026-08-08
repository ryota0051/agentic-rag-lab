import "dotenv/config";
import { isMain } from "../shared/is-main.js";
import { fetchChunks } from "./fetch.js";
import { hybridSearch } from "./hybrid-search.js";
import { vectorSearch } from "./vector-search.js";

/**
 * Phase 2 の完了条件を確認するスクリプト。
 *
 * 見るべきは「それらしい結果が返ること」だけではない。
 * **ベクトル検索とハイブリッド検索の結果が実際に異なること**を確認する。
 * 両者が完全一致するなら BM25 側が寄与しておらず、パターン1とパターン2を
 * 比較する意味が消える（＝実験が成立しない）。
 */

const K = 5;

function jaccard(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  const inter = [...setA].filter((x) => setB.has(x)).length;
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 1 : inter / union;
}

async function main() {
  const query = process.argv.slice(2).join(" ").trim();
  if (!query) {
    console.error('使い方: npx tsx src/search/verify-search.ts "検索したい質問"');
    process.exit(1);
  }

  console.log(`クエリ: ${query}\n`);

  const [vec, hyb] = await Promise.all([vectorSearch(query, K), hybridSearch(query, K)]);

  const show = (label: string, hits: typeof vec) => {
    console.log(`--- ${label} ---`);
    hits.forEach((h, i) => {
      console.log(`${i + 1}. [${h.chunk_id}] ${h.heading_path}`);
      console.log(`   ${h.snippet.replace(/\n/g, " ").slice(0, 100)}`);
    });
    console.log();
  };

  show("ベクトル検索", vec);
  show("ハイブリッド検索 (BM25 + ベクトル, RRF融合)", hyb);

  const vecIds = vec.map((h) => h.chunk_id);
  const hybIds = hyb.map((h) => h.chunk_id);
  const overlap = jaccard(vecIds, hybIds);

  console.log(`結果の重なり (Jaccard): ${(overlap * 100).toFixed(0)}%`);
  if (overlap === 1) {
    console.log(
      "⚠️  完全一致。このクエリでは BM25 が順位に寄与していません。\n" +
        "    専門用語や固有名詞を含むクエリでも一致するようなら、FTS側を疑ってください。",
    );
  } else {
    console.log("✅ 両者は異なる結果を返しています（ハイブリッドが機能）。");
  }

  // fetch の範囲拡張が効いているかも合わせて確認する
  const top = hybIds[0];
  if (top) {
    console.log("\n--- fetch の範囲拡張 ---");
    for (const scope of ["chunk_only", "with_neighbors", "whole_section"] as const) {
      const [res] = await fetchChunks([top], scope);
      console.log(
        `${scope.padEnd(15)} chunks=${res?.included_chunk_ids.length ?? 0} chars=${res?.text.length ?? 0}`,
      );
    }
  }
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
