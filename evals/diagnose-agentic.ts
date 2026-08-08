import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { hybridSearch } from "../src/search/hybrid-search.js";
import { isMain } from "../src/shared/is-main.js";
import { FINAL_CONTEXT_K } from "../src/shared/llm-client.js";
import type { GoldenItem } from "./generate-golden-set.js";
import type { RagRunResult } from "../src/shared/types.js";

/**
 * エージェント的RAGが単発ハイブリッドに負ける原因を切り分ける診断。
 *
 * 損失は3段階のどこかで起きているはず。
 *
 *   1. クエリ言い換え … エージェントは元の質問をそのまま使わず、同義語や関連語を足した
 *                      クエリを作る。それが元の質問より**悪い**なら、この時点で負ける
 *   2. 可視性        … 言い換えたクエリの上位8件（エージェントが実際に見た候補）に
 *                      正解が入っていたか
 *   3. 選択          … 候補に見えていたのに fetch しなかったか
 *
 * 生成は一切走らせない（埋め込みのみ）ので数十秒・ほぼ無料で回る。
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");
/** エージェントが search で取る候補数（search-fetch-loop.ts と一致させること） */
const AGENT_SEARCH_K = 8;

interface RawRecord {
  item: GoldenItem;
  result: RagRunResult;
  recall: { exact: number; retrievedCount: number };
}

function hit(ids: string[], golden: string[]): boolean {
  return golden.every((g) => ids.includes(g));
}
function coverage(ids: string[], golden: string[]): number {
  return golden.filter((g) => ids.includes(g)).length / golden.length;
}

async function main() {
  const rawPath = process.argv[2] ?? "experiments/raw-2026-08-08b.json";
  const raw = JSON.parse(await readFile(path.resolve(rawPath), "utf8")) as {
    records: RawRecord[];
  };
  const items = JSON.parse(await readFile(GOLDEN_PATH, "utf8")) as GoldenItem[];
  const goldenByQ = new Map(items.map((i) => [i.question, i]));

  const agentic = raw.records.filter((r) => r.result.pattern === "agentic");
  console.log(`診断対象: ${path.basename(rawPath)} / agentic ${agentic.length} 件\n`);

  let origHit = 0;
  let rewrittenHit5 = 0;
  let rewrittenHit8 = 0;
  let actuallyFetched = 0;
  let origCov = 0;
  let rewCov = 0;

  // 「見えていたのに取らなかった」ケースを集める
  const visibleButNotFetched: string[] = [];
  const lostInRewrite: string[] = [];

  for (const rec of agentic) {
    const golden = goldenByQ.get(rec.item.question)?.golden_chunk_ids;
    if (!golden) continue;

    const firstQuery = rec.result.queryTrace?.[0];
    if (!firstQuery) continue;

    // 1. 元の質問そのままで検索（= 単発ハイブリッドと同条件）
    const orig = (await hybridSearch(rec.item.question, FINAL_CONTEXT_K)).map(
      (h) => h.chunk_id,
    );
    // 2. エージェントが実際に投げたクエリ
    const rew8 = (await hybridSearch(firstQuery, AGENT_SEARCH_K)).map((h) => h.chunk_id);
    const rew5 = rew8.slice(0, FINAL_CONTEXT_K);

    const oHit = hit(orig, golden);
    const r5 = hit(rew5, golden);
    const r8 = hit(rew8, golden);
    const fetched = hit(rec.result.retrieved_chunk_ids, golden);

    if (oHit) origHit++;
    if (r5) rewrittenHit5++;
    if (r8) rewrittenHit8++;
    if (fetched) actuallyFetched++;
    origCov += coverage(orig, golden);
    rewCov += coverage(rew8, golden);

    if (r8 && !fetched) visibleButNotFetched.push(rec.item.question);
    if (oHit && !r8) lostInRewrite.push(rec.item.question);
  }

  const n = agentic.length;
  const pct = (x: number) => `${((x / n) * 100).toFixed(0)}%`;

  console.log("段階別の完全ヒット率（分母 = 全 agentic 実行）");
  console.log(`  1. 元の質問そのまま @${FINAL_CONTEXT_K}          : ${origHit}/${n} (${pct(origHit)})  ← 単発ハイブリッドと同条件`);
  console.log(`  2. エージェントの言い換え @${FINAL_CONTEXT_K}    : ${rewrittenHit5}/${n} (${pct(rewrittenHit5)})`);
  console.log(`  3. エージェントの言い換え @${AGENT_SEARCH_K}    : ${rewrittenHit8}/${n} (${pct(rewrittenHit8)})  ← エージェントが実際に見た候補`);
  console.log(`  4. 実際に fetch した根拠           : ${actuallyFetched}/${n} (${pct(actuallyFetched)})  ← 最終結果`);

  console.log("\n部分カバレッジ（正解chunkのうち何割が候補に入ったか）");
  console.log(`  元の質問 @${FINAL_CONTEXT_K}: ${((origCov / n) * 100).toFixed(0)}%`);
  console.log(`  言い換え @${AGENT_SEARCH_K}: ${((rewCov / n) * 100).toFixed(0)}%`);

  console.log("\n損失の内訳");
  console.log(`  A. クエリ言い換えで失った   : ${lostInRewrite.length} 件（元の質問なら引けたのに、言い換え後の上位${AGENT_SEARCH_K}件に入らなかった）`);
  console.log(`  B. 見えていたのに取らなかった: ${visibleButNotFetched.length} 件（上位${AGENT_SEARCH_K}件に正解があったのに fetch しなかった）`);

  if (lostInRewrite.length) {
    console.log("\n[A の例]");
    lostInRewrite.slice(0, 3).forEach((q) => console.log(`  - ${q.slice(0, 90)}`));
  }
  if (visibleButNotFetched.length) {
    console.log("\n[B の例]");
    visibleButNotFetched.slice(0, 3).forEach((q) => console.log(`  - ${q.slice(0, 90)}`));
  }

  // 言い換えクエリの長さ。keyword soup 化していないかを見る
  const qLens = agentic
    .map((r) => r.result.queryTrace?.[0]?.length)
    .filter((x): x is number => x !== undefined);
  const origLens = agentic.map((r) => r.item.question.length);
  console.log(
    `\nクエリ長: 元の質問 平均 ${Math.round(origLens.reduce((a, b) => a + b, 0) / origLens.length)} 文字 ` +
      `→ 言い換え後 平均 ${Math.round(qLens.reduce((a, b) => a + b, 0) / qLens.length)} 文字`,
  );
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
