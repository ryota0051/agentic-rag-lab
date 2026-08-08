import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { hybridSearch } from "../src/search/hybrid-search.js";
import { vectorSearch } from "../src/search/vector-search.js";
import { isMain } from "../src/shared/is-main.js";
import { FINAL_CONTEXT_K } from "../src/shared/llm-client.js";
import type { GoldenItem } from "./generate-golden-set.js";

/**
 * 検索のみの recall 測定（LLM生成を一切走らせない）。
 *
 * 用途は2つ:
 *  1. golden set が実験データとして機能するかの事前確認。
 *     全パターンが 100% か 0% に張り付くなら、そのデータでは差を観測できない
 *  2. 検索側（チャンク分割・トークナイザ・k）を変えたときの高速な回帰確認。
 *     生成を挟まないので数十秒で回り、API コストもほぼゼロ
 *
 * パターン3（エージェント的RAG）は検索だけを切り出せない（ループが本体）ため
 * ここでは測らない。3パターン比較は Phase 5 の run-comparison.ts で行う。
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");
const DRAFT_PATH = path.resolve("evals/golden-set.draft.json");

export async function loadGoldenSet(): Promise<{ items: GoldenItem[]; source: string }> {
  for (const p of [GOLDEN_PATH, DRAFT_PATH]) {
    try {
      const items = JSON.parse(await readFile(p, "utf8")) as GoldenItem[];
      return { items, source: p };
    } catch {
      continue;
    }
  }
  throw new Error(
    "golden set が見つかりません。先に `npm run gen:golden` を実行してください。",
  );
}

/**
 * 完全ヒット判定。**正解 chunk_id が全件そろって初めて正解**とみなす。
 *
 * easy（正解1件）では「いずれか1件」と同義になる。
 * multihop（正解2件）では両方必要。片方だけ取れても答えは組み立てられないので、
 * ここを緩めると「マルチホップ質問を作った意味」が消える。
 */
export function isHit(retrieved: string[], golden: string[]): boolean {
  return golden.every((g) => retrieved.includes(g));
}

/** 部分点。正解 chunk のうち何割を取得できたか（multihop の途中経過を見る用） */
export function partialRecall(retrieved: string[], golden: string[]): number {
  if (golden.length === 0) return 0;
  const found = golden.filter((g) => retrieved.includes(g)).length;
  return found / golden.length;
}

interface Bucket {
  label: string;
  items: GoldenItem[];
  vecHits: number;
  hybHits: number;
  vecPartial: number;
  hybPartial: number;
  bothMiss: string[];
}

async function measure(label: string, items: GoldenItem[]): Promise<Bucket> {
  const bucket: Bucket = {
    label,
    items,
    vecHits: 0,
    hybHits: 0,
    vecPartial: 0,
    hybPartial: 0,
    bothMiss: [],
  };

  for (const item of items) {
    const [vec, hyb] = await Promise.all([
      vectorSearch(item.question, FINAL_CONTEXT_K),
      hybridSearch(item.question, FINAL_CONTEXT_K),
    ]);
    const vecIds = vec.map((h) => h.chunk_id);
    const hybIds = hyb.map((h) => h.chunk_id);

    if (isHit(vecIds, item.golden_chunk_ids)) bucket.vecHits++;
    if (isHit(hybIds, item.golden_chunk_ids)) bucket.hybHits++;
    bucket.vecPartial += partialRecall(vecIds, item.golden_chunk_ids);
    bucket.hybPartial += partialRecall(hybIds, item.golden_chunk_ids);

    if (
      !isHit(vecIds, item.golden_chunk_ids) &&
      !isHit(hybIds, item.golden_chunk_ids)
    ) {
      bucket.bothMiss.push(item.question);
    }
  }
  return bucket;
}

function report(b: Bucket): void {
  const n = b.items.length;
  if (n === 0) return;
  const pct = (x: number) => ((x / n) * 100).toFixed(0);
  console.log(`\n### ${b.label} (${n} 問)`);
  console.log(
    `  ベクトル検索  完全ヒット ${b.vecHits}/${n} (${pct(b.vecHits)}%)  部分recall ${(
      (b.vecPartial / n) * 100
    ).toFixed(0)}%`,
  );
  console.log(
    `  ハイブリッド  完全ヒット ${b.hybHits}/${n} (${pct(b.hybHits)}%)  部分recall ${(
      (b.hybPartial / n) * 100
    ).toFixed(0)}%`,
  );
}

async function main() {
  const { items, source } = await loadGoldenSet();
  console.log(
    `golden set: ${path.basename(source)} (${items.length} 問)  k=${FINAL_CONTEXT_K}`,
  );

  const easy = items.filter((i) => i.difficulty !== "multihop");
  const multihop = items.filter((i) => i.difficulty === "multihop");

  const buckets: Bucket[] = [];
  if (easy.length) buckets.push(await measure("easy（対照群・単発で引ける想定）", easy));
  if (multihop.length) {
    buckets.push(await measure("multihop（複数チャンク必須）", multihop));
  }
  buckets.forEach(report);

  const mh = buckets.find((b) => b.label.startsWith("multihop"));
  if (mh && mh.bothMiss.length) {
    console.log("\n[multihop で両方が引けなかった質問（パターン3の見せ場）]");
    mh.bothMiss.slice(0, 6).forEach((q) => console.log(`  - ${q}`));
  }

  // データセットとしての健全性チェック。
  // 単発検索が天井なら、エージェント的RAGには改善の余地が構造的に存在せず、
  // 「ループはコストが増えるだけ」という当たり前の結論しか出ない。
  // 判定はエージェント化が効くはずの multihop バケットに対して行う。
  const target = mh ?? buckets[0];
  console.log("\n--- 実験データとしての判定 ---");
  if (!target || target.items.length === 0) {
    console.log("判定不能: 質問がありません。");
    return;
  }
  const headroom = 1 - target.hybHits / target.items.length;
  console.log(
    `単発ハイブリッドの取りこぼし（＝パターン3の伸びしろ）: ${(headroom * 100).toFixed(0)}%`,
  );
  if (headroom < 0.05) {
    console.log(
      "⚠️  単発検索が飽和しています。エージェント化で改善する余地がありません。",
    );
  } else if (target.hybHits === 0) {
    console.log("⚠️  完全ヒット0件。難しすぎるか、検索・インデックスが壊れています。");
  } else {
    console.log("✅ 伸びしろがあり、差が観測できる水準です。");
  }
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
