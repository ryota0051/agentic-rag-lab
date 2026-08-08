import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { isMain } from "../src/shared/is-main.js";
import { withRetry } from "../src/shared/retry.js";
import { runAgenticRag } from "../src/workflows/agentic-rag/index.js";
import type { RagRunResult } from "../src/shared/types.js";
import type { GoldenItem } from "./generate-golden-set.js";
import { scoreRetrievalRecall } from "./scorers/retrieval-recall.js";

/**
 * エージェント的RAGの改修を、少数の問題で素早く検証するスモークテスト。
 *
 * 全46問 × 3パターンを回すと10分以上かかる。パターン3だけをいじっている間は
 * パターン1・2の結果は変わらないので、毎回全部回すのは無駄。
 *
 * ## 失敗した問題だけを回してはいけない
 *
 * 前回失敗した問題だけで測ると、
 *  - エージェントは確率的なので、何も直していなくても何割かは「改善」する
 *  - **直したことで別の問題を壊しても気づけない**
 *
 * そこで前回**失敗した問題**と**成功した問題**を同数ずつ取り、
 * 「回復率」と「維持率」の両方を見る。維持率が落ちていたら改悪。
 *
 * 使い方:
 *   npx tsx evals/smoke-agentic.ts [baseline.json] [各群の件数]
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");
const DEFAULT_BASELINE = "experiments/raw-2026-08-08b.json";
const DEFAULT_SAMPLE = 6;
const CONCURRENCY = 2;

interface RawRecord {
  item: GoldenItem;
  result: RagRunResult;
  recall: { exact: number };
}

/** 記事が偏らないよう散らして取る */
function spread(items: GoldenItem[], n: number): GoldenItem[] {
  if (items.length <= n) return items;
  const step = items.length / n;
  return Array.from({ length: n }, (_, i) => items[Math.floor(i * step)]!);
}

async function runAll(items: GoldenItem[]): Promise<Map<string, boolean>> {
  const out = new Map<string, boolean>();
  let cursor = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (cursor < items.length) {
        const item = items[cursor++];
        if (!item) continue;
        const result = await withRetry(() => runAgenticRag(item.question), {
          label: item.question.slice(0, 20),
        });
        const score = scoreRetrievalRecall(result, item);
        out.set(item.question, score.exact === 1);
        process.stdout.write(score.exact === 1 ? "o" : "x");
      }
    }),
  );
  process.stdout.write("\n");
  return out;
}

async function main() {
  const baselinePath = path.resolve(process.argv[2] ?? DEFAULT_BASELINE);
  const sampleSize = Number(process.argv[3] ?? DEFAULT_SAMPLE);

  const raw = JSON.parse(await readFile(baselinePath, "utf8")) as {
    records: RawRecord[];
  };
  const golden = JSON.parse(await readFile(GOLDEN_PATH, "utf8")) as GoldenItem[];
  const byQuestion = new Map(golden.map((g) => [g.question, g]));

  const agentic = raw.records.filter((r) => r.result.pattern === "agentic");
  const failedQs = agentic.filter((r) => r.recall.exact === 0);
  const passedQs = agentic.filter((r) => r.recall.exact === 1);

  const failed = spread(
    failedQs.map((r) => byQuestion.get(r.item.question)).filter((x): x is GoldenItem => !!x),
    sampleSize,
  );
  const passed = spread(
    passedQs.map((r) => byQuestion.get(r.item.question)).filter((x): x is GoldenItem => !!x),
    sampleSize,
  );

  console.log(
    `ベースライン: ${path.basename(baselinePath)}\n` +
      `失敗群 ${failed.length} 件 / 成功群 ${passed.length} 件 を再実行します\n`,
  );

  process.stdout.write("失敗群: ");
  const failedNow = await runAll(failed);
  process.stdout.write("成功群: ");
  const passedNow = await runAll(passed);

  const recovered = [...failedNow.values()].filter(Boolean).length;
  const kept = [...passedNow.values()].filter(Boolean).length;

  console.log(`\n回復率（前回失敗 → 今回成功）: ${recovered}/${failed.length}`);
  console.log(`維持率（前回成功 → 今回も成功）: ${kept}/${passed.length}  ← 落ちていたら改悪`);

  const broke = passed.filter((p) => passedNow.get(p.question) === false);
  if (broke.length) {
    console.log("\n[壊れた問題]");
    broke.forEach((p) => console.log(`  - ${p.question.slice(0, 80)}`));
  }

  console.log("\n--- 判定 ---");
  if (recovered > 0 && kept === passed.length) {
    console.log("✅ 回復あり・回帰なし。全問での検証に進む価値があります。");
  } else if (kept < passed.length) {
    console.log(
      `⚠️  ${passed.length - kept} 件を壊しています。回復分と相殺されている可能性が高いので、` +
        "全問を回す前に原因を潰すこと。",
    );
  } else {
    console.log(
      "△ 回復0件。効果が見えません。別の仮説を試すか、サンプル数を増やしてください。",
    );
  }
  console.log(
    "\n※ 各群わずか数件のサンプルです。1〜2件の増減は誤差の範囲。方向性の確認にのみ使うこと。",
  );
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
