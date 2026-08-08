import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { isMain } from "../src/shared/is-main.js";
import { withRetry } from "../src/shared/retry.js";
import { selectSkill } from "../src/workflows/agentic-rag/skill-select.js";
import {
  scoreSkillSelection,
  type BoundaryCase,
  type Skill,
} from "./scorers/skill-selection-accuracy.js";

/**
 * スキル選択だけを単独で評価する。20件・数十秒で回るので description の調整を素早く反復できる。
 *
 * ⚠️ **train-on-test に注意。** ここで失敗したケースを見て description を書き換え、
 * 同じ 20 件で再評価すると、スコアは上がるが**汎化したかは分からない**。
 * 個別のケースを名指しで潰す修正（失敗した質問文を例示に加える等）は避け、
 * 判断基準そのものを一般的な形で書き換えること。
 */

const BOUNDARY_PATH = path.resolve("evals/boundary-cases.json");
const CONCURRENCY = 3;

async function main() {
  const cases = JSON.parse(await readFile(BOUNDARY_PATH, "utf8")) as BoundaryCase[];
  console.log(`boundary-cases: ${cases.length} 件\n`);

  const results: { case: BoundaryCase; selected: Skill }[] = [];
  let cursor = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (cursor < cases.length) {
        const c = cases[cursor++];
        if (!c) continue;
        const selection = await withRetry(() => selectSkill(c.question), {
          label: c.question.slice(0, 20),
        });
        results.push({ case: c, selected: selection.skill as Skill });
      }
    }),
  );

  const score = scoreSkillSelection(results);
  const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

  console.log(`正解率: ${pct(score.accuracy)}`);
  console.log(`検索 Precision: ${pct(score.searchPrecision)}`);
  console.log(`検索 Recall:    ${pct(score.searchRecall)}  ← 非対称設計上こちらを重視`);
  console.log(`取りこぼし（要検索なのに直接回答）: ${score.missedSearch} 件 ← 避けたい失敗`);
  console.log(`過剰発火（不要なのに検索）:       ${score.overTriggered} 件 ← 許容される失敗`);

  console.log("\nカテゴリ別:");
  for (const [cat, v] of Object.entries(score.byCategory)) {
    console.log(`  ${cat.padEnd(20)} ${v.correct}/${v.n}`);
  }

  const misses = results.filter((r) => r.selected !== r.case.expected_skill);
  if (misses.length) {
    console.log("\n誤判定:");
    for (const m of misses) {
      console.log(
        `  [期待 ${m.case.expected_skill} / 実際 ${m.selected}] ${m.case.question}`,
      );
    }
  }

  // 判定基準について:
  // 当初は「Recall > Precision なら非対称設計が効いている」と判定していたが、
  // この基準は誤りだった。Recall > Precision を成立させるには
  // 偽陽性（挨拶にまで検索する）が偽陰性より多い必要があり、
  // それは望ましい挙動ではなく単に指標を満たしただけの状態になる。
  //
  // 本当に見たいのは「取りこぼし（FN）を限りなく0に近づけつつ、過剰発火（FP）を
  // 実用的な範囲に抑える」こと。非対称コストの設計意図はこちらで評価する。
  console.log("\n--- 設計意図との整合 ---");
  if (score.missedSearch === 0) {
    console.log(
      `✅ 取りこぼし 0 件。過剰発火 ${score.overTriggered} 件は許容範囲の失敗。`,
    );
  } else if (score.missedSearch <= 1) {
    console.log(
      `△ 取りこぼし ${score.missedSearch} 件。ほぼ意図通りだが、` +
        "残りが構造的な誤りか偶発かを確認する価値がある。",
    );
  } else {
    console.log(
      `⚠️  取りこぼし ${score.missedSearch} 件。要検索の質問を直接回答で処理しており、` +
        "誤情報を出すリスクがある。description を検索側へ倒すこと。",
    );
  }
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
