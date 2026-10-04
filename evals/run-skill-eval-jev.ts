import "dotenv/config";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { isMain } from "../src/shared/is-main.js";
import { GENERATION_MODEL_LABEL, JEV_MODEL } from "../src/shared/llm-client.js";
import { withRetry } from "../src/shared/retry.js";
import {
  judgeSkillJev,
  skillByThreshold,
  type JevSkillJudgement,
} from "../src/workflows/agentic-rag/jev-decisions.js";
import { selectSkill } from "../src/workflows/agentic-rag/skill-select.js";
import {
  scoreSkillSelection,
  type BoundaryCase,
  type Skill,
} from "./scorers/skill-selection-accuracy.js";

/**
 * スキル分岐を LLM（selectSkill）と Jev で比べる。ADR 0015 の評価方法1。
 *
 * LLM は二値しか返さないので1点、Jev は確率を返すので閾値 τ を動かした曲線になる。
 * 「Jev の曲線のどこかに、LLM より取りこぼしが少なく過剰発火も許容範囲の点があるか」を見る。
 *
 * 両方とも 3 回ずつ回す（REPEATS）。LLM の揺れと、Jev の「同じ入力なら同じ答え」を同時に確かめるため。
 *
 * ⚠️ train-on-test に注意（run-skill-eval.ts と同じ）。τ をこの20件で選ぶと楽観的な値になる。
 * ここで見るのは「閾値で救える形の間違い方をしているか」であって、最適な τ そのものではない。
 */

const BOUNDARY_PATH = path.resolve("evals/boundary-cases.json");
const REPEATS = 3;
const CONCURRENCY = 3;
const TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

interface Row {
  c: BoundaryCase;
  llm: { skill: Skill; latencyMs: number; reason: string }[];
  jev: JevSkillJudgement[];
}

async function pool<T>(items: T[], fn: (x: T) => Promise<void>) {
  let cursor = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (cursor < items.length) {
        const x = items[cursor++];
        if (x !== undefined) await fn(x);
      }
    }),
  );
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;

async function main() {
  const cases = JSON.parse(await readFile(BOUNDARY_PATH, "utf8")) as BoundaryCase[];
  const rows: Row[] = cases.map((c) => ({ c, llm: [], jev: [] }));
  console.log(`boundary-cases: ${cases.length} 件 × ${REPEATS} 回 × (LLM, Jev)\n`);

  await pool(rows, async (row) => {
    for (let i = 0; i < REPEATS; i++) {
      const started = Date.now();
      const sel = await withRetry(() => selectSkill(row.c.question), { label: "llm" });
      row.llm.push({ skill: sel.skill, latencyMs: Date.now() - started, reason: sel.reason });
      row.jev.push(await judgeSkillJev(row.c.question));
    }
    const pd = mean(row.jev.map((j) => j.pDirect));
    console.log(
      `  [期待 ${row.c.expected_skill.padEnd(6)}] LLM ${row.llm.map((l) => l.skill[0]).join("")}  ` +
        `Jev p(direct)=${row.jev.map((j) => j.pDirect.toFixed(2)).join("/")}  ${row.c.question.slice(0, 36)}`,
    );
  });

  // ---- 集計
  const llmRuns = Array.from({ length: REPEATS }, (_, i) =>
    scoreSkillSelection(rows.map((r) => ({ case: r.c, selected: r.llm[i]!.skill }))),
  );
  /** 各 run の Jev 確率で、閾値 τ のときのスコア */
  const jevAt = (tau: number, run: number) =>
    scoreSkillSelection(
      rows.map((r) => ({ case: r.c, selected: skillByThreshold(r.jev[run]!.pDirect, tau) })),
    );

  const llmLat = rows.flatMap((r) => r.llm.map((l) => l.latencyMs));
  const jevLat = rows.flatMap((r) => r.jev.map((j) => j.latencyMs));
  const jevCost = rows.flatMap((r) => r.jev).reduce((s, j) => s + j.costUsd, 0);
  const models = [...new Set(rows.flatMap((r) => r.jev.map((j) => j.model)))];
  /** 各問題の Jev p(direct) の3回の振れ幅（最大−最小）の最大値 */
  const jevSpread = Math.max(
    ...rows.map((r) => Math.max(...r.jev.map((j) => j.pDirect)) - Math.min(...r.jev.map((j) => j.pDirect))),
  );
  const llmFlips = rows.filter((r) => new Set(r.llm.map((l) => l.skill)).size > 1);

  // ---- レポート
  const date = new Date().toLocaleDateString("sv-SE");
  const out: string[] = [];
  out.push(`# スキル分岐: LLM vs Jev（閾値で非対称コストを表せるか）`, "");
  out.push(`- 実施日: ${date}`);
  out.push(`- LLM: \`${GENERATION_MODEL_LABEL}\`（既存の \`selectSkill\`）`);
  out.push(`- Jev: \`${JEV_MODEL}\` → ${models.map((m) => `\`${m}\``).join(", ")}（\`judgeSkillJev\`）`);
  out.push(`- 判断基準の文章: 両者とも \`src/skills/descriptions.ts\` をそのまま使用（Jev 向けの調整なし）`);
  out.push(`- データ: \`evals/boundary-cases.json\` ${cases.length} 件（search ${cases.filter((c) => c.expected_skill === "search").length} / direct ${cases.filter((c) => c.expected_skill === "direct").length}）× ${REPEATS} 回`);
  out.push(`- 生成スクリプト: \`evals/run-skill-eval-jev.ts\`（生データ: \`experiments/raw-${date}-jev-skill-routing.json\`）`);
  out.push("", "## 結果", "");
  out.push("取りこぼし＝要検索なのに直接回答（**避けたい失敗**）、過剰発火＝不要なのに検索（許容される失敗）。", "");
  out.push("| 判定器 | run | 正解率 | 取りこぼし | 過剰発火 | 検索 Recall | 検索 Precision |", "|---|---|---|---|---|---|---|");
  for (const [i, s] of llmRuns.entries()) {
    out.push(`| LLM | ${i + 1} | ${pct(s.accuracy)} | ${s.missedSearch} | ${s.overTriggered} | ${pct(s.searchRecall)} | ${pct(s.searchPrecision)} |`);
  }
  for (const tau of TAUS) {
    for (let i = 0; i < REPEATS; i++) {
      const s = jevAt(tau, i);
      out.push(`| Jev τ=${tau} | ${i + 1} | ${pct(s.accuracy)} | ${s.missedSearch} | ${s.overTriggered} | ${pct(s.searchRecall)} | ${pct(s.searchPrecision)} |`);
    }
  }
  out.push("", "> Jev は「p(direct) ≥ τ のときだけ direct」。τ=0.5 がほぼ argmax（素の選択）に相当する。", "");

  out.push("## 揺れと速度", "");
  out.push(`- LLM: 3回の判定が割れた問題 ${llmFlips.length} 件${llmFlips.length ? `（${llmFlips.map((r) => `「${r.c.question.slice(0, 20)}…」`).join("、")}）` : ""}`);
  out.push(`- Jev: 同じ問題の p(direct) の振れ幅は最大 ${jevSpread.toFixed(2)}`);
  out.push(`- レイテンシ中央値: LLM ${median(llmLat)}ms ／ Jev ${median(jevLat)}ms`);
  out.push(`- Jev の総コスト: $${jevCost.toFixed(5)}（${jevLat.length} 回）`, "");

  out.push("## 問題ごと", "");
  out.push("| 期待 | カテゴリ | 質問 | LLM（3回） | Jev p(direct)（3回の平均） | Jev confidence 相当 |", "|---|---|---|---|---|---|");
  const sorted = [...rows].sort(
    (a, b) => a.c.expected_skill.localeCompare(b.c.expected_skill) || mean(b.jev.map((j) => j.pDirect)) - mean(a.jev.map((j) => j.pDirect)),
  );
  for (const r of sorted) {
    const pd = mean(r.jev.map((j) => j.pDirect));
    const llmMark = r.llm.map((l) => (l.skill === r.c.expected_skill ? "○" : "×")).join("");
    out.push(`| ${r.c.expected_skill} | ${r.c.category} | ${r.c.question} | ${llmMark} | ${pd.toFixed(2)} | ${Math.abs(pd - (1 - pd)).toFixed(2)} |`);
  }
  out.push("", "> LLM 列の ○× は期待どおりだったか。Jev の「confidence 相当」は |p(direct) − p(search)|。", "");
  out.push("## 観察", "", "> TODO", "");

  const mdPath = path.resolve(`experiments/${date}-jev-skill-routing.md`);
  await writeFile(mdPath, out.join("\n"), "utf8");
  await writeFile(
    path.resolve(`experiments/raw-${date}-jev-skill-routing.json`),
    JSON.stringify({ date, models, rows }, null, 2),
    "utf8",
  );
  console.log(`\n✅ ${path.relative(process.cwd(), mdPath)}`);
  console.log("\nLLM:", llmRuns.map((s) => `取りこぼし${s.missedSearch}/過剰${s.overTriggered}`).join("  "));
  for (const tau of TAUS) {
    console.log(`Jev τ=${tau}:`, [0, 1, 2].map((i) => { const s = jevAt(tau, i); return `取りこぼし${s.missedSearch}/過剰${s.overTriggered}`; }).join("  "));
  }
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
