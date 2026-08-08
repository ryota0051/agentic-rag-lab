import "dotenv/config";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { runAgenticRag } from "../src/workflows/agentic-rag/index.js";
import { runHybridRag } from "../src/workflows/hybrid-rag.js";
import { runNaiveRag } from "../src/workflows/naive-rag.js";
import { isMain } from "../src/shared/is-main.js";
import { withRetry } from "../src/shared/retry.js";
import { FINAL_CONTEXT_K, GENERATION_MODEL } from "../src/shared/llm-client.js";
import type { RagPattern, RagRunResult } from "../src/shared/types.js";
import type { Difficulty, GoldenItem } from "./generate-golden-set.js";
import { selectSkill } from "../src/workflows/agentic-rag/skill-select.js";
import {
  aggregateRecall,
  scoreRetrievalRecall,
  type RecallScore,
} from "./scorers/retrieval-recall.js";
import {
  scoreSkillSelection,
  type BoundaryCase,
  type Skill,
} from "./scorers/skill-selection-accuracy.js";

/**
 * 3パターン比較実験の本体。結果を experiments/ に Markdown で出力する。
 *
 * 記録する軸（docs/architecture.md「評価軸」）:
 *  - 根拠の充足度  … retrieval-recall（主指標・LLM不使用）
 *  - レイテンシ    … 単発と複数ターンの体感差
 *  - コスト        … トークン数と API 呼び出し回数
 *  - ターン数分布  … パターン3が何ターンで収束したか
 *  - スキル選択精度 … boundary-cases に対する Precision / Recall
 *
 * easy / multihop をバケットに分けて集計するのが要点。
 * 全体平均だけを見ると、easy が天井に張り付いている影響で差が薄まる。
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");
const BOUNDARY_PATH = path.resolve("evals/boundary-cases.json");
const EXPERIMENTS_DIR = path.resolve("experiments");

const RUNNERS: Record<RagPattern, (q: string) => Promise<RagRunResult>> = {
  naive: runNaiveRag,
  hybrid: runHybridRag,
  agentic: runAgenticRag,
};
const PATTERNS: RagPattern[] = ["naive", "hybrid", "agentic"];

/**
 * 同時実行数。
 *
 * gpt-5.6-luna の TPM 上限は 200,000。エージェント的RAGは1問1万トークン超を使うため、
 * 3並列だと容易に上限へ達する（実際に初回実行が 429 で全損した）。
 * レイテンシ計測が混み合いで歪むのも避けたいので控えめにする。
 */
const CONCURRENCY = Number(process.env.EVAL_CONCURRENCY ?? 2);

interface RunRecord {
  item: GoldenItem;
  result: RagRunResult;
  recall: RecallScore;
}

/** 失敗した実行。握りつぶさず記録し、レポートに件数を出す */
interface FailedRun {
  pattern: RagPattern;
  question: string;
  error: string;
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      const item = items[index];
      if (item === undefined) continue;
      results[index] = await fn(item, index);
    }
  });
  await Promise.all(workers);
  return results;
}

function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
    : (sorted[mid] ?? 0);
}

function pct(x: number): string {
  return `${(x * 100).toFixed(0)}%`;
}

async function runPattern(
  pattern: RagPattern,
  items: GoldenItem[],
  failures: FailedRun[],
): Promise<RunRecord[]> {
  console.log(`\n[${pattern}] ${items.length} 問を実行中（並列 ${CONCURRENCY}）...`);

  const records = await mapWithConcurrency(items, CONCURRENCY, async (item, i) => {
    try {
      // レート制限は指数バックオフで粘る。
      // 1問の失敗で実行全体を落とさないよう、それでも駄目なら記録して次へ進む
      const result = await withRetry(() => RUNNERS[pattern](item.question), {
        label: `${pattern} Q${i + 1}`,
      });
      if ((i + 1) % 10 === 0) console.log(`[${pattern}] ${i + 1}/${items.length}`);
      return { item, result, recall: scoreRetrievalRecall(result, item) };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[${pattern}] Q${i + 1} 失敗: ${message.slice(0, 160)}`);
      failures.push({ pattern, question: item.question, error: message });
      return undefined;
    }
  });

  return records.filter((r): r is RunRecord => r !== undefined);
}

/**
 * 途中経過をディスクに書き出す。
 *
 * 初回実行では終盤の 429 で数十分ぶんの結果が丸ごと失われた。
 * パターンごとに書き出しておけば、最後に落ちても手前までは残る。
 */
async function checkpoint(records: RunRecord[], failures: FailedRun[]): Promise<void> {
  await mkdir(EXPERIMENTS_DIR, { recursive: true });
  await writeFile(
    path.join(EXPERIMENTS_DIR, "raw-latest.json"),
    JSON.stringify({ records, failures }, null, 2),
    "utf8",
  );
}

function bucketTable(records: RunRecord[], difficulty: Difficulty): string {
  const rows = PATTERNS.map((p) => {
    const subset = records.filter(
      (r) => r.result.pattern === p && r.item.difficulty === difficulty,
    );
    const agg = aggregateRecall(subset.map((r) => r.recall));
    const latency = subset.map((r) => r.result.latencyMs);
    const inTok = subset.map((r) => r.result.usage.inputTokens);
    const outTok = subset.map((r) => r.result.usage.outputTokens);
    const calls = subset.map((r) => r.result.usage.llmCalls);
    const chars = subset.map((r) => r.result.contextChars);
    return (
      `| ${p} | ${pct(agg.exactRate)} | ${pct(agg.partialRate)} | ` +
      `${agg.meanRetrievedCount.toFixed(1)} | ${Math.round(mean(chars))} | ` +
      `${Math.round(mean(latency))} | ` +
      `${Math.round(mean(inTok))} / ${Math.round(mean(outTok))} | ${mean(calls).toFixed(1)} |`
    );
  });

  return (
    `| パターン | 完全ヒット | 部分recall | 平均chunk数 | 平均根拠文字数 | 平均レイテンシ(ms) | 平均トークン(in/out) | 平均LLM呼び出し |\n` +
    `|---|---|---|---|---|---|---|---|\n${rows.join("\n")}\n\n` +
    `> 「平均chunk数」は範囲拡張で結合されたチャンクも数えている。` +
    `パターン3は fetch の scope 次第で1件の根拠に複数チャンクが入るため、` +
    `件数だけでなく実際に与えた情報量（文字数）も併記している。`
  );
}

function buildReport(
  records: RunRecord[],
  skillScore: ReturnType<typeof scoreSkillSelection>,
  items: GoldenItem[],
  elapsedMs: number,
  failures: FailedRun[],
): string {
  const date = new Date().toISOString().slice(0, 10);
  const easyN = items.filter((i) => i.difficulty === "easy").length;
  const mhN = items.filter((i) => i.difficulty === "multihop").length;

  const agentic = records.filter((r) => r.result.pattern === "agentic");
  const turns = agentic.map((r) => r.result.turns);
  const turnDist = turns.reduce<Record<number, number>>((acc, t) => {
    acc[t] = (acc[t] ?? 0) + 1;
    return acc;
  }, {});

  // パターン別の勝敗を質問単位で突き合わせる（定性観察の材料）
  const byQuestion = new Map<string, Partial<Record<RagPattern, RunRecord>>>();
  for (const r of records) {
    const entry = byQuestion.get(r.item.question) ?? {};
    entry[r.result.pattern] = r;
    byQuestion.set(r.item.question, entry);
  }

  const agenticWins: string[] = [];
  const agenticLosses: string[] = [];
  for (const [question, entry] of byQuestion) {
    const h = entry.hybrid?.recall.exact ?? 0;
    const a = entry.agentic?.recall.exact ?? 0;
    if (a > h) agenticWins.push(question);
    if (h > a) agenticLosses.push(question);
  }

  const catRows = Object.entries(skillScore.byCategory)
    .map(([cat, v]) => `| ${cat} | ${v.correct}/${v.n} | ${pct(v.correct / v.n)} |`)
    .join("\n");

  return `# 比較実験: ナイーブRAG vs ハイブリッド vs エージェント的RAG

- 実施日: ${date}
- 生成モデル: \`${GENERATION_MODEL}\`（3パターン共通）
- golden set: ${items.length} 問（easy ${easyN} / multihop ${mhN}）
- 最終コンテキスト件数: k=${FINAL_CONTEXT_K}（3パターン共通の上限）
- 総実行時間: ${(elapsedMs / 1000 / 60).toFixed(1)} 分

## easy（単発検索で引ける想定・対照群）

${bucketTable(records, "easy")}

## multihop（複数チャンクの統合が必要・エージェント化の主戦場）

${bucketTable(records, "multihop")}

## ターン数分布（パターン3のみ）

${Object.entries(turnDist)
  .sort(([a], [b]) => Number(a) - Number(b))
  .map(([t, n]) => `- ${t} ターン: ${n} 件`)
  .join("\n")}

- 最大: ${Math.max(...turns, 0)} ターン / 中央値: ${median(turns)} ターン
- 1ターンで完結: ${turns.filter((t) => t <= 1).length} 件（＝ループが働かず単発と同じ）

> ここでの「ターン」は**確信度チェックによる再突入も含めた総検索回数**。
> 1回のループ内の上限（\`MAX_TURNS\`）とは別物なので、上限を超える値が出る。

## スキル選択精度（boundary-cases ${skillScore.n} 件）

- 正解率: ${pct(skillScore.accuracy)}
- 検索スキルの Precision: ${pct(skillScore.searchPrecision)}
- 検索スキルの **Recall: ${pct(skillScore.searchRecall)}**（非対称設計上こちらを重視）
- 取りこぼし（要検索なのに直接回答）: ${skillScore.missedSearch} 件 ← 避けたい失敗
- 過剰発火（不要なのに検索）: ${skillScore.overTriggered} 件 ← 許容される失敗

| カテゴリ | 正解 | 正解率 |
|---|---|---|
${catRows}

## パターン3が単発ハイブリッドに勝った質問（${agenticWins.length} 件）

${agenticWins.length ? agenticWins.map((q) => `- ${q}`).join("\n") : "- なし"}

## パターン3が単発ハイブリッドに負けた質問（${agenticLosses.length} 件）

${agenticLosses.length ? agenticLosses.map((q) => `- ${q}`).join("\n") : "- なし"}

## 実行の欠損

${
  failures.length === 0
    ? "なし（全パターン・全問が完走）"
    : `**${failures.length} 件が失敗しています。** 該当パターンは母数が減っているため、` +
      `他パターンとの比較は割合で見ること。\n\n` +
      Object.entries(
        failures.reduce<Record<string, number>>((acc, f) => {
          acc[f.pattern] = (acc[f.pattern] ?? 0) + 1;
          return acc;
        }, {}),
      )
        .map(([p, n]) => `- ${p}: ${n} 件`)
        .join("\n") +
      `\n\n代表的なエラー: \`${failures[0]?.error.slice(0, 200) ?? ""}\``
}

## 考察

<!-- 数値の羅列で終わらせないこと。以下を自分の言葉で埋める:
     - multihop で差がついたか。ついたなら何ターン目で回収できていたか
     - easy では差がないのにコストだけ増えているか（＝エージェント化の損益分岐点）
     - 勝った質問と負けた質問に共通する性質は何か
     - スキル選択の Recall > Precision になっているか（非対称設計が意図通りか）
-->

`;
}

/**
 * 引数を読む。
 *
 * パターン1・2は検索にLLMを使わないため**決定的**で、実行するたびに同じ結果になる
 * （実測でも5回連続で完全に同一だった）。パターン3だけをいじっている間は
 * 再実行するだけ時間とコストの無駄なので、走らせるパターンを選べるようにしてある。
 *
 *   npx tsx evals/run-comparison.ts --patterns=agentic --baseline=experiments/raw-2026-08-09e.json
 *
 * 走らせなかったパターンの結果は baseline から読み込んでレポートに合成する。
 */
function parseArgs() {
  const argv = process.argv.slice(2);
  const get = (name: string): string | undefined =>
    argv.find((a) => a.startsWith(`--${name}=`))?.split("=").slice(1).join("=");

  const patternsArg = get("patterns");
  const patterns = patternsArg
    ? (patternsArg.split(",").map((s) => s.trim()) as RagPattern[])
    : PATTERNS;

  const invalid = patterns.filter((p) => !PATTERNS.includes(p));
  if (invalid.length) {
    throw new Error(`不正なパターン: ${invalid.join(", ")}（有効: ${PATTERNS.join(", ")}）`);
  }

  return {
    patterns,
    baseline: get("baseline"),
    reuseSkill: argv.includes("--reuse-skill"),
  };
}

/** 走らせないパターンの結果を過去の実行から読み込む */
async function loadBaselineRecords(
  baselinePath: string,
  patterns: RagPattern[],
): Promise<{ records: RunRecord[]; skill?: { case: BoundaryCase; selected: Skill }[] }> {
  const raw = JSON.parse(await readFile(path.resolve(baselinePath), "utf8")) as {
    records: RunRecord[];
    skillResults?: { case: BoundaryCase; selected: Skill }[];
  };
  return {
    records: raw.records.filter((r) => patterns.includes(r.result.pattern)),
    skill: raw.skillResults,
  };
}

async function writeReport(
  report: string,
  records: RunRecord[],
  skillResults: { case: BoundaryCase; selected: Skill }[],
  failures: FailedRun[],
  started: number,
): Promise<void> {
  await mkdir(EXPERIMENTS_DIR, { recursive: true });
  const outPath = path.join(
    EXPERIMENTS_DIR,
    `${new Date().toISOString().slice(0, 10)}-naive-vs-hybrid-vs-agentic.md`,
  );
  await writeFile(outPath, report, "utf8");

  // 生ログも残す。後から失敗モードを掘り返せるようにする
  await writeFile(
    path.join(EXPERIMENTS_DIR, "raw-latest.json"),
    JSON.stringify({ records, skillResults, failures }, null, 2),
    "utf8",
  );

  const elapsed = Date.now() - started;
  console.log(`\n完了（${(elapsed / 1000 / 60).toFixed(1)} 分）`);
  if (failures.length) {
    console.warn(
      `⚠️  ${failures.length} 件が失敗しています。該当パターンの数値は母数が減っているため、` +
        "レポートの「実行の欠損」節を確認してください。",
    );
  }
  console.log(`レポート: ${outPath}`);
  console.log("考察セクションは自分の言葉で埋めてください。");
}

async function main() {
  const started = Date.now();
  const args = parseArgs();

  const items = JSON.parse(await readFile(GOLDEN_PATH, "utf8")) as GoldenItem[];
  const boundaryCases = JSON.parse(
    await readFile(BOUNDARY_PATH, "utf8"),
  ) as BoundaryCase[];

  const reused = PATTERNS.filter((p) => !args.patterns.includes(p));
  console.log(
    `golden set ${items.length} 問 × ${args.patterns.length}パターン = ` +
      `${items.length * args.patterns.length} 実行`,
  );
  if (reused.length) {
    if (!args.baseline) {
      throw new Error(
        `--patterns で一部だけ実行する場合は --baseline=<raw-*.json> も指定してください` +
          `（${reused.join(", ")} の結果をそこから読み込みます）`,
      );
    }
    console.log(`${reused.join(", ")} は ${args.baseline} から再利用します`);
  }

  const records: RunRecord[] = [];
  const failures: FailedRun[] = [];

  if (reused.length && args.baseline) {
    const { records: old } = await loadBaselineRecords(args.baseline, reused);
    if (old.length === 0) {
      throw new Error(`${args.baseline} に ${reused.join(", ")} の結果がありません`);
    }
    records.push(...old);
  }

  for (const pattern of args.patterns) {
    records.push(...(await runPattern(pattern, items, failures)));
    await checkpoint(records, failures);
  }

  // スキル選択は boundary-cases に対して単独で測る。
  // golden set 側で測ると全問 search が正解になり、境界の判定能力が見えない。
  // description を触っていない回では再利用してよい（--reuse-skill）
  if (args.reuseSkill && args.baseline) {
    const { skill } = await loadBaselineRecords(args.baseline, []);
    if (skill?.length) {
      console.log(`\n[skill] ${args.baseline} の判定結果を再利用します（${skill.length} 件）`);
      const reusedScore = scoreSkillSelection(skill);
      const report = buildReport(records, reusedScore, items, Date.now() - started, failures);
      await writeReport(report, records, skill, failures, started);
      return;
    }
    console.warn("[skill] baseline に判定結果がないため通常どおり実行します");
  }

  console.log(`\n[skill] boundary-cases ${boundaryCases.length} 件を判定中...`);
  const skillRaw = await mapWithConcurrency(boundaryCases, CONCURRENCY, async (c, i) => {
    try {
      const selection = await withRetry(() => selectSkill(c.question), {
        label: `skill Q${i + 1}`,
      });
      return { case: c, selected: selection.skill as Skill };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[skill] Q${i + 1} 失敗: ${message.slice(0, 160)}`);
      failures.push({ pattern: "agentic", question: c.question, error: message });
      return undefined;
    }
  });
  const skillResults = skillRaw.filter(
    (r): r is { case: BoundaryCase; selected: Skill } => r !== undefined,
  );
  const skillScore = scoreSkillSelection(skillResults);

  const report = buildReport(records, skillScore, items, Date.now() - started, failures);
  await writeReport(report, records, skillResults, failures, started);
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
