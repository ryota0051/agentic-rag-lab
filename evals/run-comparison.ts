import "dotenv/config";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { runAgenticRag } from "../src/workflows/agentic-rag/index.js";
import { runHybridRag } from "../src/workflows/hybrid-rag.js";
import { runNaiveRag } from "../src/workflows/naive-rag.js";
import { isMain } from "../src/shared/is-main.js";
import { withRetry } from "../src/shared/retry.js";
import {
  FINAL_CONTEXT_K,
  GENERATION_MODEL_LABEL,
  LLM_BACKEND,
} from "../src/shared/llm-client.js";
import type { FetchScope, RagPattern, RagRunResult } from "../src/shared/types.js";
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
 *
 * ローカルバックエンドでは単一GPUを奪い合うだけで並列の得が無く、レイテンシ計測も歪むので
 * 既定を 1 に落とす（`EVAL_CONCURRENCY` で明示指定すればそちらが優先される）。
 */
const CONCURRENCY = Number(
  process.env.EVAL_CONCURRENCY ?? (LLM_BACKEND === "local" ? 1 : 2),
);

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
  // --difficulty で絞ると片方のバケットが空になる。0埋めの表を出すと
  // 「全パターンが0点だった」と読めてしまうので、対象外だと明示する
  if (!records.some((r) => r.item.difficulty === difficulty)) {
    return `（このバケットは今回の実行対象外です。\`--difficulty\` で絞り込んでいます）`;
  }

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

/**
 * パターン3のツール使用の内訳。
 *
 * ここが「エージェントを駆動するLLMを差し替えると何が変わるか」の本体。
 * recall やターン数だけを見ていると、**ツールを使えていないのか、使った上で不要と判断したのか**
 * が区別できない（`ToolUseStats` のコメント参照）。
 */
function toolUseSection(records: RunRecord[]): string {
  const agentic = records.filter((r) => r.result.pattern === "agentic");
  if (agentic.length === 0) return "パターン3を実行していないため記録なし。";

  // 旧レポート（計測導入前）を baseline から読み込むと toolUse が無い。
  // **平均は計測できた実行だけで取る。** 未計測を 0 として混ぜると、
  // 「ツールを使わなかった」ことにされて平均が不当に下がる
  const stats = agentic
    .map((r) => r.result.toolUse)
    .filter((t): t is NonNullable<typeof t> => t !== undefined);

  if (stats.length === 0) {
    return (
      "**未計測**（ツール使用の計測を導入する前の実行結果を再利用しています）。" +
      "この節を比較に使う場合は、baseline 側も計測付きで取り直すこと。"
    );
  }
  const measured = stats.length;

  const scopes: FetchScope[] = ["chunk_only", "with_neighbors", "whole_section"];
  const scopeTotal = scopes.reduce(
    (acc, s) => acc + stats.reduce((a, t) => a + t.fetchScopes[s], 0),
    0,
  );

  const sum = (pick: (t: (typeof stats)[number]) => number) =>
    stats.reduce((a, t) => a + pick(t), 0);

  const failures = sum((t) => t.structuredOutputFailures);
  const blocked = sum((t) => t.searchBlockedCalls);
  const unresolved = sum((t) => t.fetchUnresolvedIds);
  const noop = sum((t) => t.fetchNoOpCalls);

  const rows = [
    `| search 呼び出し | ${mean(stats.map((t) => t.searchToolCalls)).toFixed(2)} | ${sum((t) => t.searchToolCalls)} |`,
    `| fetch 呼び出し | ${mean(stats.map((t) => t.fetchToolCalls)).toFixed(2)} | ${sum((t) => t.fetchToolCalls)} |`,
    `| search 上限超過で拒否 | ${mean(stats.map((t) => t.searchBlockedCalls)).toFixed(2)} | ${blocked} |`,
    `| fetch 空振り（取得済みの再要求） | ${mean(stats.map((t) => t.fetchNoOpCalls)).toFixed(2)} | ${noop} |`,
    `| fetch 未解決ID | ${mean(stats.map((t) => t.fetchUnresolvedIds)).toFixed(2)} | ${unresolved} |`,
    `| 構造化出力の失敗 | ${mean(stats.map((t) => t.structuredOutputFailures)).toFixed(2)} | ${failures} |`,
  ].join("\n");

  const scopeRows = scopes
    .map((s) => {
      const n = stats.reduce((a, t) => a + t.fetchScopes[s], 0);
      return `| ${s} | ${n} | ${scopeTotal ? pct(n / scopeTotal) : "-"} |`;
    })
    .join("\n");

  const warnings: string[] = [];
  if (failures > 0) {
    warnings.push(
      `> ⚠️ **構造化出力が ${failures} 回失敗しています。** スキル選択は "search"、` +
        "充足チェック・確信度チェックは sufficient=true にフォールバックする実装なので、" +
        "**ターン数分布とスキル選択精度は「モデルの判断」ではなく「フォールバックの結果」を" +
        "含んでいます。** この実験の該当数値を能力比較の根拠に使わないこと。",
    );
  }
  if (blocked > 0) {
    warnings.push(
      `> 検索回数の上限を ${blocked} 回超過して呼びに来ています` +
        "（プロンプトの回数制約に従えていない）。ツール側で機械的に打ち切っているため" +
        "実害はないが、指示追従性の差として記録に値する。",
    );
  }
  if (unresolved > 0) {
    warnings.push(
      `> 本文が返らなかった chunk_id の要求が ${unresolved} 件あります。` +
        "存在しないIDの生成とコンテキスト予算切れの**合算**なので、" +
        "内訳は raw JSON と突き合わせて確認すること。",
    );
  }

  return (
    `対象: パターン3 ${agentic.length} 件（うち計測あり ${measured} 件）\n\n` +
    `| 項目 | 1問あたり平均 | 合計 |\n|---|---|---|\n${rows}\n\n` +
    `### fetch の scope 選択（読む範囲を自分で決められているか）\n\n` +
    `| scope | 回数 | 割合 |\n|---|---|---|\n${scopeRows}\n\n` +
    `> \`chunk_only\` に張り付いている場合、search/fetch を分離した狙い` +
    `（docs/decisions/0003）が活きていない。範囲拡張を使い分けられるかは、` +
    `エージェントを駆動するモデルの能力差が最も出やすい箇所。\n` +
    (warnings.length ? `\n${warnings.join("\n>\n")}\n` : "")
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
- 生成モデル: \`${GENERATION_MODEL_LABEL}\`（3パターン共通）
- バックエンド: \`${LLM_BACKEND}\`${
    LLM_BACKEND === "local"
      ? "（サーバ設定は `docker/compose.yaml` を参照。文脈長・量子化・reasoning_effort はそこに固定されている）"
      : ""
  }
- 埋め込み・LanceDBインデックス: **無変更**（検索側は固定。変数は生成・エージェントのLLMのみ）
- golden set: ${items.length} 問（easy ${easyN} / multihop ${mhN}）
- 最終コンテキスト件数: k=${FINAL_CONTEXT_K}（3パターン共通の上限）
- 同時実行数: ${CONCURRENCY}
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

## ツール使用の内訳（パターン3のみ）

${toolUseSection(records)}

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

  // 難易度で golden set を絞る。ローカルバックエンドは実質直列で時間がかかるため、
  // エージェント化の主戦場である multihop だけを回せるようにしてある
  const difficultyArg = get("difficulty");
  const DIFFICULTIES: Difficulty[] = ["easy", "multihop"];
  if (difficultyArg && !DIFFICULTIES.includes(difficultyArg as Difficulty)) {
    throw new Error(
      `不正な難易度: ${difficultyArg}（有効: ${DIFFICULTIES.join(", ")}）`,
    );
  }

  return {
    patterns,
    baseline: get("baseline"),
    reuseSkill: argv.includes("--reuse-skill"),
    difficulty: difficultyArg as Difficulty | undefined,
    slug: get("slug"),
  };
}

/**
 * レポートのファイル名スラグ。
 *
 * 固定名にすると、同じ日にクラウド版とローカル版を回したときに
 * **片方がもう片方を上書きして消える**（1実験1ファイルという約束が壊れる）。
 * 既定でバックエンドと難易度を織り込み、`--slug=` で上書きできるようにする。
 */
function defaultSlug(difficulty?: Difficulty): string {
  const backend = LLM_BACKEND === "local" ? "local" : "cloud";
  return [backend, difficulty ?? "full", "naive-vs-hybrid-vs-agentic"].join("-");
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
  slug: string,
): Promise<void> {
  await mkdir(EXPERIMENTS_DIR, { recursive: true });
  const outPath = path.join(
    EXPERIMENTS_DIR,
    `${new Date().toISOString().slice(0, 10)}-${slug}.md`,
  );
  await writeFile(outPath, report, "utf8");

  // 生ログも残す。後から失敗モードを掘り返せるようにする。
  // raw-latest.json は「直近の実行」を指す固定名だが、それだけだと同じ日に
  // クラウド版とローカル版を回したときに互いを上書きしてしまうので、
  // スラグ付きのコピーも書いて --baseline から名指しできるようにする
  const raw = JSON.stringify({ records, skillResults, failures }, null, 2);
  const rawPath = path.join(
    EXPERIMENTS_DIR,
    `raw-${new Date().toISOString().slice(0, 10)}-${slug}.json`,
  );
  await writeFile(rawPath, raw, "utf8");
  await writeFile(path.join(EXPERIMENTS_DIR, "raw-latest.json"), raw, "utf8");

  const elapsed = Date.now() - started;
  console.log(`\n完了（${(elapsed / 1000 / 60).toFixed(1)} 分）`);
  if (failures.length) {
    console.warn(
      `⚠️  ${failures.length} 件が失敗しています。該当パターンの数値は母数が減っているため、` +
        "レポートの「実行の欠損」節を確認してください。",
    );
  }
  console.log(`レポート: ${outPath}`);
  console.log(`生ログ:   ${rawPath}`);
  console.log("考察セクションは自分の言葉で埋めてください。");
}

async function main() {
  const started = Date.now();
  const args = parseArgs();

  const allItems = JSON.parse(await readFile(GOLDEN_PATH, "utf8")) as GoldenItem[];
  const items = args.difficulty
    ? allItems.filter((i) => i.difficulty === args.difficulty)
    : allItems;
  if (items.length === 0) {
    throw new Error(`難易度 ${args.difficulty} の質問が golden set にありません`);
  }
  const boundaryCases = JSON.parse(
    await readFile(BOUNDARY_PATH, "utf8"),
  ) as BoundaryCase[];

  const slug = args.slug ?? defaultSlug(args.difficulty);
  const reused = PATTERNS.filter((p) => !args.patterns.includes(p));
  if (args.difficulty) {
    console.log(
      `難易度 ${args.difficulty} のみを対象にします（${allItems.length} 問中 ${items.length} 問）`,
    );
  }
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
    // 難易度で絞っている場合、baseline 側も同じ問題集合に揃える。
    // 揃えないと再利用したパターンだけ easy を含んだまま集計され、
    // バケット別の表が別々の母数で並ぶ
    const questions = new Set(items.map((i) => i.question));
    const scoped = old.filter((r) => questions.has(r.item.question));
    if (scoped.length === 0) {
      throw new Error(
        `${args.baseline} に ${reused.join(", ")} の結果がありません` +
          (args.difficulty ? `（難易度 ${args.difficulty} で絞り込んだ結果）` : ""),
      );
    }
    records.push(...scoped);
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
      await writeReport(report, records, skill, failures, started, slug);
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
  await writeReport(report, records, skillResults, failures, started, slug);
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
