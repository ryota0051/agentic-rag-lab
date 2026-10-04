import "dotenv/config";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fetchChunks } from "../src/search/fetch.js";
import { hybridSearch } from "../src/search/hybrid-search.js";
import { collectRetrievedChunkIds } from "../src/shared/evidence-stats.js";
import { isMain } from "../src/shared/is-main.js";
import { FINAL_CONTEXT_K, GENERATION_MODEL_LABEL, JEV_MODEL } from "../src/shared/llm-client.js";
import { withRetry } from "../src/shared/retry.js";
import type { FetchResult } from "../src/shared/types.js";
import { checkEvidenceLlm as checkEvidence } from "../src/workflows/agentic-rag/evidence-check.js";
import {
  JEV_PER_DOC_CHARS,
  judgeEvidenceJev,
  truncateForJev,
  type JevEvidenceJudgement,
} from "../src/workflows/agentic-rag/jev-decisions.js";
import type { GoldenItem } from "./generate-golden-set.js";

/**
 * 収集完了の判定（evidence-check）を LLM と Jev で比べる。ADR 0015 の評価方法2。
 *
 * **エージェントは回さない。** 判定器に渡す「根拠のスナップショット」を検索だけで作り、
 * 正解ラベルは golden の正解 chunk が揃っているかで決める（LLM を使わない決定的なラベル）。
 *
 * スナップショットは4種類。自然な検索結果だけだと「十分」に偏るので、両側を意図的に作る:
 *   - natural_k5  … hybridSearch(q, 5)。パターン2が見るのと同じもの
 *   - natural_k2  … hybridSearch(q, 2)。件数が少なく不足しがち
 *   - forced_full … 正解 chunk を必ず含めて5件にしたもの（十分側）
 *   - drop_one    … 正解 chunk を1件抜き、**その記事ごと**除外して検索し直したもの（不足側）。
 *                   話題としては関連しているのに肝心の記述が無い、いちばん難しい不足
 *
 * 判定器のエラーには非対称なコストがある（0008 と同じ構造）:
 *   - 見逃し: 不足なのに「十分」→ ループが止まり、足りない根拠で回答する。**避けたい失敗**
 *   - 空振り: 十分なのに「不足」→ 追加検索が1ラウンド走るだけ。許容される失敗
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");
const REPEATS = 2;
const CONCURRENCY = 3;
const TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

type SnapshotKind = "natural_k5" | "natural_k2" | "forced_full" | "drop_one";

interface Snapshot {
  kind: SnapshotKind;
  item: GoldenItem;
  docs: FetchResult[];
  /** 正解 chunk がすべて含まれているか */
  sufficient: boolean;
  /** 含まれていた正解 chunk の割合 */
  goldCoverage: number;
}

interface Judged extends Snapshot {
  llm: { sufficient: boolean; missing: string; latencyMs: number; failed: boolean }[];
  jev: JevEvidenceJudgement[];
}

const articleOf = (chunkId: string) => chunkId.split("#")[0]!;
const uniq = (xs: string[]) => [...new Set(xs)];

async function buildSnapshots(items: GoldenItem[]): Promise<Snapshot[]> {
  const out: Snapshot[] = [];
  const make = async (kind: SnapshotKind, item: GoldenItem, ids: string[]) => {
    const docs = truncateForJev(await fetchChunks(ids, "chunk_only"));
    const got = collectRetrievedChunkIds(docs);
    const found = item.golden_chunk_ids.filter((g) => got.includes(g)).length;
    out.push({
      kind,
      item,
      docs,
      sufficient: found === item.golden_chunk_ids.length,
      goldCoverage: found / item.golden_chunk_ids.length,
    });
  };

  for (const [i, item] of items.entries()) {
    const gold = item.golden_chunk_ids;
    const hits8 = (await hybridSearch(item.question, 8)).map((h) => h.chunk_id);

    await make("natural_k5", item, hits8.slice(0, FINAL_CONTEXT_K));
    await make("natural_k2", item, hits8.slice(0, 2));

    // 検索順位を保ったまま正解を差し込む。検索に出てこなかった正解は末尾に置く
    const fillers = hits8.filter((id) => !gold.includes(id)).slice(0, FINAL_CONTEXT_K - gold.length);
    const forced = [
      ...hits8.filter((id) => gold.includes(id) || fillers.includes(id)),
      ...gold.filter((g) => !hits8.includes(g)),
    ];
    await make("forced_full", item, forced);

    // 抜く正解は問題ごとに交互に選ぶ（multihop の「前半の対象」ばかり抜かないように）
    const dropped = gold[i % gold.length]!;
    const kept = gold.filter((g) => g !== dropped);
    const hitsNo = (
      await hybridSearch(item.question, 8, { excludeArticleIds: [articleOf(dropped)] })
    ).map((h) => h.chunk_id);
    await make("drop_one", item, uniq([...kept, ...hitsNo]).slice(0, FINAL_CONTEXT_K));

    process.stdout.write(`\r  スナップショット作成 ${i + 1}/${items.length}`);
  }
  console.log();
  return out;
}

// ---------------------------------------------------------------------------
// 集計

interface Confusion {
  n: number;
  accuracy: number;
  /** 不足なのに十分と判定（避けたい） */
  missed: number;
  /** 十分なのに不足と判定（許容） */
  overSearch: number;
  /** 不足を不足と言えた割合 */
  insufficientRecall: number;
}

function confusion(pairs: { label: boolean; pred: boolean }[]): Confusion {
  let correct = 0, missed = 0, over = 0, insuff = 0, insuffHit = 0;
  for (const { label, pred } of pairs) {
    if (label === pred) correct++;
    if (!label && pred) missed++;
    if (label && !pred) over++;
    if (!label) {
      insuff++;
      if (!pred) insuffHit++;
    }
  }
  return {
    n: pairs.length,
    accuracy: correct / (pairs.length || 1),
    missed,
    overSearch: over,
    insufficientRecall: insuffHit / (insuff || 1),
  };
}

/** ROC AUC（順位統計で計算。同点は 0.5） */
function auc(scored: { label: boolean; score: number }[]): number {
  const pos = scored.filter((s) => s.label);
  const neg = scored.filter((s) => !s.label);
  if (!pos.length || !neg.length) return NaN;
  let sum = 0;
  for (const p of pos) for (const q of neg) sum += p.score > q.score ? 1 : p.score === q.score ? 0.5 : 0;
  return sum / (pos.length * neg.length);
}

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;

async function main() {
  const items = JSON.parse(await readFile(GOLDEN_PATH, "utf8")) as GoldenItem[];
  console.log(`golden set ${items.length} 問 → スナップショット4種類`);
  const snapshots = await buildSnapshots(items);
  const judged: Judged[] = snapshots.map((s) => ({ ...s, llm: [], jev: [] }));
  console.log(
    `  ${judged.length} 件（十分 ${judged.filter((j) => j.sufficient).length} / 不足 ${judged.filter((j) => !j.sufficient).length}）× ${REPEATS} 回 × (LLM, Jev)\n`,
  );

  let done = 0;
  let cursor = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (cursor < judged.length) {
        const j = judged[cursor++];
        if (!j) continue;
        for (let r = 0; r < REPEATS; r++) {
          const started = Date.now();
          const v = await withRetry(() => checkEvidence(j.item.question, j.docs, [j.item.question]), {
            label: "llm",
          });
          j.llm.push({
            sufficient: v.sufficient,
            missing: v.missing,
            latencyMs: Date.now() - started,
            failed: v.structuredOutputFailed,
          });
          j.jev.push(await judgeEvidenceJev(j.item.question, j.docs));
        }
        done++;
        process.stdout.write(`\r  判定 ${done}/${judged.length}`);
      }
    }),
  );
  console.log("\n");

  // ---- 集計
  const llmRun = (r: number, subset = judged) =>
    confusion(subset.map((j) => ({ label: j.sufficient, pred: j.llm[r]!.sufficient })));
  const jevRun = (r: number, tau: number, key: "pSufficient" | "pFullCoverage", subset = judged) =>
    confusion(subset.map((j) => ({ label: j.sufficient, pred: j.jev[r]![key] >= tau })));
  const jevAuc = (key: "pSufficient" | "pFullCoverage", subset = judged) =>
    auc(subset.map((j) => ({ label: j.sufficient, score: mean(j.jev.map((x) => x[key])) })));

  const kinds: SnapshotKind[] = ["natural_k5", "natural_k2", "forced_full", "drop_one"];
  const llmLat = judged.flatMap((j) => j.llm.map((l) => l.latencyMs));
  const jevLat = judged.flatMap((j) => j.jev.map((x) => x.latencyMs));
  const jevCost = judged.flatMap((j) => j.jev).reduce((s, x) => s + x.costUsd, 0);
  const models = [...new Set(judged.flatMap((j) => j.jev.map((x) => x.model)))];
  const llmFlips = judged.filter((j) => new Set(j.llm.map((l) => l.sufficient)).size > 1).length;
  const jevFlips = judged.filter((j) => new Set(j.jev.map((x) => x.pSufficient >= 0.5)).size > 1).length;
  const llmFailed = judged.flatMap((j) => j.llm).filter((l) => l.failed).length;

  // ---- レポート
  const date = new Date().toLocaleDateString("sv-SE");
  const o: string[] = [];
  o.push(`# 収集完了の判定: LLM vs Jev（この根拠で答えられるか）`, "");
  o.push(`- 実施日: ${date}`);
  o.push(`- LLM: \`${GENERATION_MODEL_LABEL}\`（既存の \`checkEvidence\`）`);
  o.push(`- Jev: \`${JEV_MODEL}\` → ${models.map((m) => `\`${m}\``).join(", ")}（\`judgeEvidenceJev\`: noul「答えられるか」＋ score「カバー率」）`);
  o.push(`- 入力: 両者に同じ根拠を渡す（1件 ${JEV_PER_DOC_CHARS} 字で切り詰め。Jev の state 上限 8,000 字のため）`);
  o.push(`- 正解ラベル: golden の正解 chunk がすべて根拠に含まれていれば「十分」`);
  o.push(`- 各判定器 ${REPEATS} 回ずつ。生成スクリプト: \`evals/run-evidence-eval-jev.ts\`（生データ: \`experiments/raw-${date}-jev-evidence-gate.json\`）`);
  o.push("", "## データセット", "", "| 種類 | 作り方 | 件数 | うち十分 |", "|---|---|---|---|");
  const how: Record<SnapshotKind, string> = {
    natural_k5: "hybridSearch 上位5件（パターン2と同じ）",
    natural_k2: "hybridSearch 上位2件",
    forced_full: "正解 chunk を必ず含めて5件",
    drop_one: "正解を1件抜き、その記事ごと除外して検索し直した5件",
  };
  for (const k of kinds) {
    const sub = judged.filter((j) => j.kind === k);
    o.push(`| ${k} | ${how[k]} | ${sub.length} | ${sub.filter((j) => j.sufficient).length} |`);
  }

  o.push("", "## 結果", "");
  o.push("見逃し＝不足なのに「十分」（ループが止まる。**避けたい失敗**）／空振り＝十分なのに「不足」（追加検索が走るだけ）。", "");
  o.push("| 判定器 | run | 正解率 | 見逃し | 空振り | 不足の検出率 |", "|---|---|---|---|---|---|");
  for (let r = 0; r < REPEATS; r++) {
    const c = llmRun(r);
    o.push(`| LLM | ${r + 1} | ${pct(c.accuracy)} | ${c.missed} | ${c.overSearch} | ${pct(c.insufficientRecall)} |`);
  }
  for (const key of ["pSufficient", "pFullCoverage"] as const) {
    for (const tau of TAUS) {
      const c = jevRun(0, tau, key);
      const label = key === "pSufficient" ? "Jev noul" : "Jev score";
      o.push(`| ${label} τ=${tau} | 1 | ${pct(c.accuracy)} | ${c.missed} | ${c.overSearch} | ${pct(c.insufficientRecall)} |`);
    }
  }
  o.push("", `> Jev は「p ≥ τ なら十分」。noul は p(答えられる)、score は p(すべてカバー)。表は run 1 の値（run 2 との差は「揺れ」を参照）。`, "");
  o.push(`- ROC AUC: Jev noul **${jevAuc("pSufficient").toFixed(3)}** ／ Jev score **${jevAuc("pFullCoverage").toFixed(3)}**（2回の平均確率で計算）`, "");

  o.push("## 種類ごとの内訳", "");
  o.push("| 種類 | 件数 | LLM 正解率 (run1/run2) | Jev noul τ=0.5 正解率 | Jev noul AUC | Jev 平均 p(十分)：十分 / 不足 |", "|---|---|---|---|---|---|");
  for (const k of kinds) {
    const sub = judged.filter((j) => j.kind === k);
    // 該当が0件の側は「—」。mean([]) は 0 を返すので、そのまま出すと「確率0」と誤読される
    const ps = (lab: boolean) => {
      const xs = sub.filter((j) => j.sufficient === lab).map((j) => mean(j.jev.map((x) => x.pSufficient)));
      return xs.length ? mean(xs).toFixed(2) : "—";
    };
    const a = jevAuc("pSufficient", sub);
    o.push(
      `| ${k} | ${sub.length} | ${pct(llmRun(0, sub).accuracy)} / ${pct(llmRun(1, sub).accuracy)} | ${pct(jevRun(0, 0.5, "pSufficient", sub).accuracy)} | ${Number.isNaN(a) ? "—" : a.toFixed(3)} | ${ps(true)} / ${ps(false)} |`,
    );
  }

  o.push("", "## キャリブレーション（Jev noul）", "", "p(十分) を区間に分け、実際に十分だった割合と比べる。キャリブレーションが取れていれば両者は近くなる。", "");
  o.push("| p(十分) の区間 | 件数 | 実際に十分だった割合 |", "|---|---|---|");
  for (let b = 0; b < 5; b++) {
    const lo = b / 5, hi = (b + 1) / 5;
    const sub = judged.filter((j) => {
      const p = mean(j.jev.map((x) => x.pSufficient));
      return p >= lo && (b === 4 ? p <= hi : p < hi);
    });
    o.push(`| ${lo.toFixed(1)}–${hi.toFixed(1)} | ${sub.length} | ${sub.length ? pct(sub.filter((j) => j.sufficient).length / sub.length) : "—"} |`);
  }

  o.push("", "## 揺れ・速度・コスト", "");
  o.push(`- 2回で判定が割れた件数: LLM ${llmFlips} 件 ／ Jev（τ=0.5）${jevFlips} 件`);
  o.push(`- LLM の構造化出力失敗（→「十分」にフォールバック）: ${llmFailed} 回`);
  o.push(`- レイテンシ中央値: LLM ${median(llmLat)}ms ／ Jev ${median(jevLat)}ms`);
  o.push(`- Jev の総コスト: $${jevCost.toFixed(4)}（${jevLat.length} 回）`, "");

  o.push("## LLM と Jev（noul, τ=0.5）で判定が分かれたもの", "");
  o.push("| 種類 | 正解 | LLM | Jev p(十分) | 質問 | LLM の missing |", "|---|---|---|---|---|---|");
  const disagree = judged.filter((j) => j.llm[0]!.sufficient !== j.jev[0]!.pSufficient >= 0.5);
  for (const j of disagree) {
    const tf = (b: boolean) => (b ? "十分" : "不足");
    o.push(
      `| ${j.kind} | ${tf(j.sufficient)} | ${j.llm.map((l) => tf(l.sufficient)).join("/")} | ${j.jev.map((x) => x.pSufficient.toFixed(2)).join("/")} | ${j.item.question.slice(0, 60).replace(/\|/g, "\\|")}… | ${(j.llm[0]!.missing || "—").slice(0, 80).replace(/\|/g, "\\|").replace(/\n/g, " ")} |`,
    );
  }
  o.push("", "## 観察", "", "> TODO", "");
  o.push("## 注意", "", "- 正解ラベルは「正解 chunk が揃っているか」であって、真の充足度ではない。別の chunk に同じ情報があれば、ラベル上は「不足」なのに実際は答えられる。**「空振り」には判定器ではなくラベル側の誤りが混ざっている**可能性がある。", "");

  const mdPath = path.resolve(`experiments/${date}-jev-evidence-gate.md`);
  await writeFile(mdPath, o.join("\n"), "utf8");
  await writeFile(
    path.resolve(`experiments/raw-${date}-jev-evidence-gate.json`),
    JSON.stringify(
      {
        date,
        models,
        records: judged.map(({ docs, item, ...rest }) => ({
          ...rest,
          question: item.question,
          difficulty: item.difficulty,
          golden_chunk_ids: item.golden_chunk_ids,
          chunk_ids: collectRetrievedChunkIds(docs),
        })),
      },
      null,
      2,
    ),
    "utf8",
  );
  console.log(`✅ ${path.relative(process.cwd(), mdPath)}`);
  for (let r = 0; r < REPEATS; r++) console.log(`LLM run${r + 1}:`, llmRun(r));
  console.log("Jev noul τ=0.5:", jevRun(0, 0.5, "pSufficient"), "AUC", jevAuc("pSufficient").toFixed(3));
  console.log("Jev score τ=0.5:", jevRun(0, 0.5, "pFullCoverage"), "AUC", jevAuc("pFullCoverage").toFixed(3));
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
