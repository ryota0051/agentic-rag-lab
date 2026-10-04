import "dotenv/config";
import { existsSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fetchChunks } from "../src/search/fetch.js";
import { hybridSearch } from "../src/search/hybrid-search.js";
import { isMain } from "../src/shared/is-main.js";
import { evaluateJev, type JevRequest, type JevResult } from "../src/shared/jev-client.js";
import { FINAL_CONTEXT_K, GENERATION_MODEL_LABEL, JEV_MODEL } from "../src/shared/llm-client.js";
import type { FetchResult } from "../src/shared/types.js";
import { checkEvidenceLlm as checkEvidence } from "../src/workflows/agentic-rag/evidence-check.js";

/**
 * Jev を触ってみるスクリプト。**「試してみた」記事の素材を残すこと**が目的なので、
 * 1回の実行で人が読む Markdown と生の JSON を experiments/ に書き出す。
 *
 * ケースは記事の見出し単位で分けている:
 *   1. 疎通（公式の例そのまま）
 *   2. 日本語でスキル分岐できるか
 *   3. 「この根拠で答えられるか」を確率で聞く ← 本題
 *   4. 同じ入力を3回 — Jev と LLM（checkEvidence）の揺れを比べる
 *   5. 質問を言い換えても確率は安定か
 *   6. 選択肢や問いの書き方に引きずられるか
 *
 * Markdown の「観察」欄は TODO のまま出力される。実行後に結果を読んで人（or Claude）が埋める。
 * 埋めた後に再実行して上書きしないよう、既存ファイルがあれば --force なしでは止まる。
 *
 *   npm run jev:hello            # experiments/YYYY-MM-DD-jev-hello.md を作る
 *   npm run jev:hello -- --force # 上書き
 */

// ---------------------------------------------------------------------------
// ログ

interface JevEntry {
  kind: "jev";
  label: string;
  request: JevRequest;
  result: JevResult;
}
interface LlmEntry {
  kind: "llm";
  label: string;
  sufficient: boolean;
  missing: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
}
type Entry = JevEntry | LlmEntry;

interface CaseLog {
  id: string;
  title: string;
  purpose: string;
  /** state に入れた根拠の内訳など、リクエスト以外に残しておきたい前提 */
  notes: string[];
  entries: Entry[];
}

const cases: CaseLog[] = [];
let current: CaseLog | undefined;

function startCase(id: string, title: string, purpose: string): void {
  current = { id, title, purpose, notes: [], entries: [] };
  cases.push(current);
  console.log(`\n━━ ${id}. ${title}\n   ${purpose}`);
}

function note(text: string): void {
  current!.notes.push(text);
  console.log(`   · ${text}`);
}

async function ask(label: string, request: JevRequest): Promise<JevResult> {
  const result = await evaluateJev(request);
  current!.entries.push({ kind: "jev", label, request, result });
  console.log(`   [Jev] ${label}  (${result.latencyMs}ms)`);
  for (const line of answerLines(result)) console.log(`         ${line}`);
  return result;
}

async function askLlm(label: string, question: string, docs: FetchResult[]): Promise<void> {
  const started = Date.now();
  const v = await checkEvidence(question, docs, [question]);
  const entry: LlmEntry = {
    kind: "llm",
    label,
    sufficient: v.sufficient,
    missing: v.missing,
    latencyMs: Date.now() - started,
    inputTokens: v.inputTokens,
    outputTokens: v.outputTokens,
  };
  current!.entries.push(entry);
  console.log(`   [LLM] ${label}  sufficient=${v.sufficient}  (${entry.latencyMs}ms)`);
}

// ---------------------------------------------------------------------------
// 表示

const BAR_WIDTH = 20;
const bar = (p: number) => {
  const n = Math.round(p * BAR_WIDTH);
  return "█".repeat(n) + "·".repeat(BAR_WIDTH - n);
};
const pct = (p: number) => `${(p * 100).toFixed(0).padStart(3)}%`;

/** 回答をターミナル・Markdown 共通の行に整形する */
function answerLines(r: JevResult): string[] {
  const lines: string[] = [];
  for (const [name, a] of Object.entries(r.answers)) {
    if (a.type === "noul") {
      lines.push(`${name} (noul)  yes ${bar(a.noul)} ${pct(a.noul)}`);
    } else if (a.type === "choice") {
      lines.push(`${name} (choice) → ${a.choice}  confidence ${a.confidence.toFixed(2)}`);
      for (const [opt, p] of Object.entries(a.probabilities)) {
        lines.push(`    ${opt.padEnd(12)} ${bar(p)} ${pct(p)}`);
      }
    } else {
      lines.push(`${name} (score) → ${a.score.toFixed(2)}  confidence ${a.confidence.toFixed(2)}`);
      for (const [idx, p] of Object.entries(a.probabilities)) {
        lines.push(`    ${idx}:${(a.legend[idx] ?? "").padEnd(14)} ${bar(p)} ${pct(p)}`);
      }
    }
  }
  return lines;
}

const STATE_PREVIEW = 400;
const preview = (s: string) =>
  s.length <= STATE_PREVIEW ? s : `${s.slice(0, STATE_PREVIEW)}\n…（以下略、全 ${s.length} 文字）`;

function renderMarkdown(date: string, models: Set<string>): string {
  const jevEntries = cases.flatMap((c) => c.entries).filter((e): e is JevEntry => e.kind === "jev");
  const llmEntries = cases.flatMap((c) => c.entries).filter((e): e is LlmEntry => e.kind === "llm");
  const totalCost = jevEntries.reduce((s, e) => s + e.result.costUsd, 0);
  const totalIn = jevEntries.reduce((s, e) => s + e.result.inputTokens, 0);
  const lat = jevEntries.map((e) => e.result.latencyMs).sort((a, b) => a - b);
  const llmLat = llmEntries.map((e) => e.latencyMs).sort((a, b) => a - b);
  const median = (xs: number[]) => (xs.length ? xs[Math.floor(xs.length / 2)]! : 0);

  const out: string[] = [];
  out.push(`# Jev を触ってみた（判定専用モデルは「検索は十分か？」に使えるか）`, "");
  out.push(`- 実施日: ${date}`);
  out.push(`- Jev: \`${JEV_MODEL}\` → 実体 ${[...models].map((m) => `\`${m}\``).join(", ")}`);
  out.push(`- 比較用 LLM（ケース4のみ）: \`${GENERATION_MODEL_LABEL}\`（既存の \`checkEvidence\`）`);
  out.push(`- 生成スクリプト: \`scripts/jev-hello.ts\`（生データ: \`experiments/raw-${date}-jev-hello.json\`）`);
  out.push(
    `- Jev 呼び出し ${jevEntries.length} 回 / 入力 ${totalIn.toLocaleString()} トークン / ` +
      `概算 $${totalCost.toFixed(5)} / レイテンシ中央値 ${median(lat)}ms（最大 ${lat.at(-1) ?? 0}ms）`,
  );
  if (llmEntries.length) {
    out.push(`- 参考: LLM 判定のレイテンシ中央値 ${median(llmLat)}ms（${llmEntries.length} 回）`);
  }
  out.push("", "## やってみて分かったこと", "", "> TODO: 実行結果を読んで埋める", "");

  for (const c of cases) {
    out.push(`## ${c.id}. ${c.title}`, "", `**確かめたいこと**: ${c.purpose}`, "");
    if (c.notes.length) {
      for (const n of c.notes) out.push(`- ${n}`);
      out.push("");
    }

    // 同じリクエスト形を繰り返すケース（4〜6）は、最初の1件だけ全文を載せる
    let shownRequest = false;
    for (const e of c.entries) {
      out.push(`### ${e.label}`, "");
      if (e.kind === "llm") {
        out.push(
          `LLM（checkEvidence）: **sufficient = ${e.sufficient}** ／ ${e.latencyMs}ms ／ ` +
            `in ${e.inputTokens} / out ${e.outputTokens} トークン`,
        );
        if (e.missing) out.push("", `> missing: ${e.missing}`);
        out.push("");
        continue;
      }
      if (!shownRequest) {
        out.push("<details><summary>リクエスト</summary>", "");
        out.push("state:", "", "```text", preview(e.request.state), "```", "");
        out.push("questions:", "", "```json", JSON.stringify(e.request.questions, null, 2), "```", "");
        out.push("</details>", "");
        shownRequest = true;
      } else {
        out.push("state 冒頭: `" + e.request.state.slice(0, 80).replace(/\n/g, " ") + "…`", "");
      }
      out.push("```text", ...answerLines(e.result), "```", "");
      out.push(
        `${e.result.latencyMs}ms ／ in ${e.result.inputTokens} トークン ／ $${e.result.costUsd.toFixed(6)}`,
        "",
      );
    }
    out.push("> 観察: TODO", "");
  }
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// 共通の素材

/** 根拠を state に詰める。Jev の state 上限（8,000字）に収まるよう1件ずつ切り詰める */
const PER_DOC_CHARS = 1200;
function evidenceState(question: string, docs: FetchResult[]): string {
  const body = docs
    .map((d, i) => `[${i + 1}] ${d.heading_path}\n${d.text.slice(0, PER_DOC_CHARS)}`)
    .join("\n\n");
  return `質問: ${question}\n\n集まった根拠（${docs.length}件）:\n${body}`;
}

/** 「答えられるか」を3つの型で同時に聞く。型ごとの癖を見比べるため */
const SUFFICIENCY_QUESTIONS: JevRequest["questions"] = {
  sufficient: {
    type: "noul",
    instructions:
      "集まった根拠だけで、質問のすべての要素に答えられますか？" +
      "関連しているだけで、問われている値や手順そのものが書かれていない場合は「いいえ」です。",
  },
  coverage: {
    type: "score",
    instructions: "質問の要素のうち、根拠でカバーされている割合はどの程度ですか？",
    criteria: ["まったくカバーしていない", "一部だけカバーしている", "すべてカバーしている"],
  },
};

/** スキル分岐の選択肢。docs/decisions/0001 の2スキルを短く言い直したもの */
const ROUTE_NEUTRAL: Record<string, string> = {
  search: "技術記事のコーパスを検索してから答えるべき質問",
  direct: "検索せずにそのまま答えてよい発話（挨拶・会話メタ・意見・計算など）",
};

async function docsFor(ids: string[]): Promise<FetchResult[]> {
  return fetchChunks(ids, "chunk_only");
}

// ---------------------------------------------------------------------------
// ケース

async function case1() {
  startCase("1", "疎通確認（公式の例そのまま）", "最小の呼び出しで何が返ってくるか");
  await ask("英語の請求チケット", {
    state: "The customer has been billed twice and wants a refund today.",
    questions: {
      route: {
        type: "choice",
        instructions: "Which queue should own this ticket?",
        criteria: {
          billing: "Payments, refunds, invoices",
          technical: "Product errors and usage problems",
        },
      },
      escalate: { type: "noul", instructions: "Does this need a human right now?" },
    },
  });
}

async function case2() {
  startCase(
    "2",
    "日本語でスキル分岐できるか",
    "境界ケースの日本語の質問を search / direct に振り分けさせ、確率が意味のある形で出るか",
  );
  note("質問は evals/boundary-cases.json から、両側と際どいもの（jargon-but-general）を選んだ");
  const qs: [string, string][] = [
    ["direct / greeting", "こんにちは。今日はいい天気ですね"],
    ["direct / calculation", "1024 × 768 はいくつになりますか？"],
    ["direct / opinion", "一般論として、技術記事は結論から書くのと背景から書くのとではどちらが読まれやすいと思いますか？"],
    ["search / domain-casual", "max-model-len ってどれくらいにしておけばいいんだっけ？"],
    ["search / domain-explicit", "Slurm でジョブを投入するコマンドを教えてください"],
    ["search / jargon-but-general", "トークンとは何ですか？"],
  ];
  for (const [label, q] of qs) {
    await ask(`期待 ${label}: ${q}`, {
      state: q,
      questions: {
        skill: {
          type: "choice",
          instructions: "この質問に答えるとき、どちらのスキルを使うべきですか？",
          criteria: ROUTE_NEUTRAL,
        },
      },
    });
  }
}

// ケース3〜5で使う題材: golden set の multihop から、読者に説明しやすい LLaVA / LLaVA-Med の問題
const Q = "LLaVAはLLMの学習段階のうち何を画像に対して実施しており、LLaVA-Medは医療画像を用いてどのような2段階の学習を行うことで医療画像チャットモデルを構築していますか？";
const GOLD_LLAVA = "7ae8f81007d15be3b28c#2";
const GOLD_MED = "ac094cf15928483b16c1#2";
const MED_ARTICLE = "ac094cf15928483b16c1";

interface EvidenceSets {
  full: FetchResult[];
  partial: FetchResult[];
  irrelevant: FetchResult[];
}

async function buildEvidenceSets(): Promise<EvidenceSets> {
  const uniq = (xs: string[]) => [...new Set(xs)];
  const hits = await hybridSearch(Q, 8);
  const full = uniq([GOLD_LLAVA, GOLD_MED, ...hits.map((h) => h.chunk_id)]).slice(0, FINAL_CONTEXT_K);

  // LLaVA-Med の記事を丸ごと除外して検索し直す。同じ記事の別チャンクに
  // 「2段階」の記述が残っていると「足りない」という前提が崩れるため
  const hitsNoMed = await hybridSearch(Q, 8, { excludeArticleIds: [MED_ARTICLE] });
  const partial = uniq([GOLD_LLAVA, ...hitsNoMed.map((h) => h.chunk_id)]).slice(0, FINAL_CONTEXT_K);

  const irrelevantHits = await hybridSearch("Slurm でジョブを投入するコマンド", FINAL_CONTEXT_K);
  const irrelevant = irrelevantHits.map((h) => h.chunk_id);

  return {
    full: await docsFor(full),
    partial: await docsFor(partial),
    irrelevant: await docsFor(irrelevant),
  };
}

const idsOf = (docs: FetchResult[]) => docs.map((d) => `\`${d.chunk_id}\``).join(", ");

async function case3(ev: EvidenceSets) {
  startCase(
    "3",
    "「この根拠で答えられるか」を確率で聞く",
    "根拠が十分／肝心の片方が欠けている／無関係、の3状態で確率がどう動くか",
  );
  note(`質問（golden set の multihop）: ${Q}`);
  note(`正解 chunk: \`${GOLD_LLAVA}\`（LLaVA）と \`${GOLD_MED}\`（LLaVA-Med）の両方が必要`);
  note(`十分: ${idsOf(ev.full)}`);
  note(`片方欠け（LLaVA-Med の記事を除外して検索）: ${idsOf(ev.partial)}`);
  note(`無関係（Slurm の検索結果）: ${idsOf(ev.irrelevant)}`);
  note(`各根拠は ${PER_DOC_CHARS} 字で切り詰め（Jev の state 上限 8,000 字に収めるため）`);

  await ask("十分（正解2件を含む）", { state: evidenceState(Q, ev.full), questions: SUFFICIENCY_QUESTIONS });
  await ask("片方欠け（LLaVA-Med の記述なし）", {
    state: evidenceState(Q, ev.partial),
    questions: SUFFICIENCY_QUESTIONS,
  });
  await ask("無関係（Slurm の記事）", {
    state: evidenceState(Q, ev.irrelevant),
    questions: SUFFICIENCY_QUESTIONS,
  });
}

async function case4(ev: EvidenceSets) {
  startCase(
    "4",
    "同じ入力を3回 — Jev と LLM の揺れ",
    "いちばん判定が割れやすい「片方欠け」を3回ずつ投げ、確率・判定が毎回同じか",
  );
  note("LLM 側は本番パイプラインの checkEvidence をそのまま使う（ただし根拠は Jev と同じく切り詰めたもの）");
  const truncated = ev.partial.map((d) => ({ ...d, text: d.text.slice(0, PER_DOC_CHARS) }));
  for (let i = 1; i <= 3; i++) {
    await ask(`Jev ${i}回目`, { state: evidenceState(Q, ev.partial), questions: SUFFICIENCY_QUESTIONS });
  }
  for (let i = 1; i <= 3; i++) {
    await askLlm(`LLM ${i}回目`, Q, truncated);
  }
}

async function case5(ev: EvidenceSets) {
  startCase(
    "5",
    "質問を言い換えても確率は安定か",
    "意味が同じ質問の言い回しを変えたとき、十分／片方欠けの確率がどれだけ動くか",
  );
  const paraphrases = [
    Q,
    "LLaVAが画像に対して行っている学習段階は何か。また、LLaVA-Medが医療画像チャットモデルを作るために行う2段階の学習とは？",
    "LLaVA と LLaVA-Med の学習方法を教えて。LLaVA は画像で何をしていて、LLaVA-Med はどんな2ステップで学習してる？",
  ];
  for (const [i, q] of paraphrases.entries()) {
    await ask(`言い換え${i} × 十分`, { state: evidenceState(q, ev.full), questions: SUFFICIENCY_QUESTIONS });
    await ask(`言い換え${i} × 片方欠け`, {
      state: evidenceState(q, ev.partial),
      questions: SUFFICIENCY_QUESTIONS,
    });
  }
}

async function case6(ev: EvidenceSets) {
  startCase(
    "6",
    "選択肢や問いの書き方に引きずられるか",
    "(a) criteria の説明文を検索寄り／直接回答寄りに書き換える (b) 問いを裏返す（十分か？ ↔ 足りないか？）",
  );
  const variants: [string, Record<string, string>][] = [
    ["中立", ROUTE_NEUTRAL],
    [
      "検索寄り（ADR 0001 の「迷ったら検索」を入れる）",
      {
        search:
          "既定のスキル。技術用語・製品名・数値が1つでも含まれるなら、一般知識で答えられそうでもこちら。迷ったらこちら",
        direct: "挨拶・会話メタ・意見・計算など、明確に情報を求めていない発話のみ",
      },
    ],
    [
      "直接回答寄り",
      {
        search: "文書を参照しないと絶対に答えられない、固有の事実を問う質問のみ",
        direct: "一般知識で答えられる質問。用語の説明や概念の解説を含む",
      },
    ],
  ];
  for (const q of ["トークンとは何ですか？", "vLLM って結局どうなんですか？"]) {
    for (const [label, criteria] of variants) {
      await ask(`(a) ${label}: ${q}`, {
        state: q,
        questions: {
          skill: { type: "choice", instructions: "この質問に答えるとき、どちらのスキルを使うべきですか？", criteria },
        },
      });
    }
  }

  // 裏返した問い。整合していれば p(十分) ≈ 1 − p(足りない) になるはず
  for (const [label, docs] of [["十分", ev.full], ["片方欠け", ev.partial]] as const) {
    await ask(`(b) 表裏を同時に聞く × ${label}`, {
      state: evidenceState(Q, docs),
      questions: {
        sufficient: SUFFICIENCY_QUESTIONS.sufficient!,
        missing: {
          type: "noul",
          instructions: "質問に答えるために、集まった根拠には書かれていない情報が残っていますか？",
        },
      },
    });
  }
}

// ---------------------------------------------------------------------------

async function main() {
  const force = process.argv.includes("--force");
  const date = new Date().toLocaleDateString("sv-SE"); // YYYY-MM-DD（ローカル日付）
  const mdPath = path.resolve(`experiments/${date}-jev-hello.md`);
  const rawPath = path.resolve(`experiments/raw-${date}-jev-hello.json`);
  if (existsSync(mdPath) && !force) {
    throw new Error(`${mdPath} が既にあります。観察を書いた後なら上書きしないこと（上書きは --force）`);
  }

  await case1();
  await case2();
  const ev = await buildEvidenceSets();
  await case3(ev);
  await case4(ev);
  await case5(ev);
  await case6(ev);

  const models = new Set(
    cases.flatMap((c) => c.entries).flatMap((e) => (e.kind === "jev" ? [e.result.model] : [])),
  );
  await writeFile(mdPath, renderMarkdown(date, models), "utf8");
  await writeFile(rawPath, JSON.stringify({ date, models: [...models], cases }, null, 2), "utf8");
  console.log(`\n✅ ${path.relative(process.cwd(), mdPath)}\n✅ ${path.relative(process.cwd(), rawPath)}`);
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
