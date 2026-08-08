import "dotenv/config";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { openChunksTable } from "../src/index/build-index.js";
import { isMain } from "../src/shared/is-main.js";
import { JUDGE_MODEL } from "../src/shared/llm-client.js";

/**
 * golden set のドラフト生成。
 *
 * 各チャンクを LLM に渡し「このチャンクを読まないと答えられない質問」を作らせる。
 * 自動生成を選ぶ最大の理由は、**正解 chunk_id が生成時点で自動的に紐づく**こと。
 * 手書きだと「この質問の根拠はどのチャンクか」を人手で対応付ける作業が発生し、
 * そこが一番のボトルネックになる。
 *
 * 出力は evals/golden-set.draft.json（レビュー前）。
 * 人手レビュー後に evals/golden-set.json として確定させる。
 *
 * ⚠️ **絶対にやってはいけないこと: 検索でヒットするかどうかで質問を絞り込むこと。**
 * 「今の検索器で引ける質問だけ残す」と、評価対象の検索器で評価データを選別することになり、
 * 3パターンすべてのスコアが実力以上に嵩上げされ、比較そのものが無意味になる（循環論法）。
 * このスクリプトのフィルタは**質問文と回答文の性質だけ**を見ており、検索結果は一切参照しない。
 */

const OUT_PATH = path.resolve("evals/golden-set.draft.json");

/** 1記事あたり何チャンクから質問を作るか。20記事 × 3 = 60問の候補 */
const CHUNKS_PER_ARTICLE = 3;
/** 質問生成の対象にする最小文字数。短すぎるチャンクは質問が作れない */
const MIN_CHUNK_CHARS = 250;
/** 同時実行数。API を叩きすぎない程度に */
const CONCURRENCY = 4;

const GeneratedQuestion = z.object({
  question: z
    .string()
    .describe(
      "この文書を読まないと答えられない、具体的で自己完結した日本語の質問。" +
        "文書が手元にない人がそのまま検索窓に打てる形にすること",
    ),
  expected_answer: z.string().describe("文書に基づく簡潔な想定回答"),
  answerable_without_document: z
    .boolean()
    .describe("一般的な技術知識だけで答えられてしまう質問なら true"),
  question_leaks_answer: z
    .boolean()
    .describe("質問文自体に答えが書かれてしまっているなら true"),
  is_too_generic: z
    .boolean()
    .describe("この記事以外にも同じ答えを持つ文書がありそうな一般的すぎる質問なら true"),
  is_self_contained: z
    .boolean()
    .describe(
      "「この記事」「本文」「この実験」などの指示語に頼らず、主題を名指ししていて、" +
        "文書を見ていない人でも何について尋ねているか分かるなら true",
    ),
});

const generatorAgent = new Agent({
  id: "golden-set-generator",
  name: "Golden Set Generator",
  instructions: `あなたは技術文書に対する評価用の質問セットを作成します。

与えられた文書の断片から、**その断片を読まないと答えられない質問**を1つ作ってください。

## 最重要の条件: 自己完結していること

この質問は、**文書が手元にない人が検索窓に打ち込む**ものとして使われます。
「この記事では」「本文の」「提示された」「この実験で」「上記の」といった指示語は
**絶対に使わないでください**。読み手はどの文書を指しているか分かりません。

代わりに、**主題を具体的に名指し**してください。製品名・ライブラリ名・コマンド名・
手法名・コンペ名など、その文書を特定できる固有名詞を質問文に含めます。

悪い例: 「この実験で使用したデータ拡張手法を3つ挙げてください」
  → 何の実験か分からず、検索の手がかりになる語が1つもない
良い例: 「schedule-free optimizer の CIFAR-10 実験で使われたデータ拡張手法は？」
  → 主題が名指しされており、そのまま検索できる

悪い例: 「提示された sinfo -s の出力で debug パーティションの空きノードは何台ですか？」
良い例: 「Slurm の sinfo -s で debug パーティションの NODES(A/I/O/T) 欄はどう読みますか？」

## その他の条件
- 文書中の具体的な事実（設定値、手順、理由、固有名詞、数値、比較結果）を問う
- 実際に人が検索しそうな自然な日本語
- 質問文だけを読んで答えが分からない
- この文書に固有の内容を問う（一般的な技術知識で答えられない）

## 避けるべき質問
- 「〜について説明してください」のような漠然としたもの
- 「この記事では何を扱っていますか」のようなメタな質問
- 文書を見なくても常識で答えられるもの
- 質問文の中に答えが含まれているもの

作成後、その質問が上記の落とし穴に該当するかを自己判定し、各フラグを正直に埋めてください。
自己判定は厳しめにつけてください（後段で自動除外されます）。`,
  model: JUDGE_MODEL,
});

interface ChunkRow {
  chunk_id: string;
  article_id: string;
  title: string;
  heading_path: string;
  text: string;
}

/**
 * 質問の難易度カテゴリ。
 *
 * - `easy`     … 1チャンクから生成した単発検索で引ける質問。**対照群**。
 *                 単発検索で recall がほぼ100%になることを実測済み。
 *                 ここでエージェント化が無駄になることを示すのが役割
 * - `multihop` … 複数チャンク（別記事・離れたセクション）を組み合わせないと答えられない質問。
 *                 単発 top-k では片方しか取れず、ループして二度検索するパターン3が
 *                 初めて優位になりうる。**実験の主戦場**
 */
export type Difficulty = "easy" | "multihop";

export interface GoldenItem {
  question: string;
  expected_answer: string;
  /** 正解の根拠。multihop では複数入り、**全件そろって初めて正解**とみなす */
  golden_chunk_ids: string[];
  article_id: string;
  difficulty: Difficulty;
  /** レビュー時の判断材料。確定版では削っても構わない */
  note: string;
}

/**
 * 「文書が手元にある前提」の指示語。これを含む質問は検索クエリとして成立しない。
 *
 * 自己完結していない質問は、主題を特定する語を持たないため BM25 もベクトル検索も
 * 手がかりを得られず、3パターンとも一様に失敗する。結果としてパターン間の差が
 * ノイズに埋もれ、実験から信号が消える。LLMの自己判定だけに任せず決定論的にも弾く。
 *
 * ※ これは質問文の性質だけを見る検査であり、検索結果は一切参照していない。
 *    検索でヒットするかで絞ると循環論法になる（ファイル冒頭の警告を参照）。
 */
const DEICTIC_PATTERNS = [
  /本文|本記事|この記事|当記事/,
  /提示され|上記|下記|前述|後述/,
  /この(文書|資料|ページ|セクション|章)/,
  /この(実験|実装|検証|取り組み|コンペ|手順|構成|設定|例|ケース|方法|アプローチ|スクリプト|プロジェクト|フロー|環境)/,
  /ここでは|今回の記事/,
];

function findDeicticPhrase(question: string): string | undefined {
  for (const pattern of DEICTIC_PATTERNS) {
    const m = pattern.exec(question);
    if (m) return m[0];
  }
  return undefined;
}

/** コードブロックが大半を占めるチャンクは質問が作りにくいので避ける */
function isMostlyCode(text: string): boolean {
  const codeChars = [...text.matchAll(/```[\s\S]*?```/g)].reduce(
    (n, m) => n + m[0].length,
    0,
  );
  return codeChars > text.length * 0.5;
}

/** 記事ごとに、情報量の多そうなチャンクを選ぶ */
function selectChunks(rows: ChunkRow[]): ChunkRow[] {
  const byArticle = new Map<string, ChunkRow[]>();
  for (const row of rows) {
    const list = byArticle.get(row.article_id) ?? [];
    list.push(row);
    byArticle.set(row.article_id, list);
  }

  const selected: ChunkRow[] = [];
  for (const chunks of byArticle.values()) {
    const eligible = chunks
      .filter((c) => c.text.length >= MIN_CHUNK_CHARS && !isMostlyCode(c.text))
      // 長いチャンクほど問える事実が多い
      .sort((a, b) => b.text.length - a.text.length);

    // 先頭だけ取ると同じセクションに偏るので、見出しが重複しないよう散らす
    const picked: ChunkRow[] = [];
    const usedHeadings = new Set<string>();
    for (const chunk of eligible) {
      if (picked.length >= CHUNKS_PER_ARTICLE) break;
      if (usedHeadings.has(chunk.heading_path)) continue;
      usedHeadings.add(chunk.heading_path);
      picked.push(chunk);
    }
    // 見出しの種類が足りなければ残りから補充する
    for (const chunk of eligible) {
      if (picked.length >= CHUNKS_PER_ARTICLE) break;
      if (!picked.includes(chunk)) picked.push(chunk);
    }

    selected.push(...picked);
  }
  return selected;
}

async function generateForChunk(
  chunk: ChunkRow,
  retryFeedback?: string,
): Promise<GoldenItem | undefined> {
  const base = `文書のタイトルと見出し: ${chunk.heading_path}\n\n本文:\n${chunk.text}`;
  const prompt = retryFeedback
    ? `${base}\n\n---\n前回の質問は次の理由で不採用でした: ${retryFeedback}\n` +
      `主題を固有名詞で名指しし、指示語を使わずに書き直してください。`
    : base;

  const result = await generatorAgent.generate(prompt, {
    structuredOutput: { schema: GeneratedQuestion },
  });

  const gen = result.object;
  if (!gen) return undefined;

  const reasons: string[] = [];
  if (gen.answerable_without_document) reasons.push("一般知識で答えられる");
  if (gen.question_leaks_answer) reasons.push("質問文に答えが含まれる");
  if (gen.is_too_generic) reasons.push("一般的すぎる");
  if (!gen.is_self_contained) reasons.push("自己完結していない(自己判定)");
  // 短すぎる質問は具体性を欠いていることが多い
  if (gen.question.length < 12) reasons.push("質問が短すぎる");

  // 自己判定は甘く出るので決定論的にも検査する
  const deictic = findDeicticPhrase(gen.question);
  if (deictic) reasons.push(`指示語 "${deictic}" を含む`);

  if (reasons.length > 0) {
    // 指示語が原因なら書き直しで救える見込みがあるので1度だけ再試行する。
    // 「一般知識で答えられる」等は文書側の性質なので再試行しても変わらない
    if (!retryFeedback && deictic) {
      return generateForChunk(chunk, reasons.join(" / "));
    }
    console.log(`  除外 [${chunk.chunk_id}] ${reasons.join(" / ")}: ${gen.question}`);
    return undefined;
  }

  return {
    question: gen.question,
    expected_answer: gen.expected_answer,
    // 生成元チャンクを正解とする。回答が前後チャンクにまたがる場合は
    // レビュー時に手で追加してよい（スコアラーはいずれか1件ヒットで正解とみなす）
    golden_chunk_ids: [chunk.chunk_id],
    article_id: chunk.article_id,
    difficulty: "easy",
    note: chunk.heading_path,
  };
}

/** 単純な並列実行プール */
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

export async function generateGoldenSet(): Promise<GoldenItem[]> {
  const table = await openChunksTable();
  const rows = (await table
    .query()
    .select(["chunk_id", "article_id", "title", "heading_path", "text"])
    .limit(10000)
    .toArray()) as ChunkRow[];

  const targets = selectChunks(rows);
  console.log(
    `[golden] ${rows.length} チャンクから ${targets.length} 件を質問生成の対象に選定`,
  );

  const generated = await mapWithConcurrency(targets, CONCURRENCY, async (chunk, i) => {
    const item = await generateForChunk(chunk);
    if ((i + 1) % 10 === 0) console.log(`[golden] ${i + 1}/${targets.length}`);
    return item;
  });

  const items = generated.filter((x): x is GoldenItem => x !== undefined);

  // 質問文が重複したものを落とす（別チャンクから同じ質問が出ることがある）
  const seen = new Set<string>();
  const deduped = items.filter((item) => {
    const key = item.question.replace(/\s+/g, "");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  await mkdir(path.dirname(OUT_PATH), { recursive: true });
  await writeFile(OUT_PATH, JSON.stringify(deduped, null, 2), "utf8");

  console.log(
    `\n[golden] ${deduped.length} 問を ${OUT_PATH} に出力しました` +
      `（候補 ${targets.length} → 自動除外後 ${items.length} → 重複除去後 ${deduped.length}）`,
  );
  return deduped;
}

if (isMain(import.meta.url)) {
  generateGoldenSet()
    .then((items) => {
      console.log(
        "\n次の手順:\n" +
          "  1. evals/golden-set.draft.json を目視レビューする\n" +
          "  2. 「質問文に答えが書いてある」「複数記事に同じ答えがある」「一般知識で答えられる」を削る\n" +
          "  3. 残ったものを evals/golden-set.json として保存する\n" +
          `  現在 ${items.length} 問。目標は40〜60問。`,
      );
    })
    .catch((err) => {
      console.error(err instanceof Error ? err.stack : err);
      process.exit(1);
    });
}
