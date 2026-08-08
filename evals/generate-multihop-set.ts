import "dotenv/config";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { Agent } from "@mastra/core/agent";
import { z } from "zod";
import { openChunksTable } from "../src/index/build-index.js";
import { isMain } from "../src/shared/is-main.js";
import { JUDGE_MODEL } from "../src/shared/llm-client.js";
import type { GoldenItem } from "./generate-golden-set.js";

/**
 * マルチホップ質問の生成と、golden set の最終組み立て。
 *
 * ## なぜ必要か
 * 1チャンクから作った easy 質問は、単発ハイブリッド検索で recall@5 = 100% に張り付く
 * ことを実測した。単発が満点ならエージェント化に改善の余地は構造的に存在せず、
 * 「ループはコストが増えるだけ」という、データの性質に由来する結論しか出ない。
 *
 * マルチホップ質問は、**別々の記事にある2つのチャンクを組み合わせないと答えられない**。
 * 単発 top-5 では片方しか入らないことが起こりうる一方、search→評価→再検索のループを
 * 持つパターン3は2回目の検索でもう片方を取りに行ける。ここで初めて
 * 「エージェント化が何のために必要か」が観測可能になる。
 *
 * ## ペアの選び方
 * あるチャンクのベクトル近傍を**別記事から**引くことで、
 * 「話題は近いが別文書にある」ペアを作る。同一記事内だと近接チャンクが
 * 1回の検索でまとめて取れてしまい、マルチホップにならない。
 */

const DRAFT_PATH = path.resolve("evals/golden-set.draft.json");

/** easy 質問を対照群として何問残すか（1記事1問で20記事ぶん） */
const EASY_KEEP_PER_ARTICLE = 1;
/** マルチホップ質問の生成試行数 */
const MULTIHOP_TARGETS = 30;
const MIN_CHUNK_CHARS = 250;
const CONCURRENCY = 4;

const MultiHopQuestion = z.object({
  question: z
    .string()
    .describe(
      "2つの文書の情報を両方使わないと答えられない、自己完結した日本語の質問",
    ),
  expected_answer: z.string().describe("両方の文書に基づく簡潔な想定回答"),
  requires_both: z
    .boolean()
    .describe(
      "本当に両方の文書が必要なら true。片方だけで答えられてしまうなら false",
    ),
  is_self_contained: z
    .boolean()
    .describe("指示語に頼らず主題を名指ししていれば true"),
  is_natural: z
    .boolean()
    .describe(
      "人が実際に尋ねそうな自然な質問なら true。" +
        "無関係な2件を強引に接続しただけの不自然な質問なら false",
    ),
});

const multihopAgent = new Agent({
  id: "multihop-generator",
  name: "Multi-hop Question Generator",
  instructions: `あなたは RAG 検索の評価用に、**マルチホップ質問**を作成します。

2つの異なる文書の断片が与えられます。**両方の情報を使わないと答えられない**質問を1つ作ってください。

## 良いマルチホップ質問
- 比較: 「AとBのそれぞれの〜は何が違うか」（各文書に片方ずつ書かれている）
- 統合: 「Aの手順で使う〜を、Bではどう設定しているか」
- 因果の接続: 「Aで報告された問題は、Bのどの設定で回避できるか」

## 必須の条件
- **自己完結**: 「この記事」「本文」「上記」などの指示語は使わない。
  製品名・ライブラリ名・手法名など固有名詞で主題を名指しすること
- **両方が必要**: 片方の文書だけで答えられる質問は失格
- **自然さ**: 実際に人が尋ねそうな質問であること。
  全く無関係な2件を無理に接続した不自然な質問なら is_natural=false にすること

2つの文書に共通の話題が見出せず、自然な質問が作れない場合は、
無理に作らず is_natural=false としてください。正直な自己判定をしてください。`,
  model: JUDGE_MODEL,
});

interface ChunkRow {
  chunk_id: string;
  article_id: string;
  title: string;
  heading_path: string;
  text: string;
  vector: number[];
}

const DEICTIC =
  /本文|本記事|この記事|提示され|上記|下記|前述|後述|この(文書|実験|実装|検証|取り組み|コンペ|手順|構成|設定|例|ケース|方法|アプローチ)/;

function isMostlyCode(text: string): boolean {
  const codeChars = [...text.matchAll(/```[\s\S]*?```/g)].reduce(
    (n, m) => n + m[0].length,
    0,
  );
  return codeChars > text.length * 0.5;
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

/** easy 質問を1記事あたり N 問に間引く */
function trimEasy(items: GoldenItem[]): GoldenItem[] {
  const perArticle = new Map<string, number>();
  const kept: GoldenItem[] = [];
  for (const item of items) {
    const n = perArticle.get(item.article_id) ?? 0;
    if (n >= EASY_KEEP_PER_ARTICLE) continue;
    perArticle.set(item.article_id, n + 1);
    kept.push({ ...item, difficulty: "easy" });
  }
  return kept;
}

export async function generateMultihopSet(): Promise<GoldenItem[]> {
  const table = await openChunksTable();
  const rows = (await table.query().limit(10000).toArray()) as ChunkRow[];

  const eligible = rows.filter(
    (r) => r.text.length >= MIN_CHUNK_CHARS && !isMostlyCode(r.text),
  );

  // 記事が偏らないよう、記事を巡回しながらシードを選ぶ
  const byArticle = new Map<string, ChunkRow[]>();
  for (const row of eligible) {
    const list = byArticle.get(row.article_id) ?? [];
    list.push(row);
    byArticle.set(row.article_id, list);
  }
  const articleLists = [...byArticle.values()].map((list) =>
    list.slice().sort((a, b) => b.text.length - a.text.length),
  );

  const seeds: ChunkRow[] = [];
  for (let round = 0; seeds.length < MULTIHOP_TARGETS; round++) {
    let added = false;
    for (const list of articleLists) {
      const chunk = list[round];
      if (!chunk) continue;
      seeds.push(chunk);
      added = true;
      if (seeds.length >= MULTIHOP_TARGETS) break;
    }
    if (!added) break;
  }

  console.log(`[multihop] ${seeds.length} 件のシードからペアを構成します`);

  const generated = await mapWithConcurrency(seeds, CONCURRENCY, async (seed, i) => {
    // 別記事から話題の近いチャンクを1件引く（同一記事は除外）
    const neighbors = (await table
      .query()
      .nearestTo(seed.vector)
      .where(`article_id != '${seed.article_id.replace(/'/g, "''")}'`)
      .limit(1)
      .toArray()) as ChunkRow[];

    const partner = neighbors[0];
    if (!partner) return undefined;

    const result = await multihopAgent.generate(
      `## 文書A\n見出し: ${seed.heading_path}\n本文:\n${seed.text}\n\n` +
        `## 文書B\n見出し: ${partner.heading_path}\n本文:\n${partner.text}`,
      { structuredOutput: { schema: MultiHopQuestion } },
    );

    if ((i + 1) % 10 === 0) console.log(`[multihop] ${i + 1}/${seeds.length}`);

    const gen = result.object;
    if (!gen) return undefined;

    const reasons: string[] = [];
    if (!gen.requires_both) reasons.push("片方の文書だけで答えられる");
    if (!gen.is_self_contained) reasons.push("自己完結していない");
    if (!gen.is_natural) reasons.push("不自然な接続");
    if (DEICTIC.test(gen.question)) reasons.push("指示語を含む");

    if (reasons.length > 0) {
      console.log(`  除外 [${seed.chunk_id}] ${reasons.join(" / ")}`);
      return undefined;
    }

    const item: GoldenItem = {
      question: gen.question,
      expected_answer: gen.expected_answer,
      // 2件そろって初めて正解。スコアラーはこれを厳密に見る
      golden_chunk_ids: [seed.chunk_id, partner.chunk_id],
      article_id: seed.article_id,
      difficulty: "multihop",
      note: `${seed.heading_path} ＋ ${partner.heading_path}`,
    };
    return item;
  });

  return generated.filter((x): x is GoldenItem => x !== undefined);
}

async function main() {
  const existing = JSON.parse(await readFile(DRAFT_PATH, "utf8")) as GoldenItem[];
  const easy = trimEasy(existing.filter((x) => x.difficulty !== "multihop"));
  console.log(`[multihop] easy を ${existing.length} → ${easy.length} 問に間引きました`);

  const multihop = await generateMultihopSet();

  // 質問文の重複を除去
  const seen = new Set<string>();
  const combined = [...easy, ...multihop].filter((item) => {
    const key = item.question.replace(/\s+/g, "");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  await writeFile(DRAFT_PATH, JSON.stringify(combined, null, 2), "utf8");
  console.log(
    `\n[multihop] 合計 ${combined.length} 問（easy ${easy.length} / multihop ${multihop.length}）を ` +
      `${DRAFT_PATH} に出力しました`,
  );
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
