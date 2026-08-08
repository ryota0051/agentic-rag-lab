import "dotenv/config";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import * as lancedb from "@lancedb/lancedb";
import type { BaseTokenizer } from "@lancedb/lancedb";
import { embedBatch } from "../search/embed.js";
import { isMain } from "../shared/is-main.js";
import type { Chunk, ChunkRow, RawArticle } from "../shared/types.js";
import { chunkArticle } from "./chunking.js";

/**
 * data/raw/*.json → clean → chunk → embed → LanceDB 投入。
 *
 * 3パターン全てがこの単一テーブルを共有する（CLAUDE.md の不変条件1）。
 * ここで作ったインデックスは実験中は固定し、作り直したら全パターンを回し直すこと。
 */

const RAW_DIR = path.resolve("data/raw");
export const DB_DIR = path.resolve("data/index");
export const TABLE_NAME = "chunks";

/**
 * 日本語BM25のためのトークナイザ。
 *
 * デフォルトの "simple" は空白と句読点で分割するため、分かち書きしない日本語では
 * 記事本文が実質1トークンになり BM25 が壊滅する。パターン2（ハイブリッド検索）が
 * 無意味になるので明示指定が必須。
 *
 * "icu" は ICU の辞書ベース単語分割で、外部モデルのダウンロードが不要。
 * 効きが悪ければ "lindera/ipadic"（要 LANCE_LANGUAGE_MODEL_HOME）か
 * "ngram" に切り替える。判断は `npm run verify-fts` の実測で行い、
 * 結果を docs/decisions/0007-japanese-fts-tokenizer.md に記録すること。
 */
export const FTS_TOKENIZER: BaseTokenizer =
  (process.env.FTS_TOKENIZER as BaseTokenizer | undefined) ?? "icu";

export async function loadRawArticles(): Promise<RawArticle[]> {
  let files: string[];
  try {
    files = (await readdir(RAW_DIR)).filter((f) => f.endsWith(".json"));
  } catch {
    throw new Error(`${RAW_DIR} がありません。先に \`npm run ingest\` を実行してください。`);
  }
  if (files.length === 0) {
    throw new Error(`${RAW_DIR} が空です。先に \`npm run ingest\` を実行してください。`);
  }
  return Promise.all(
    files.map(async (f) => JSON.parse(await readFile(path.join(RAW_DIR, f), "utf8")) as RawArticle),
  );
}

/** 既存テーブルを開く。検索側から使う */
export async function openChunksTable(): Promise<lancedb.Table> {
  const db = await lancedb.connect(DB_DIR);
  const names = await db.tableNames();
  if (!names.includes(TABLE_NAME)) {
    throw new Error(
      `テーブル "${TABLE_NAME}" がありません。先に \`npm run build-index\` を実行してください。`,
    );
  }
  return db.openTable(TABLE_NAME);
}

export async function buildIndex(): Promise<{ chunks: number; articles: number }> {
  const articles = await loadRawArticles();
  console.log(`[build] ${articles.length} 記事を読み込みました`);

  const chunks: Chunk[] = articles.flatMap(chunkArticle);
  if (chunks.length === 0) {
    throw new Error("チャンクが0件です。clean.ts / chunking.ts を確認してください。");
  }
  const avgChars = Math.round(
    chunks.reduce((sum, c) => sum + c.text.length, 0) / chunks.length,
  );
  console.log(`[build] ${chunks.length} チャンク（平均 ${avgChars} 文字）`);

  const vectors = await embedBatch(chunks.map((c) => c.text));

  const rows: ChunkRow[] = chunks.map((c, i) => {
    const vector = vectors[i];
    if (!vector) throw new Error(`チャンク ${c.chunk_id} の埋め込みが取得できませんでした`);
    return { ...c, vector };
  });

  const db = await lancedb.connect(DB_DIR);
  // 実験の再現性のため常に作り直す。差分更新すると「いつのインデックスか」が曖昧になる。
  // createTable は Record<string, unknown>[] を要求するが、ChunkRow はインデックスシグネチャを
  // 持たない（持たせると型の誤りを拾えなくなる）ので、投入時のみキャストする
  const table = await db.createTable(
    TABLE_NAME,
    rows as unknown as Record<string, unknown>[],
    { mode: "overwrite" },
  );
  console.log(`[build] テーブル "${TABLE_NAME}" に ${rows.length} 行を投入しました`);

  // 全文検索（BM25）インデックス。トークナイザ指定が日本語では必須
  await table.createIndex("text", {
    config: lancedb.Index.fts({ baseTokenizer: FTS_TOKENIZER }),
    replace: true,
  });
  console.log(`[build] FTSインデックス作成（baseTokenizer: "${FTS_TOKENIZER}"）`);

  // ベクトルインデックス。20記事規模では総当たりでも十分速いが、
  // IVF_PQ は学習に一定の行数を要求するため、少数行では作成をスキップする
  if (rows.length >= 256) {
    await table.createIndex("vector", { replace: true });
    console.log("[build] ベクトルインデックス作成");
  } else {
    console.log(
      `[build] ベクトルインデックスはスキップ（${rows.length} 行）。総当たり検索で動作します`,
    );
  }

  return { chunks: rows.length, articles: articles.length };
}

if (isMain(import.meta.url)) {
  buildIndex()
    .then(({ chunks, articles }) => {
      console.log(`\n[build] 完了: ${articles} 記事 / ${chunks} チャンク`);
      console.log("次は `npm run verify-fts` で日本語BM25が効いているか確認してください。");
    })
    .catch((err) => {
      console.error(err instanceof Error ? err.message : err);
      process.exit(1);
    });
}
