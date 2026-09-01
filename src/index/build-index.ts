import "dotenv/config";
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import * as lancedb from "@lancedb/lancedb";
import type { BaseTokenizer } from "@lancedb/lancedb";
import { embedBatch } from "../search/embed.js";
import { isMain } from "../shared/is-main.js";
import {
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL_LABEL,
  EMBEDDING_PROFILE,
  EMBEDDING_SLUG,
  type EmbeddingBackend,
} from "../shared/llm-client.js";
import type { Chunk, ChunkRow, RawArticle } from "../shared/types.js";
import { chunkArticle } from "./chunking.js";

/**
 * data/raw/*.json → clean → chunk → embed → LanceDB 投入。
 *
 * 3パターン全てがこの単一テーブルを共有する（CLAUDE.md の不変条件1）。
 * ここで作ったインデックスは実験中は固定し、作り直したら全パターンを回し直すこと。
 */

const RAW_DIR = path.resolve("data/raw");

/**
 * インデックスの置き場所。**埋め込みプロファイルごとに分ける。**
 *
 * 分離が必須な理由: 3072次元と768次元なら LanceDB が次元不一致で落ちてくれるが、
 * bge-m3(1024) と別の1024次元モデルは**同じ形のまま意味が違うベクトル空間**になり、
 * 検索は成功して結果もそれらしく返る。不変条件3が壊れたことに誰も気づけない。
 * ディレクトリを分けるのは、この破滅ケースを物理的に不可能にするための措置
 * （docs/decisions/0014-local-embedding-backend.md）。
 */
export const DB_DIR = path.resolve(`data/index-${EMBEDDING_SLUG}`);
export const META_PATH = path.join(DB_DIR, "index-meta.json");
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

/**
 * インデックスの指紋。検索側が「自分の設定と同じ埋め込みで作られたか」を確認するために使う。
 *
 * LanceDB の別テーブルではなく sidecar JSON にしているのは:
 *   - 接続前に読めるので**日本語で throw** できる（Arrow の次元不一致エラーは原因が読めない）
 *   - `db.tableNames()` を汚さない（`openChunksTable` は includes で判定している）
 *   - 現状 build-index には Arrow の明示スキーマが一切ない（投入データから推論）。その美点を崩さない
 *   - `cat` で人間が読め、実験レポートにそのまま貼れる
 */
export interface IndexMeta {
  /** 将来メタの構造を変えたときに古いインデックスを弾くための版番号 */
  metaVersion: 1;
  embeddingBackend: EmbeddingBackend;
  embeddingModel: string;
  embeddingDimensions: number;
  embeddingPooling: string;
  queryPrefix: string;
  documentPrefix: string;
  ftsTokenizer: string;
  chunks: number;
  articles: number;
  builtAt: string;
}

function currentMeta(chunks: number, articles: number): IndexMeta {
  return {
    metaVersion: 1,
    embeddingBackend: EMBEDDING_PROFILE.backend,
    embeddingModel: EMBEDDING_PROFILE.model,
    embeddingDimensions: EMBEDDING_PROFILE.dimensions,
    embeddingPooling: EMBEDDING_PROFILE.pooling,
    queryPrefix: EMBEDDING_PROFILE.queryPrefix,
    documentPrefix: EMBEDDING_PROFILE.documentPrefix,
    ftsTokenizer: String(FTS_TOKENIZER),
    chunks,
    articles,
    builtAt: new Date().toISOString(),
  };
}

/** メタを読む。存在しない・壊れている場合は undefined（レポート表示用にも使う） */
export async function readIndexMeta(): Promise<IndexMeta | undefined> {
  try {
    return JSON.parse(await readFile(META_PATH, "utf8")) as IndexMeta;
  } catch {
    return undefined;
  }
}

/** 指紋照合は1プロセス1回でよい。hybridSearch は毎クエリ openChunksTable を通るため */
let verifiedDir: string | undefined;

async function assertIndexMatchesProfile(table: lancedb.Table): Promise<void> {
  if (verifiedDir === DB_DIR) return;

  // --- (1) 実テーブルの次元。sidecar を手で書き換えても、ここは嘘をつけない ---
  const field = (await table.schema()).fields.find((f) => f.name === "vector");
  const actualDim = (field?.type as { listSize?: number } | undefined)?.listSize;
  if (typeof actualDim === "number" && actualDim !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `インデックスのベクトル次元が設定と一致しません。\n` +
        `  インデックス実測: ${actualDim} 次元 (${DB_DIR})\n` +
        `  現在の設定:       ${EMBEDDING_DIMENSIONS} 次元 (${EMBEDDING_MODEL_LABEL})\n` +
        `  → 該当プロファイルで \`npm run build-index\` を実行し直してください。`,
    );
  }

  // --- (2) sidecar。モデルの同一性はテーブルからは分からないのでこちらで見る ---
  const meta = await readIndexMeta();
  if (!meta) {
    throw new Error(
      `${META_PATH} がありません（または壊れています）。\n` +
        `  このインデックスが**どの埋め込みモデルで作られたか検証できません**。\n` +
        `  次元が同じで中身が違うベクトル空間は検索が成功してしまい、壊れていることに\n` +
        `  気づけません（CLAUDE.md 不変条件3）。\`npm run build-index\` で作り直してください。`,
    );
  }

  const mismatches: string[] = [];
  const cmp = (name: string, want: unknown, got: unknown) => {
    if (want !== got) {
      mismatches.push(`  ${name}: インデックス=${String(got)} / 現在の設定=${String(want)}`);
    }
  };
  cmp("embeddingBackend", EMBEDDING_PROFILE.backend, meta.embeddingBackend);
  cmp("embeddingModel", EMBEDDING_PROFILE.model, meta.embeddingModel);
  cmp("embeddingDimensions", EMBEDDING_PROFILE.dimensions, meta.embeddingDimensions);
  cmp("embeddingPooling", EMBEDDING_PROFILE.pooling, meta.embeddingPooling);
  cmp("queryPrefix", EMBEDDING_PROFILE.queryPrefix, meta.queryPrefix);
  cmp("documentPrefix", EMBEDDING_PROFILE.documentPrefix, meta.documentPrefix);

  if (mismatches.length > 0) {
    throw new Error(
      `インデックスと現在の埋め込み設定が食い違っています（${DB_DIR}）。\n` +
        `${mismatches.join("\n")}\n` +
        `  この状態の検索は**エラーにならずに壊れた結果を返します**。\n` +
        `  設定を戻すか、\`npm run build-index\` で作り直してください。`,
    );
  }

  // FTS トークナイザはベクトル空間を壊さない（BM25 の質が変わるだけ）ので警告に留める
  if (meta.ftsTokenizer !== String(FTS_TOKENIZER)) {
    console.warn(
      `⚠️  FTS_TOKENIZER が構築時と異なります` +
        `（構築時="${meta.ftsTokenizer}" / 現在="${FTS_TOKENIZER}"）。\n` +
        `    ディスク上の FTS インデックスは構築時のトークナイザのままです。`,
    );
  }

  verifiedDir = DB_DIR;
}

/** 既存テーブルを開く。検索側から使う。**開く前に必ず指紋を照合する** */
export async function openChunksTable(): Promise<lancedb.Table> {
  const db = await lancedb.connect(DB_DIR);
  const names = await db.tableNames();
  if (!names.includes(TABLE_NAME)) {
    throw new Error(
      `テーブル "${TABLE_NAME}" がありません（${DB_DIR}）。\n` +
        `  埋め込み: ${EMBEDDING_MODEL_LABEL}\n` +
        `  先に \`npm run build-index\` を実行してください` +
        `（インデックスは埋め込みモデルごとに別ディレクトリです）。`,
    );
  }
  const table = await db.openTable(TABLE_NAME);
  await assertIndexMatchesProfile(table);
  return table;
}

export async function buildIndex(): Promise<{ chunks: number; articles: number }> {
  console.log(`[build] 埋め込み: ${EMBEDDING_MODEL_LABEL} / ${EMBEDDING_DIMENSIONS}次元`);
  console.log(`[build] 出力先:   ${DB_DIR}`);

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

  // 指紋を書き出す。**これを書けるのは build-index だけ**にしておくこと
  // （手で置くとメタと中身が食い違い、検証が意味を失う）
  await writeFile(
    META_PATH,
    `${JSON.stringify(currentMeta(rows.length, articles.length), null, 2)}\n`,
    "utf8",
  );
  console.log(`[build] メタを書き出しました: ${META_PATH}`);

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
