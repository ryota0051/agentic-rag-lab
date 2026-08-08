import "dotenv/config";
import * as lancedb from "@lancedb/lancedb";
import type { BaseTokenizer } from "@lancedb/lancedb";
import { isMain } from "../shared/is-main.js";
import { DB_DIR, FTS_TOKENIZER, TABLE_NAME } from "./build-index.js";

/**
 * 日本語BM25が実際に効いているかのスモークテスト。**Phase 2 に進む前の必須ゲート。**
 *
 * LanceDB の全文検索は Tantivy ベースで、デフォルトの "simple" トークナイザは
 * 空白と句読点でしか分割しない。日本語は分かち書きしないので、何も指定しないと
 * 記事本文が実質1トークンになり BM25 が機能しない。
 *
 * ここを無検証で通すと「ハイブリッド検索はナイーブと大差ない」という**誤った結論**が出る。
 * 実装のバグではなくトークナイザの設定ミスなので、スコアを見ても原因に気づけない。
 *
 * テスト語はコーパスから自動抽出する（記事の内容に依存したハードコードを避けるため）。
 * 「あるチャンクにしか出てこない語」で検索して、そのチャンクが上位に返るかを見る。
 */

const TOP_K = 10;
/** 検証に使うチャンク数 */
const SAMPLE_SIZE = 25;

interface Row {
  chunk_id: string;
  text: string;
}

/**
 * チャンクから検索テスト用の語を抜き出す。
 *
 * カタカナ語（4文字以上）と英数字混じりの技術用語を狙う。日本語の BM25 が壊れていると
 * これらが引けなくなる。ひらがな主体の語は助詞と混ざって判定がぶれるので使わない。
 */
function extractCandidateTerms(text: string): string[] {
  const terms = new Set<string>();

  // カタカナ連続（長音符・中黒含む）4文字以上
  for (const m of text.matchAll(/[゠-ヿ]{4,}/g)) {
    if (m[0]) terms.add(m[0]);
  }
  // 漢字の連続 2〜6文字（複合語を狙う）
  for (const m of text.matchAll(/[一-鿿]{2,6}/g)) {
    if (m[0]) terms.add(m[0]);
  }

  return [...terms];
}

/** そのコーパス内で出現チャンク数が少ない＝識別力の高い語を選ぶ */
function pickDistinctiveTerm(rows: Row[], target: Row): string | undefined {
  const candidates = extractCandidateTerms(target.text);
  let best: { term: string; df: number } | undefined;

  for (const term of candidates) {
    const df = rows.reduce((n, r) => (r.text.includes(term) ? n + 1 : n), 0);
    if (df === 0) continue;
    // 1チャンクにしか出ない語が理想。同点なら長い語を優先（より特徴的）
    if (
      !best ||
      df < best.df ||
      (df === best.df && term.length > best.term.length)
    ) {
      best = { term, df };
    }
  }

  // コーパスの半分以上に出る語は識別力がなく、ヒットしなくても異常とは言えない
  if (!best || best.df > Math.max(3, rows.length * 0.5)) return undefined;
  return best.term;
}

export interface VerifyResult {
  tokenizer: BaseTokenizer;
  total: number;
  passed: number;
  failures: { term: string; expected: string; got: string[] }[];
}

async function runProbes(table: lancedb.Table, rows: Row[]): Promise<VerifyResult> {
  // 偏りを避けるためコーパス全体から等間隔にサンプリングする
  const step = Math.max(1, Math.floor(rows.length / SAMPLE_SIZE));
  const samples: Row[] = [];
  for (let i = 0; i < rows.length && samples.length < SAMPLE_SIZE; i += step) {
    const row = rows[i];
    if (row) samples.push(row);
  }

  let total = 0;
  let passed = 0;
  const failures: VerifyResult["failures"] = [];

  for (const sample of samples) {
    const term = pickDistinctiveTerm(rows, sample);
    if (!term) continue;

    total++;
    const hits = (await table
      .query()
      .fullTextSearch(term)
      .limit(TOP_K)
      .toArray()) as Row[];
    const hitIds = hits.map((h) => h.chunk_id);

    if (hitIds.includes(sample.chunk_id)) {
      passed++;
    } else {
      failures.push({ term, expected: sample.chunk_id, got: hitIds.slice(0, 3) });
    }
  }

  return { tokenizer: FTS_TOKENIZER, total, passed, failures };
}

/**
 * 候補トークナイザを順に張り替えて合格率を比較する。
 * ADR（0007-japanese-fts-tokenizer.md）に載せる実測値はこれで取る。
 * 埋め込みは再計算しないので何度回しても API コストはかからない。
 */
export async function compareTokenizers(
  tokenizers: BaseTokenizer[] = ["simple", "icu", "ngram"],
): Promise<VerifyResult[]> {
  const db = await lancedb.connect(DB_DIR);
  const table = await db.openTable(TABLE_NAME);
  const rows = (await table.query().select(["chunk_id", "text"]).toArray()) as Row[];

  const results: VerifyResult[] = [];
  for (const tokenizer of tokenizers) {
    const config =
      tokenizer === "ngram"
        ? lancedb.Index.fts({ baseTokenizer: "ngram", ngramMinLength: 2, ngramMaxLength: 3 })
        : lancedb.Index.fts({ baseTokenizer: tokenizer });

    try {
      await table.createIndex("text", { config, replace: true });
    } catch (err) {
      console.warn(
        `[verify] トークナイザ "${tokenizer}" のインデックス作成に失敗（スキップ）: ` +
          (err instanceof Error ? err.message : String(err)),
      );
      continue;
    }

    const result = await runProbes(table, rows);
    results.push({ ...result, tokenizer });
    const rate = result.total ? ((result.passed / result.total) * 100).toFixed(0) : "-";
    console.log(
      `[verify] baseTokenizer="${tokenizer}": ${result.passed}/${result.total} 合格 (${rate}%)`,
    );
  }

  // 比較で index を張り替えたので、本来の設定に戻しておく
  const restore =
    FTS_TOKENIZER === "ngram"
      ? lancedb.Index.fts({ baseTokenizer: "ngram", ngramMinLength: 2, ngramMaxLength: 3 })
      : lancedb.Index.fts({ baseTokenizer: FTS_TOKENIZER });
  await table.createIndex("text", { config: restore, replace: true });
  console.log(`[verify] インデックスを baseTokenizer="${FTS_TOKENIZER}" に戻しました`);

  return results;
}

if (isMain(import.meta.url)) {
  const compare = process.argv.includes("--compare");

  (async () => {
    if (compare) {
      const results = await compareTokenizers();
      console.log("\n=== トークナイザ比較 ===");
      for (const r of results) {
        const rate = r.total ? ((r.passed / r.total) * 100).toFixed(0) : "-";
        console.log(`  ${String(r.tokenizer).padEnd(12)} ${r.passed}/${r.total} (${rate}%)`);
      }
      console.log("\nこの数値を docs/decisions/0007-japanese-fts-tokenizer.md に記録してください。");
      return;
    }

    const db = await lancedb.connect(DB_DIR);
    const table = await db.openTable(TABLE_NAME);
    const rows = (await table.query().select(["chunk_id", "text"]).toArray()) as Row[];
    const result = await runProbes(table, rows);

    const rate = result.total ? (result.passed / result.total) * 100 : 0;
    console.log(
      `\n[verify] baseTokenizer="${result.tokenizer}": ${result.passed}/${result.total} 合格 (${rate.toFixed(0)}%)`,
    );

    if (result.failures.length) {
      console.log("\n失敗した検索語（上位3件のヒットを表示）:");
      for (const f of result.failures.slice(0, 10)) {
        console.log(`  "${f.term}" → 期待 ${f.expected} / 実際 [${f.got.join(", ")}]`);
      }
    }

    if (result.total === 0) {
      console.error("\n判定不能: テスト語を抽出できませんでした。コーパスを確認してください。");
      process.exit(1);
    }

    // 完全一致は求めない。同じ語が複数チャンクに出れば上位10件から溢れることは正常にありうる。
    // 8割を下回るならトークナイズが壊れていると判断する。
    if (rate < 80) {
      console.error(
        `\n❌ 日本語BM25が十分に機能していません（合格率 ${rate.toFixed(0)}%）。\n` +
          `   \`npm run verify-fts -- --compare\` でトークナイザを比較し、\n` +
          `   FTS_TOKENIZER 環境変数（icu / lindera/ipadic / ngram）を切り替えて再構築してください。`,
      );
      process.exit(1);
    }

    console.log("\n✅ 日本語BM25は機能しています。Phase 2 に進めます。");
  })().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
