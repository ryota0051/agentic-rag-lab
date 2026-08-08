import { openChunksTable } from "../index/build-index.js";
import type { FetchResult, FetchScope } from "../shared/types.js";

/**
 * search が返した chunk_id から本文を引き当てる（分離型の要）。
 *
 * search を軽量な参照だけに留め、本文取得をここに分けている理由は
 * 「どこまで広げて読むか」をエージェントに判断させたいから
 * （docs/decisions/0003-search-fetch-separation.md）。
 * ノイズ除去は ingestion 時に完了している前提なので、ここでは結合・重複除去・
 * メタデータ付与・予算管理だけを行う。
 */

/** 1回の fetch で返す本文の合計文字数上限。エージェントのコンテキストを圧迫させない */
const CONTEXT_BUDGET_CHARS = 6000;

const TRUNCATION_NOTICE = "\n…（コンテキスト予算により省略）";

interface ChunkRecord {
  chunk_id: string;
  article_id: string;
  title: string;
  url: string;
  heading_path: string;
  chunk_index: number;
  text: string;
  updated_at: string;
}

/** SQL 文字列リテラルへの埋め込み用。シングルクォートをエスケープする */
function sqlQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

/**
 * 同一ループ内での再取得を防ぐキャッシュ。
 * 1回の実行（1質問）ごとにインスタンスを作る。プロセス全体で共有すると
 * 実験間で状態が漏れるので、必ずワークフロー側で作り直すこと。
 */
export class FetchCache {
  private readonly seen = new Set<string>();

  /** 既に取得済みの chunk_id を除いた配列を返す */
  filterUnseen(chunkIds: string[]): string[] {
    return chunkIds.filter((id) => !this.seen.has(id));
  }

  markFetched(chunkIds: string[]): void {
    for (const id of chunkIds) this.seen.add(id);
  }

  get size(): number {
    return this.seen.size;
  }
}

/**
 * 指定した chunk_id 群の本文を、scope に応じて範囲拡張しつつ取得する。
 *
 * - `chunk_only`     … そのチャンクのみ
 * - `with_neighbors` … 前後1チャンクを含める（境界にまたがる記述を拾う）
 * - `whole_section`  … 同一 heading_path のチャンクをすべて（章単位で読む）
 */
export async function fetchChunks(
  chunkIds: string[],
  scope: FetchScope = "chunk_only",
  budgetChars: number = CONTEXT_BUDGET_CHARS,
): Promise<FetchResult[]> {
  if (chunkIds.length === 0) return [];

  const table = await openChunksTable();

  // 起点となるチャンクを引く
  const idList = chunkIds.map(sqlQuote).join(", ");
  const seeds = (await table
    .query()
    .where(`chunk_id IN (${idList})`)
    .limit(chunkIds.length)
    .toArray()) as ChunkRecord[];

  // 呼び出し側が指定した順序を保つ（検索スコア順が意味を持つため）
  const seedById = new Map(seeds.map((s) => [s.chunk_id, s]));
  const ordered = chunkIds
    .map((id) => seedById.get(id))
    .filter((s): s is ChunkRecord => s !== undefined);

  const results: FetchResult[] = [];
  // 複数の seed が同じ範囲に展開されることがあるので、全体で重複を排除する
  const usedChunkIds = new Set<string>();
  let spent = 0;

  for (const seed of ordered) {
    const expanded = await expandScope(table, seed, scope);

    const parts: string[] = [];
    const included: string[] = [];
    for (const chunk of expanded) {
      if (usedChunkIds.has(chunk.chunk_id)) continue;
      usedChunkIds.add(chunk.chunk_id);
      included.push(chunk.chunk_id);
      parts.push(chunk.text);
    }
    if (included.length === 0) continue;

    let text = parts.join("\n\n");

    // コンテキスト予算。優先度（＝検索順位）の高いものから詰め、
    // 溢れる分は切り詰める。予算を使い切ったらそれ以降は返さない。
    // 省略注記も予算の内側に収める（注記分だけ超過させない）
    if (spent >= budgetChars) break;
    if (spent + text.length > budgetChars) {
      const room = budgetChars - spent - TRUNCATION_NOTICE.length;
      // 注記すら入らないほど残りが少ないなら、このチャンクは載せずに打ち切る
      if (room <= 0) break;
      text = text.slice(0, room) + TRUNCATION_NOTICE;
    }
    spent += text.length;

    results.push({
      chunk_id: seed.chunk_id,
      title: seed.title,
      url: seed.url,
      heading_path: seed.heading_path,
      text,
      included_chunk_ids: included,
      updated_at: seed.updated_at,
    });
  }

  return results;
}

async function expandScope(
  table: Awaited<ReturnType<typeof openChunksTable>>,
  seed: ChunkRecord,
  scope: FetchScope,
): Promise<ChunkRecord[]> {
  if (scope === "chunk_only") return [seed];

  const article = sqlQuote(seed.article_id);

  if (scope === "with_neighbors") {
    const rows = (await table
      .query()
      .where(
        `article_id = ${article} AND chunk_index >= ${seed.chunk_index - 1} ` +
          `AND chunk_index <= ${seed.chunk_index + 1}`,
      )
      .limit(3)
      .toArray()) as ChunkRecord[];
    return rows.sort((a, b) => a.chunk_index - b.chunk_index);
  }

  // whole_section: 同じ見出しパスを持つチャンクをすべて集める
  const rows = (await table
    .query()
    .where(`article_id = ${article} AND heading_path = ${sqlQuote(seed.heading_path)}`)
    .limit(50)
    .toArray()) as ChunkRecord[];
  return rows.sort((a, b) => a.chunk_index - b.chunk_index);
}
