/** Qiita API から取得した記事の生データ（data/raw/*.json の形） */
export interface RawArticle {
  article_id: string;
  title: string;
  url: string;
  tags: string[];
  body: string;
  created_at: string;
  updated_at: string;
}

/** チャンク分割後、インデックス投入前の単位 */
export interface Chunk {
  /** `${article_id}#${chunk_index}`。golden set の正解ラベルもこの形式 */
  chunk_id: string;
  article_id: string;
  title: string;
  url: string;
  /** 例: "記事タイトル > セットアップ > Docker"。文脈補強と fetch の範囲拡張に使う */
  heading_path: string;
  chunk_index: number;
  text: string;
  tags: string[];
  updated_at: string;
}

/** LanceDB のテーブル行。Chunk に埋め込みベクトルが付いたもの */
export interface ChunkRow extends Chunk {
  vector: number[];
}

/**
 * search が返す軽量な参照。
 *
 * 本文 (`text`) は含めない。ここで本文まで返すとエージェントが snippet で判断せず
 * 何でも読みに行き、コンテキストが肥大化する（docs/architecture.md の設計方針）。
 */
export interface SearchHit {
  chunk_id: string;
  title: string;
  heading_path: string;
  /** 先頭数百字のみ */
  snippet: string;
  score: number;
}

/** fetch の取得範囲。エージェントに選ばせる */
export type FetchScope = "chunk_only" | "with_neighbors" | "whole_section";

/** fetch が返す本文＋引用メタデータ */
export interface FetchResult {
  chunk_id: string;
  title: string;
  url: string;
  heading_path: string;
  /** 範囲拡張・結合済みの本文 */
  text: string;
  /** 実際に結合された chunk_id の一覧（重複排除の検証用） */
  included_chunk_ids: string[];
  updated_at: string;
}

/** 回答に付ける引用 */
export interface Citation {
  chunk_id: string;
  title: string;
  url: string;
  heading_path: string;
}

export type RagPattern = "naive" | "hybrid" | "agentic";

/**
 * エージェントがツールをどう使ったかの記録（パターン3のみ）。
 *
 * ## なぜ必要か
 *
 * ツール呼び出しや構造化出力に失敗しても、コード側のフォールバックが
 * **もっともらしい正常値**に化けてしまう:
 *
 * - `skill-select.ts`   構造化出力が取れないと "search" に倒す → 「正しく検索を選んだ」に見える
 * - `evidence-check.ts` / `confidence-check.ts` 判定失敗を sufficient=true に倒す
 *                        → 「1回で足りた」＝ループ不要に見える
 * - `query-decompose.ts` 分解失敗で空配列 → 「複合質問ではなかった」に見える
 * - `fetch.ts`          存在しない chunk_id を黙って捨てる → 掴んだ事実自体が消える
 *
 * つまり計測を入れないと、モデルがツールを使えていない状態が
 * 「ターン数中央値1、ループが働かなかった」という観測に化ける。
 * バグを実験結果として記録しないために、失敗を明示的に数える。
 */
export interface ToolUseStats {
  /** エージェント自身が search を呼んだ回数。前処理の種検索は含めない */
  searchToolCalls: number;
  /**
   * search の回数上限に達して弾かれた回数。
   * 予算の指示を守れないモデルほど増えるので、指示追従性の指標になる
   */
  searchBlockedCalls: number;
  /** fetch を呼んだ回数 */
  fetchToolCalls: number;
  /**
   * fetch の scope 選択の内訳。
   * 「どこまで広げて読むかをエージェント自身に決めさせる」という
   * search/fetch 分離の狙い（docs/decisions/0003）が機能しているかの直接指標
   */
  fetchScopes: Record<FetchScope, number>;
  /** 全 chunk_id が取得済みで空返しになった fetch の回数。同じ行動の反復を捉える */
  fetchNoOpCalls: number;
  /**
   * fetch を要求したのに本文が返らなかった chunk_id の数。
   *
   * **存在しないIDとコンテキスト予算切れの合算**である点に注意。
   * 「ハルシネーションしたID数」と断定してはいけない
   */
  fetchUnresolvedIds: number;
  /**
   * 構造化出力が取れずフォールバックした回数（スキル選択・充足チェック・
   * 確信度チェック・質問分解の合計）。
   *
   * **ここが0でない実験結果は、ターン数もスキル選択精度も「モデルの判断」ではなく
   * 「フォールバックの結果」を測っている。** レポートには必ずその旨を書くこと
   */
  structuredOutputFailures: number;
}

/** ツール未使用パターン用のゼロ値。集計側で undefined を分岐しなくて済む */
export function emptyToolUseStats(): ToolUseStats {
  return {
    searchToolCalls: 0,
    searchBlockedCalls: 0,
    fetchToolCalls: 0,
    fetchScopes: { chunk_only: 0, with_neighbors: 0, whole_section: 0 },
    fetchNoOpCalls: 0,
    fetchUnresolvedIds: 0,
    structuredOutputFailures: 0,
  };
}

/**
 * 3パターン共通の戻り値形状（CLAUDE.md の不変条件5）。
 * 比較実験はこの形状に依存しているので、パターンごとに勝手に増やさないこと。
 */
export interface RagRunResult {
  pattern: RagPattern;
  question: string;
  answer: string;
  citations: Citation[];
  /**
   * 実際にプロンプトへ入った chunk_id。retrieval-recall スコアラーの採点対象。
   *
   * **fetch の範囲拡張で結合されたチャンクもすべて含める**こと。
   * seed の chunk_id だけを記録すると、`with_neighbors` / `whole_section` で
   * 実際に読んだ隣接チャンクが計上されず、範囲拡張を使うパターン3だけが
   * 一方的に減点される（実際にそうなっていた）。
   */
  retrieved_chunk_ids: string[];
  /**
   * プロンプトへ入った根拠の総文字数。
   *
   * 件数だけ見ると、範囲拡張で1件に何チャンクも詰め込んだパターン3が
   * 「少ない根拠で当てた」ように見えてしまう。実際に与えた情報量を併記する。
   */
  contextChars: number;
  usage: {
    inputTokens: number;
    outputTokens: number;
    /** LLM API を何回叩いたか。ループのコストが見える */
    llmCalls: number;
  };
  latencyMs: number;
  /** search→fetch ループのターン数。パターン1・2は常に 0 */
  turns: number;
  /** パターン3のみ。どちらのスキルが選ばれたか */
  selectedSkill?: "search" | "direct";
  /** パターン3のみ。各ターンで実際に投げたクエリ（言い換えの観察用） */
  queryTrace?: string[];
  /**
   * パターン3のみ。ツールの使われ方（`ToolUseStats` の説明を参照）。
   * パターン1・2はそもそもツールを持たないので付かない
   */
  toolUse?: ToolUseStats;
}
