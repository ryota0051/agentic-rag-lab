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
}
