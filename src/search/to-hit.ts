import type { SearchHit } from "../shared/types.js";

/**
 * search が返すスニペットの長さ。
 *
 * ## 200 → 500 に変更した理由
 *
 * 診断（`npm run diagnose`）で、エージェントが見た候補の完全ヒット率は 83% なのに
 * 最終結果は 74% で、**9ポイントすべてが「候補に見えていたのに fetch しなかった」**
 * ことによる損失だと分かった。7件で正解チャンクが候補リストに表示されていながら
 * 取得されていない（うち3件は1〜3位）。
 *
 * 原因は判断材料の薄さ。200字では「この見出しの、この書き出しの文書が、
 * 本当に質問に答えているか」を判断できない。
 *
 * ## それでも本文全部は返さない
 *
 * 長くしすぎると、エージェントが fetch を呼ばずスニペットだけで回答を組み立ててしまい、
 * search/fetch を分離した意味が消える（`0003-search-fetch-separation.md`）。
 * チャンクの平均は約412字なので、500字は「短いチャンクならほぼ全文、
 * 長いチャンクなら冒頭」という水準。範囲拡張（`with_neighbors` / `whole_section`）の
 * 判断は依然として fetch を呼ばないとできないため、分離の意義は保たれる。
 *
 * ## パターン1・2への影響はない
 *
 * スニペットを読むのはエージェントだけ。パターン1・2は search の結果から
 * chunk_id を取り出して直接 fetch するため、この値を変えても挙動は変わらない。
 */
export const SNIPPET_CHARS = 500;

/** LanceDB が返す生の行。距離/スコア列は検索方法によって名前が変わる */
export interface RawHit {
  chunk_id: string;
  title: string;
  heading_path: string;
  text: string;
  _distance?: number;
  _score?: number;
  _relevance_score?: number;
}

/**
 * LanceDB の行を SearchHit に落とす。**本文 (`text`) は載せない。**
 *
 * スコアは検索方法ごとに別の列で返る:
 *  - ベクトル検索 → `_distance`（小さいほど良い）
 *  - 全文検索/ハイブリッド → `_score` / `_relevance_score`（大きいほど良い）
 * 比較実験ではスコアの絶対値ではなく順位しか使わないので、
 * 「大きいほど良い」向きに揃えるだけにして正規化はしない。
 */
export function toSearchHit(row: RawHit): SearchHit {
  const score =
    row._relevance_score ??
    row._score ??
    (row._distance !== undefined ? -row._distance : 0);

  // 本文先頭には見出しパスを前置してあるので、スニペットではそれを取り除いて
  // 実際の中身が見えるようにする
  const body = row.text.startsWith(row.heading_path)
    ? row.text.slice(row.heading_path.length).trimStart()
    : row.text;

  return {
    chunk_id: row.chunk_id,
    title: row.title,
    heading_path: row.heading_path,
    snippet:
      body.length > SNIPPET_CHARS ? body.slice(0, SNIPPET_CHARS) + "…" : body,
    score,
  };
}
