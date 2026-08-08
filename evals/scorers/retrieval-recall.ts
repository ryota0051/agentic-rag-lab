import type { GoldenItem } from "../generate-golden-set.js";
import type { RagRunResult } from "../../src/shared/types.js";

/**
 * 根拠の充足度スコア。**この実験の主指標。**
 *
 * 実際にプロンプトへ入った chunk_id と golden set の正解 chunk_id を突き合わせる。
 * **LLM を一切使わない**ので、安定・安価で、実行ごとにブレない。
 * faithfulness などの LLM 採点は生成品質を測るが、それらは
 * 「検索が正解を引けたか」を直接には表さない。3パターンの検索能力の差を
 * もっとも素直に映すのがこの指標。
 *
 * 完全ヒットは **正解 chunk がすべて揃って初めて成立**する。multihop では
 * 片方だけ取れても答えは組み立てられないため、ここを緩めると
 * マルチホップ質問を用意した意味が消える。
 */

export interface RecallScore {
  /** 正解 chunk が全件揃ったか（0 or 1） */
  exact: number;
  /** 正解 chunk のうち取得できた割合（部分点） */
  partial: number;
  /** 実際にプロンプトへ入った根拠の件数。少ない件数で当てたのか、多く取って当てたのかを見る */
  retrievedCount: number;
}

export function scoreRetrievalRecall(
  result: Pick<RagRunResult, "retrieved_chunk_ids">,
  item: Pick<GoldenItem, "golden_chunk_ids">,
): RecallScore {
  const retrieved = result.retrieved_chunk_ids;
  const golden = item.golden_chunk_ids;

  if (golden.length === 0) {
    return { exact: 0, partial: 0, retrievedCount: retrieved.length };
  }

  const found = golden.filter((g) => retrieved.includes(g)).length;

  return {
    exact: found === golden.length ? 1 : 0,
    partial: found / golden.length,
    retrievedCount: retrieved.length,
  };
}

export interface AggregatedRecall {
  n: number;
  exactRate: number;
  partialRate: number;
  meanRetrievedCount: number;
}

export function aggregateRecall(scores: RecallScore[]): AggregatedRecall {
  if (scores.length === 0) {
    return { n: 0, exactRate: 0, partialRate: 0, meanRetrievedCount: 0 };
  }
  const sum = scores.reduce(
    (acc, s) => ({
      exact: acc.exact + s.exact,
      partial: acc.partial + s.partial,
      count: acc.count + s.retrievedCount,
    }),
    { exact: 0, partial: 0, count: 0 },
  );
  return {
    n: scores.length,
    exactRate: sum.exact / scores.length,
    partialRate: sum.partial / scores.length,
    meanRetrievedCount: sum.count / scores.length,
  };
}
