/**
 * スキル選択の精度。パターン3のみが対象。
 *
 * ## 単なる正解率を見ない理由
 * architecture.md の設計は**非対称**である:
 *   - 検索スキル側は「迷ったらこちら」
 *   - 直接回答側は「少しでも疑わしければ使わない」
 * つまり意図的に検索側へ倒してあり、検索の過剰発火（＝雑談にも検索してしまう）は
 * **許容された失敗**、検索の取りこぼし（＝要検索なのに直接回答してしまう）は
 * **避けたい失敗**という重み付けになっている。
 *
 * 正解率だけを見るとこの非対称性が潰れて見えなくなるので、
 * 検索スキルを陽性クラスとして Precision / Recall を分けて出す。
 *
 * 設計が意図通り機能していれば **Recall が高く、Precision はそれより低い**という
 * 形になるはず。Precision の方が高く出ているなら、description が保守的すぎて
 * 検索に倒せていないことを意味する（＝設計意図の未達）。
 */

export type Skill = "search" | "direct";

export interface BoundaryCase {
  question: string;
  expected_skill: Skill;
  category: string;
  note: string;
}

export interface SkillSelectionScore {
  n: number;
  accuracy: number;
  /** 検索と判定したもののうち、本当に検索が必要だった割合 */
  searchPrecision: number;
  /** 本当に検索が必要だったもののうち、検索と判定できた割合。設計上こちらを重視する */
  searchRecall: number;
  /** 検索すべきなのに直接回答した件数。**設計上もっとも避けたい失敗** */
  missedSearch: number;
  /** 検索不要なのに検索した件数。設計上は許容される失敗 */
  overTriggered: number;
  /** カテゴリ別の正解率。どの類型で崩れるかを見る */
  byCategory: Record<string, { n: number; correct: number }>;
}

export function scoreSkillSelection(
  results: { case: BoundaryCase; selected: Skill }[],
): SkillSelectionScore {
  let truePositive = 0; // 期待 search / 判定 search
  let falsePositive = 0; // 期待 direct / 判定 search（過剰発火・許容）
  let falseNegative = 0; // 期待 search / 判定 direct（取りこぼし・避けたい）
  let correct = 0;
  const byCategory: Record<string, { n: number; correct: number }> = {};

  for (const { case: c, selected } of results) {
    const isCorrect = selected === c.expected_skill;
    if (isCorrect) correct++;

    if (c.expected_skill === "search" && selected === "search") truePositive++;
    if (c.expected_skill === "direct" && selected === "search") falsePositive++;
    if (c.expected_skill === "search" && selected === "direct") falseNegative++;

    const bucket = byCategory[c.category] ?? { n: 0, correct: 0 };
    bucket.n++;
    if (isCorrect) bucket.correct++;
    byCategory[c.category] = bucket;
  }

  const n = results.length;
  const precisionDenom = truePositive + falsePositive;
  const recallDenom = truePositive + falseNegative;

  return {
    n,
    accuracy: n === 0 ? 0 : correct / n,
    searchPrecision: precisionDenom === 0 ? 0 : truePositive / precisionDenom,
    searchRecall: recallDenom === 0 ? 0 : truePositive / recallDenom,
    missedSearch: falseNegative,
    overTriggered: falsePositive,
    byCategory,
  };
}
