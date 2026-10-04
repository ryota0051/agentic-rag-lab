import { evaluateJev, JEV_LIMITS, type JevResult } from "../../shared/jev-client.js";
import type { FetchResult } from "../../shared/types.js";
import {
  DIRECT_ANSWER_SKILL_DESCRIPTION,
  SEARCH_SKILL_DESCRIPTION,
  SKILL_SELECTION_POLICY,
} from "../../skills/descriptions.js";

/**
 * パターン3の判定ステップを Jev（判定専用モデル）で行う版。
 * → docs/decisions/0015-jev-decision-model.md
 *
 * LLM 版（skill-select.ts / evidence-check.ts）と**同じ判断基準の文章**を渡すこと。
 * jev-hello のケース6で、criteria の書き方ひとつで p(search) が 0%〜89% まで動くと分かっている。
 * Jev 向けに文言を調整すると、比べているのが「モデル」なのか「プロンプト」なのか分からなくなる。
 */

// ---------------------------------------------------------------------------
// A. スキル分岐

export interface JevSkillJudgement {
  pSearch: number;
  pDirect: number;
  latencyMs: number;
  inputTokens: number;
  costUsd: number;
  model: string;
}

/**
 * 確率だけを返す。どちらを選ぶかは呼び出し側が閾値で決める
 * （閾値で非対称コストを表すのがこの判定器を使う理由なので、ここで argmax を取らない）。
 */
export async function judgeSkillJev(question: string): Promise<JevSkillJudgement> {
  const r = await evaluateJev({
    state: question,
    questions: {
      skill: {
        type: "choice",
        instructions:
          "あなたはユーザーの質問に対して、使用するスキルを1つ選ぶルーターです。\n\n" +
          SKILL_SELECTION_POLICY,
        criteria: {
          search: SEARCH_SKILL_DESCRIPTION,
          direct: DIRECT_ANSWER_SKILL_DESCRIPTION,
        },
      },
    },
  });
  const a = r.answers.skill;
  if (a?.type !== "choice") throw new Error("Jev: skill の回答が choice ではない");
  return {
    pSearch: a.probabilities.search ?? 0,
    pDirect: a.probabilities.direct ?? 0,
    ...meta(r),
  };
}

/** `p(direct) ≥ τ` のときだけ direct。それ以外は検索に倒す（ADR 0001 の非対称コスト） */
export function skillByThreshold(pDirect: number, tau: number): "search" | "direct" {
  return pDirect >= tau ? "direct" : "search";
}

// ---------------------------------------------------------------------------
// B. 収集完了の判定

export interface JevEvidenceJudgement {
  /** noul: 根拠だけで質問のすべての要素に答えられる確率 */
  pSufficient: number;
  /** score: カバー率の期待値（0=なし, 1=一部, 2=すべて） */
  coverage: number;
  /** score の「すべてカバー」の確率 */
  pFullCoverage: number;
  latencyMs: number;
  inputTokens: number;
  costUsd: number;
  model: string;
}

/** 1件あたりの切り詰め字数。5件 × 1,200 字 + 見出し・質問で state 上限 8,000 字に収まる */
export const JEV_PER_DOC_CHARS = 1200;

/** Jev に渡す形に切り詰めた根拠。**LLM 側にも同じものを渡して入力条件を揃える**こと */
export function truncateForJev(docs: FetchResult[]): FetchResult[] {
  return docs.map((d) => ({ ...d, text: d.text.slice(0, JEV_PER_DOC_CHARS) }));
}

export function evidenceState(question: string, docs: FetchResult[]): string {
  const body = docs.length
    ? docs.map((d, i) => `[${i + 1}] ${d.heading_path}\n${d.text}`).join("\n\n")
    : "(根拠が1件も集まっていません)";
  const s = `質問: ${question}\n\n集まった根拠（${docs.length}件）:\n${body}`;
  // 見出しが長い場合の保険。末尾を落とす（先頭の質問は必ず残す）
  return s.slice(0, JEV_LIMITS.stateChars);
}

/**
 * 問いは肯定形の1本に固定する。否定形（「足りない情報が残っているか」）は
 * yes 側に寄った（jev-hello ケース6b）。判断基準は evidence-check.ts の instructions に揃えている。
 */
const SUFFICIENT_INSTRUCTIONS =
  "集まった根拠だけで、質問のすべての要素に答えられますか？\n" +
  "質問を構成要素に分解し、それぞれに対応する記述が根拠の中にあるかを確認してください。\n" +
  "次の場合は「いいえ」です: 質問が2つ以上の対象を扱っているのに根拠が片方しかカバーしていない／" +
  "具体的な数値・設定値・手順を求めているのに根拠にその値が無い／" +
  "根拠が主題と関連しているだけで、問われている論点そのものには触れていない／根拠が0件。\n" +
  "「だいたい関連しているから十分」とは判断しないでください。";

export async function judgeEvidenceJev(
  question: string,
  docs: FetchResult[],
): Promise<JevEvidenceJudgement> {
  const r = await evaluateJev({
    state: evidenceState(question, docs),
    questions: {
      sufficient: { type: "noul", instructions: SUFFICIENT_INSTRUCTIONS },
      coverage: {
        type: "score",
        instructions: "質問の要素のうち、根拠でカバーされている割合はどの程度ですか？",
        criteria: ["まったくカバーしていない", "一部だけカバーしている", "すべてカバーしている"],
      },
    },
  });
  const s = r.answers.sufficient;
  const c = r.answers.coverage;
  if (s?.type !== "noul" || c?.type !== "score") throw new Error("Jev: 回答の型が想定外");
  return {
    pSufficient: s.noul,
    coverage: c.score,
    pFullCoverage: c.probabilities["2"] ?? 0,
    ...meta(r),
  };
}

function meta(r: JevResult) {
  return { latencyMs: r.latencyMs, inputTokens: r.inputTokens, costUsd: r.costUsd, model: r.model };
}
