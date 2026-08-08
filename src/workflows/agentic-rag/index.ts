import { toCitations } from "../../shared/answer-prompt.js";
import { collectRetrievedChunkIds, countContextChars } from "../../shared/evidence-stats.js";
import { generateAnswer, generateDirectAnswer } from "../../shared/generate.js";
import { FINAL_CONTEXT_K } from "../../shared/llm-client.js";
import type { FetchResult, RagRunResult } from "../../shared/types.js";
import { MAX_RECHECKS, checkConfidence } from "./confidence-check.js";
import { checkEvidence } from "./evidence-check.js";

/**
 * 根拠収集のラウンド上限。
 *
 * 1ラウンド = search/fetch ループ1回 + 充足チェック1回。
 * ループ内の search 回数上限（MAX_TURNS）とは別軸で、
 * 「集め直しを何度まで許すか」を決める。
 */
const MAX_COLLECTION_ROUNDS = 3;
import { runSearchFetchLoop } from "./search-fetch-loop.js";
import { selectSkill } from "./skill-select.js";

/**
 * パターン3: エージェント的RAG。
 *
 * 全体フロー（docs/architecture.md）:
 *   質問 → スキル選択 →（直接回答 | 検索スキル: search→評価→fetch のループ）
 *        → 回答生成 → 確信度チェック →（不確実なら追加検索へ戻る）→ 出力
 *
 * パターン2との差分は「ループの有無」と「読む範囲をエージェントが決めること」。
 * 検索エンジン自体（ハイブリッド検索）は同一のものを使っている。
 */
export async function runAgenticRag(question: string): Promise<RagRunResult> {
  const started = Date.now();

  let inputTokens = 0;
  let outputTokens = 0;
  let llmCalls = 0;
  const queryTrace: string[] = [];

  const accumulate = (u: {
    inputTokens: number;
    outputTokens: number;
    llmCalls: number;
  }) => {
    inputTokens += u.inputTokens;
    outputTokens += u.outputTokens;
    llmCalls += u.llmCalls;
  };

  // 1. スキル選択（検索要否の判定を兼ねる）
  const selection = await selectSkill(question);
  accumulate(selection);

  // 2a. 直接回答スキル: 検索せずに答える。
  // 文書接地プロンプトではなく直接回答専用プロンプトを使う（理由は answer-prompt.ts）
  if (selection.skill === "direct") {
    const gen = await generateDirectAnswer(question);
    accumulate(gen);
    return {
      pattern: "agentic",
      question,
      answer: gen.text,
      citations: [],
      retrieved_chunk_ids: [],
      contextChars: 0,
      usage: { inputTokens, outputTokens, llmCalls },
      latencyMs: Date.now() - started,
      turns: 0,
      selectedSkill: "direct",
      queryTrace: [],
    };
  }

  // 2b. 検索スキル: search→fetch ループ。
  //     **回答を書く前に**根拠の充足をチェックし、足りなければ収集し直す。
  //     回答後のチェックだと、生成された文章の流暢さが根拠の不足を覆い隠してしまい、
  //     ループがほとんど働かなかった（初回実験でターン数の中央値が1）。
  let docs: FetchResult[] = [];
  let turns = 0;
  let missingHint: string | undefined;

  for (let round = 0; round < MAX_COLLECTION_ROUNDS; round++) {
    const loop = await runSearchFetchLoop(question, missingHint);
    accumulate(loop);
    queryTrace.push(...loop.queryTrace);
    turns += loop.turns;

    // 既出 chunk を除いて統合し、上位 k 件に切る。
    // 件数を揃えないとパターン1・2との比較が崩れる（CLAUDE.md の不変条件4）
    //
    // ⚠️ **新しい根拠を先頭に置くこと。** 素直に [...既存, ...新規] の順で結合して
    // 先頭 k 件を取ると、1ラウンド目で k 件埋まっている場合に
    // **2ラウンド目に見つけた根拠が丸ごと切り捨てられる**。
    // 2ラウンド目が走るのは「1ラウンド目の根拠では足りない」と判定されたときなので、
    // 捨ててよいのは既存の方であり、新規の方ではない。
    // （この順序を逆にしていたため、ループが働くほど精度が下がっていた）
    const seen = new Set(docs.map((d) => d.chunk_id));
    const fresh = loop.docs.filter((d) => !seen.has(d.chunk_id));
    const merged = round === 0 ? [...docs, ...fresh] : [...fresh, ...docs];
    const nextDocs = merged.slice(0, FINAL_CONTEXT_K);
    const gained = fresh.length > 0;
    docs = nextDocs;

    // 最終ラウンドならもうチェックしても行動に移せないので省く（無駄なLLM呼び出しを避ける）
    if (round === MAX_COLLECTION_ROUNDS - 1) break;

    const verdict = await checkEvidence(question, docs, queryTrace);
    accumulate(verdict);
    if (verdict.sufficient) break;

    // 2周目以降で新しい根拠が1件も増えなかったなら、これ以上回しても収束しない
    if (round > 0 && !gained) break;

    missingHint = verdict.missing.trim() || verdict.followupQuery.trim() || undefined;
  }

  // 3. 回答生成
  let gen = await generateAnswer(question, docs);
  accumulate(gen);

  // 4. 確信度チェック（回答後）。
  //    根拠の充足は 2b で見ているので、ここが捉えるのは別の失敗
  //    ——根拠は足りているのに回答がそれを正しく使えていないケース。
  //    役割が絞られたぶん再突入の上限も 1 に減らした
  for (let recheck = 0; recheck < MAX_RECHECKS; recheck++) {
    const verdict = await checkConfidence(question, gen.text, docs);
    accumulate(verdict);
    if (verdict.sufficient) break;

    const extra = await runSearchFetchLoop(
      verdict.followupQuery.trim() || question,
      verdict.missing.trim() || undefined,
    );
    accumulate(extra);
    queryTrace.push(...extra.queryTrace);
    turns += extra.turns;

    // 収集段階と同じ理由で、追加取得したぶんを先頭に置く
    const seen = new Set(docs.map((d) => d.chunk_id));
    const fresh = extra.docs.filter((d) => !seen.has(d.chunk_id));
    if (fresh.length === 0) break;
    docs = [...fresh, ...docs].slice(0, FINAL_CONTEXT_K);

    gen = await generateAnswer(question, docs);
    accumulate(gen);
  }

  return {
    pattern: "agentic",
    question,
    answer: gen.text,
    citations: toCitations(docs),
    retrieved_chunk_ids: collectRetrievedChunkIds(docs),
    contextChars: countContextChars(docs),
    usage: { inputTokens, outputTokens, llmCalls },
    latencyMs: Date.now() - started,
    turns,
    selectedSkill: "search",
    queryTrace,
  };
}
