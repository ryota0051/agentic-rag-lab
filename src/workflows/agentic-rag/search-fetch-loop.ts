import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { FetchCache, fetchChunks } from "../../search/fetch.js";
import { hybridSearch } from "../../search/hybrid-search.js";
import { readUsage } from "../../shared/generate.js";
import { FINAL_CONTEXT_K, GENERATION_MODEL } from "../../shared/llm-client.js";
import type { FetchResult } from "../../shared/types.js";

/**
 * 検索スキルの内部処理: search → 評価 → fetch のループ。
 *
 * search は軽量な参照（chunk_id・タイトル・スニペット・スコア）のみを返し、
 * 本文は fetch でしか取れない。この分離により「どこまで読むか」の判断を
 * エージェント自身に行わせる（docs/decisions/0003-search-fetch-separation.md）。
 *
 * turn 上限は無限ループ防止のガード。確信度チェック（事後補正）とは
 * **独立した安全装置**として併用する（docs/architecture.md）。
 */

/** search→fetch ループの上限。architecture.md の想定は3〜6ターン */
export const MAX_TURNS = 3;

/** search が返す候補数 */
const SEARCH_K = 8;

export interface LoopOutcome {
  docs: FetchResult[];
  turns: number;
  queryTrace: string[];
  inputTokens: number;
  outputTokens: number;
  llmCalls: number;
}

/**
 * 1回の実行ごとにツールを作り直す。
 *
 * 取得したドキュメントと発行クエリを収集する必要があり、モジュールスコープの
 * 可変状態にすると質問間で汚染される（評価を並列実行すると即座に壊れる）。
 */
function createSearchTools(collected: FetchResult[], queryTrace: string[]) {
  const cache = new FetchCache();

  // turn 上限の**強制**。プロンプトに「最大N回」と書くだけでは守られず、
  // 初回実験では1問で10回検索したケースがあった（無限ループ防止という
  // ガードの目的を果たしていなかった）。ツール側で機械的に打ち切る。
  //
  // 1ターン目のシード検索も1回として数えるため、カウンタは呼び出し側と共有する
  const counter = { searchCalls: 0 };

  const searchTool = createTool({
    id: "search",
    description:
      "文書インデックスを検索し、候補の参照情報（chunk_id・見出し・冒頭抜粋・スコア）を返す。" +
      "本文は返らないため、読むべきと判断した候補は fetch で取得すること。" +
      "結果のスコアが全体的に低い、または件数が少ない場合は、クエリを言い換えて再検索するとよい。",
    inputSchema: z.object({
      query: z
        .string()
        .describe(
          "検索クエリ。語を継ぎ足して長くするのではなく、別の言い方に**置き換えた**短い文にすること",
        ),
    }),
    execute: async ({ query }) => {
      if (counter.searchCalls >= MAX_TURNS) {
        return {
          hits: [],
          note:
            `検索回数の上限（${MAX_TURNS}回）に達しました。` +
            `これ以上検索できません。取得済みの情報で回答を組み立ててください。`,
        };
      }
      counter.searchCalls++;
      queryTrace.push(query);
      const hits = await hybridSearch(query, SEARCH_K);
      return {
        hits: hits.map((h) => ({
          chunk_id: h.chunk_id,
          heading: h.heading_path,
          snippet: h.snippet,
          score: Number(h.score.toFixed(4)),
        })),
      };
    },
  });

  const fetchTool = createTool({
    id: "fetch",
    description:
      "search で得た chunk_id の本文を取得する。scope で読む範囲を選べる: " +
      "chunk_only=そのチャンクのみ / with_neighbors=前後1チャンクを含む / " +
      "whole_section=同じ見出しのセクション全体。文脈が足りないと感じたら範囲を広げること。",
    inputSchema: z.object({
      chunk_ids: z.array(z.string()).describe("取得する chunk_id（search の結果から選ぶ）"),
      scope: z
        .enum(["chunk_only", "with_neighbors", "whole_section"])
        .default("chunk_only")
        .describe("読む範囲"),
    }),
    execute: async ({ chunk_ids, scope }) => {
      // 同一ループ内での再取得を防ぐ。無駄なトークンとレイテンシを避ける
      const unseen = cache.filterUnseen(chunk_ids);
      if (unseen.length === 0) {
        return { documents: [], note: "指定された chunk は取得済みです。" };
      }
      const docs = await fetchChunks(unseen, scope);
      cache.markFetched(docs.flatMap((d) => d.included_chunk_ids));
      collected.push(...docs);

      return {
        documents: docs.map((d) => ({
          chunk_id: d.chunk_id,
          heading: d.heading_path,
          text: d.text,
        })),
      };
    },
  });

  return { searchTool, fetchTool, counter };
}

export async function runSearchFetchLoop(
  question: string,
  /** 前ラウンドの充足チェックで指摘された不足。2周目以降に渡される */
  missingHint?: string,
): Promise<LoopOutcome> {
  const collected: FetchResult[] = [];
  const queryTrace: string[] = [];
  const { searchTool, fetchTool, counter } = createSearchTools(collected, queryTrace);

  const agent = new Agent({
    id: "search-agent",
    name: "Search Agent",
    instructions: `あなたは文書インデックスから、質問に答えるための根拠を集める調査担当です。

手順:
1. search で候補を探す。まずは質問の要点を表すクエリを投げる。
2. 返ってきた抜粋を見て、本文を読む価値がある候補を選び fetch で取得する。
3. 根拠が不十分なら、クエリを言い換えて search からやり直す。

クエリの言い換え方（重要）:
- **語を継ぎ足さないこと。** 元の質問に関連語を足して長くすると、検索精度はむしろ**下がる**。
  キーワードを並べた長い文字列は、質問文としての意味を失い、
  余分な語が一致度を薄めるため。
- 正しいやり方は**置き換え**。元の質問と同じ長さかそれ以下の、別の言い方の文にする。
  - 略語 ⇄ 正式名称を入れ替える（「KVキャッシュ」⇄「key-value cache」）
  - 質問の言葉を、文書側で使われていそうな言葉に置き換える
    （「速くなるのはなぜ」→「スループット改善の理由」）
  - 質問が複数の対象を含むなら、**対象を1つに絞った**短いクエリにする
- 同じ語彙の言い換えを繰り返さないこと。前回と違う切り口にすること。
4. 十分な根拠が集まったら、収集した内容を要約せずに「収集完了」とだけ答えて終了する。

根拠の量について:
- 回答生成に渡せる根拠は**最大 ${FINAL_CONTEXT_K} 件**。それを超えた分は捨てられる。
- 逆に ${FINAL_CONTEXT_K} 件に満たない場合は**枠が余ったまま**回答生成に進む。
  関連しそうな候補が残っているなら、余った枠のぶんは fetch しておくこと。
- ただし無関係なものを埋め合わせで入れないこと。枠を埋めること自体が目的ではない。

制約:
- search を最大 ${MAX_TURNS} 回まで。それ以上は呼ばないこと。
- 抜粋（snippet）だけで回答を組み立てないこと。必ず fetch で本文を確認する。
- あなたの仕事は根拠を集めることであり、回答を書くことではない。`,
    model: GENERATION_MODEL,
    tools: { search: searchTool, fetch: fetchTool },
  });

  // maxSteps はツール呼び出しを含むLLMステップ数の上限。
  // search + fetch を1ターンとみなし、往復と最終応答の余裕を見て設定する
  // 1ターン目は**元の質問をそのまま**ハイブリッド検索に通し、その結果を渡した状態から
  // エージェントを開始する。
  //
  // 診断で、エージェントが自作した言い換えクエリは元の質問より検索性能が低いことが
  // 分かっている（元の質問 @5 で 80% に対し、言い換え @5 は 63%）。
  // 言い換えを最初から使うと、一番良いクエリを試さないまま劣化した地点から始まる。
  // 言い換えは「1回目で足りなかったとき」の手段として2ターン目以降に回す。
  //
  // architecture.md の「1ターン目は必ずハイブリッド検索を通す」というガイドとも一致する。
  const seeded = await hybridSearch(question, SEARCH_K);
  queryTrace.push(question);
  counter.searchCalls++;

  const seededList = seeded
    .map(
      (h, i) =>
        `${i + 1}. chunk_id=${h.chunk_id}\n   見出し: ${h.heading_path}\n   抜粋: ${h.snippet.replace(/\n/g, " ")}`,
    )
    .join("\n");

  const parts = [
    question,
    "\n---\n元の質問をそのまま検索した結果です。まずここから読むべきものを選んで fetch してください。\n",
    seededList,
  ];
  if (missingHint) {
    parts.push(
      `\n---\n前回の収集では以下が不足していました。これを埋める根拠を探してください:\n${missingHint}`,
    );
  }

  const result = await agent.generate(parts.join("\n"), {
    maxSteps: MAX_TURNS * 2 + 2,
  });

  const { inputTokens, outputTokens } = readUsage(result.totalUsage ?? result.usage);

  // 集めすぎた場合は上位 k 件に切る。パターン1・2と根拠件数を揃えるため
  // （CLAUDE.md の不変条件4）。これをしないと「根拠が多いから強い」だけの結論になる
  const docs = collected.slice(0, FINAL_CONTEXT_K);

  return {
    docs,
    turns: queryTrace.length,
    queryTrace,
    inputTokens,
    outputTokens,
    llmCalls: result.steps?.length ?? 1,
  };
}
