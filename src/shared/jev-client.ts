import { z } from "zod";
import { JEV_BASE_URL, JEV_MODEL, JEV_USD_PER_M_INPUT } from "./llm-client.js";
import { withRetry } from "./retry.js";

/**
 * Jev（TypeSafe System One）の薄いクライアント。
 *
 * SDK は使わず fetch で直接叩く。エンドポイントは1本（`POST /v1/systemone`）で、
 * 依存を増やすほどの厚みがないため。
 *
 * 質問の型は3つ:
 * - `choice` … criteria（選択肢名 → 説明）から1つ選び、各選択肢の確率を返す
 * - `score`  … 順序付きの段階（2〜10）のどこに当たるかを返す
 * - `noul`   … yes/no の「yes である確率」だけを返す
 *
 * API の制約はリクエスト前にここで検査する。超えると API 側で 4xx になるが、
 * どの値が原因かが分かりにくいので、手元で落とす方が調べやすい。
 */

export const JEV_LIMITS = {
  stateChars: 8000,
  instructionsChars: 1800,
  questionsPerRequest: 8,
  questionNameChars: 64,
  minOptions: 2,
  maxOptions: 20,
} as const;

export type JevQuestion =
  | { type: "choice"; instructions: string; criteria: Record<string, string> }
  /** score の criteria は**配列**（低い順）。オブジェクトで渡すと 400 になる */
  | { type: "score"; instructions: string; criteria: string[] }
  | { type: "noul"; instructions: string };

export interface JevRequest {
  state: string;
  questions: Record<string, JevQuestion>;
}

const ChoiceAnswer = z.object({
  type: z.literal("choice"),
  choice: z.string(),
  confidence: z.number(),
  probabilities: z.record(z.string(), z.number()),
});
const ScoreAnswer = z.object({
  type: z.literal("score"),
  /** 期待値（0 始まりの段階インデックスで、小数になりうる） */
  score: z.number(),
  confidence: z.number(),
  /** キーは段階インデックス（"0", "1", ...） */
  probabilities: z.record(z.string(), z.number()),
  legend: z.record(z.string(), z.string()),
});
const NoulAnswer = z.object({
  type: z.literal("noul"),
  noul: z.number(),
});

const JevResponseSchema = z.object({
  /** エイリアスではなく解決後の実バージョン（例: jev-1.13.0）が入る */
  model: z.string(),
  answers: z.record(z.string(), z.union([ChoiceAnswer, ScoreAnswer, NoulAnswer])),
  usage: z.object({ input_tokens: z.number(), output_tokens: z.number() }),
});

export type JevAnswer = z.infer<typeof JevResponseSchema>["answers"][string];

export interface JevResult {
  /** 解決後のモデルバージョン */
  model: string;
  answers: Record<string, JevAnswer>;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  costUsd: number;
}

function validate(req: JevRequest): void {
  const names = Object.keys(req.questions);
  if (req.state.length > JEV_LIMITS.stateChars) {
    throw new Error(`Jev: state が ${req.state.length} 文字（上限 ${JEV_LIMITS.stateChars}）`);
  }
  if (names.length === 0 || names.length > JEV_LIMITS.questionsPerRequest) {
    throw new Error(`Jev: 質問数 ${names.length}（1〜${JEV_LIMITS.questionsPerRequest}）`);
  }
  for (const name of names) {
    const q = req.questions[name]!;
    if (name.length > JEV_LIMITS.questionNameChars) {
      throw new Error(`Jev: 質問名 "${name}" が長すぎる`);
    }
    if (q.instructions.length > JEV_LIMITS.instructionsChars) {
      throw new Error(
        `Jev: "${name}" の instructions が ${q.instructions.length} 文字（上限 ${JEV_LIMITS.instructionsChars}）`,
      );
    }
    if (q.type !== "noul") {
      const n = Array.isArray(q.criteria) ? q.criteria.length : Object.keys(q.criteria).length;
      if (n < JEV_LIMITS.minOptions || n > JEV_LIMITS.maxOptions) {
        throw new Error(`Jev: "${name}" の選択肢数 ${n}（${JEV_LIMITS.minOptions}〜${JEV_LIMITS.maxOptions}）`);
      }
    }
  }
}

/**
 * 遅延取得のまま維持すること。モジュール読み込み時に検査すると、
 * Jev を使わない既存の実験（DECISION_BACKEND=llm）までキー無しで落ちる。
 */
function apiKey(): string {
  const key = process.env.TYPESAFE_API_KEY;
  if (!key) throw new Error("TYPESAFE_API_KEY が未設定です（.env を確認）");
  return key;
}

export async function evaluateJev(req: JevRequest): Promise<JevResult> {
  validate(req);

  return withRetry(
    async () => {
      const started = Date.now();
      const res = await fetch(`${JEV_BASE_URL}/v1/systemone`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey()}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: JEV_MODEL, ...req }),
      });
      const latencyMs = Date.now() - started;
      const text = await res.text();
      if (!res.ok) {
        // withRetry は 429 をメッセージで拾うので、ステータスを必ず含める
        throw new Error(`Jev API ${res.status}: ${text.slice(0, 500)}`);
      }
      const parsed = JevResponseSchema.parse(JSON.parse(text));
      return {
        model: parsed.model,
        answers: parsed.answers,
        inputTokens: parsed.usage.input_tokens,
        outputTokens: parsed.usage.output_tokens,
        latencyMs,
        costUsd: (parsed.usage.input_tokens / 1_000_000) * JEV_USD_PER_M_INPUT,
      };
    },
    { label: "jev" },
  );
}

/** noul の yes 確率を取り出す。型が合わなければ throw（取り違えを黙って通さない） */
export function noulOf(result: JevResult, name: string): number {
  const a = result.answers[name];
  if (a?.type !== "noul") throw new Error(`Jev: "${name}" は noul の回答ではありません`);
  return a.noul;
}

/** choice の各選択肢の確率を取り出す */
export function choiceOf(
  result: JevResult,
  name: string,
): { choice: string; probabilities: Record<string, number>; confidence: number } {
  const a = result.answers[name];
  if (a?.type !== "choice") throw new Error(`Jev: "${name}" は choice の回答ではありません`);
  return { choice: a.choice, probabilities: a.probabilities, confidence: a.confidence };
}
