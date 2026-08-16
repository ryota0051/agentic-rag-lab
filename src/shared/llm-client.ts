import "dotenv/config";
import type { MastraModelConfig } from "@mastra/core/llm";
import OpenAI from "openai";

/**
 * モデル文字列の唯一の定義箇所。
 *
 * 比較実験の妥当性はここが1箇所であることに依存している（CLAUDE.md の不変条件2・3）。
 * ワークフローやスコアラーにモデル名をベタ書きしないこと。
 * ローカルLLMへの差し替えもこのファイルだけで完結する（実際そうなった。下記参照）。
 */

/** 生成・エージェントを回すバックエンド。既定はクラウド */
export type LlmBackend = "openai" | "local";

export const LLM_BACKEND: LlmBackend =
  process.env.LLM_BACKEND === "local" ? "local" : "openai";

/** ローカルバックエンドの接続先。docker/compose.yaml が公開しているポートに対応する */
const LOCAL_LLM_BASE_URL = process.env.LOCAL_LLM_BASE_URL ?? "http://127.0.0.1:8080/v1";

/**
 * ローカルモデルの識別子。llama.cpp の OpenAI 互換APIはモデル名を検証しないので、
 * ここは実験ログ上の呼び名を兼ねる。
 */
const LOCAL_LLM_MODEL = process.env.LOCAL_LLM_MODEL ?? "qwen3.8-27b";

/**
 * 生成・エージェント用。3パターン全てがこれを使う。
 *
 * Mastra の `MastraModelConfig` は model router の文字列に加えて、OpenAI 互換
 * エンドポイントの設定オブジェクト（`{ id, url, apiKey }`）も受け付ける。
 * そのため llama.cpp への差し替えがこの分岐だけで済んでいる。
 */
export const GENERATION_MODEL: MastraModelConfig =
  LLM_BACKEND === "local"
    ? {
        id: `local/${LOCAL_LLM_MODEL}`,
        url: LOCAL_LLM_BASE_URL,
        // llama.cpp は既定で認証しないが、OpenAI互換クライアントが未設定を嫌うので置く
        apiKey: process.env.LOCAL_LLM_API_KEY ?? "not-needed",
      }
    : "openai/gpt-5.6-luna";

/**
 * レポート・ログ表示用の文字列。
 *
 * `GENERATION_MODEL` はローカル時にオブジェクトになるため、テンプレートリテラルへ
 * 直接埋めると `[object Object]` になる。表示にはこちらを使うこと。
 */
export const GENERATION_MODEL_LABEL =
  LLM_BACKEND === "local"
    ? `${LOCAL_LLM_MODEL} @ ${LOCAL_LLM_BASE_URL} (llama.cpp / local)`
    : "openai/gpt-5.6-luna";

/**
 * LLM-as-judge 用。既定は生成と同じモデル。
 *
 * **バックエンドを切り替えてもここはクラウドのまま固定する。** golden set はこのモデルで
 * 生成されている。判定側までローカルに差し替えるとものさし自体が変わり、過去の実験レポートと
 * 比較できなくなる（「生成・エージェントのモデルだけを変数にする」という実験設計が崩れる）。
 *
 * 採点役に被験者と同じモデルを使うと自己評価バイアスが入りうる。3パターン全てが同一モデル生成
 * なので相対比較は成立するが、faithfulness / answer-relevancy の絶対値は甘めに出る可能性がある。
 * 3パターンのスコアが頭打ちして差がつかなくなったら "openai/gpt-5.6-terra" に上げる。
 */
export const JUDGE_MODEL = "openai/gpt-5.6-luna";

/** 埋め込み。ingestion と検索で必ず同一でなければならない。 */
export const EMBEDDING_MODEL = "text-embedding-3-large";

/**
 * text-embedding-3-large のネイティブ次元は3072。Matryoshka で切り詰め可能。
 * 3072 のままにしておく（20記事規模ではストレージも速度も問題にならず、精度を落とす理由がない）。
 */
export const EMBEDDING_DIMENSIONS = 3072;

/** 最終的にプロンプトへ入る根拠の件数。3パターン共通（CLAUDE.md の不変条件4）。 */
export const FINAL_CONTEXT_K = 5;

let client: OpenAI | undefined;

/** 埋め込み用の OpenAI クライアント。生成は Mastra model router 経由なのでここでは使わない。 */
export function getOpenAI(): OpenAI {
  if (!client) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OPENAI_API_KEY が未設定です。.env.example をコピーして .env を作成してください。",
      );
    }
    client = new OpenAI({ apiKey });
  }
  return client;
}
