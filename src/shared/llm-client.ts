import "dotenv/config";
import OpenAI from "openai";

/**
 * モデル文字列の唯一の定義箇所。
 *
 * 比較実験の妥当性はここが1箇所であることに依存している（CLAUDE.md の不変条件2・3）。
 * ワークフローやスコアラーにモデル名をベタ書きしないこと。
 * ローカルLLM（vLLM / llama.cpp）へ差し替える際もこのファイルだけを触れば済む。
 */

/** 生成・エージェント用。3パターン全てがこれを使う。Mastra model router 形式。 */
export const GENERATION_MODEL = "openai/gpt-5.6-luna";

/**
 * LLM-as-judge 用。既定は生成と同じモデル。
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
