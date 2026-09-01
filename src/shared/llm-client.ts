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

/* ===== 埋め込み ===================================================== */

/**
 * 埋め込みバックエンド。**`LLM_BACKEND` とは独立に切り替える。**
 *
 * 従属させない理由: 実験軸は1本ずつ動かす（ADR 0013 の原則）。
 * 「生成はクラウドのまま・埋め込みだけローカル」が検索側だけを変数にした比較であり、
 * 従属させるとこの組み合わせが作れなくなる。
 */
export type EmbeddingBackend = "openai" | "local";

export const EMBEDDING_BACKEND: EmbeddingBackend =
  process.env.EMBEDDING_BACKEND === "local" ? "local" : "openai";

/** ローカル埋め込みサーバ。docker/compose.yaml の llama-embed-* が公開するポート */
export const LOCAL_EMBEDDING_BASE_URL =
  process.env.LOCAL_EMBEDDING_BASE_URL ?? "http://127.0.0.1:8081/v1";

export interface EmbeddingProfile {
  /** インデックスディレクトリ名・レポートのスラグに入る識別子。[a-z0-9-] のみ */
  slug: string;
  backend: EmbeddingBackend;
  /** API の model フィールドに渡す文字列。llama.cpp は検証しないので実験ログ上の呼び名を兼ねる */
  model: string;
  /** ベクトル次元。**インデックスと検索で必ず一致していなければならない** */
  dimensions: number;
  /**
   * 非対称モデルの prefix。ruri v3 は付け忘れるとエラーにならず精度だけが落ちる
   * （＝典型的な「もっともらしい正常値」）。空文字なら対称モデル。
   */
  queryPrefix: string;
  documentPrefix: string;
  /**
   * サーバ側の pooling。**コードは使わないが docker/compose.yaml の --pooling と
   * 一致していること。** 記録と照合のためだけに持つ（ここと compose がズレたら実験は無効）。
   */
  pooling: "mean" | "cls" | "none";
  /** 1リクエストに詰める本数。ローカルは llama.cpp の -b / -ub に律速される */
  batchSize: number;
  /** 表示・レポート用 */
  label: string;
  /** 補足（ADR から参照する運用メモ） */
  note?: string;
}

/**
 * 埋め込みプロファイル表。**モデル・次元・prefix・pooling の唯一の定義箇所**
 * （CLAUDE.md の不変条件3）。ここ以外にモデル名や次元を書かないこと。
 */
const EMBEDDING_PROFILES = {
  "text-embedding-3-large": {
    slug: "openai-3large",
    backend: "openai",
    model: "text-embedding-3-large",
    // ネイティブ次元。Matryoshka で切り詰め可能だがしない
    // （20記事規模ではストレージも速度も問題にならず、精度を落とす理由がない）
    dimensions: 3072,
    queryPrefix: "",
    documentPrefix: "",
    pooling: "none", // サーバ側 pooling の概念がない
    batchSize: 96, // OpenAI の1リクエスト上限に対して余裕を持たせた値
    label: "openai/text-embedding-3-large (3072d)",
  },
  /**
   * 既定のローカルモデル。日本語特化・768次元・mean pooling・prefix必須。
   * JMTEB 77.2 で日本語 SOTA、実測 P@1 は text-embedding-3-large とほぼ同等。
   *
   * **GGUF は「ModernBERT + SentencePiece 対応のパッチ版 llama.cpp で変換」と明記されている。**
   * ピン留めした b10450 でロードできるかは未検証で、落ちる場合はサーバが起動時に即死する
   * （Node 側からは ECONNREFUSED に見える）。そのときは bge-m3 に倒すこと。
   */
  "ruri-v3-310m": {
    slug: "ruri-v3-310m",
    backend: "local",
    model: "ruri-v3-310m",
    dimensions: 768,
    queryPrefix: "検索クエリ: ",
    documentPrefix: "検索文書: ",
    pooling: "mean",
    // llama.cpp の埋め込みは非causal で、1系列が丸ごと physical batch (-ub) に載る必要がある。
    // チャンクは目標600文字 ≒ 400〜700トークンなので -ub 8192 なら約11本が上限。安全側で 8。
    // 「input is too large to process」が出たら 4 へ、それでも駄目なら compose の -b/-ub を 16384 へ。
    batchSize: 8,
    label: "ruri-v3-310m (768d, mean, prefix必須)",
    note: "Targoyle/ruri-v3-310m-GGUF",
  },
  /** ruri がロードできない場合のフォールバック。多言語・1024次元・cls・prefix不要 */
  "bge-m3": {
    slug: "bge-m3",
    backend: "local",
    model: "bge-m3",
    dimensions: 1024,
    queryPrefix: "",
    documentPrefix: "",
    pooling: "cls",
    batchSize: 8,
    label: "bge-m3 (1024d, cls, prefixなし)",
    note: "gpustack/bge-m3-GGUF",
  },
} as const satisfies Record<string, EmbeddingProfile>;

const LOCAL_EMBEDDING_MODEL = process.env.LOCAL_EMBEDDING_MODEL ?? "ruri-v3-310m";

function resolveEmbeddingProfile(): EmbeddingProfile {
  if (EMBEDDING_BACKEND === "openai") {
    return EMBEDDING_PROFILES["text-embedding-3-large"];
  }
  const profiles: Record<string, EmbeddingProfile> = EMBEDDING_PROFILES;
  const profile = profiles[LOCAL_EMBEDDING_MODEL];
  if (!profile || profile.backend !== "local") {
    const valid = Object.values(profiles)
      .filter((p) => p.backend === "local")
      .map((p) => p.model)
      .join(" / ");
    throw new Error(
      `LOCAL_EMBEDDING_MODEL="${LOCAL_EMBEDDING_MODEL}" は未知の埋め込みプロファイルです。\n` +
        `  有効な値: ${valid}\n` +
        `  プロファイル表は src/shared/llm-client.ts にあります。`,
    );
  }
  return profile;
}

/** 現在有効な埋め込みプロファイル。embed.ts / build-index.ts はこれだけを見る */
export const EMBEDDING_PROFILE: EmbeddingProfile = resolveEmbeddingProfile();

/** プロファイル由来の派生値。既存の import を壊さないため名前を維持している */
export const EMBEDDING_MODEL = EMBEDDING_PROFILE.model;
export const EMBEDDING_DIMENSIONS = EMBEDDING_PROFILE.dimensions;

/** インデックスディレクトリ名・レポートのスラグに使う識別子 */
export const EMBEDDING_SLUG = EMBEDDING_PROFILE.slug;

/**
 * 表示用。`GENERATION_MODEL_LABEL` と同じ役割で、レポートやログにはこちらを使う。
 * プロファイルのオブジェクトを直接テンプレートリテラルへ埋めないこと。
 */
export const EMBEDDING_MODEL_LABEL =
  EMBEDDING_BACKEND === "local"
    ? `${EMBEDDING_PROFILE.label} @ ${LOCAL_EMBEDDING_BASE_URL} (llama.cpp / local)`
    : EMBEDDING_PROFILE.label;

/** 最終的にプロンプトへ入る根拠の件数。3パターン共通（CLAUDE.md の不変条件4）。 */
export const FINAL_CONTEXT_K = 5;

let client: OpenAI | undefined;

/**
 * OpenAI クライアント。埋め込み専用（生成は Mastra model router 経由）。
 *
 * **遅延初期化のまま維持すること。** eager にすると、ローカル埋め込みだけを検証したい場面で
 * OPENAI_API_KEY が要求されてしまう（`npm run verify-embedding` はキー無しで通るのが正しい）。
 */
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

let localEmbedClient: OpenAI | undefined;

/**
 * ローカル埋め込みサーバ用の OpenAI 互換クライアント。
 *
 * `getOpenAI()` と分けているのは、**ローカル埋め込み時に OPENAI_API_KEY を要求しないため**。
 * ただし judge は依然 OpenAI 固定なので、比較実験を回すならキーは結局必要になる。
 *
 * timeout: CPU 実行は1バッチ数秒〜数十秒かかる。SDK 既定の10分でも足りるが、
 * 「固まったのか遅いだけなのか」を切り分けたいので明示する。
 * maxRetries: 再試行されるのは接続エラー・5xx・429 のみ。バッチ過大などの 4xx は
 * 再試行されない ＝ 設定ミスが握り潰されずに即座に落ちる。これは意図した挙動。
 */
export function getLocalEmbeddingClient(): OpenAI {
  localEmbedClient ??= new OpenAI({
    baseURL: LOCAL_EMBEDDING_BASE_URL,
    // llama.cpp は既定で認証しないが、OpenAI互換クライアントが未設定を嫌うので置く
    apiKey: process.env.LOCAL_EMBEDDING_API_KEY ?? "not-needed",
    timeout: 300_000,
    maxRetries: 3,
  });
  return localEmbedClient;
}
