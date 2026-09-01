import {
  EMBEDDING_BACKEND,
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  EMBEDDING_MODEL_LABEL,
  EMBEDDING_PROFILE,
  getLocalEmbeddingClient,
  getOpenAI,
} from "../shared/llm-client.js";

/**
 * 埋め込みの薄いラッパ。
 *
 * ingestion（build-index.ts）と検索（vector-search.ts / hybrid-search.ts）が
 * **必ずこの同じ関数を通る**ようにしてある。モデルや次元数がズレると
 * ベクトル空間が変わり全実験が無効になるため（CLAUDE.md の不変条件3）。
 *
 * ## role を内側に閉じ込めている理由
 *
 * ruri v3 のような非対称モデルはクエリと文書で別の prefix を要求する。
 * **付け忘れてもエラーにはならず精度だけが落ちる**ため、呼び出し側に選ばせず
 * `embedQuery` / `embedBatch` の内側で固定した。role を受け取る公開関数は作らないこと。
 */
type EmbedRole = "query" | "document";

const BATCH_SIZE = EMBEDDING_PROFILE.batchSize;

/**
 * L2正規化。**バックエンドを問わず無条件に通す。**
 *
 * LanceDB の `nearestTo()` の既定距離は L2。単位ベクトル同士なら
 * ‖a−b‖² = 2 − 2·cos(a,b) なので L2 の順位は cos の順位と完全に一致する。
 * 既存パイプラインはこの前提の上に立っている（OpenAI の埋め込みは正規化済み）。
 *
 * llama-server は `--embd-normalize` を持たず、pooling によっては**非正規化ベクトル**を返す。
 * そのまま L2 で引くと、意味的な近さではなく**ベクトルの長さ（≒トークン数）が順位を支配する**。
 * 検索は成功し、5件返り、それらしい回答が生成される——つまり ADR 0013 が潰した
 * 「もっともらしい正常値」の埋め込み版になる。
 *
 * 正規化は冪等（‖v/‖v‖‖ = 1）なので、既に単位ベクトルなら値は変わらない。
 * よってバックエンドで分岐する必要がなく、分岐が無い方が事故が起きない。
 */
function l2normalize(vec: number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const norm = Math.sqrt(sum);
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error(
      `埋め込みベクトルのノルムが ${norm} です（${EMBEDDING_MODEL_LABEL}）。\n` +
        `  サーバの --pooling 設定か入力テキストを確認してください。`,
    );
  }
  // 既に単位ベクトルなら触らない（浮動小数の無駄な揺らぎを避ける）
  if (Math.abs(norm - 1) < 1e-6) return vec;
  return vec.map((v) => v / norm);
}

async function embed(texts: string[], role: EmbedRole): Promise<number[][]> {
  const prefix =
    role === "query" ? EMBEDDING_PROFILE.queryPrefix : EMBEDDING_PROFILE.documentPrefix;

  // prefix は埋め込みに渡すテキストにだけ付ける。
  // LanceDB の `text` 列（BM25 の対象であり fetch が返す本文でもある）には絶対に混ぜないこと
  const input = prefix ? texts.map((t) => prefix + t) : texts;

  const isLocal = EMBEDDING_BACKEND === "local";
  const client = isLocal ? getLocalEmbeddingClient() : getOpenAI();

  const res = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input,
    // llama.cpp が base64 を返さないよう明示する
    encoding_format: "float",
    // `dimensions` は OpenAI 固有。**llama.cpp に送ってはいけない**
    // （未知パラメータで 400 になるか、黙って無視されて「指定したつもり」になる）
    ...(isLocal ? {} : { dimensions: EMBEDDING_DIMENSIONS }),
  });

  if (res.data.length !== input.length) {
    throw new Error(
      `埋め込みの件数が一致しません（要求 ${input.length} 件 / 応答 ${res.data.length} 件）。\n` +
        `  ${EMBEDDING_MODEL_LABEL} / バッチサイズ ${BATCH_SIZE}\n` +
        `  llama.cpp の -b / -ub を超えている可能性があります。` +
        `プロファイル表の batchSize を下げてください。`,
    );
  }

  // API はリクエスト順を保証しないので index で並べ直す
  const vectors = res.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding as number[]);

  // 設定とサーバの実体がズレたら、壊れたインデックスを作る前にここで即死させる
  const first = vectors[0];
  if (!first) throw new Error("埋め込みの取得に失敗しました（応答が空）");
  if (first.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `埋め込みの次元が設定と一致しません。\n` +
        `  サーバの応答: ${first.length} 次元\n` +
        `  設定:         ${EMBEDDING_DIMENSIONS} 次元 (${EMBEDDING_MODEL_LABEL})\n` +
        `  → LOCAL_EMBEDDING_MODEL と llama-embed サービスが同じモデルを指しているか、\n` +
        `    src/shared/llm-client.ts のプロファイル表の dimensions が正しいかを確認してください。\n` +
        `    応答が極端に大きい場合、サーバの --pooling が none でトークン単位の出力に\n` +
        `    なっている可能性があります。`,
    );
  }

  return vectors.map(l2normalize);
}

/** 検索クエリ1本を埋め込む。**クエリ側 prefix はここで固定的に付く** */
export async function embedQuery(text: string): Promise<number[]> {
  const [vec] = await embed([text], "query");
  if (!vec) throw new Error("埋め込みの取得に失敗しました");
  return vec;
}

/** チャンク群をバッチで埋め込む。入力と同じ順序で返す。**文書側 prefix が付く** */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  const started = Date.now();
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const vectors = await embed(batch, "document");
    out.push(...vectors);

    // ローカル（CPU実行）は遅い。残り時間が見えないと「固まった」と誤認して中断してしまう
    const done = Math.min(i + BATCH_SIZE, texts.length);
    const perSec = done / Math.max((Date.now() - started) / 1000, 1e-6);
    const etaSec = Math.round((texts.length - done) / Math.max(perSec, 1e-6));
    console.log(
      `[embed] ${done}/${texts.length} (${perSec.toFixed(1)} 件/s, 残り約 ${etaSec}s)`,
    );
  }
  return out;
}
