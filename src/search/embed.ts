import {
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  getOpenAI,
} from "../shared/llm-client.js";

/**
 * OpenAI 埋め込みの薄いラッパ。
 *
 * ingestion（build-index.ts）と検索（vector-search.ts / hybrid-search.ts）が
 * **必ずこの同じ関数を通る**ようにしてある。モデルや次元数がズレると
 * ベクトル空間が変わり全実験が無効になるため（CLAUDE.md の不変条件3）。
 */

/** OpenAI 埋め込みAPIの1リクエストあたり上限に対して余裕を持たせたバッチサイズ */
const BATCH_SIZE = 96;

async function embed(texts: string[]): Promise<number[][]> {
  const openai = getOpenAI();
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
    dimensions: EMBEDDING_DIMENSIONS,
  });
  // API はリクエスト順を保証しないので index で並べ直す
  return res.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

/** 検索クエリ1本を埋め込む */
export async function embedQuery(text: string): Promise<number[]> {
  const [vec] = await embed([text]);
  if (!vec) throw new Error("埋め込みの取得に失敗しました");
  return vec;
}

/** チャンク群をバッチで埋め込む。入力と同じ順序で返す */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const vectors = await embed(batch);
    out.push(...vectors);
    console.log(`[embed] ${Math.min(i + BATCH_SIZE, texts.length)}/${texts.length}`);
  }
  return out;
}
