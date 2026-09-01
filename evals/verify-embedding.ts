import "dotenv/config";
import { DB_DIR, loadRawArticles, readIndexMeta } from "../src/index/build-index.js";
import { chunkArticle } from "../src/index/chunking.js";
import { embedBatch, embedQuery } from "../src/search/embed.js";
import { isMain } from "../src/shared/is-main.js";
import {
  EMBEDDING_BACKEND,
  EMBEDDING_DIMENSIONS,
  EMBEDDING_MODEL,
  EMBEDDING_MODEL_LABEL,
  EMBEDDING_PROFILE,
  getLocalEmbeddingClient,
  getOpenAI,
} from "../src/shared/llm-client.js";
import type { Chunk } from "../src/shared/types.js";

/**
 * 埋め込みバックエンドが「検索に使える状態か」のスモークテスト。
 * **`npm run build-index` の前に必ず通すゲート。**
 *
 * ## なぜ独立したスクリプトなのか
 *
 * 埋め込みの失敗は生成の失敗と違って**一切エラーを出さない**。
 * pooling を取り違えても、prefix を付け忘れても、正規化を忘れても、
 * 検索は成功し 5件返り、それらしい回答が生成される。recall だけが静かに落ちる。
 * ADR 0013 の `ToolUseStats`（もっともらしい正常値を潰す）と同じ思想を埋め込みに適用したもの
 * （docs/decisions/0014-local-embedding-backend.md）。
 *
 * ## インデックスに依存させていない理由
 *
 * 検証材料は `data/raw` から `loadRawArticles()` + `chunkArticle()` で作る。
 * インデックスに依存させると「壊れた埋め込みでインデックスを作ってから壊れていると分かる」
 * 順序になり、ゲートとして機能しない。
 *
 * 使い方:
 *   $env:EMBEDDING_BACKEND="local"; npm run verify-embedding
 */

interface CheckResult {
  name: string;
  ok: boolean;
  /** 落ちても実験は続けられる項目（情報として出すだけ） */
  soft?: boolean;
  detail: string;
}

/* ===== ベクトルのユーティリティ ===================================== */

function norm(v: number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

/** 正規化されていないベクトルでも正しく出る cos 類似度 */
function cosine(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += (a[i] ?? 0) * (b[i] ?? 0);
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot / d;
}

/**
 * **`embed.ts` を経由せず**サーバを直接叩く。
 *
 * 正規化前の生ベクトルが必要な検査（ノルム測定）と、prefix を任意に差し替える検査
 * （prefix 効果の測定）で使う。`embed.ts` の出力でノルムを測ると、
 * 自分がかけた正規化のせいで必ず 1 になり**同義反復**になってしまう。
 */
async function rawEmbed(texts: string[], prefix = ""): Promise<number[][]> {
  const isLocal = EMBEDDING_BACKEND === "local";
  const client = isLocal ? getLocalEmbeddingClient() : getOpenAI();
  const res = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: prefix ? texts.map((t) => prefix + t) : texts,
    encoding_format: "float",
    ...(isLocal ? {} : { dimensions: EMBEDDING_DIMENSIONS }),
  });
  return res.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding as number[]);
}

/** プロファイルの batchSize を尊重しつつ rawEmbed を回す */
async function rawEmbedAll(texts: string[], prefix = ""): Promise<number[][]> {
  const size = EMBEDDING_PROFILE.batchSize;
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += size) {
    out.push(...(await rawEmbed(texts.slice(i, i + size), prefix)));
  }
  return out;
}

/* ===== 材料 ========================================================= */

/** 意味的に無関係な文。等方性（ベクトル崩壊）の検査に使う */
const UNRELATED_SENTENCES = [
  "味噌汁の出汁は昆布と鰹節でとるのが基本です。",
  "台風が接近しているため、明日の午後は電車が止まる見込みです。",
  "バイオリンの弓は馬の尾の毛で張られています。",
  "確定申告の期限は毎年3月15日です。",
  "南極大陸には常設の氷床観測基地が複数あります。",
  "この靴は26.5センチで、幅が少しきついです。",
  "江戸時代の飛脚は東海道を約3日で走破したと言われます。",
  "深夜のコンビニでおでんの匂いがしていた。",
  "光合成は葉緑体のチラコイド膜で始まります。",
  "проверка кодировки при чтении файла",
  "The quarterly earnings call is scheduled for next Tuesday.",
  "彼女はピアノの発表会で緊張して手が震えた。",
  "オリーブオイルは低温で白く濁ることがあります。",
  "住宅ローンの金利は固定と変動で総返済額が変わります。",
  "サッカーのオフサイドはボールが蹴られた瞬間で判定されます。",
  "この地域の方言では語尾が「〜じゃけん」になります。",
  "顕微鏡の対物レンズは倍率ごとに交換します。",
  "冷蔵庫の霜取りヒーターが故障していた。",
  "俳句は五・七・五の十七音で構成されます。",
  "登山では標高が100m上がるごとに気温が約0.6度下がります。",
];

interface Corpus {
  /** 文書プール（ベクトル計算の対象） */
  pool: Chunk[];
  /** 正規化済みの文書ベクトル */
  poolVectors: number[][];
  /** 1件あたりの埋め込み所要秒 */
  secPerItem: number;
  /** コーパス全体のチャンク数（ingest 所要時間の見積もりに使う） */
  totalChunks: number;
}

/** 文書プールのサイズ。CPU 実行でも数分で終わる規模に抑える */
const POOL_SIZE = 120;
/** retrieval サニティで投げるクエリ数 */
const PROBE_COUNT = 30;

async function buildCorpus(): Promise<Corpus> {
  const articles = await loadRawArticles();
  const chunks = articles.flatMap(chunkArticle);

  // heading_path をクエリに使うので、プール内で一意なものだけを対象にする
  // （同じ heading_path が複数あると「正解」が定義できない）
  const byHeading = new Map<string, Chunk[]>();
  for (const c of chunks) {
    const list = byHeading.get(c.heading_path);
    if (list) list.push(c);
    else byHeading.set(c.heading_path, [c]);
  }
  const unique = [...byHeading.values()].filter((v) => v.length === 1).map((v) => v[0]!);

  // 決定的にサンプリングする（実行ごとに結果が揺れると判断材料にならない）
  const step = Math.max(1, Math.floor(unique.length / POOL_SIZE));
  const pool = unique.filter((_, i) => i % step === 0).slice(0, POOL_SIZE);

  const started = Date.now();
  const raw = await rawEmbedAll(
    pool.map((c) => c.text),
    EMBEDDING_PROFILE.documentPrefix,
  );
  const secPerItem = (Date.now() - started) / 1000 / Math.max(pool.length, 1);

  return {
    pool,
    poolVectors: raw.map((v) => v.map((x) => x / (norm(v) || 1))),
    secPerItem,
    totalChunks: chunks.length,
  };
}

/**
 * クエリベクトル群を文書プールに当てて、正解チャンクが top-N に入る率を返す。
 * 「retrieval として使い物になるか」を直接測る唯一の手段。
 */
function topNRate(
  queryVectors: number[][],
  answerIndexes: number[],
  poolVectors: number[][],
  n: number,
): number {
  let hit = 0;
  for (let q = 0; q < queryVectors.length; q++) {
    const qv = queryVectors[q]!;
    const scored = poolVectors
      .map((pv, i) => ({ i, score: cosine(qv, pv) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, n);
    if (scored.some((s) => s.i === answerIndexes[q])) hit++;
  }
  return hit / Math.max(queryVectors.length, 1);
}

/* ===== 検査（プール不要・安いものから） ============================= */

/** 1. 接続と次元一致 */
async function checkConnectionAndDimensions(): Promise<CheckResult> {
  const vec = await embedQuery("LanceDB の日本語全文検索でトークナイザを指定する方法");
  const ok = vec.length === EMBEDDING_DIMENSIONS;
  return {
    name: "接続と次元一致",
    ok,
    detail: ok
      ? `${vec.length} 次元を取得`
      : `応答 ${vec.length} 次元 / 設定 ${EMBEDDING_DIMENSIONS} 次元`,
  };
}

/**
 * 2. 生ノルムの実測と、正規化が効いていることの確認。
 *
 * llama-server は `--embd-normalize` を持たず、pooling によっては非正規化ベクトルを返す。
 * そのまま L2 で引くと**ベクトル長（≒トークン数）が順位を支配する**——最大の静かな失敗。
 * `embed.ts` が正規化しているかを、生ベクトルと突き合わせて確認する。
 */
async function checkNormalization(): Promise<CheckResult> {
  const text = "LanceDB のハイブリッド検索は BM25 とベクトル検索を RRF で融合する。";
  const [raw] = await rawEmbed([text]);
  if (!raw) return { name: "正規化", ok: false, detail: "生ベクトルを取得できませんでした" };

  const rawNorm = norm(raw);
  const normalized = await embedQuery(text);
  const outNorm = norm(normalized);
  const ok = Math.abs(outNorm - 1) < 1e-3;

  return {
    name: "正規化",
    ok,
    detail:
      `サーバの生ノルム=${rawNorm.toFixed(4)} → embed.ts 出力ノルム=${outNorm.toFixed(6)}` +
      (Math.abs(rawNorm - 1) > 1e-3
        ? "（サーバは非正規化。embed.ts の L2正規化が必須の構成です）"
        : "（サーバ側で既に正規化済み）") +
      (ok ? "" : " ← **出力が単位ベクトルになっていません**"),
  };
}

/** 3. 決定性。スロット再利用やコンテキスト汚染で揺れるサーバを弾く */
async function checkDeterminism(): Promise<CheckResult> {
  const text = "エージェント的RAGでは search と fetch を分離して予算を管理する。";
  const a = await embedQuery(text);
  const b = await embedQuery(text);
  const sim = cosine(a, b);
  return {
    name: "決定性",
    ok: sim > 0.9999,
    detail: `同一文2回の cos=${sim.toFixed(6)}`,
  };
}

/**
 * 4. 等方性（ベクトル崩壊の検出）。
 *
 * **pooling が壊れたモデルは「全部ほぼ同じベクトル」を返す。**
 * この状態でも検索は5件返り、回答も生成され、recall だけが静かに落ちる。
 * 本スクリプトが最も検出したい失敗モード。
 */
async function checkIsotropy(): Promise<CheckResult> {
  const vectors = await rawEmbedAll(UNRELATED_SENTENCES, EMBEDDING_PROFILE.documentPrefix);
  let sum = 0;
  let pairs = 0;
  let max = -1;
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      const c = cosine(vectors[i]!, vectors[j]!);
      sum += c;
      max = Math.max(max, c);
      pairs++;
    }
  }
  const mean = sum / Math.max(pairs, 1);
  return {
    name: "等方性（ベクトル崩壊）",
    ok: mean < 0.95,
    detail:
      `無関係${vectors.length}文の総当たり cos: 平均=${mean.toFixed(4)} 最大=${max.toFixed(4)}` +
      (mean < 0.95 ? "" : " ← **ベクトルが崩壊しています（pooling を疑ってください）**"),
  };
}

/**
 * 5. prefix 適用の実在確認。
 *
 * プロファイル表と `embed.ts` の実装がズレていないかを見る。
 * prefix ありなら query 側と document 側は別ベクトルになるはずで、
 * prefix なしなら完全一致するはず。
 */
async function checkPrefixApplied(): Promise<CheckResult> {
  const text = "LanceDB のFTSインデックスは baseTokenizer を指定して作成する。";
  const asQuery = await embedQuery(text);
  const [asDocument] = await embedBatch([text]);
  const sim = cosine(asQuery, asDocument!);

  const hasPrefix =
    EMBEDDING_PROFILE.queryPrefix !== "" || EMBEDDING_PROFILE.documentPrefix !== "";
  const ok = hasPrefix ? sim < 0.999 : sim > 0.9999;

  return {
    name: "prefix 適用",
    ok,
    detail: hasPrefix
      ? `prefix あり（"${EMBEDDING_PROFILE.queryPrefix}" / "${EMBEDDING_PROFILE.documentPrefix}"）` +
        ` → query と document の cos=${sim.toFixed(4)}` +
        (ok ? "（別ベクトルになっている）" : " ← **prefix が適用されていません**")
      : `prefix なしプロファイル → cos=${sim.toFixed(6)}` +
        (ok ? "（一致）" : " ← **付くはずのない差が出ています**"),
  };
}

/**
 * 6. `embedBatch` の順序保証。
 *
 * **必ずバッチ境界を跨ぐ本数**で試す。ループの push 順や
 * `sort((a, b) => a.index - b.index)` の欠落は、ここでしか表に出ない。
 */
async function checkBatchOrder(): Promise<CheckResult> {
  const count = EMBEDDING_PROFILE.batchSize * 2 + 3;
  const texts = Array.from(
    { length: count },
    (_, i) => `${i}番目の検証用テキスト。内容はそれぞれ異なる話題を扱います。番号は${i}です。`,
  );

  // 決定的なシャッフル（実行ごとに結果が揺れると再現性がない）
  const order = texts.map((_, i) => i).sort((a, b) => ((a * 7919) % 101) - ((b * 7919) % 101));
  const shuffled = order.map((i) => texts[i]!);

  const batched = await embedBatch(shuffled);
  const individually = await Promise.all(order.map((i) => embedQuery(texts[i]!)));

  // prefix 非対称モデルでは query と document のベクトルが違うので、
  // 「i 番目の出力が i 番目の入力に対応しているか」は
  // **document 埋め込み同士の総当たりで最も近いものが自分自身か**で見る
  let wrong = 0;
  for (let i = 0; i < batched.length; i++) {
    let best = -1;
    let bestScore = -Infinity;
    for (let j = 0; j < batched.length; j++) {
      const s = cosine(individually[i]!, batched[j]!);
      if (s > bestScore) {
        bestScore = s;
        best = j;
      }
    }
    if (best !== i) wrong++;
  }

  return {
    name: "embedBatch の順序保証",
    ok: wrong === 0,
    detail:
      `${count} 本（バッチサイズ ${EMBEDDING_PROFILE.batchSize} を跨ぐ）をシャッフルして検証: ` +
      (wrong === 0 ? "全件が入力順に対応" : `**${wrong} 件が入れ替わっています**`),
  };
}

/** 7. 境界入力。ゼロベクトルや NaN が返らないこと */
async function checkBoundaryInputs(): Promise<CheckResult> {
  const cases: { label: string; text: string }[] = [
    { label: "空白のみ", text: "   " },
    { label: "1文字", text: "あ" },
    { label: "長文(n_ctx超え狙い)", text: "検索".repeat(20000) },
  ];
  const notes: string[] = [];
  let ok = true;

  for (const c of cases) {
    try {
      const v = await embedQuery(c.text);
      const n = norm(v);
      const finite = v.every((x) => Number.isFinite(x));
      if (!finite || Math.abs(n - 1) > 1e-3) {
        ok = false;
        notes.push(`${c.label}: **不正なベクトル**(finite=${finite}, norm=${n.toFixed(4)})`);
      } else {
        notes.push(`${c.label}: OK`);
      }
    } catch (err) {
      // 落ちるのは許容する（黙って壊れたベクトルを返すより遥かによい）
      notes.push(`${c.label}: 例外（許容）${(err instanceof Error ? err.message : "").slice(0, 60)}`);
    }
  }

  return { name: "境界入力", ok, soft: true, detail: notes.join(" / ") };
}

/* ===== 検査（文書プールが必要なもの） =============================== */

/** 8. 日本語の意味的サニティ。実コーパスの語彙で、関連 > 無関係 が成り立つか */
async function checkSemanticSanity(corpus: Corpus): Promise<CheckResult> {
  const probes = corpus.pool.slice(0, 10);
  const queries = await rawEmbedAll(
    probes.map((c) => c.heading_path),
    EMBEDDING_PROFILE.queryPrefix,
  );

  let relatedSum = 0;
  let unrelatedSum = 0;
  for (let i = 0; i < probes.length; i++) {
    const self = corpus.pool.indexOf(probes[i]!);
    relatedSum += cosine(queries[i]!, corpus.poolVectors[self]!);
    // 別記事のチャンクを無関係サンプルとして使う
    const other = corpus.pool.findIndex((c) => c.article_id !== probes[i]!.article_id);
    unrelatedSum += cosine(queries[i]!, corpus.poolVectors[other]!);
  }
  const related = relatedSum / probes.length;
  const unrelated = unrelatedSum / probes.length;
  const gap = related - unrelated;

  return {
    name: "日本語の意味的サニティ",
    ok: gap > 0.05,
    detail: `関連 cos=${related.toFixed(4)} / 無関係 cos=${unrelated.toFixed(4)} / 差=${gap.toFixed(4)}`,
  };
}

/**
 * 9. 実コーパス上の retrieval サニティ。
 *
 * 「この埋め込みが検索として使い物になるか」を直接測る唯一の項目。
 * 見出しパスをクエリに仕立てて、対応するチャンクが top-3 に入る率を見る。
 */
async function checkRetrievalSanity(corpus: Corpus): Promise<CheckResult> {
  const probeIdx = corpus.pool
    .map((_, i) => i)
    .filter((_, i) => i % Math.max(1, Math.floor(corpus.pool.length / PROBE_COUNT)) === 0)
    .slice(0, PROBE_COUNT);

  const queries = await rawEmbedAll(
    probeIdx.map((i) => corpus.pool[i]!.heading_path),
    EMBEDDING_PROFILE.queryPrefix,
  );
  const rate = topNRate(queries, probeIdx, corpus.poolVectors, 3);
  const chance = 3 / corpus.pool.length;

  return {
    name: "実コーパス retrieval サニティ",
    ok: rate >= 0.6,
    detail:
      `${probeIdx.length}問 / 文書プール${corpus.pool.length}件で top-3 命中率 ` +
      `${(rate * 100).toFixed(0)}%（ランダム相当 ${(chance * 100).toFixed(1)}%）` +
      (rate >= 0.6 ? "" : " ← **検索として機能していません**"),
  };
}

/**
 * 10. prefix の効果。
 *
 * prefix を逆に付けて項目9を再実行し、命中率が落ちるかを見る。
 * 落ちなければ prefix が実質効いていないという情報になる（ruri で無効なら bge-m3 でよい、
 * という判断材料）。落ちること自体は失敗ではないので soft。
 */
async function checkPrefixEffect(corpus: Corpus): Promise<CheckResult> {
  if (EMBEDDING_PROFILE.queryPrefix === EMBEDDING_PROFILE.documentPrefix) {
    return {
      name: "prefix の効果",
      ok: true,
      soft: true,
      detail: "対称モデル（prefix なし）のためスキップ",
    };
  }

  const probeIdx = corpus.pool
    .map((_, i) => i)
    .filter((_, i) => i % Math.max(1, Math.floor(corpus.pool.length / PROBE_COUNT)) === 0)
    .slice(0, PROBE_COUNT);

  const correct = await rawEmbedAll(
    probeIdx.map((i) => corpus.pool[i]!.heading_path),
    EMBEDDING_PROFILE.queryPrefix,
  );
  // 文書側の prefix をクエリに付ける ＝ 誤用したときの再現
  const swapped = await rawEmbedAll(
    probeIdx.map((i) => corpus.pool[i]!.heading_path),
    EMBEDDING_PROFILE.documentPrefix,
  );

  const rateCorrect = topNRate(correct, probeIdx, corpus.poolVectors, 3);
  const rateSwapped = topNRate(swapped, probeIdx, corpus.poolVectors, 3);

  return {
    name: "prefix の効果",
    ok: true,
    soft: true,
    detail:
      `正しい prefix: ${(rateCorrect * 100).toFixed(0)}% / ` +
      `文書側 prefix を誤用: ${(rateSwapped * 100).toFixed(0)}%` +
      (rateCorrect > rateSwapped
        ? "（prefix が効いています）"
        : "（差がありません。prefix が実質無効の可能性）"),
  };
}

/** 11. スループット。build-index が5分で終わるのか3時間かかるのかを先に知る */
function checkThroughput(corpus: Corpus): CheckResult {
  const perSec = 1 / Math.max(corpus.secPerItem, 1e-9);
  const etaMin = (corpus.totalChunks * corpus.secPerItem) / 60;
  return {
    name: "スループット",
    ok: true,
    soft: true,
    detail:
      `${perSec.toFixed(1)} 件/s → 全 ${corpus.totalChunks} チャンクの build-index は約 ` +
      `${etaMin < 1 ? "1分未満" : `${etaMin.toFixed(1)}分`}`,
  };
}

/** 12. 既存インデックスとの整合。ここでは落とさない（次にやるのが build-index なので） */
async function checkIndexConsistency(): Promise<CheckResult> {
  const meta = await readIndexMeta();
  if (!meta) {
    return {
      name: "インデックス整合",
      ok: true,
      soft: true,
      detail: `${DB_DIR} にインデックスがありません（build-index が必要）`,
    };
  }
  const same =
    meta.embeddingModel === EMBEDDING_PROFILE.model &&
    meta.embeddingDimensions === EMBEDDING_PROFILE.dimensions &&
    meta.embeddingPooling === EMBEDDING_PROFILE.pooling &&
    meta.queryPrefix === EMBEDDING_PROFILE.queryPrefix &&
    meta.documentPrefix === EMBEDDING_PROFILE.documentPrefix;
  return {
    name: "インデックス整合",
    ok: true,
    soft: true,
    detail: same
      ? `既存インデックスは同一設定（${meta.chunks} チャンク / ${meta.builtAt}）`
      : `既存インデックスは別設定（${meta.embeddingModel} / ${meta.embeddingDimensions}次元）。build-index が必要`,
  };
}

/* ===== 実行 ========================================================= */

function report(r: CheckResult): void {
  const mark = r.ok ? "✅" : r.soft ? "⚠️ " : "❌";
  console.log(`${mark} ${r.name}\n   ${r.detail}\n`);
}

async function runStage(
  checks: { label: string; run: () => Promise<CheckResult> | CheckResult }[],
): Promise<CheckResult[]> {
  const results: CheckResult[] = [];
  for (const check of checks) {
    try {
      results.push(await check.run());
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({ name: check.label, ok: false, detail: `例外: ${message.slice(0, 400)}` });
    }
    report(results[results.length - 1]!);
  }
  return results;
}

/** 失敗の症状から原因の当たりを付けて助言する */
function advise(results: CheckResult[]): void {
  const failed = results.filter((r) => !r.ok && !r.soft);
  const names = new Set(failed.map((f) => f.name));
  const details = failed.map((f) => f.detail).join(" ");

  if (
    /ECONNREFUSED|APIConnectionError|fetch failed|Connection error/i.test(details) ||
    (names.has("接続と次元一致") && /例外/.test(details))
  ) {
    console.log(
      "\n[接続] 埋め込みサーバに接続できません。`npm run serve:embed:logs` を確認してください。\n" +
        "  `unknown model architecture` / `unknown pre-tokenizer type` / `missing tokenizer` が\n" +
        "  出ている場合、この GGUF はパッチ版 llama.cpp で変換されたもので、ピン留めした\n" +
        "  b10450 ではロードできません。フォールバックに切り替えてください:\n" +
        "    npm run serve:embed:down\n" +
        "    npm run serve:embed:bge\n" +
        '    $env:LOCAL_EMBEDDING_MODEL="bge-m3"; npm run verify-embedding',
    );
  }
  if (/501|does not support embeddings|not supported/i.test(details)) {
    console.log(
      "\n[エンドポイント] docker/compose.yaml の command に `--embeddings` があるか確認してください。",
    );
  }
  if (names.has("等方性（ベクトル崩壊）") || names.has("接続と次元一致")) {
    console.log(
      "\n[pooling] compose の `--pooling` が src/shared/llm-client.ts のプロファイル表と\n" +
        `  一致しているか確認してください（このプロファイルの期待値: ${EMBEDDING_PROFILE.pooling}）。\n` +
        "  次元が極端に大きい場合は --pooling none でトークン単位の出力になっています。",
    );
  }
  if (names.has("正規化")) {
    console.log(
      "\n[正規化] embed.ts の L2正規化が効いていません。非正規化ベクトルを LanceDB の L2 距離で\n" +
        "  引くと、意味的な近さではなくベクトル長（≒トークン数）が順位を支配します。\n" +
        "  検索は成功して結果も返るため、このまま実験を回すと嘘の数値が出ます。",
    );
  }
  if (names.has("prefix 適用")) {
    console.log(
      "\n[prefix] プロファイル表の queryPrefix / documentPrefix と embed.ts の実装がズレています。\n" +
        "  prefix の付け忘れはエラーにならず精度だけが落ちます。",
    );
  }
  if (names.has("実コーパス retrieval サニティ") || names.has("日本語の意味的サニティ")) {
    console.log(
      "\n[retrieval] 埋め込みが検索として機能していません。**この状態で build-index を実行しないこと。**\n" +
        "  pooling / prefix / 正規化のいずれかが誤っている可能性が高いので、上の項目を先に見てください。",
    );
  }
  if (names.has("embedBatch の順序保証")) {
    console.log(
      "\n[順序] チャンクとベクトルの対応が崩れています。インデックス全体が無意味になるので、\n" +
        "  embed.ts の sort((a, b) => a.index - b.index) と push 順を確認してください。",
    );
  }
}

async function main() {
  console.log(`埋め込みバックエンド: ${EMBEDDING_BACKEND}`);
  console.log(`モデル:               ${EMBEDDING_MODEL_LABEL}`);
  console.log(`次元 / pooling:       ${EMBEDDING_DIMENSIONS} / ${EMBEDDING_PROFILE.pooling}`);
  console.log(
    `prefix:               query="${EMBEDDING_PROFILE.queryPrefix}" document="${EMBEDDING_PROFILE.documentPrefix}"`,
  );
  console.log(`バッチサイズ:         ${EMBEDDING_PROFILE.batchSize}`);
  console.log(`インデックス:         ${DB_DIR}\n`);

  if (EMBEDDING_BACKEND !== "local") {
    console.warn(
      "⚠️  EMBEDDING_BACKEND が local ではありません。ローカル埋め込みを検証するなら\n" +
        '    $env:EMBEDDING_BACKEND="local" を設定してから実行してください。\n',
    );
  }

  // 安い検査から順に。ここで落ちたら文書プールの構築（高コスト）へ進まない
  const results = await runStage([
    { label: "接続と次元一致", run: checkConnectionAndDimensions },
    { label: "正規化", run: checkNormalization },
    { label: "決定性", run: checkDeterminism },
    { label: "等方性（ベクトル崩壊）", run: checkIsotropy },
    { label: "prefix 適用", run: checkPrefixApplied },
    { label: "embedBatch の順序保証", run: checkBatchOrder },
    { label: "境界入力", run: checkBoundaryInputs },
  ]);

  if (results.every((r) => r.ok || r.soft)) {
    console.log(`--- 実コーパスでの検証（文書プール ${POOL_SIZE} 件を埋め込みます）---\n`);
    const corpus = await buildCorpus();
    results.push(
      ...(await runStage([
        { label: "日本語の意味的サニティ", run: () => checkSemanticSanity(corpus) },
        { label: "実コーパス retrieval サニティ", run: () => checkRetrievalSanity(corpus) },
        { label: "prefix の効果", run: () => checkPrefixEffect(corpus) },
        { label: "スループット", run: () => checkThroughput(corpus) },
        { label: "インデックス整合", run: checkIndexConsistency },
      ])),
    );
  } else {
    console.log("--- 基本検査が失敗したため、実コーパスでの検証はスキップします ---\n");
  }

  const failed = results.filter((r) => !r.ok && !r.soft);
  console.log("--- 判定 ---");
  if (failed.length === 0) {
    console.log("必須項目はすべて成立。`npm run build-index` に進めます。");
    return;
  }
  console.log(`${failed.map((f) => f.name).join(" / ")} が失敗しています。`);
  console.log("**この状態で build-index を実行してはいけません。**");
  advise(results);
  process.exitCode = 1;
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
