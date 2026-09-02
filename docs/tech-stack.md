# 技術スタックと選定理由

| レイヤー | 選定 | 理由 |
|---|---|---|
| エージェント／ワークフロー | **Mastra**（`@mastra/core`） | スキル選択と search/fetch ループをワークフローとして表現でき、トレースが自動で残る。`decisions/0002-mastra-typescript.md` |
| 評価 | **自前の決定的スコアラー**（`evals/scorers/`） | retrieval-recall（chunk_id の集合比較）と skill-selection-accuracy（ラベル一致）。**LLM を使わないので実行ごとにブレず、安価**。LLM-as-judge 指標は採用していない（下記） |
| LLM（生成・エージェント・golden set 生成） | **OpenAI `gpt-5.6-luna`** | Mastra model router 経由（`"openai/gpt-5.6-luna"`）。5.6系の最速・最安ティアで、試行回数を稼ぐ実験用途に合う。入力 $0.20 / 出力 $1.20 per 1M |
| LLM（ローカルバックエンド） | **Qwen3.8-27B**（unsloth GGUF, UD-Q4_K_XL） | `LLM_BACKEND=local` のときの生成・エージェント用。llama.cpp 公式Dockerイメージで OpenAI 互換サーバとしてホストする。`decisions/0013-local-llm-backend.md` |
| 埋め込み | **OpenAI `text-embedding-3-large`** | 既定。3072次元。LLM と同じ `OPENAI_API_KEY` で済み、必要なキーが1つになる |
| 埋め込み（ローカルバックエンド） | **`ruri-v3-310m`**（cl-nagoya / GGUF Q8_0） | `EMBEDDING_BACKEND=local` のとき。日本語特化・768次元・JMTEB 77.2（日本語SOTA）。337MB で llama.cpp の CPU 実行が現実的。フォールバックに `bge-m3`。`decisions/0014-local-embedding-backend.md` |
| ベクトルDB／全文検索 | **LanceDB**（`@lancedb/lancedb`） | 埋め込み型でサーバー不要。全文検索（Tantivy = BM25相当）も標準搭載しており、ベクトル＋BM25 のハイブリッドを1本で完結できる |
| データ取り込み | **Qiita API v2** | Markdown 本文とメタデータを直接取得でき、HTML由来のノイズをそもそも回避できる。`decisions/0004-cleaning-at-ingestion.md` |
| ストレージ（トレース） | `@mastra/libsql` | ローカルファイル。実行ログを `traces/` に永続化 |

## モデル文字列は1箇所に閉じる

`src/shared/llm-client.ts` が唯一の定義箇所。

```ts
// 3パターン共通。LLM_BACKEND=local ならローカルサーバの設定オブジェクトになる
export const GENERATION_MODEL: MastraModelConfig = /* openai/gpt-5.6-luna | { id, url, apiKey } */;
export const GENERATION_MODEL_LABEL = /* 表示用の文字列 */;
// golden set を作るモデル。**採点には使われない**（下記「LLM-as-judge を使っていない」）
export const JUDGE_MODEL      = "openai/gpt-5.6-luna";  // バックエンドに関わらず固定

// 埋め込みはプロファイル表が唯一の定義。モデル名・次元・prefix・pooling・batchSize を束ねる
const EMBEDDING_PROFILES = { "text-embedding-3-large": {...}, "ruri-v3-310m": {...}, "bge-m3": {...} };
export const EMBEDDING_PROFILE    = /* EMBEDDING_BACKEND と LOCAL_EMBEDDING_MODEL から解決 */;
export const EMBEDDING_MODEL      = EMBEDDING_PROFILE.model;       // 派生値
export const EMBEDDING_DIMENSIONS = EMBEDDING_PROFILE.dimensions;  // 派生値
export const EMBEDDING_SLUG       = EMBEDDING_PROFILE.slug;        // インデックスのディレクトリ名になる
export const FINAL_CONTEXT_K = 5;
```

ワークフローやスコアラーにモデル名をベタ書きしないこと。
ローカルLLMへの差し替えも実際にここだけで完結した（ワークフローは無変更）。

`GENERATION_MODEL` はローカル時にオブジェクトになるため、**レポートやログに埋めるときは
`GENERATION_MODEL_LABEL` を使う**こと。直接テンプレートリテラルへ入れると `[object Object]` になる。

## 生成バックエンドの切り替え

`LLM_BACKEND` 環境変数で、生成・エージェントのモデルだけを差し替えられる。

| | `openai`（既定） | `local` |
|---|---|---|
| 生成・エージェント | `gpt-5.6-luna` | Qwen3.8-27B（llama.cpp / Docker / GPU / port 8080） |
| 埋め込み | **影響を受けない**（`EMBEDDING_BACKEND` が決める） | **同左** |
| golden set 生成モデル | `gpt-5.6-luna` | **同じ（変更しない）** |
| LanceDBインデックス | **影響を受けない** | **同左** |

検索側を完全に固定することで、変数がエージェントを駆動するLLM1つだけになる。
`LLM_BACKEND=local` でも `EMBEDDING_BACKEND` が既定なら `OPENAI_API_KEY` は必要（埋め込みが使う）。
**両方 local なら比較実験はキー無しで回る**（採点に LLM を使っていないため。下記）。

```bash
npm run serve:local        # llama.cpp サーバを起動（docker/compose.yaml）
npm run verify-local-llm   # 生成／ツール呼び出し／構造化出力を個別に確認（本番前に必須）
```

`JUDGE_MODEL` をローカルに倒してはいけない理由と、ランタイムの選定理由は
`decisions/0013-local-llm-backend.md` を参照。

## 埋め込みバックエンドの切り替え

`EMBEDDING_BACKEND` は **`LLM_BACKEND` とは独立の軸**。両方を同時に動かすと
どちらの寄与か分離できなくなるので、必ず1本ずつ動かす。

| | `openai`（既定） | `local` |
|---|---|---|
| 埋め込み | `text-embedding-3-large`（3072d） | `ruri-v3-310m`（768d）/ `bge-m3`（1024d） |
| 実行場所 | クラウドAPI | llama.cpp **CPU**（GPU非占有 / port 8081） |
| LanceDBインデックス | `data/index-openai-3large/` | `data/index-ruri-v3-310m/` 等 |
| golden set 生成モデル | **OpenAI 固定** | **OpenAI 固定** |

インデックスを分けたうえで `index-meta.json`（指紋）を照合するのは、
**次元が同じで意味が違うベクトル空間は検索が成功してしまい、壊れていることに
気づけない**ため。詳細は `decisions/0014-local-embedding-backend.md`。

```bash
npm run serve:embed        # 埋め込みサーバを起動（生成サーバと同時起動できる）
npm run verify-embedding   # 次元・正規化・等方性・prefix・順序保証を確認（build-index の前に必須）
npm run build-index        # 埋め込みを変えたら必ず作り直す
```

## LLM-as-judge を使っていない

`JUDGE_MODEL` という定数名に反して、**比較実験の採点に LLM は一切使っていない。**
この定数を実際に呼び出しているのは `evals/generate-golden-set.ts` /
`generate-multihop-set.ts` の2箇所、つまり**問題作成時のみ**。

`npm run eval` の採点は `evals/scorers/` の2つだけ:

| スコアラー | 判定方法 |
|---|---|
| retrieval-recall（**主指標**） | `golden_chunk_ids` と `retrieved_chunk_ids` の集合比較 |
| skill-selection-accuracy | 期待ラベルとの一致 |

どちらも決定的な関数で、**実行ごとにブレず、追加のAPI課金も発生しない**。
faithfulness / answer-relevancy を採らないのは、生成品質は測れても
「検索が正解を引けたか」を直接には表さないため（`evals/scorers/retrieval-recall.ts` のコメント）。

したがって `JUDGE_MODEL` を固定すべき理由は「採点に使うから」ではなく
「**問題を作ったモデルだから**」。作り直すとものさし自体が変わる。
`decisions/0014-local-embedding-backend.md` に経緯を記録している。

なお `@mastra/evals` は package.json の依存に残っているが、現状どこからも import していない。

## 使わなかったもの

| 候補 | 不採用の理由 |
|---|---|
| `@mastra/lance` | ベクトル検索しか叩けず、BM25 と RRF 融合を制御できない。ハイブリッド検索がパターン2の本体なので、抽象を挟まず `@lancedb/lancedb` を直接使う |
| vLLM（ローカルホスト） | unsloth が案内する経路は NVFP4 量子化が前提で、これは Blackwell 世代の機能。検証機の RTX 4090（Ada）では動かない |
| Ollama（ローカルホスト） | 導入は最も楽だが、サンプラーと `reasoning_effort` の指定が Modelfile 経由の間接的なものになる。思考モードを実験条件として固定したいので、フラグで直接書ける llama.cpp を採った |
| `create-mastra` | `src/mastra/` 配下に固める構成を強制され、本プロジェクトの関心ごと分割と衝突する |
| Cross-Encoder リランカー（Voyage 等） | 比較実験の変数が増える。`decisions/0005-no-cross-encoder-rerank.md` |
| deepeval / MLflow | Python プロセスの別立てが必要になる。Mastra Scorers で足りる |
| grep 系ツール | BM25 と役割が重複し、エージェントのツール選択の余地を無駄に広げる。`decisions/0006-grep-prototyping-only.md` |
| GPU 常駐の埋め込みサーバ | 24GB に 18GB の生成モデルが載っている。数百MBでも VRAM 競合の可能性を作ると `0013` の前提（生成の再現条件を固定する）が崩れる。CPU で 5.1 件/s 出るので困らない |
| compose の `${VAR}` 展開で pooling を切り替える案 | このファイルが「そのとき何で回したか」の記録でなくなる。モデルごとにサービスを分け、全フラグを literal に書く |
| `Qwen3-Embedding-0.6B` | 多言語 MTEB では強いが、**日本語 Retrieval で ruri-v3-310m に約9pt 劣る**。`decisions/0014-local-embedding-backend.md` |

## 日本語で必ず踏む地雷

LanceDB の全文検索はデフォルトの `baseTokenizer: "simple"` だと**日本語をほぼ分割できない**。
実測で `simple` は実コーパス16%（合成データでは0%）まで落ちる。
FTSインデックス作成時に `"icu"` を明示すること。`decisions/0007-japanese-fts-tokenizer.md`

`npm run verify-fts` で常に検証できる。
