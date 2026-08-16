# 技術スタックと選定理由

| レイヤー | 選定 | 理由 |
|---|---|---|
| エージェント／ワークフロー | **Mastra**（`@mastra/core`） | スキル選択と search/fetch ループをワークフローとして表現でき、トレースが自動で残る。`decisions/0002-mastra-typescript.md` |
| 評価 | **Mastra Scorers**（`@mastra/core/evals` + `@mastra/evals`） | `createScorer` / `runEvals`。TypeScript 単独で評価まで完結し、Python プロセスを別立てしなくて済む |
| LLM（生成・エージェント・judge） | **OpenAI `gpt-5.6-luna`** | Mastra model router 経由（`"openai/gpt-5.6-luna"`）。5.6系の最速・最安ティアで、試行回数を稼ぐ実験用途に合う。入力 $0.20 / 出力 $1.20 per 1M |
| LLM（ローカルバックエンド） | **Qwen3.8-27B**（unsloth GGUF, UD-Q4_K_XL） | `LLM_BACKEND=local` のときの生成・エージェント用。llama.cpp 公式Dockerイメージで OpenAI 互換サーバとしてホストする。`decisions/0013-local-llm-backend.md` |
| 埋め込み | **OpenAI `text-embedding-3-large`** | 3072次元。LLM と同じ `OPENAI_API_KEY` で済み、必要なキーが1つになる |
| ベクトルDB／全文検索 | **LanceDB**（`@lancedb/lancedb`） | 埋め込み型でサーバー不要。全文検索（Tantivy = BM25相当）も標準搭載しており、ベクトル＋BM25 のハイブリッドを1本で完結できる |
| データ取り込み | **Qiita API v2** | Markdown 本文とメタデータを直接取得でき、HTML由来のノイズをそもそも回避できる。`decisions/0004-cleaning-at-ingestion.md` |
| ストレージ（トレース） | `@mastra/libsql` | ローカルファイル。実行ログを `traces/` に永続化 |

## モデル文字列は1箇所に閉じる

`src/shared/llm-client.ts` が唯一の定義箇所。

```ts
// 3パターン共通。LLM_BACKEND=local ならローカルサーバの設定オブジェクトになる
export const GENERATION_MODEL: MastraModelConfig = /* openai/gpt-5.6-luna | { id, url, apiKey } */;
export const GENERATION_MODEL_LABEL = /* 表示用の文字列 */;
export const JUDGE_MODEL      = "openai/gpt-5.6-luna";
export const EMBEDDING_MODEL  = "text-embedding-3-large";
export const EMBEDDING_DIMENSIONS = 3072;
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
| 生成・エージェント | `gpt-5.6-luna` | Qwen3.8-27B（llama.cpp / Docker） |
| 埋め込み | `text-embedding-3-large` | **同じ（変更しない）** |
| LLM-as-judge | `gpt-5.6-luna` | **同じ（変更しない）** |
| LanceDBインデックス | 共有 | **同じものを共有** |

検索側を完全に固定することで、変数がエージェントを駆動するLLM1つだけになる。
**`LLM_BACKEND=local` でも `OPENAI_API_KEY` は必要**（埋め込みと judge が使う）。

```bash
npm run serve:local        # llama.cpp サーバを起動（docker/compose.yaml）
npm run verify-local-llm   # 生成／ツール呼び出し／構造化出力を個別に確認（本番前に必須）
```

judge をローカルに倒してはいけない理由と、ランタイムの選定理由は
`decisions/0013-local-llm-backend.md` を参照。

## LLM-as-judge に同じモデルを使うことについて

既定では judge も生成と同じ `gpt-5.6-luna`。採点役に被験者と同じモデルを使うと
自己評価バイアスが入りうる。3パターンすべてが同一モデル生成なので**相対比較は成立する**が、
faithfulness / answer-relevancy の**絶対値は甘めに出る**可能性がある。

3パターンのスコアが頭打ちして差がつかなくなったら `JUDGE_MODEL` だけを
`"openai/gpt-5.6-terra"` に上げる。

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

## 日本語で必ず踏む地雷

LanceDB の全文検索はデフォルトの `baseTokenizer: "simple"` だと**日本語をほぼ分割できない**。
実測で `simple` は実コーパス16%（合成データでは0%）まで落ちる。
FTSインデックス作成時に `"icu"` を明示すること。`decisions/0007-japanese-fts-tokenizer.md`

`npm run verify-fts` で常に検証できる。
