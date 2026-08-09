# agentic-rag-lab

ナイーブRAG / ハイブリッド検索 / エージェント的RAG の3パターンを比較し、
「検索精度の改善」と「エージェント化（ループ）による効果」を切り分けて観察する実験リポジトリ。

設計の詳細は `docs/architecture.md`、意思決定の経緯は `docs/decisions/` を参照。

## 絶対に壊してはいけない不変条件

比較実験の妥当性がここに依存している。変更する場合は必ず理由をADRに残すこと。

1. **3パターンは同一のLanceDBインデックスを共有する。** 同一コーパス・同一チャンク分割・同一埋め込みモデル。
   検索戦略とループの有無だけが変数。パターンごとにインデックスを作り直してはいけない。
2. **生成モデルとプロンプトは3パターン共通。** モデル文字列は `src/shared/llm-client.ts`、
   生成プロンプトは `src/shared/answer-prompt.ts` に唯一の定義を置き、全パターンがそれを import する。
   ワークフロー内にモデル名やプロンプトをベタ書きしない。
3. **埋め込みモデルと次元数は ingestion と検索で必ず同一。** ここがズレると全実験が無効になる。
   `src/shared/llm-client.ts` の `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` が唯一の定義。
4. **最終的にプロンプトへ入る根拠の件数は3パターンとも k=5 に揃える。**
   パターン3も fetch 後に件数上限を掛ける。ここが揃っていないと「エージェントは根拠が多いから強い」
   という当たり前の結論しか出ない。
5. **3パターンとも同一形状 (`RagRunResult`) を返す。** `answer` / `citations` / `usage` / `latencyMs` / `turns`。

## リポジトリの約束

- 実験結果は必ず `experiments/` に **1実験1ファイル** で残す（ファイル名は `YYYY-MM-DD-<slug>.md`）。
  数値表だけでなく「どの質問でどのパターンが勝ち/負けたか」の定性観察を必ず書く。ここが記事の素材になる。
- 設計判断が発生したらその場で `docs/decisions/NNNN-*.md` を書く。後追いで書くと理由が失われる。
- `docs/architecture.md` は「現在の最終設計」を書く生きたドキュメント。経緯はADRに追い出して簡潔に保つ。
- 作業が発生した場合は適切なブランチを作成する。完了次第mainブランチにマージしてマージすること。

## 技術スタック

| レイヤー | 選定 |
|---|---|
| エージェント/ワークフロー | Mastra (`@mastra/core`) |
| 評価 | Mastra Scorers (`@mastra/core/evals` + `@mastra/evals`) |
| LLM（生成・エージェント・judge） | OpenAI `gpt-5.6-luna`（Mastra model router 経由） |
| 埋め込み | OpenAI `text-embedding-3-large` |
| ベクトルDB / 全文検索 | LanceDB (`@lancedb/lancedb`) を直接利用 |
| データ取り込み | Qiita API v2 |

`@mastra/lance` は使わない。ベクトル検索しか叩けず BM25 と融合を制御できないため、
`@lancedb/lancedb` を直接使っている（`docs/decisions/` 参照）。

## 日本語BM25の注意

LanceDB の全文検索は Tantivy ベースで、デフォルトの `simple` トークナイザは日本語を分割できない。
FTSインデックス作成時のトークナイザ指定は必須。採用値と実測根拠は
`docs/decisions/0007-japanese-fts-tokenizer.md` にある。`npm run verify-fts` で常に検証できる。

## よく使うコマンド

```bash
npm run ingest       # Qiita から記事取得 → clean → chunk → embed → LanceDB 投入
npm run verify-fts   # 日本語BM25が効いているかのスモークテスト
npm run eval         # 3パターン比較を実行し experiments/ にレポート出力
npm run typecheck
```
