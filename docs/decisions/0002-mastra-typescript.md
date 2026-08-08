# 0002. TypeScript + Mastra で全体を構成する

- ステータス: 採択
- 日付: 2026-08-08
- 関連: `src/mastra.ts`, `src/workflows/index.ts`, `package.json`

## 決定

エージェント、ワークフロー、評価のすべてを **TypeScript + Mastra** で構成し、
Python を使わない単一言語構成にする。

## 理由

### 評価基盤を Mastra に寄せたことで Python が不要になった

当初は評価に deepeval や MLflow（Python エコシステム）を使う案もあった。
その場合、エージェント本体は TypeScript、評価は Python という二言語構成になり、
プロセス間のデータ受け渡しと二重の環境構築が発生する。

Mastra の Scorers（`@mastra/core/evals` の `createScorer` / `runEvals`、
および `@mastra/evals` の組み込みスコアラー）で評価まで完結できるため、
**TypeScript 単独で閉じられる**。学習用プロジェクトとして、環境構築の摩擦は小さいほどよい。

### ワークフローとして表現できる

3パターンの比較実験は「同じ入力を別の経路に流して結果を比べる」構造なので、
ワークフローという抽象がそのまま当てはまる。`createWorkflow` / `createStep` で
定義しておくと、トレースが自動で記録され、後から失敗モードを掘り返せる。

### モデルルーターでプロバイダを差し替えられる

Mastra の model router は `"openai/gpt-5.6-luna"` のような文字列でモデルを指定し、
`OPENAI_API_KEY` を自動で読む。SDK の追加インストールが不要で、
将来ローカルLLM（vLLM / llama.cpp）に差し替える際も
`src/shared/llm-client.ts` の1箇所を変えるだけで済む。

## ワークフローは薄いラッパにする

実装本体は `runNaiveRag()` のような**素の非同期関数**に置き、
`src/workflows/index.ts` でそれを `createStep` / `createWorkflow` に包んでいる。

理由は2つ。

1. スクリプトから直接関数を呼べるため、手動確認とデバッグが速い
   （`scripts/run-workflow.ts`）
2. `runEvals({ target })` は Agent か Workflow を要求するので、
   評価から使うにはワークフロー化が必要

ロジックをワークフローの `execute` に直接書くと 1 が失われる。

## `@mastra/lance` を使わない

ベクトルDBは `@lancedb/lancedb` を**直接**使っている。
Mastra のベクトルストア抽象（`@mastra/lance`）経由だとベクトル検索しか叩けず、
**BM25 と RRF 融合を制御できない**。ハイブリッド検索がパターン2の本体なので、
ここは抽象を挟まず生の API を使う判断をした。

## `create-mastra` を使わない

`create-mastra` は `src/mastra/` 配下にエージェント・ワークフローを固める構成を前提とする。
本プロジェクトは `src/workflows/` `src/search/` `src/index/` `evals/` という
関心ごとの分割を採っており衝突するため、手動セットアップにした。
