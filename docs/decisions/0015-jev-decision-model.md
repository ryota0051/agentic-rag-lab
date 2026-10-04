# 0015. パターン3の判定ステップに判定専用モデル Jev を試す

- 状態: 採用（**オプションの実験軸として**。既定は `llm` のまま）
- 日付: 2026-10-04
- 関連: [`0001`](0001-skill-based-routing.md)（スキル分岐）、[`0008`](0008-evidence-check-before-generation.md)（収集段階の充足チェック）、
  [`0013`](0013-local-llm-backend.md) / [`0014`](0014-local-embedding-backend.md)（実験軸は1本ずつ動かす）

## 背景

パターン3には、LLM に structured output で二値を言わせている**判定ステップ**が4つある。
いずれも「LLM は自分の判断の確からしさを自己申告できない」（calibration 問題）という
弱点を抱えており、0001 と 0008 にもそれぞれ書いてある。

- 0001: スキル選択は Precision 100% / Recall 82%。「迷ったら検索」と書いても、モデルは「迷っていない」と判断してしまう
- 0008: 回答文を見ると流暢さが根拠の不足を覆い隠す。根拠だけを見るように変えたが、それでも「だいたい関連しているから十分」と判定しがち

TypeSafe AI の Jev（2026-09 公開）は、状態を受け取り、**有限の選択肢と確率だけ**を返す判定専用モデルである。
文章は生成しない。`experiments/2026-10-04-jev-hello.md` で触ってみて、次のことが分かった。

- 根拠充足の yes 確率は、十分 84% ／ 片方欠け 8% ／ 無関係 2% と明確に分かれる
- 1回 150〜240ms で、同じ判定を LLM にさせると 3.6〜5.4 秒。コストは無視できる水準
- **間違えるときは確信度が低い**。スキル分岐で期待と逆になった2問は confidence 0.25 / 0.53 だった
- ただし criteria（選択肢の説明文）の書き方には強く引きずられる。否定形の問いは yes 側に寄る

この「間違えるときは確信度が低い」性質が本当なら、0001 で言葉では実現できなかった非対称コストを
**閾値という数値で**表現できる。

## 判定ステップの棚卸しと提案

| 箇所 | 現状の出力 | Jev に任せられる部分 | 正解ラベル | 判断 |
|---|---|---|---|---|
| **A. スキル分岐** `skill-select.ts` | `skill` + `reason` | `skill` 全体（reason はログ用なので不要） | `evals/boundary-cases.json` 20件 | **試す** |
| **B. 収集完了の判定** `evidence-check.ts` | `sufficient` + `missing` + `followup_query` | `sufficient` のみ。`missing` と `followup_query` は文章なので LLM に残す | golden の正解 chunk が根拠に揃っているか（決定的に計算できる） | **試す（本命）** |
| C. 分解要否 `query-decompose.ts` | `is_compound` + `sub_queries` | `is_compound` のみ | difficulty（easy / multihop）で代用 | 見送り |
| D. 回答後の確信度チェック `confidence-check.ts` | `sufficient` + `missing` + `followup_query` | `sufficient` のみ | 無し | 見送り |

C を見送る理由: 分解すると判定された場合は結局 LLM でサブクエリを作るので、LLM 呼び出しが減らない。
しかも分解が要る質問は multihop に偏っていて、判定を先に回してもほとんど節約にならない。

D を見送る理由: 0008 以降、根拠の充足は B で見ている。D が捉えるのは「根拠はあるのに回答が使えていない」という
別の失敗で、正解ラベルを機械的に作れない。

### A. スキル分岐

- choice で `search` / `direct` の確率を得る。**`p(direct) ≥ τ` のときだけ direct を選び、それ以外は search にする**
- criteria には `SEARCH_SKILL_DESCRIPTION` / `DIRECT_ANSWER_SKILL_DESCRIPTION` を要約して入れる
  （instructions は 1,800 字、criteria も短い方がよいため）。jev-hello のケース6で、説明文の書き方が結果を大きく左右すると分かっている。
  そのため、**LLM 側と同じ判断基準を書くこと**を優先し、Jev 向けのチューニングはしない
- τ を 0.5〜0.99 で動かし、取りこぼしと過剰発火の曲線で LLM（1点）と比べる

### B. 収集完了の判定

- 1回のリクエストで2つの型を同時に聞く
  - noul「根拠だけで質問のすべての要素に答えられるか」
  - score「カバー率（なし／一部／すべて）」
  - jev-hello では score の方が確信度が高く出たので、どちらがよく分けられるかも見る
- 問いは肯定形の1本に固定する（否定形は yes 側に寄った。jev-hello ケース6b）
- 不足と判定されたときだけ、既存の LLM に `missing` / `followup_query` を書かせる。足りている場合は LLM 呼び出しが1回減る

## 実験軸と不変条件

- 判定器はパターン3にしか無い。**不変条件2（生成モデルと生成プロンプトは3パターン共通）には抵触しない**。
  回答生成は従来どおり `src/shared/generate.ts` を通る
- 新しい軸 `DECISION_BACKEND=llm|jev`（既定 `llm`）として導入する。`LLM_BACKEND` / `EMBEDDING_BACKEND` とは独立で、
  **同時に動かさない**
- Jev は API でしか提供されていない。`DECISION_BACKEND=jev` にすると、「両方 local ならキー無しで回る」性質は失われる
- モデル文字列は `src/shared/llm-client.ts` の `JEV_MODEL`（`jev-latest`）に置く。エイリアスなので、
  **実験ログにはレスポンスの `model`（例: `jev-1.13.0`）を記録する**

## 評価方法

1. **A のオフライン比較** — boundary-cases 20件を LLM と Jev に3回ずつ通す。採点は既存の `scoreSkillSelection` を使う
2. **B のオフライン比較** — エージェントは回さず、`hybridSearch` + `fetchChunks` で判定スナップショットを作る
   （k=5、k=2、multihop から正解を1件抜いたもの、の3種類）。正解ラベルは正解 chunk が揃っているかどうか。
   LLM と Jev に同じ（切り詰めた）根拠を渡して比べる
3. **B の E2E** — 2 で Jev が LLM と同等以上だった場合だけ、パターン3の `evidence-check` を差し替えて multihop を回す

注意: B の正解ラベルは「正解 chunk が揃っているか」であって、真の充足度ではない。
別の chunk で答えられる場合は、ラベル上は「不足」なのに実際は答えられる。

## 結果（2026-10-04）

| 実験 | 結果 | レポート |
|---|---|---|
| A. スキル分岐（オフライン） | Jev は τ=0.5 で3回とも 20/20。LLM は3回中2回、取りこぼし1件 | `experiments/2026-10-04-jev-skill-routing.md` |
| B. 収集完了の判定（オフライン） | Jev の AUC 0.962。τ=0.5 で正解率は LLM と同等（87% 対 86〜88%）、見逃しは約半分（5 対 10〜12） | `experiments/2026-10-04-jev-evidence-gate.md` |
| B. E2E（multihop 26問） | 完全ヒット 16 → 17、部分recall 73% → 81%、入力トークン −14% | `experiments/2026-10-04-jev-e2e-decision-jev.md` |

どの実験でも、**Jev が間違えるのは確率が中間（おおむね 0.2〜0.55）のときだけ**だった。
確率が端に寄っているときに LLM と食い違った例は無い。

### 既定を llm のままにする理由

- E2E は n=26 で各1回しか回していない。1問差はノイズの範囲で、「Jev の方が良い」とはまだ言えない
- 既定を変えると、過去の実験レポートと agentic の数値が比べられなくなる
- Jev は API でしか提供されていないため、既定にすると「両方 local ならキー無しで回る」性質が失われる

### 宿題

- E2E を複数回回して、ばらつきの幅を出す
- スキル分岐の Jev 化を `DECISION_BACKEND` に載せる（現状はオフライン比較だけ）
- τ を、boundary-cases や golden set とは別の質問セットで決める
- 「全部揃っている」ケースでの Jev の弱さ（forced_full 74%）が、長い前置き付きの質問に偏っているかを確かめる
