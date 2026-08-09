# agentic-rag-lab

「普通のRAGとエージェント的RAGは何が違うのか」を、自分のQiita記事をコーパスにして
実測するための実験リポジトリ。

3パターンを**同一のインデックス・同一の生成モデル・同一の生成プロンプト**で走らせ、
検索戦略とループの有無だけを変数にして比較する。

| # | パターン | 検索 | ループ |
|---|---|---|---|
| 1 | ナイーブRAG | ベクトル top-5 | なし |
| 2 | ハイブリッド | BM25＋ベクトル（RRF融合）top-5 | なし |
| 3 | エージェント的RAG | ハイブリッド + スキル選択 | あり（turn上限3） |

1 → 2 の差が「検索精度の改善による効果」、2 → 3 の差が「エージェント化による効果」になる。

## セットアップ

```bash
npm install
cp .env.example .env   # OPENAI_API_KEY と QIITA_USERNAME を記入
```

必要な API キーは `OPENAI_API_KEY` のみ（LLM と埋め込みの両方に使う）。
`QIITA_TOKEN` は任意だが、設定すると API のレート制限が 60req/h → 1000req/h に緩和される。

## 使い方

```bash
# 1. コーパス構築
npm run ingest        # Qiita API から記事を取得 → data/raw/
npm run build-index   # clean → chunk → embed → LanceDB
npm run verify-fts    # 日本語BM25が効いているかの検証（必須ゲート）

# 2. 評価データ作成
npm run gen:golden     # easy 質問を生成
npm run gen:multihop   # multihop 質問を生成し golden set を組み立て
#    → evals/golden-set.draft.json を目視レビューして golden-set.json に確定
npm run validate:golden   # 形式と chunk_id の実在を検証
npm run probe:retrieval   # 単発検索の recall を測定（生成なし・数十秒）

# 3. 比較実験
npm run eval          # 3パターン × golden set → experiments/ にレポート出力
```

手動で1問だけ試す場合:

```bash
npx tsx scripts/run-workflow.ts all "vLLMのmax-model-lenを小さくすると速度が改善するのはなぜ？"
```

## ドキュメント

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 現在の最終設計 |
| [docs/comparison-experiment.md](docs/comparison-experiment.md) | 比較実験の設計と評価軸 |
| [docs/search-fetch.md](docs/search-fetch.md) | search/fetch 内部処理の詳細 |
| [docs/tech-stack.md](docs/tech-stack.md) | 技術スタックと選定理由 |
| [docs/decisions/](docs/decisions/) | ADR（意思決定記録） |
| [experiments/](experiments/) | 実験ログ（1実験1ファイル） |

## 結果（2026-08-09 時点）

golden set 46問（easy 20 / multihop 26）に対する完全ヒット。

| | easy | multihop | 入力トークン | レイテンシ |
|---|---|---|---|---|
| ナイーブRAG | 19/20 | 13/26 | 1,591 | 2.8s |
| ハイブリッド | **20/20** | **17/26** | 1,620 | 3.0s |
| エージェント的RAG | 19/20 | 17/25 | **41,009** | **24.0s** |

**エージェント化は精度上のメリットを生まなかった。** 同じ精度に25倍のトークンと8倍の
レイテンシを払っている。単発で足りる easy 領域では6回の実験を通じて一度も勝てていない。

ただし初回は multihop 10/26 の「完敗」だった。そこから 17 まで詰めた7問ぶんは
**新機能の追加ではなく、すべて実装バグと計測の交絡の除去**による。
経緯は [experiments/](experiments/) に (a)〜(f) として残してある。
「エージェント的RAGは効果がない」と結論する前に潰すべきものが5つあった、というのが
このリポジトリの一番の収穫かもしれない。

## 実装で踏んだ地雷

このリポジトリを読む価値があるとすれば、たぶんここ。

**LanceDB の全文検索はデフォルト設定だと日本語がほぼ引けない。**
`baseTokenizer` は既定が `"simple"`（空白と句読点で分割）で、分かち書きしない日本語では
BM25 が機能しない。実測で合成データ 0%、実コーパス 16%。エラーも警告も出ないため、
気づかないまま「ハイブリッド検索は効果がない」と誤結論しうる。`"icu"` を明示すること。
→ [0007](docs/decisions/0007-japanese-fts-tokenizer.md)

**LLM に生成させた golden set は簡単すぎて差が出ない。**
1チャンクから質問を作ると、その1チャンクと語彙・意味の両方がほぼ完全一致するため、
単発ハイブリッド検索の recall@5 が 100% に達した。単発が満点ではエージェント化に
改善の余地が構造的に存在せず、「ループはコストが増えるだけ」という*データの性質に由来する*
結論しか出ない。複数チャンクの統合を要する multihop 質問を別カテゴリとして追加した。
→ [docs/comparison-experiment.md](docs/comparison-experiment.md)

**質問文の指示語が検索を壊す。**
「この実験で使われたデータ拡張手法は？」のような質問は、主題を特定する語を持たないため
BM25 もベクトル検索も手がかりを得られない。3パターンとも一様に失敗し、差がノイズに埋もれる。
生成時に自己完結性を強制している。

**共通プロンプトの流用が直接回答スキルを壊した。**
挨拶に対してパターン3が「提供された文書からは判断できません」と回答していた。
スキル選択は正しく `direct` を選んでいたのに、文書接地プロンプトを流用していたのが原因。
放置すると「パターン3は雑談に弱い」とスキル選択の問題に誤診断するところだった。

**片方のパターンだけが使う機能が、採点対象から漏れていた。**
`retrieved_chunk_ids` に fetch の起点チャンクしか記録しておらず、範囲拡張
（`with_neighbors` / `whole_section`）で実際に読んだ隣接チャンクが計上されていなかった。
範囲拡張はエージェント的RAGだけが使う機能なので、**エージェント側だけを一方的に減点**していた。
回答内容が完全に正解なのに不正解と採点されていたケースがある。
→ [0009](docs/decisions/0009-agentic-loss-decomposition.md)

**「足りない」と気づかせないとループは働かない。**
確信度チェックを回答生成の**後**に置いていたとき、ターン数の中央値は1だった。
一度回答文が生成されると、その流暢さが根拠の不足を覆い隠して「十分」と判定されてしまう。
チェックを収集段階に移し、回答文を渡さず根拠だけを見せるようにして初めてループが回った。
→ [0008](docs/decisions/0008-evidence-check-before-generation.md)

**単発の結果を根拠に設計変更を重ねてはいけない。**
分散を測らないまま4回チューニングを回し、途中で「マージ順のバグが原因」と因果を断定したが、
修正しても改善しなかった（仮説がデータに支持されなかった）。
n=26 で確率的に振る舞うエージェントでは、2〜4問の差に意味を読んではいけない。

## ライセンス / 注意

学習用の実験リポジトリ。`data/` と `traces/` は再生成可能なので gitignore してある。

MIT License。詳細は [LICENSE](LICENSE) を参照。
