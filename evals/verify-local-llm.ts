import "dotenv/config";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { isMain } from "../src/shared/is-main.js";
import { stripReasoningTags } from "../src/shared/generate.js";
import {
  GENERATION_MODEL,
  GENERATION_MODEL_LABEL,
  LLM_BACKEND,
} from "../src/shared/llm-client.js";

/**
 * 生成バックエンドが「エージェント的RAGに必要な3つの能力」を満たしているかのスモークテスト。
 *
 * ## なぜ独立したスクリプトなのか
 *
 * 比較実験1回はローカルだと長時間かかる。その最後に「実はツール呼び出しが
 * 一度も成立していなかった」と分かるのが最悪の失敗で、しかも
 * **コード側のフォールバックのせいで結果はもっともらしく見えてしまう**
 * （`ToolUseStats` のコメント参照）。
 *
 * そこで本番実行の前に、3つの能力を**個別に**切り分けて確認する:
 *
 *   1. 素の生成が返るか            … 接続とチャットテンプレートの確認
 *   2. ツール呼び出しが往復するか  … search/fetch ループの前提。llama.cpp では --jinja 必須
 *   3. 構造化出力が取れるか        … スキル選択・充足チェック・確信度チェック・質問分解の前提
 *
 * 3 が落ちる場合、ワークフローは動くには動くが、全ての判定がフォールバックに倒れて
 * 「ループが働かなかった」という誤った観測を生む。ここで止めること。
 *
 * 使い方:
 *   $env:LLM_BACKEND="local"; npx tsx evals/verify-local-llm.ts
 */

interface CheckResult {
  name: string;
  ok: boolean;
  detail: string;
}

/** 1. 素の生成 */
async function checkPlainGeneration(): Promise<CheckResult> {
  const agent = new Agent({
    id: "verify-plain",
    name: "Verify Plain",
    instructions: "簡潔に日本語で答えてください。",
    model: GENERATION_MODEL,
  });

  const result = await agent.generate("1+1はいくつですか。数字だけ答えてください。");
  const text = stripReasoningTags(result.text ?? "");
  return {
    name: "素の生成",
    ok: text.trim().length > 0,
    detail: text.trim().slice(0, 120) || "(空の応答)",
  };
}

/**
 * 2. ツール呼び出し。
 *
 * ダミーのツールを1つだけ持たせ、**実際に execute が走ったか**で判定する。
 * 応答テキストから推測すると、ツールを呼ばずに「呼びました」と書くモデルを
 * 通してしまう（それこそが検出したい失敗モード）。
 */
async function checkToolCalling(): Promise<CheckResult> {
  let called = false;
  let receivedArg = "";

  const echoTool = createTool({
    id: "lookup_stock",
    description:
      "商品の在庫数を調べる。在庫を聞かれたら必ずこのツールを使うこと。推測で答えてはいけない。",
    inputSchema: z.object({
      item: z.string().describe("調べる商品名"),
    }),
    execute: async ({ item }) => {
      called = true;
      receivedArg = item;
      return { item, stock: 42 };
    },
  });

  const agent = new Agent({
    id: "verify-tools",
    name: "Verify Tools",
    instructions:
      "あなたは在庫を答える担当です。在庫数は必ず lookup_stock ツールで調べてから答えてください。",
    model: GENERATION_MODEL,
    tools: { lookup_stock: echoTool },
  });

  const result = await agent.generate("りんごの在庫はいくつですか。", { maxSteps: 4 });
  const text = stripReasoningTags(result.text ?? "");

  return {
    name: "ツール呼び出し",
    ok: called,
    detail: called
      ? `execute が実行された (item="${receivedArg}") / 応答: ${text.trim().slice(0, 80)}`
      : `**ツールが一度も呼ばれなかった** / 応答: ${text.trim().slice(0, 120)}`,
  };
}

/**
 * 3. 構造化出力。
 *
 * `skill-select.ts` と同じ形のスキーマで試す。enum・string・boolean・配列を
 * ひと通り含めているのは、単純な1フィールドだけ通って実際のスキーマで落ちるのを避けるため。
 */
const VerifySchema = z.object({
  skill: z.enum(["search", "direct"]).describe("使用するスキル"),
  reason: z.string().describe("選んだ理由"),
  is_compound: z.boolean().describe("複合質問なら true"),
  sub_queries: z.array(z.string()).max(2).describe("分解した検索クエリ"),
});

async function checkStructuredOutput(): Promise<CheckResult> {
  const agent = new Agent({
    id: "verify-structured",
    name: "Verify Structured",
    instructions:
      "質問に対して、文書検索が必要なら search、不要なら direct を選ぶルーターです。",
    model: GENERATION_MODEL,
  });

  const result = await agent.generate(
    "LanceDBの日本語全文検索でトークナイザを指定する方法と、埋め込みの次元数の決め方を教えてください。",
    { structuredOutput: { schema: VerifySchema } },
  );

  const object = result.object;
  if (!object) {
    return {
      name: "構造化出力",
      ok: false,
      detail:
        "**result.object が undefined**（スキーマに沿ったJSONを取り出せていない）。" +
        `素のテキスト: ${stripReasoningTags(result.text ?? "").slice(0, 160)}`,
    };
  }

  return {
    name: "構造化出力",
    ok: true,
    detail: JSON.stringify(object),
  };
}

async function main() {
  console.log(`バックエンド: ${LLM_BACKEND}`);
  console.log(`モデル:       ${GENERATION_MODEL_LABEL}\n`);

  if (LLM_BACKEND !== "local") {
    console.warn(
      "⚠️  LLM_BACKEND が local ではありません。ローカルモデルを検証するなら\n" +
        '    $env:LLM_BACKEND="local" を設定してから実行してください。\n',
    );
  }

  // ラベルを関数と一緒に持つ。例外時に fn.name（英語の関数名）を出すと、
  // 下の判定メッセージの分岐と名前が食い違って助言が出なくなる
  const checks: { label: string; run: () => Promise<CheckResult> }[] = [
    { label: "素の生成", run: checkPlainGeneration },
    { label: "ツール呼び出し", run: checkToolCalling },
    { label: "構造化出力", run: checkStructuredOutput },
  ];
  const results: CheckResult[] = [];

  for (const check of checks) {
    try {
      const r = await check.run();
      results.push(r);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({ name: check.label, ok: false, detail: `例外: ${message.slice(0, 300)}` });
    }
    const last = results[results.length - 1]!;
    console.log(`${last.ok ? "✅" : "❌"} ${last.name}\n   ${last.detail}\n`);
  }

  const failed = results.filter((r) => !r.ok);
  console.log("--- 判定 ---");
  if (failed.length === 0) {
    console.log("3項目すべて成立。比較実験を回して問題ありません。");
    return;
  }

  console.log(`${failed.map((f) => f.name).join(" / ")} が失敗しています。`);
  if (failed.some((f) => f.name === "ツール呼び出し")) {
    console.log(
      "\n[ツール呼び出し] llama.cpp では --jinja を付けないとチャットテンプレートが使われず、\n" +
        "  tool call が解釈されません。docker/compose.yaml の command を確認してください。",
    );
  }
  if (failed.some((f) => f.name === "構造化出力")) {
    console.log(
      "\n[構造化出力] これが通らないまま比較実験を回してはいけません。\n" +
        "  スキル選択は search、充足/確信度チェックは sufficient=true にフォールバックするため、\n" +
        "  「ループが働かなかった」というもっともらしい嘘の結果が出ます。",
    );
  }
  process.exitCode = 1;
}

if (isMain(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.stack : err);
    process.exit(1);
  });
}
