import "dotenv/config";
import { runAgenticRag } from "../src/workflows/agentic-rag/index.js";
import { runHybridRag } from "../src/workflows/hybrid-rag.js";
import { runNaiveRag } from "../src/workflows/naive-rag.js";
import type { RagPattern, RagRunResult } from "../src/shared/types.js";

/** 手動確認用。1パターン × 1質問を実行して結果を表示する */

const RUNNERS: Record<RagPattern, (q: string) => Promise<RagRunResult>> = {
  naive: runNaiveRag,
  hybrid: runHybridRag,
  agentic: runAgenticRag,
};

async function main() {
  const [pattern, ...rest] = process.argv.slice(2);
  const question = rest.join(" ").trim();

  const valid = pattern === "all" || (pattern !== undefined && pattern in RUNNERS);
  if (!valid || !question) {
    console.error(
      ' 使い方: npx tsx scripts/run-workflow.ts <naive|hybrid|agentic|all> "質問"',
    );
    process.exit(1);
  }

  const patterns: RagPattern[] =
    pattern === "all" ? ["naive", "hybrid", "agentic"] : [pattern as RagPattern];

  for (const p of patterns) {
    const result = await RUNNERS[p](question);
    console.log(`\n${"=".repeat(60)}\n[${result.pattern}]`);
    console.log(`回答:\n${result.answer}\n`);
    console.log(`引用: ${result.retrieved_chunk_ids.join(", ") || "(なし)"}`);
    if (result.selectedSkill) console.log(`選択スキル: ${result.selectedSkill}`);
    if (result.queryTrace?.length) {
      console.log(`クエリ履歴: ${result.queryTrace.map((q) => `"${q}"`).join(" → ")}`);
    }
    console.log(
      `turns=${result.turns} llmCalls=${result.usage.llmCalls} ` +
        `tokens=${result.usage.inputTokens}/${result.usage.outputTokens} ` +
        `latency=${result.latencyMs}ms`,
    );
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack : err);
  process.exit(1);
});
