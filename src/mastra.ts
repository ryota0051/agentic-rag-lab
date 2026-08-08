import { Mastra } from "@mastra/core/mastra";
import { LibSQLStore } from "@mastra/libsql";
import { ragWorkflows } from "./workflows/index.js";

/**
 * Mastra インスタンス。ワークフロー登録とトレースの永続化を担う。
 *
 * トレースを残しておくことで、パターン3が「何ターン使ったか」「クエリをどう言い換えたか」を
 * 後から失敗モード分析にかけられる（docs/architecture.md「ログ設計への反映」）。
 * 実験ログに書く定性観察の材料はここから拾う。
 */
export const mastra = new Mastra({
  workflows: ragWorkflows,
  storage: new LibSQLStore({ id: "traces", url: "file:./traces/mastra.db" }),
});
