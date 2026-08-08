import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { openChunksTable } from "../src/index/build-index.js";
import { isMain } from "../src/shared/is-main.js";
import type { GoldenItem } from "./generate-golden-set.js";

/**
 * 人手レビュー後の golden set を検証する。
 *
 * 手編集の JSON は壊れやすく、しかも**壊れても実験は最後まで走ってしまう**。
 * 存在しない chunk_id が1つ混ざるとその質問は誰にも正解できなくなり、
 * 3パターンすべてのスコアが等しく下がる。原因が見えないまま
 * 「エージェント化しても改善しなかった」という誤った結論に着地する。
 *
 * とくに注意すべきなのが **chunk_id の陳腐化**。chunk_id は `article_id#連番` なので、
 * チャンク分割の設定を変えて `npm run build-index` をやり直すと連番がずれ、
 * 既存の golden set の正解 ID が静かに別のチャンクを指すようになる。
 * インデックスを作り直したら必ずこれを流すこと。
 */

const GOLDEN_PATH = path.resolve("evals/golden-set.json");

interface Problem {
  index: number;
  message: string;
}

export async function validateGoldenSet(filePath = GOLDEN_PATH): Promise<boolean> {
  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch {
    console.error(
      `${filePath} がありません。draft をコピーしてレビューしてください:\n` +
        "  cp evals/golden-set.draft.json evals/golden-set.json",
    );
    return false;
  }

  let items: GoldenItem[];
  try {
    items = JSON.parse(raw) as GoldenItem[];
  } catch (err) {
    console.error(
      `JSON として読めません（末尾カンマや括弧の閉じ忘れが多いです）:\n  ${
        err instanceof Error ? err.message : err
      }`,
    );
    return false;
  }

  if (!Array.isArray(items)) {
    console.error("トップレベルが配列ではありません。");
    return false;
  }

  // インデックスに実在する chunk_id を集める
  const table = await openChunksTable();
  const rows = (await table.query().select(["chunk_id"]).limit(100000).toArray()) as {
    chunk_id: string;
  }[];
  const known = new Set(rows.map((r) => r.chunk_id));

  const problems: Problem[] = [];
  const seenQuestions = new Map<string, number>();

  items.forEach((item, i) => {
    const at = (msg: string) => problems.push({ index: i, message: msg });

    if (!item.question?.trim()) at("question が空です");
    if (!item.expected_answer?.trim()) at("expected_answer が空です");
    if (!item.article_id?.trim()) at("article_id が空です");

    if (!Array.isArray(item.golden_chunk_ids) || item.golden_chunk_ids.length === 0) {
      at("golden_chunk_ids が空です");
    } else {
      for (const id of item.golden_chunk_ids) {
        if (!known.has(id)) {
          at(
            `golden_chunk_ids に実在しない chunk_id "${id}" があります` +
              "（インデックスを作り直した場合、連番がずれている可能性があります）",
          );
        }
      }
      const unique = new Set(item.golden_chunk_ids);
      if (unique.size !== item.golden_chunk_ids.length) {
        at("golden_chunk_ids に重複があります");
      }
    }

    if (item.difficulty !== "easy" && item.difficulty !== "multihop") {
      at(`difficulty が不正です: ${String(item.difficulty)}`);
    }
    if (item.difficulty === "multihop" && item.golden_chunk_ids?.length < 2) {
      at("multihop なのに golden_chunk_ids が1件しかありません（easy に変えてください）");
    }

    const key = item.question?.replace(/\s+/g, "");
    if (key) {
      const prev = seenQuestions.get(key);
      if (prev !== undefined) at(`質問が ${prev} 番目と重複しています`);
      else seenQuestions.set(key, i);
    }
  });

  const easy = items.filter((i) => i.difficulty === "easy").length;
  const multihop = items.filter((i) => i.difficulty === "multihop").length;

  console.log(`${path.basename(filePath)}: ${items.length} 問（easy ${easy} / multihop ${multihop}）`);

  if (problems.length > 0) {
    console.error(`\n❌ ${problems.length} 件の問題:`);
    for (const p of problems.slice(0, 30)) {
      console.error(`  [${p.index}] ${p.message}`);
    }
    if (problems.length > 30) console.error(`  ... 他 ${problems.length - 30} 件`);
    return false;
  }

  console.log("✅ 形式・chunk_id ともに問題ありません。");
  if (multihop < 10) {
    console.warn(
      `⚠️  multihop が ${multihop} 問しかありません。エージェント化の効果を測る主戦場なので、` +
        "10問を下回ると結果が偶然に左右されやすくなります。",
    );
  }
  return true;
}

if (isMain(import.meta.url)) {
  validateGoldenSet()
    .then((ok) => process.exit(ok ? 0 : 1))
    .catch((err) => {
      console.error(err instanceof Error ? err.stack : err);
      process.exit(1);
    });
}
