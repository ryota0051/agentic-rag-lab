import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * このモジュールが `tsx foo.ts` として直接実行されたかを判定する。
 *
 * `import.meta.url === \`file://${process.argv[1]}\`` という素朴な比較は Windows で壊れる
 * （パス区切りがバックスラッシュ、ドライブレターの大小、file:/// のスラッシュ数が一致しない）。
 * 一度 fileURLToPath で実パスに戻してから正規化して比較する。
 */
export function isMain(importMetaUrl: string): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return path.resolve(fileURLToPath(importMetaUrl)) === path.resolve(entry);
  } catch {
    return false;
  }
}
