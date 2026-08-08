/**
 * 整形・ノイズ除去。**インデックス構築時に一度だけ**行う（fetch 時にはやらない）。
 *
 * 理由は docs/architecture.md / docs/decisions/0004-cleaning-at-ingestion.md の通り:
 *  - 同じ文書が複数クエリから何度も fetch されるたびにクリーニングするのは無駄
 *  - embedding と BM25 のインデックス自体をノイズ入りの生テキストに対して作ると search の精度が落ちる
 *  - クリーニング改善時もインデックス再構築だけで済み、search/fetch 側に手を入れずに済む
 *
 * 重要: **見出し構造は壊さない。** 次段の chunking.ts が `#` 見出しに依存している。
 */

/** コードブロックとして残す最大行数。これを超えたら先頭だけ残して省略する */
const MAX_CODE_LINES = 20;

/**
 * 長大なコードブロックを切り詰める。
 *
 * 技術記事なのでコードは検索対象として価値がある（エラーメッセージや API 名で引ける）が、
 * 200行の設定ファイル全文などはチャンクを食い潰して周囲の説明文を押し出してしまう。
 * 言語名と先頭 MAX_CODE_LINES 行を残す方針。
 */
function truncateCodeBlocks(md: string): string {
  return md.replace(/^```([^\n]*)\n([\s\S]*?)^```/gm, (_all, lang: string, body: string) => {
    const lines = body.split("\n");
    if (lines.length <= MAX_CODE_LINES) return "```" + lang + "\n" + body + "```";
    const kept = lines.slice(0, MAX_CODE_LINES).join("\n");
    const omitted = lines.length - MAX_CODE_LINES;
    return "```" + lang + "\n" + kept + `\n... (以下 ${omitted} 行省略)\n` + "```";
  });
}

export function cleanMarkdown(md: string): string {
  let out = md;

  // 改行コードを LF に統一（CRLF が混ざると見出し正規表現と文字数カウントがズレる）
  out = out.replace(/\r\n?/g, "\n");

  out = truncateCodeBlocks(out);

  // 画像は本文として意味を持たないので alt テキストだけ残す。alt が空なら丸ごと削除
  out = out.replace(/!\[([^\]]*)\]\([^)]*\)/g, (_all, alt: string) =>
    alt.trim() ? `（図: ${alt.trim()}）` : "",
  );

  // Qiita 独自の :::note info / :::note warn 記法。中身は残し、マーカーだけ剥がす
  out = out.replace(/^:::\s*note\s*\w*\s*$/gm, "");
  out = out.replace(/^:::\s*$/gm, "");

  // リンクはテキストだけ残す。URL がチャンクの文字数を無駄に食い、BM25 のノイズにもなる
  out = out.replace(/\[([^\]]+)\]\((?:https?:\/\/)[^)]*\)/g, "$1");

  // HTML タグ残骸（<br>, <details>, <div align="center"> など）を除去。
  // コードブロック内の < > を壊さないよう、行頭がタグのみの行と inline の既知タグに限定する
  out = out.replace(/<\/?(?:br|hr|details|summary|div|span|font|center|img|p)\b[^>]*>/gi, "");

  // 水平線は文脈的な意味を持たないので削除
  out = out.replace(/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/gm, "");

  // 行末の全角スペースや連続空白を正規化（BM25 のトークナイズを安定させる）
  out = out
    .split("\n")
    .map((line) => line.replace(/[ \t　]+$/, ""))
    .join("\n");

  // 3行以上の連続空行を2行に圧縮
  out = out.replace(/\n{3,}/g, "\n\n");

  return out.trim();
}
