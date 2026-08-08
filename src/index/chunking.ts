import type { Chunk, RawArticle } from "../shared/types.js";
import { cleanMarkdown } from "./clean.js";

/**
 * 意味的境界（Markdown の見出し）でのチャンク分割。
 *
 * 固定長スライディングウィンドウを使わないのは、見出しをまたいで切ると
 * 「何についての説明か」を失ったチャンクができ、embedding も BM25 も劣化するため。
 */

/** 1チャンクの目標文字数。日本語なので文字数ベースで管理する */
const TARGET_CHARS = 600;
/** これを下回るチャンクは前のチャンクに併合する（見出しだけの断片を作らない） */
const MIN_CHARS = 120;
/** 分割時のオーバーラップ。境界にまたがる記述を両方のチャンクから引けるようにする */
const OVERLAP_CHARS = 80;

interface Section {
  /** 例: ["セットアップ", "Docker"]（記事タイトルは含めない。呼び出し側で前置する） */
  headings: string[];
  body: string;
}

/** Markdown を見出し単位のセクションに切る。コードブロック内の `#` は見出しとみなさない */
function splitIntoSections(md: string): Section[] {
  const lines = md.split("\n");
  const sections: Section[] = [];
  // headingStack[level] = そのレベルの直近の見出しテキスト
  const headingStack: string[] = [];
  let current: Section = { headings: [], body: "" };
  let inCodeBlock = false;

  const pushCurrent = () => {
    if (current.body.trim()) sections.push({ ...current, body: current.body.trim() });
  };

  for (const line of lines) {
    if (/^```/.test(line)) {
      inCodeBlock = !inCodeBlock;
      current.body += line + "\n";
      continue;
    }

    const m = !inCodeBlock ? /^(#{1,4})\s+(.*)$/.exec(line) : null;
    if (m) {
      pushCurrent();
      const level = m[1]!.length;
      const text = m[2]!.trim();
      // 同レベル以下の見出しを捨てて、このレベルを差し替える
      headingStack.length = level - 1;
      headingStack[level - 1] = text;
      current = { headings: headingStack.filter(Boolean), body: "" };
    } else {
      current.body += line + "\n";
    }
  }
  pushCurrent();

  return sections;
}

/**
 * 長いセクションを段落境界でさらに分割する。
 * 単語（というより文字）の途中で切らないよう、必ず段落単位で積み上げる。
 */
function splitLongBody(body: string): string[] {
  if (body.length <= TARGET_CHARS) return [body];

  const paragraphs = body.split(/\n{2,}/).filter((p) => p.trim());
  const parts: string[] = [];
  let buf = "";

  for (const para of paragraphs) {
    // 単一段落が既に長すぎる場合はそれ自体を1チャンクにする（無理に切らない）
    if (!buf && para.length > TARGET_CHARS) {
      parts.push(para);
      continue;
    }
    if (buf && (buf + "\n\n" + para).length > TARGET_CHARS) {
      parts.push(buf);
      // 直前チャンクの末尾をオーバーラップとして引き継ぐ
      const tail = buf.slice(-OVERLAP_CHARS);
      buf = tail + "\n\n" + para;
    } else {
      buf = buf ? buf + "\n\n" + para : para;
    }
  }
  if (buf.trim()) parts.push(buf);

  return parts;
}

/** 短すぎるチャンクを直前に併合する */
function mergeTinyParts(parts: string[]): string[] {
  const out: string[] = [];
  for (const part of parts) {
    const prev = out[out.length - 1];
    if (prev !== undefined && part.length < MIN_CHARS) {
      out[out.length - 1] = prev + "\n\n" + part;
    } else {
      out.push(part);
    }
  }
  return out;
}

export function chunkArticle(article: RawArticle): Chunk[] {
  const cleaned = cleanMarkdown(article.body);
  const sections = splitIntoSections(cleaned);

  const chunks: Chunk[] = [];
  let index = 0;

  for (const section of sections) {
    // 記事タイトルを常に先頭に置く。チャンク単体で読んだときに何の記事か分かるようにする
    const headingPath = [article.title, ...section.headings].join(" > ");
    const parts = mergeTinyParts(splitLongBody(section.body));

    for (const part of parts) {
      if (part.trim().length < MIN_CHARS && chunks.length > 0) {
        // セクション全体が短い場合。見出しパスが違うので併合はせず、そのまま採用する
        // （見出しだけの目次セクションなどはここで落ちる）
        if (part.trim().length < 40) continue;
      }
      chunks.push({
        chunk_id: `${article.article_id}#${index}`,
        article_id: article.article_id,
        title: article.title,
        url: article.url,
        heading_path: headingPath,
        chunk_index: index,
        // embedding と BM25 の両方が見出し文脈を利用できるよう、本文に見出しパスを前置する
        text: `${headingPath}\n\n${part.trim()}`,
        tags: article.tags,
        updated_at: article.updated_at,
      });
      index++;
    }
  }

  return chunks;
}
