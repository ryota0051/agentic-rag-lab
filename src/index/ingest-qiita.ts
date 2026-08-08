import "dotenv/config";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { isMain } from "../shared/is-main.js";
import type { RawArticle } from "../shared/types.js";

/**
 * Qiita API v2 から自分の記事を取得し data/raw/*.json に保存する。
 *
 * HTML スクレイピングではなく API を使うので、Markdown 本文 (`body`) がそのまま取れる。
 * HTML由来のノイズ（ナビゲーション、広告、タグ残骸）が原理的に発生しないのが選定理由
 * （docs/decisions/0004-cleaning-at-ingestion.md）。
 */

const RAW_DIR = path.resolve("data/raw");
const PER_PAGE = 100;

/** Qiita API のレスポンスのうち今回使うフィールドだけ */
interface QiitaItem {
  id: string;
  title: string;
  url: string;
  body: string;
  created_at: string;
  updated_at: string;
  tags: { name: string }[];
}

async function fetchPage(
  username: string,
  page: number,
  token: string | undefined,
): Promise<QiitaItem[]> {
  const url = `https://qiita.com/api/v2/users/${encodeURIComponent(username)}/items?page=${page}&per_page=${PER_PAGE}`;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(url, { headers });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    if (res.status === 403 || res.status === 429) {
      throw new Error(
        `Qiita API のレート制限に達しました (${res.status})。` +
          `QIITA_TOKEN を設定すると 60req/h → 1000req/h に緩和されます。\n${body}`,
      );
    }
    if (res.status === 404) {
      throw new Error(
        `Qiita ユーザー "${username}" が見つかりません。QIITA_USERNAME を確認してください。`,
      );
    }
    throw new Error(`Qiita API エラー ${res.status}: ${body}`);
  }

  return (await res.json()) as QiitaItem[];
}

export async function ingestQiita(): Promise<RawArticle[]> {
  const username = process.env.QIITA_USERNAME;
  if (!username) {
    throw new Error("QIITA_USERNAME が未設定です。.env を確認してください。");
  }
  const token = process.env.QIITA_TOKEN || undefined;
  if (!token) {
    console.warn("[ingest] QIITA_TOKEN 未設定。未認証のため 60req/h 制限で動作します。");
  }

  await mkdir(RAW_DIR, { recursive: true });

  const articles: RawArticle[] = [];
  for (let page = 1; ; page++) {
    const items = await fetchPage(username, page, token);
    if (items.length === 0) break;

    for (const item of items) {
      // 本文が実質空の記事（限定共有の下書き等）はインデックスを汚すので落とす
      if (!item.body || item.body.trim().length < 50) {
        console.warn(`[ingest] スキップ（本文が短すぎる）: ${item.title}`);
        continue;
      }
      const article: RawArticle = {
        article_id: item.id,
        title: item.title,
        url: item.url,
        tags: item.tags.map((t) => t.name),
        body: item.body,
        created_at: item.created_at,
        updated_at: item.updated_at,
      };
      articles.push(article);
      await writeFile(
        path.join(RAW_DIR, `${article.article_id}.json`),
        JSON.stringify(article, null, 2),
        "utf8",
      );
    }

    if (items.length < PER_PAGE) break;
  }

  console.log(`[ingest] ${articles.length} 記事を ${RAW_DIR} に保存しました。`);
  return articles;
}

if (isMain(import.meta.url)) {
  ingestQiita().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
