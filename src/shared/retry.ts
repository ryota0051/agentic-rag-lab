/**
 * レート制限に対する指数バックオフ付きリトライ。
 *
 * gpt-5.6-luna の TPM 上限は 200,000。エージェント的RAGは1問あたり1万トークン超を
 * 使うため、並列実行すると容易に上限へ達する。上流のリトライだけでは回数が足りず、
 * **1回の 429 で数十分の実行が丸ごと失われる**事故が起きた。
 */

export interface RetryOptions {
  maxAttempts?: number;
  /** 初回の待機時間(ms)。以降 2倍ずつ伸びる */
  baseDelayMs?: number;
  maxDelayMs?: number;
  label?: string;
}

function isRateLimitError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    /rate.?limit/i.test(msg) ||
    /429/.test(msg) ||
    /tokens per min|TPM|requests per min|RPM/i.test(msg)
  );
}

/** エラーメッセージ内の "Please try again in 743ms" / "in 1.5s" を拾う */
function parseSuggestedDelay(err: unknown): number | undefined {
  const msg = err instanceof Error ? err.message : String(err);
  const ms = /try again in (\d+(?:\.\d+)?)ms/i.exec(msg);
  if (ms?.[1]) return Number(ms[1]);
  const s = /try again in (\d+(?:\.\d+)?)s/i.exec(msg);
  if (s?.[1]) return Number(s[1]) * 1000;
  return undefined;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const {
    maxAttempts = 6,
    baseDelayMs = 2000,
    maxDelayMs = 60000,
    label = "request",
  } = options;

  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      // レート制限以外（プロンプト不正など）はリトライしても直らないので即座に投げる
      if (!isRateLimitError(err)) throw err;
      if (attempt === maxAttempts) break;

      // サーバーが提示した待ち時間があればそれを尊重しつつ、
      // 並列ワーカーが同時に復帰して再び上限を叩かないようジッターを足す
      const suggested = parseSuggestedDelay(err);
      const backoff = Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs);
      const jitter = Math.random() * 1000;
      const delay = Math.max(suggested ?? 0, backoff) + jitter;

      console.warn(
        `[retry] ${label}: レート制限 (${attempt}/${maxAttempts})。${Math.round(delay)}ms 待機`,
      );
      await sleep(delay);
    }
  }
  throw lastError;
}
