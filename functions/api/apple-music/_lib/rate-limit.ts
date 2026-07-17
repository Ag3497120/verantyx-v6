import { RATE_LIMIT } from './types';

type Bucket = {
  count: number;
  windowStart: number;
};

/** In-memory per isolate. Resets on cold start; not shared across isolates. */
const buckets = new Map<string, Bucket>();

export type RateLimitResult =
  | { allowed: true; remaining: number; resetMs: number }
  | { allowed: false; remaining: 0; resetMs: number };

export function checkRateLimit(apiKey: string): RateLimitResult {
  const now = Date.now();
  let bucket = buckets.get(apiKey);

  if (!bucket || now - bucket.windowStart >= RATE_LIMIT.windowMs) {
    bucket = { count: 0, windowStart: now };
    buckets.set(apiKey, bucket);
  }

  const resetMs = bucket.windowStart + RATE_LIMIT.windowMs;

  if (bucket.count >= RATE_LIMIT.maxRequests) {
    return { allowed: false, remaining: 0, resetMs };
  }

  bucket.count += 1;
  return {
    allowed: true,
    remaining: RATE_LIMIT.maxRequests - bucket.count,
    resetMs,
  };
}

export function rateLimitHeaders(result: RateLimitResult): HeadersInit {
  const retryAfter = Math.max(1, Math.ceil((result.resetMs - Date.now()) / 1000));
  return {
    'X-RateLimit-Limit': String(RATE_LIMIT.maxRequests),
    'X-RateLimit-Remaining': String(result.remaining),
    'X-RateLimit-Reset': String(Math.ceil(result.resetMs / 1000)),
    ...(result.allowed ? {} : { 'Retry-After': String(retryAfter) }),
  };
}
