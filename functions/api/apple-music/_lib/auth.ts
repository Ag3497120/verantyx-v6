import type { Env } from './types';

export function parseFriendApiKeys(env: Env): Set<string> {
  const raw = env.FRIEND_API_KEYS ?? '';
  return new Set(
    raw
      .split(',')
      .map((k) => k.trim())
      .filter(Boolean),
  );
}

export function extractApiKey(request: Request): string | null {
  const header = request.headers.get('x-api-key')?.trim();
  if (header) return header;

  const auth = request.headers.get('authorization');
  if (auth?.toLowerCase().startsWith('bearer ')) {
    const token = auth.slice(7).trim();
    return token || null;
  }

  return null;
}

export function authorizeFriend(request: Request, env: Env): { ok: true; key: string } | { ok: false; status: number; error: string } {
  const keys = parseFriendApiKeys(env);
  if (keys.size === 0) {
    return { ok: false, status: 503, error: 'FRIEND_API_KEYS is not configured' };
  }

  const key = extractApiKey(request);
  if (!key) {
    return {
      ok: false,
      status: 401,
      error: 'Missing API key. Send x-api-key header (or Authorization: Bearer <key>).',
    };
  }

  if (!keys.has(key)) {
    return { ok: false, status: 403, error: 'Invalid API key' };
  }

  return { ok: true, key };
}
