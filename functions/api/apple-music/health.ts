import type { Env } from './_lib/types';
import { parseFriendApiKeys } from './_lib/auth';
import { appleCredentialsConfigured } from './_lib/jwt';
import { RATE_LIMIT } from './_lib/types';
import { json } from './_lib/response';

/**
 * Public health check — never returns secrets or key material.
 * GET /api/apple-music/health
 */
export const onRequestGet: PagesFunction<Env> = async (context) => {
  const friendKeysConfigured = parseFriendApiKeys(context.env).size > 0;
  const appleConfigured = appleCredentialsConfigured(context.env);

  return json({
    ok: true,
    service: 'apple-music-proxy',
    catalogOnly: true,
    appleCredentialsConfigured: appleConfigured,
    friendApiKeysConfigured: friendKeysConfigured,
    rateLimit: {
      maxRequests: RATE_LIMIT.maxRequests,
      windowMs: RATE_LIMIT.windowMs,
      note: 'In-memory per Cloudflare isolate; not global across all edges.',
    },
    endpoints: {
      search: 'GET /api/apple-music/search?term=&types=&storefront=',
      health: 'GET /api/apple-music/health',
    },
  });
};
