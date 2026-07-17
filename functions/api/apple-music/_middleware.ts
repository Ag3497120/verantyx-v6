import type { Env } from './_lib/types';
import { authorizeFriend } from './_lib/auth';
import { checkRateLimit, rateLimitHeaders } from './_lib/rate-limit';
import { corsHeaders, json, withCors } from './_lib/response';

/**
 * Auth + rate-limit for /api/apple-music/* except health (public, no secrets).
 */
export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, next } = context;
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, '') || '/';

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(request) });
  }

  const isHealth = path.endsWith('/apple-music/health');
  if (isHealth) {
    const response = await next();
    return withCors(response, request);
  }

  if (request.method !== 'GET') {
    return withCors(
      json({ error: 'Method not allowed' }, { status: 405 }),
      request,
    );
  }

  const auth = authorizeFriend(request, context.env);
  if (!auth.ok) {
    return withCors(json({ error: auth.error }, { status: auth.status }), request);
  }

  const limit = checkRateLimit(auth.key);
  if (!limit.allowed) {
    return withCors(
      json(
        {
          error: 'Rate limit exceeded',
          limit: 60,
          window: '1 minute per API key (per isolate)',
        },
        { status: 429, headers: rateLimitHeaders(limit) },
      ),
      request,
    );
  }

  const response = await next();
  const headers = new Headers(response.headers);
  for (const [k, v] of Object.entries(rateLimitHeaders(limit))) {
    headers.set(k, v);
  }
  for (const [k, v] of Object.entries(corsHeaders(request))) {
    headers.set(k, v);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
};
