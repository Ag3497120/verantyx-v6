import type { Env } from './_lib/types';
import { APPLE_MUSIC_API } from './_lib/types';
import { getDeveloperToken } from './_lib/jwt';
import { json } from './_lib/response';

const ALLOWED_TYPES = new Set([
  'activities',
  'albums',
  'apple-curators',
  'artists',
  'curators',
  'music-videos',
  'playlists',
  'record-labels',
  'songs',
  'stations',
]);

/**
 * Catalog search proxy.
 * GET /api/apple-music/search?term=&types=&storefront=
 *
 * Requires x-api-key. Catalog-only (developer token). No Music User Token.
 */
export const onRequestGet: PagesFunction<Env> = async (context) => {
  const url = new URL(context.request.url);
  const term = url.searchParams.get('term')?.trim() ?? '';
  const storefront = (url.searchParams.get('storefront')?.trim() || 'us').toLowerCase();
  const typesRaw = url.searchParams.get('types')?.trim() || 'songs,albums,artists';

  if (!term) {
    return json(
      {
        error: 'Missing required query parameter: term',
        example: '/api/apple-music/search?term=beatles&types=songs,albums&storefront=us',
      },
      { status: 400 },
    );
  }

  if (!/^[a-z]{2}$/.test(storefront)) {
    return json(
      { error: 'Invalid storefront. Use a 2-letter country code (e.g. us, jp).' },
      { status: 400 },
    );
  }

  const types = typesRaw
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);

  if (types.length === 0 || types.some((t) => !ALLOWED_TYPES.has(t))) {
    return json(
      {
        error: 'Invalid types parameter',
        allowed: [...ALLOWED_TYPES],
      },
      { status: 400 },
    );
  }

  let developerToken: string;
  try {
    developerToken = await getDeveloperToken(context.env);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Token generation failed';
    return json({ error: 'Apple Music credentials misconfigured', detail: message }, { status: 503 });
  }

  const upstream = new URL(`${APPLE_MUSIC_API}/catalog/${storefront}/search`);
  upstream.searchParams.set('term', term);
  upstream.searchParams.set('types', types.join(','));

  const limit = url.searchParams.get('limit');
  if (limit) {
    const n = Number(limit);
    if (Number.isFinite(n) && n >= 1 && n <= 25) {
      upstream.searchParams.set('limit', String(Math.floor(n)));
    }
  }

  const offset = url.searchParams.get('offset');
  if (offset && /^\d+$/.test(offset)) {
    upstream.searchParams.set('offset', offset);
  }

  let appleRes: Response;
  try {
    appleRes = await fetch(upstream.toString(), {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${developerToken}`,
        Accept: 'application/json',
      },
    });
  } catch {
    return json({ error: 'Failed to reach Apple Music API' }, { status: 502 });
  }

  const bodyText = await appleRes.text();
  const headers = new Headers({
    'Content-Type': appleRes.headers.get('Content-Type') || 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });

  // Pass through Apple status; do not leak Authorization.
  return new Response(bodyText, {
    status: appleRes.status,
    headers,
  });
};
