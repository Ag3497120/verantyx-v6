import type { Env } from './types';
import { TOKEN } from './types';

type CachedToken = {
  token: string;
  exp: number;
};

let cached: CachedToken | null = null;

function base64UrlEncode(data: ArrayBuffer | Uint8Array | string): string {
  let bytes: Uint8Array;
  if (typeof data === 'string') {
    bytes = new TextEncoder().encode(data);
  } else if (data instanceof Uint8Array) {
    bytes = data;
  } else {
    bytes = new Uint8Array(data);
  }
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function normalizePem(pem: string): string {
  // Dashboard / .dev.vars often store literal \n sequences.
  return pem.replace(/\\n/g, '\n').trim();
}

async function importEcPrivateKey(pem: string): Promise<CryptoKey> {
  const normalized = normalizePem(pem);
  const body = normalized
    .replace(/-----BEGIN PRIVATE KEY-----/g, '')
    .replace(/-----END PRIVATE KEY-----/g, '')
    .replace(/-----BEGIN EC PRIVATE KEY-----/g, '')
    .replace(/-----END EC PRIVATE KEY-----/g, '')
    .replace(/\s+/g, '');

  const der = Uint8Array.from(atob(body), (c) => c.charCodeAt(0));

  return crypto.subtle.importKey(
    'pkcs8',
    der,
    { name: 'ECDSA', namedCurve: 'P-256' },
    false,
    ['sign'],
  );
}

async function signEs256Jwt(
  header: Record<string, string>,
  payload: Record<string, string | number>,
  privateKeyPem: string,
): Promise<string> {
  const encodedHeader = base64UrlEncode(JSON.stringify(header));
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const signingInput = `${encodedHeader}.${encodedPayload}`;

  const key = await importEcPrivateKey(privateKeyPem);
  const signature = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' },
    key,
    new TextEncoder().encode(signingInput),
  );

  return `${signingInput}.${base64UrlEncode(signature)}`;
}

function requireEnv(env: Env): void {
  if (!env.APPLE_TEAM_ID?.trim()) throw new Error('APPLE_TEAM_ID is not configured');
  if (!env.APPLE_MUSIC_KEY_ID?.trim()) throw new Error('APPLE_MUSIC_KEY_ID is not configured');
  if (!env.APPLE_MUSIC_PRIVATE_KEY?.trim()) {
    throw new Error('APPLE_MUSIC_PRIVATE_KEY is not configured');
  }
}

/**
 * Returns a cached Apple Music developer token (JWT ES256).
 * Regenerates when missing or within the refresh margin of expiry.
 */
export async function getDeveloperToken(env: Env): Promise<string> {
  requireEnv(env);

  const now = Math.floor(Date.now() / 1000);
  if (cached && cached.exp - TOKEN.refreshMarginSeconds > now) {
    return cached.token;
  }

  const iat = now;
  const exp = iat + TOKEN.ttlSeconds;

  const token = await signEs256Jwt(
    { alg: 'ES256', kid: env.APPLE_MUSIC_KEY_ID.trim() },
    { iss: env.APPLE_TEAM_ID.trim(), iat, exp },
    env.APPLE_MUSIC_PRIVATE_KEY,
  );

  cached = { token, exp };
  return token;
}

export function appleCredentialsConfigured(env: Env): boolean {
  return Boolean(
    env.APPLE_TEAM_ID?.trim() &&
      env.APPLE_MUSIC_KEY_ID?.trim() &&
      env.APPLE_MUSIC_PRIVATE_KEY?.trim(),
  );
}

/** Test helper: clear in-memory token cache (local only). */
export function clearTokenCache(): void {
  cached = null;
}
