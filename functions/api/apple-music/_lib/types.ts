export interface Env {
  APPLE_TEAM_ID: string;
  APPLE_MUSIC_KEY_ID: string;
  APPLE_MUSIC_PRIVATE_KEY: string;
  /** Comma-separated friend API keys */
  FRIEND_API_KEYS: string;
}

export const APPLE_MUSIC_API = 'https://api.music.apple.com/v1';

/** Per-key rate limit (in-memory per CF isolate). */
export const RATE_LIMIT = {
  windowMs: 60_000,
  maxRequests: 60,
} as const;

/** Developer token lifetime (~6 months) with refresh margin. */
export const TOKEN = {
  /** Apple allows up to 6 months; we use ~5.5 months. */
  ttlSeconds: 15768000,
  /** Refresh when fewer than 7 days remain. */
  refreshMarginSeconds: 7 * 24 * 60 * 60,
} as const;
