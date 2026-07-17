'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

const BASE = 'https://verantyx.ai/api/apple-music';

const CURL_SEARCH = `curl -sS "${BASE}/search?term=beatles&types=songs,albums&storefront=us" \\
  -H "x-api-key: YOUR_FRIEND_API_KEY"`;

const CURL_HEALTH = `curl -sS "${BASE}/health"`;

export default function AppleMusicApiPage() {
  const { lang } = useLanguage();
  const t = (o: { en: string; ja: string }) => o[lang];

  return (
    <main className="relative text-white overflow-x-hidden min-h-screen">
      <Navbar />

      <section className="relative px-6 pt-32 pb-16">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 60% 45% at 50% 20%, rgba(var(--accent-rgb), 0.1), transparent 70%)',
          }}
        />
        <div className="max-w-3xl mx-auto relative z-10">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xs tracking-[0.35em] uppercase mb-5"
            style={{ color: 'rgba(var(--accent-rgb), 0.85)' }}
          >
            Docs · Friends API · Cloudflare Pages Functions
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 20, filter: 'blur(6px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ duration: 0.75 }}
            className="font-display text-4xl md:text-6xl font-extrabold tracking-tight gradient-brand mb-5"
          >
            Apple Music API
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="text-lg font-light leading-relaxed text-slate-300"
          >
            {t({
              en: 'Catalog-only proxy for friends. Your app calls verantyx.ai; the edge holds the Apple developer token. Never share the .p8 private key.',
              ja: '友人向けカタログ専用プロキシ。アプリは verantyx.ai を呼び、エッジが Apple 開発者トークンを保持します。.p8 秘密鍵は絶対に共有しないでください。',
            })}
          </motion.p>
        </div>
      </section>

      <section className="px-6 pb-24">
        <div className="max-w-3xl mx-auto space-y-14">
          <DocBlock
            title={t({ en: 'Architecture', ja: 'アーキテクチャ' })}
            body={t({
              en: 'Friend app → https://verantyx.ai/api/apple-music/* (API key + developer JWT) → api.music.apple.com. Implemented as Cloudflare Pages Functions — not Next.js API routes (this site is a static export).',
              ja: '友人アプリ → https://verantyx.ai/api/apple-music/*（API キー + 開発者 JWT）→ api.music.apple.com。Cloudflare Pages Functions で実装（静的エクスポートのため Next.js API ルートは使えません）。',
            })}
          />

          <DocBlock
            title={t({ en: 'Security', ja: 'セキュリティ' })}
            body={t({
              en: 'Rotate FRIEND_API_KEYS if a key leaks. Never commit .p8 files or paste them into client apps. This proxy is catalog-only by default: search and metadata only. Playback and library APIs need a Music User Token from MusicKit on the user device — that token is not issued or stored here.',
              ja: 'キー漏洩時は FRIEND_API_KEYS をローテーション。.p8 をリポジトリやクライアントに置かないこと。既定はカタログ専用（検索・メタデータ）。再生やライブラリには端末上の MusicKit が発行する Music User Token が必要で、このプロキシでは発行・保管しません。',
            })}
          />

          <div>
            <h2 className="font-display text-2xl font-bold mb-3">Endpoints</h2>
            <ul className="space-y-3 text-sm text-slate-300">
              <li>
                <code className="text-[rgba(var(--accent-rgb),0.95)]">GET /api/apple-music/search</code>
                <span className="text-slate-500"> — </span>
                {t({
                  en: 'Query params: term (required), types (default songs,albums,artists), storefront (default us), optional limit (1–25), offset. Header: x-api-key.',
                  ja: 'クエリ: term（必須）、types（既定 songs,albums,artists）、storefront（既定 us）、任意で limit（1–25）、offset。ヘッダ: x-api-key。',
                })}
              </li>
              <li>
                <code className="text-[rgba(var(--accent-rgb),0.95)]">GET /api/apple-music/health</code>
                <span className="text-slate-500"> — </span>
                {t({
                  en: 'Public. Confirms the service is up and whether credentials are configured — never returns secrets.',
                  ja: '公開。稼働と認証情報の有無のみ。秘密は返しません。',
                })}
              </li>
            </ul>
          </div>

          <div>
            <h2 className="font-display text-2xl font-bold mb-3">
              {t({ en: 'Rate limits', ja: 'レート制限' })}
            </h2>
            <p className="text-slate-300 text-sm leading-relaxed">
              {t({
                en: '60 requests per API key per rolling 60-second window, counted in-memory per Cloudflare isolate (not a global edge-wide counter). Responses include X-RateLimit-* headers; 429 includes Retry-After.',
                ja: 'API キーごとに 60 秒あたり 60 リクエスト。Cloudflare isolate 内のインメモリ集計（全エッジ共通ではありません）。応答に X-RateLimit-*、429 時は Retry-After。',
              })}
            </p>
          </div>

          <div>
            <h2 className="font-display text-2xl font-bold mb-4">
              {t({ en: 'Example curl', ja: 'curl 例' })}
            </h2>
            <CodeBlock label="Search" code={CURL_SEARCH} />
            <div className="h-4" />
            <CodeBlock label="Health" code={CURL_HEALTH} />
          </div>

          <div>
            <h2 className="font-display text-2xl font-bold mb-3">
              {t({ en: 'Cloudflare secrets', ja: 'Cloudflare シークレット' })}
            </h2>
            <p className="text-slate-300 text-sm leading-relaxed mb-4">
              {t({
                en: 'Pages → your project → Settings → Environment variables (Production). Mark secrets as encrypted. Set:',
                ja: 'Pages → プロジェクト → Settings → Environment variables（Production）。秘密は Encrypted で設定:',
              })}
            </p>
            <ul className="text-sm text-slate-400 space-y-2 font-mono">
              <li>APPLE_TEAM_ID</li>
              <li>APPLE_MUSIC_KEY_ID</li>
              <li>APPLE_MUSIC_PRIVATE_KEY</li>
              <li>FRIEND_API_KEYS</li>
            </ul>
            <p className="text-slate-500 text-xs mt-4 leading-relaxed">
              {t({
                en: 'For APPLE_MUSIC_PRIVATE_KEY, paste the full PEM. Newlines may be stored as \\n. See .env.example in the repo. Never commit real .p8 files.',
                ja: 'APPLE_MUSIC_PRIVATE_KEY には PEM 全文を。改行は \\n でも可。リポジトリの .env.example を参照。本物の .p8 はコミットしないこと。',
              })}
            </p>
          </div>

          <div>
            <h2 className="font-display text-2xl font-bold mb-3">
              {t({ en: 'Local test', ja: 'ローカル検証' })}
            </h2>
            <CodeBlock
              label="wrangler"
              code={`cp .env.example .dev.vars
# edit .dev.vars with real values (gitignored)

npm run build
npx wrangler pages dev out --compatibility-date=2026-07-15

curl -sS "http://127.0.0.1:8788/api/apple-music/health"
curl -sS "http://127.0.0.1:8788/api/apple-music/search?term=beatles&types=songs&storefront=us" \\
  -H "x-api-key: friend-key-one"`}
            />
          </div>

          <DocBlock
            title={t({ en: 'Music User Token', ja: 'Music User Token' })}
            body={t({
              en: 'Developer tokens authorize catalog access. User-specific playback, library, and recommendations require a Music User Token from Apple MusicKit JS / native MusicKit after the user signs in. Do not send user tokens to this proxy unless you intentionally extend it — today it only attaches the server developer JWT.',
              ja: '開発者トークンはカタログ向け。ユーザー固有の再生・ライブラリ・レコメンドには MusicKit サインイン後の Music User Token が必要です。このプロキシは現状サーバー側の開発者 JWT のみを付与します。',
            })}
          />
        </div>
      </section>

      <Footer />
    </main>
  );
}

function DocBlock({ title, body }: { title: string; body: string }) {
  return (
    <div>
      <h2 className="font-display text-2xl font-bold mb-3">{title}</h2>
      <p className="text-slate-300 text-sm leading-relaxed">{body}</p>
    </div>
  );
}

function CodeBlock({ label, code }: { label: string; code: string }) {
  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        border: '1px solid rgba(var(--accent-rgb), 0.18)',
        background: 'rgba(10, 10, 20, 0.85)',
      }}
    >
      <div
        className="px-4 py-2 text-[11px] tracking-widest uppercase text-slate-500"
        style={{ borderBottom: '1px solid rgba(148,163,184,0.12)' }}
      >
        {label}
      </div>
      <pre className="p-4 text-xs md:text-sm text-slate-200 overflow-x-auto leading-relaxed whitespace-pre-wrap break-all">
        {code}
      </pre>
    </div>
  );
}
