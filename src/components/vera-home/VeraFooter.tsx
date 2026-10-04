'use client';

import type { HomeLang, L } from './useHomeLang';

/* Own footer for the Vera home and /try/. The shared Footer links "Cleanroom / Home" to /,
 * which is no longer Cleanroom; it cannot be edited in this change, so it is not used here.
 * Plain <a> everywhere: /cleanroom/ is a separately built app and must be a full page load. */

const LINKS: { href: string; label: L; external?: boolean }[] = [
  { href: '/', label: { ja: 'Vera', en: 'Vera' } },
  { href: '/try/', label: { ja: '体験', en: 'Try it' } },
  { href: '/vera/', label: { ja: 'Vera の測定', en: 'Measurements' } },
  { href: '/cleanroom/', label: { ja: 'Cleanroom', en: 'Cleanroom' } },
  { href: '/apps/', label: { ja: 'Apps', en: 'Apps' } },
  { href: '/author/', label: { ja: 'Author', en: 'Author' } },
  { href: '/privacy/', label: { ja: 'Privacy', en: 'Privacy' } },
  { href: '/support/', label: { ja: 'Support', en: 'Support' } },
  { href: '/terms/', label: { ja: 'Terms', en: 'Terms' } },
  { href: 'https://github.com/Ag3497120/vera', label: { ja: 'GitHub', en: 'GitHub' }, external: true },
];

export default function VeraFooter({ view }: { view: HomeLang }) {
  return (
    <footer className="relative px-5 sm:px-6 py-14 sm:py-16">
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(var(--accent-rgb), 0.28), transparent)',
        }}
      />
      <div className="mx-auto w-full max-w-4xl flex flex-col md:flex-row md:items-end md:justify-between gap-6">
        <div>
          <div
            className="font-display font-extrabold"
            style={{
              fontSize: 'clamp(1.6rem, 5vw, 2.1rem)',
              background: 'linear-gradient(135deg, var(--accent), var(--accent-2))',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Verantyx
          </div>
          <p className="mt-1" style={{ color: 'var(--ink-4)', fontSize: '0.8rem' }}>
            {view === 'ja' ? 'Vera — 嘘をつけない枠をつくる AI' : 'Vera — an AI that builds a frame it cannot lie in'}
          </p>
        </div>
        <nav className="flex flex-wrap gap-x-4 gap-y-2" aria-label="footer">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              {...(l.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
              style={{ color: 'var(--ink-3)', fontSize: '0.84rem', textDecoration: 'none' }}
            >
              {l.label[view]}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  );
}
