'use client';

/* The catalogue — every repository, read by the engine the site is about.
 *
 * Nothing on this page is written for it. Summaries are the first real
 * sentence of each README, topics are the cores Vera extracted ranked by mass
 * within that document, and both are shown as such. A catalogue whose entries
 * were rewritten to sound good would be a catalogue of the writing, not of
 * the work.
 */

import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';
import { CATALOGUE, CATALOGUE_CHARS } from '@/lib/catalogue';

type L = { en: string; ja: string };

export default function CataloguePage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';
  const [query, setQuery] = useState('');

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = [...CATALOGUE].sort(
      (a, b) => b.stars - a.stars || a.name.localeCompare(b.name)
    );
    if (!q) return rows;
    return rows.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.description.toLowerCase().includes(q) ||
        e.summary.toLowerCase().includes(q) ||
        e.topics.some((topic) => topic.toLowerCase().includes(q))
    );
  }, [query]);

  return (
    <main lang={lang} className="relative text-white min-h-screen" style={{ overflowX: 'clip' }}>
      <Navbar />

      <section className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-10 sm:pb-14">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 70% 55% at 30% 25%, rgba(var(--accent-rgb), 0.10), transparent 62%)',
          }}
        />
        <div className="mx-auto w-full max-w-4xl relative z-10">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="uppercase mb-5"
            style={{
              color: 'rgba(var(--accent-rgb), 0.9)',
              fontSize: 'clamp(0.62rem, 1.6vw, 0.72rem)',
              letterSpacing: '0.3em',
            }}
          >
            {t({ en: 'Catalogue', ja: '図鑑' })}
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.05 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: ja
                ? 'clamp(1.9rem, 6.5vw, 3.5rem)'
                : 'clamp(2.1rem, 7.5vw, 4rem)',
              lineHeight: ja ? 1.25 : 1.05,
              letterSpacing: ja ? '-0.01em' : '-0.03em',
              marginBottom: '1rem',
            }}
          >
            {t({ en: 'Every repository, read', ja: '全リポジトリを、読んだ' })}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.14 }}
            className="max-w-2xl"
            style={{
              color: 'var(--ink-2)',
              fontSize: 'clamp(0.95rem, 2.3vw, 1.1rem)',
              lineHeight: ja ? 1.95 : 1.7,
              fontWeight: 300,
            }}
          >
            {t({
              en: `${CATALOGUE.length} repositories, ${CATALOGUE_CHARS.toLocaleString()} characters of README run through Vera itself — the same loaders, the same splitter, the same segmentation it uses on anything else.`,
              ja: `${CATALOGUE.length} リポジトリ、README ${CATALOGUE_CHARS.toLocaleString()}字を Vera 自身に読ませました。他のコーパスと同じローダー、同じ分割、同じ形態処理です。`,
            })}
          </motion.p>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.24 }}
            className="max-w-2xl mt-4"
            style={{
              color: 'var(--ink-4)',
              fontSize: 'clamp(0.8rem, 2vw, 0.875rem)',
              lineHeight: ja ? 1.9 : 1.65,
            }}
          >
            {t({
              en: 'Every line below is verbatim. Summaries are the first real sentence of the README; topics are the cores the engine extracted, ranked by mass within that document. Nothing was rewritten to read better.',
              ja: 'すべて原文のままです。要約は README の最初の実文、話題はエンジンが抽出したコアをその文書内の重みで並べたもの。読みやすくするための書き換えは一切していません。',
            })}
          </motion.p>
        </div>
      </section>

      {/* Filter */}
      <section className="relative px-5 sm:px-6 pb-6">
        <div className="mx-auto w-full max-w-4xl">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t({
              en: 'Filter by name, topic or subject',
              ja: '名前・話題・分野で絞り込む',
            })}
            style={{
              width: '100%',
              padding: '11px 15px',
              borderRadius: 12,
              border: '1px solid var(--line-strong)',
              background: 'var(--surface-2)',
              color: 'var(--ink)',
              fontSize: '0.9rem',
              outline: 'none',
            }}
          />
          <div
            style={{
              marginTop: 8,
              fontSize: '0.74rem',
              color: 'var(--ink-4)',
            }}
          >
            {shown.length} / {CATALOGUE.length}
          </div>
        </div>
      </section>

      {/* Entries */}
      <section className="relative px-5 sm:px-6 pb-16 sm:pb-24">
        <div className="mx-auto w-full max-w-4xl grid gap-4 sm:gap-5">
          {shown.map((e) => (
            <a
              key={e.name}
              href={e.url}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-2xl border p-5 sm:p-6 block"
              style={{
                borderColor: 'var(--line)',
                background: 'var(--surface)',
                textDecoration: 'none',
                transition: 'border-color 0.25s ease, transform 0.25s ease',
              }}
              onMouseEnter={(ev) => {
                ev.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.4)';
                ev.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(ev) => {
                ev.currentTarget.style.borderColor = 'var(--line)';
                ev.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-2">
                <span
                  className="font-display font-bold"
                  style={{
                    color: 'var(--ink)',
                    fontSize: 'clamp(1rem, 2.6vw, 1.15rem)',
                  }}
                >
                  {e.name}
                </span>
                {e.language && (
                  <span
                    style={{
                      fontSize: '0.68rem',
                      color: 'var(--ink-4)',
                      border: '1px solid var(--line)',
                      borderRadius: 999,
                      padding: '2px 8px',
                    }}
                  >
                    {e.language}
                  </span>
                )}
                <span
                  className="ml-auto font-mono shrink-0"
                  style={{ fontSize: '0.7rem', color: 'var(--ink-4)' }}
                >
                  {e.sentences} {t({ en: 'sentences', ja: '文' })}
                </span>
              </div>

              <p
                style={{
                  color: 'var(--ink-2)',
                  fontSize: 'clamp(0.85rem, 2.2vw, 0.94rem)',
                  lineHeight: ja ? 1.9 : 1.65,
                  marginBottom: e.topics.length ? 12 : 0,
                  overflowWrap: 'anywhere',
                }}
              >
                {e.description || e.summary}
              </p>

              {e.topics.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {e.topics.slice(0, 9).map((topic) => (
                    <span
                      key={topic}
                      style={{
                        fontSize: '0.7rem',
                        color: 'var(--ink-3)',
                        border: '1px solid var(--line)',
                        borderRadius: 7,
                        padding: '3px 8px',
                        overflowWrap: 'anywhere',
                      }}
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              )}
            </a>
          ))}

          {shown.length === 0 && (
            <div
              className="rounded-2xl border p-6 text-center"
              style={{ borderColor: 'var(--line)', color: 'var(--ink-4)' }}
            >
              {t({
                en: 'Nothing in the catalogue matches that.',
                ja: '図鑑に該当するものがありません。',
              })}
            </div>
          )}
        </div>
      </section>

      <Footer />
    </main>
  );
}
