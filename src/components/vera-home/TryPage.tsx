'use client';

import Navbar from '@/components/Navbar';
import TryVera from './TryVera';
import VeraFooter from './VeraFooter';
import { Body, H2 } from './ui';
import { useHomeLang, type L } from './useHomeLang';

export default function TryPage() {
  const view = useHomeLang();
  const t = (o: L) => o[view];
  const ja = view === 'ja';
  return (
    <main lang={view} className="relative min-h-screen" style={{ overflowX: 'clip', color: 'var(--ink)' }}>
      <Navbar />
      <section className="relative px-5 sm:px-6 pt-28 sm:pt-32 pb-12 sm:pb-16">
        <div className="mx-auto w-full max-w-4xl">
          <p
            className="uppercase mb-4"
            style={{ color: 'rgba(var(--accent-rgb), 0.9)', fontSize: '0.68rem', letterSpacing: '0.3em' }}
          >
            {t({ ja: '体験', en: 'Try it' })}
          </p>
          <H2 ja={ja}>{t({ ja: 'Vera を体験する', en: 'Try Vera' })}</H2>
          <Body ja={ja} className="mb-6">
            {t({
              ja: '固定の文書に問いを投げます。答えた問いには出所の文が付き、読めない問いには型つきで黙ります。',
              en: 'Ask questions about a fixed document. Answers come with the sentence they rest on; questions it cannot read get a typed silence.',
            })}
          </Body>
          <TryVera view={view} />
          <p className="mt-8" style={{ fontSize: '0.88rem' }}>
            <a href="/vera/about/" style={{ color: 'var(--accent)', textDecoration: 'none' }}>
              ← {t({ ja: 'Vera のホームへ', en: 'Back to Vera' })}
            </a>
          </p>
        </div>
      </section>
      <VeraFooter view={view} />
    </main>
  );
}
