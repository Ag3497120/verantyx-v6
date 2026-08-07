'use client';

/* The front door.
 *
 * It used to open on Verantyx-CLI — one project's hero, one project's install
 * block — which made the flagship the whole of what Verantyx is. The CLI has
 * had its own page for a while, so this one is now about the position the
 * projects share: systems that say what they do not know, in types, with the
 * line the answer came from.
 *
 * Every number here is measured and links to where it was measured. A
 * principles page that asserts principles is a manifesto; one that can be
 * checked is a claim.
 */

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import Logo from '@/components/Logo';
import { useLanguage } from '@/lib/i18n';
import { CATALOGUE, CATALOGUE_CHARS } from '@/lib/catalogue';

type L = { en: string; ja: string };

const PRINCIPLES: { n: string; title: L; body: L }[] = [
  {
    n: '01',
    title: {
      en: 'Not knowing is an answer, and it has a type',
      ja: '知らないことは答えであり、それには型がある',
    },
    body: {
      en: 'A system that always produces something produces something when it has nothing. Ours return UNKNOWN_NO_EVIDENCE, UNKNOWN_LOW_COVERAGE, UNKNOWN_DOMINANT_SOURCE — each naming what is missing, so the next step is a procedure rather than a matter of taste.',
      ja: '常に何かを返す仕組みは、何も無いときにも何かを返します。ここでは UNKNOWN_NO_EVIDENCE、UNKNOWN_LOW_COVERAGE、UNKNOWN_DOMINANT_SOURCE を返します。何が欠けているかを名指すので、次の一手が好みではなく手順になります。',
    },
  },
  {
    n: '02',
    title: {
      en: 'An answer names the line it came from',
      ja: '答えは、出典の行を名指す',
    },
    body: {
      en: 'Not a summary of the sources — the sentence itself, with the file. A person can then disagree with the machine, which is the only way anyone finds out it is wrong.',
      ja: '出典の要約ではなく、文そのものと、ファイル名。そうして初めて人が機械に反論でき、それが機械の誤りが見つかる唯一の道です。',
    },
  },
  {
    n: '03',
    title: {
      en: 'The same input gives the same output',
      ja: '同じ入力からは、同じ出力',
    },
    body: {
      en: 'No model sits in the answer path, so there is nothing to be non-deterministic about. A finding that changes between runs cannot be cited, and a finding that cannot be cited cannot carry a decision.',
      ja: '答えの経路にモデルが無いので、非決定的になる余地がありません。実行ごとに変わる所見は引用できず、引用できない所見は判断を支えられません。',
    },
  },
  {
    n: '04',
    title: {
      en: 'Limits are published, not discovered',
      ja: '限界は、見つけられる前に公開する',
    },
    body: {
      en: 'Every measurement here is stated with the corpus it was measured on, and every known gap is written down beside it. A number without its corpus is exactly the shape of claim these systems exist to refuse.',
      ja: 'すべての測定を、どのコーパスで測ったかとセットで書きます。既知の穴もその隣に書きます。コーパスを伴わない数字は、これらの仕組みが拒否する形そのものです。',
    },
  },
];

const MEASURED: { value: string; label: L; note: L; href: string }[] = [
  {
    value: '8 / 8',
    label: { en: 'recall, real documents', ja: '再現率(実文書)' },
    note: {
      en: 'Cabinet Office damage reports, 4 revisions',
      ja: '内閣府 被害状況速報 4版',
    },
    href: '/vera/#measured',
  },
  {
    value: '6 / 6',
    label: { en: 'recall, read blind', ja: '再現率(ブラインド)' },
    note: {
      en: 'MLIT series, no code changes, read after',
      ja: '国交省 系列。コード無変更、読んだのは後',
    },
    href: '/vera/#measured',
  },
  {
    value: '0',
    label: { en: 'false positives', ja: '誤検出' },
    note: {
      en: 'across both corpora, every finding read',
      ja: '両コーパス。全所見を人が照合',
    },
    href: '/vera/#measured',
  },
  {
    value: String(CATALOGUE.length),
    label: { en: 'repositories read', ja: '読んだリポジトリ' },
    note: {
      en: `${CATALOGUE_CHARS.toLocaleString()} characters of README`,
      ja: `README ${CATALOGUE_CHARS.toLocaleString()}字`,
    },
    href: '/catalogue/',
  },
];

const ENTRANCES: { title: string; body: L; href: string; external?: boolean }[] = [
  {
    title: 'Vera-α',
    body: {
      en: 'The engine. Documents in, disagreement out — measured on two government report series.',
      ja: 'エンジン本体。文書を入れ、食い違いを出す。2つの官庁報告系列で実測済み。',
    },
    href: '/vera/',
  },
  {
    title: 'Verantyx-CLI',
    body: {
      en: 'A resident local router that wakes larger models only when the task needs them.',
      ja: 'ローカル常駐ルーター。必要なときだけ大型モデルを起こします。',
    },
    href: '/verantyx-cli/',
  },
  {
    title: { en: 'Catalogue', ja: '図鑑' }.en,
    body: {
      en: 'Every repository, read by the engine itself and listed verbatim.',
      ja: '全リポジトリを、エンジン自身に読ませて原文のまま並べたもの。',
    },
    href: '/catalogue/',
  },
  {
    title: 'Apps',
    body: {
      en: 'The iOS side — mouth-controlled games and related projects.',
      ja: 'iOS 側。口の動きで遊ぶゲームと関連プロジェクト。',
    },
    href: '/apps/',
  },
];

export default function Home() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';
  const heroRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);

  return (
    <main lang={lang} className="relative text-white min-h-screen" style={{ overflowX: 'clip' }}>
      <Navbar />

      {/* ── Hero ───────────────────────────────────────────────── */}
      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity }}
        className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-16 sm:pb-24"
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 70% 50% at 50% 30%, rgba(var(--accent-rgb), 0.11), transparent 64%)',
          }}
        />
        <div className="mx-auto w-full max-w-4xl relative z-10 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="flex justify-center mb-7"
          >
            <Logo size="clamp(56px, 14vw, 88px)" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: 'clamp(2.6rem, 11vw, 6rem)',
              lineHeight: 1,
              letterSpacing: '-0.035em',
              marginBottom: '1.4rem',
            }}
          >
            Verantyx
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.22 }}
            className="mx-auto max-w-2xl"
            style={{
              color: 'var(--ink-2)',
              fontSize: 'clamp(1.02rem, 2.6vw, 1.35rem)',
              lineHeight: ja ? 1.95 : 1.62,
              fontWeight: 300,
            }}
          >
            {t({
              en: 'Systems that say what they do not know — in types, with the line the answer came from.',
              ja: '知らないことを、知らないと言う仕組み。型で、そして答えの出典の行とともに。',
            })}
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="mt-9 flex flex-wrap justify-center gap-3"
          >
            <a
              href="/vera/"
              className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {t({ en: 'What is measured', ja: '実測値を見る' })} →
            </a>
            <a
              href="/catalogue/"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{
                border: '1px solid var(--line-strong)',
                color: 'var(--ink-3)',
                textDecoration: 'none',
              }}
            >
              {t({ en: 'Catalogue', ja: '図鑑' })}
            </a>
          </motion.div>
        </div>
      </motion.section>

      {/* ── Measured ───────────────────────────────────────────── */}
      <Section label={t({ en: 'Measured', ja: '実測値' })}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
          {MEASURED.map((m) => (
            <a
              key={m.label.en}
              href={m.href}
              className="rounded-2xl border p-4 sm:p-5 block"
              style={{
                borderColor: 'var(--line)',
                background: 'var(--surface)',
                textDecoration: 'none',
                transition: 'border-color 0.25s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--line)';
              }}
            >
              <div
                className="font-display font-bold gradient-brand"
                style={{ fontSize: 'clamp(1.5rem, 5vw, 2rem)', lineHeight: 1.1 }}
              >
                {m.value}
              </div>
              <div
                className="mt-2 font-semibold"
                style={{ color: 'var(--ink-2)', fontSize: 'clamp(0.74rem, 1.9vw, 0.82rem)' }}
              >
                {t(m.label)}
              </div>
              <div
                className="mt-1.5"
                style={{
                  color: 'var(--ink-4)',
                  fontSize: 'clamp(0.68rem, 1.7vw, 0.74rem)',
                  lineHeight: ja ? 1.8 : 1.5,
                }}
              >
                {t(m.note)}
              </div>
            </a>
          ))}
        </div>
      </Section>

      {/* ── Principles ─────────────────────────────────────────── */}
      <Section label={t({ en: 'The position', ja: '立場' })}>
        <div className="grid gap-4 sm:gap-5 md:grid-cols-2">
          {PRINCIPLES.map((p) => (
            <div
              key={p.n}
              className="rounded-2xl border p-5 sm:p-6"
              style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}
            >
              <div
                className="font-mono mb-3"
                style={{
                  color: 'rgba(var(--accent-rgb), 0.8)',
                  fontSize: '0.7rem',
                  letterSpacing: '0.16em',
                }}
              >
                {p.n}
              </div>
              <h3
                className="font-semibold mb-3"
                style={{
                  color: 'var(--ink)',
                  fontSize: 'clamp(0.98rem, 2.5vw, 1.12rem)',
                  lineHeight: ja ? 1.6 : 1.32,
                }}
              >
                {t(p.title)}
              </h3>
              <p
                style={{
                  color: 'var(--ink-3)',
                  fontSize: 'clamp(0.85rem, 2.2vw, 0.92rem)',
                  lineHeight: ja ? 1.95 : 1.68,
                }}
              >
                {t(p.body)}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Where to go ────────────────────────────────────────── */}
      <Section label={t({ en: 'Where to go', ja: '入口' })}>
        <div className="grid gap-4 sm:gap-5 sm:grid-cols-2">
          {ENTRANCES.map((e) => (
            <a
              key={e.href}
              href={e.href}
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
              <div
                className="font-display font-bold mb-2"
                style={{ color: 'var(--ink)', fontSize: 'clamp(1rem, 2.6vw, 1.15rem)' }}
              >
                {e.title}
              </div>
              <p
                style={{
                  color: 'var(--ink-3)',
                  fontSize: 'clamp(0.83rem, 2.1vw, 0.9rem)',
                  lineHeight: ja ? 1.9 : 1.62,
                }}
              >
                {t(e.body)}
              </p>
            </a>
          ))}
        </div>
      </Section>

      {/* ── What this is not ───────────────────────────────────── */}
      <Section label={t({ en: 'What this is not', ja: 'これでないもの' })}>
        <p
          className="max-w-2xl"
          style={{
            color: 'var(--ink-3)',
            fontSize: 'clamp(0.9rem, 2.3vw, 1rem)',
            lineHeight: ja ? 1.95 : 1.72,
          }}
        >
          {t({
            en: 'None of this writes for you. No free-form prose, no summarisation, no translation, no open-domain chat. That is not a weakness being worked on — it is the trade that buys the rest: a system that will not compose a sentence also cannot compose a fact.',
            ja: 'ここにあるものは、あなたの代わりに文章を書きません。自由作文も要約も翻訳も雑談もしません。これは改善中の弱点ではなく、残り全部を買うための取引です。文を作らない仕組みは、事実も作れません。',
          })}
        </p>
      </Section>

      <Footer />
    </main>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="relative px-5 sm:px-6 py-12 sm:py-16 md:py-20">
      <div className="mx-auto w-full max-w-4xl">
        <div className="flex items-center gap-4 mb-6 sm:mb-8">
          <span
            className="h-px w-8 sm:w-12 shrink-0"
            style={{
              background:
                'linear-gradient(90deg, rgba(var(--accent-rgb), 0.5), transparent)',
            }}
          />
          <span
            className="uppercase font-semibold"
            style={{
              color: 'var(--ink-4)',
              fontSize: '0.66rem',
              letterSpacing: '0.26em',
            }}
          >
            {label}
          </span>
        </div>
        {children}
      </div>
    </section>
  );
}
