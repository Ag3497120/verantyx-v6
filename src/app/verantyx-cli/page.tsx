'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import StickyCliCta from '@/components/StickyCliCta';
import { useLanguage } from '@/lib/i18n';

const GITHUB = 'https://github.com/Ag3497120/verantyx-cli';
const RELEASE = 'https://github.com/Ag3497120/verantyx-cli/releases/tag/v3.0.0-alpha';
const DOCS = '/verantyx-cli-docs.html';

const FEATURES = [
  {
    title: { en: 'Resident 0.5B router', ja: '0.5B 常駐ルーター' },
    body: {
      en: 'Classify-only routing stays local and cheap. Larger models wake only when the task needs them.',
      ja: '分類専用のルーティングをローカルに常駐。必要なときだけ大型モデルを起こします。',
    },
  },
  {
    title: { en: 'Vector council', ja: 'ベクトル評議会' },
    body: {
      en: 'Optional multi-voice deliberation for harder prompts — without pretending structure invents world knowledge.',
      ja: '難しいプロンプト向けの任意の合議。構造が世界知識を生むとは主張しません。',
    },
  },
  {
    title: { en: 'Eternal memory', ja: '永遠の記憶' },
    body: {
      en: 'Carry durable context across restarts so the harness improves overnight instead of amnesia each boot.',
      ja: '再起動をまたいで記憶を保持。起動のたびに忘れるのではなく、一晩かけて改善します。',
    },
  },
  {
    title: { en: 'Honest benchmarks', ja: '誠実なベンチマーク' },
    body: {
      en: 'Claim boundaries live in-repo. This is a local harness — not a magic accuracy booster for tiny models.',
      ja: '主張の境界はリポジトリに公開。小さなモデルを魔法で強くするツールではありません。',
    },
  },
];

const ARCH = [
  {
    step: '01',
    title: { en: 'Ingest', ja: '入力' },
    body: {
      en: 'CLI / agent tools accept prompts and tool calls with local-first defaults.',
      ja: 'CLI / エージェントツールが、ローカル優先のデフォルトで入力を受け付けます。',
    },
  },
  {
    step: '02',
    title: { en: 'Route', ja: 'ルーティング' },
    body: {
      en: 'The resident router classifies — it does not invent answers.',
      ja: '常駐ルーターは分類のみ。答えを捏造しません。',
    },
  },
  {
    step: '03',
    title: { en: 'Escalate', ja: 'エスカレート' },
    body: {
      en: 'Wake larger local models, council, or tools only when needed.',
      ja: '必要時だけ大型ローカルモデル・評議会・ツールを起動。',
    },
  },
  {
    step: '04',
    title: { en: 'Remember', ja: '記憶' },
    body: {
      en: 'Feedback loops and eternal memory feed the next session.',
      ja: 'フィードバックと永遠の記憶が次のセッションへつながります。',
    },
  },
];

export default function VerantyxCLIPage() {
  const { lang } = useLanguage();
  const t = (o: { en: string; ja: string }) => o[lang];

  return (
    <main className="relative text-white overflow-x-hidden min-h-screen">
      <Navbar />
      <StickyCliCta />

      {/* Hero */}
      <section className="relative min-h-[78vh] flex items-center px-6 pt-32 pb-20">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 65% 50% at 40% 40%, rgba(var(--accent-rgb), 0.12), transparent 65%)',
          }}
        />
        <div className="max-w-5xl mx-auto relative z-10 w-full">
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-xs tracking-[0.35em] uppercase mb-6"
            style={{ color: 'rgba(var(--accent-rgb), 0.85)' }}
          >
            Flagship · Open Source · v3.0.0-alpha
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 24, filter: 'blur(6px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ duration: 0.85, delay: 0.1 }}
            className="font-display text-5xl md:text-7xl font-extrabold tracking-tight gradient-brand mb-6"
          >
            Verantyx-CLI
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.25 }}
            className="text-lg md:text-2xl font-light max-w-3xl leading-relaxed"
            style={{ color: 'rgba(226,232,240,0.9)' }}
          >
            {t({
              en: 'Keep a small local router resident. Wake larger local models only when needed. Carry memory across restarts — without pretending structure replaces world knowledge.',
              ja: '小さなローカルルーターを常駐させ、必要なときだけ大型モデルを起こす。再起動をまたいで記憶を運ぶ。構造が世界知識の代わりになるとは言いません。',
            })}
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.4 }}
            className="mt-10 flex flex-wrap gap-4"
          >
            <a
              href={GITHUB}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-accent rounded-xl px-7 py-3.5 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {t({ en: 'GitHub repository', ja: 'GitHub リポジトリ' })} →
            </a>
            <a
              href={RELEASE}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-7 py-3.5 text-sm font-semibold text-slate-300"
              style={{ border: '1px solid rgba(148,163,184,0.28)', textDecoration: 'none' }}
            >
              {t({ en: 'Release notes', ja: 'リリースノート' })}
            </a>
            <a
              href="#install"
              className="rounded-xl px-5 py-3.5 text-sm font-medium text-slate-500"
              style={{ textDecoration: 'none' }}
            >
              {t({ en: 'Install ↓', ja: 'インストール ↓' })}
            </a>
          </motion.div>
        </div>
      </section>

      {/* Install */}
      <Section id="install" label="INSTALL">
        <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
          {t({ en: 'Get started in minutes', ja: '数分で始められる' })}
        </h2>
        <p className="text-slate-400 mb-8 max-w-2xl leading-relaxed">
          {t({
            en: 'Prefer the curated stable branch or the v3.0.0-alpha release. main moves fast as a research workbench.',
            ja: '訪問者向けには stable ブランチか v3.0.0-alpha を推奨。main は研究用ワークベンチとして高速に動きます。',
          })}
        </p>
        <pre
          className="overflow-x-auto rounded-2xl border p-5 md:p-6 text-sm leading-relaxed font-mono"
          style={{
            borderColor: 'rgba(var(--accent-rgb), 0.2)',
            background: 'rgba(0,0,0,0.55)',
            color: 'rgba(224, 242, 254, 0.92)',
          }}
        >
{`git clone https://github.com/Ag3497120/verantyx-cli.git
cd verantyx-cli
git checkout stable   # curated snapshot for visitors
python3 verantyx.py`}
        </pre>
        <div className="mt-6 flex flex-wrap gap-3">
          <a
            href={GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent rounded-lg px-5 py-2.5 text-sm font-semibold"
            style={{ textDecoration: 'none' }}
          >
            Clone on GitHub
          </a>
          <a
            href={DOCS}
            className="rounded-lg px-5 py-2.5 text-sm font-semibold text-slate-400"
            style={{ border: '1px solid rgba(148,163,184,0.2)', textDecoration: 'none' }}
          >
            {t({ en: 'Legacy HTML docs', ja: '旧 HTML ドキュメント' })}
          </a>
        </div>
      </Section>

      {/* Features */}
      <Section label="FEATURES">
        <h2 className="font-display text-3xl md:text-4xl font-bold mb-10">
          {t({ en: 'What ships in the box', ja: '同梱されるもの' })}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title.en}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ duration: 0.55, delay: i * 0.06 }}
              className="rounded-2xl p-6"
              style={{
                background: 'rgba(10,10,20,0.65)',
                border: '1px solid rgba(var(--accent-rgb), 0.12)',
              }}
            >
              <h3 className="font-display text-xl font-semibold mb-3" style={{ color: '#f1f5f9' }}>
                {t(f.title)}
              </h3>
              <p className="text-slate-400 leading-relaxed">{t(f.body)}</p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* Architecture */}
      <Section label="ARCHITECTURE">
        <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
          {t({ en: 'Pipeline highlights', ja: 'パイプラインの要点' })}
        </h2>
        <p className="text-slate-400 mb-10 max-w-2xl">
          {t({
            en: 'A local-first harness: classify, escalate, remember — with clear claim boundaries.',
            ja: 'ローカル優先のハーネス：分類 → エスカレート → 記憶。主張の境界は明確です。',
          })}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {ARCH.map((a, i) => (
            <motion.div
              key={a.step}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="relative rounded-2xl p-5"
              style={{
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid rgba(var(--accent-rgb), 0.1)',
              }}
            >
              <span
                className="font-display text-xs tracking-[0.25em] font-semibold"
                style={{ color: 'rgba(var(--accent-rgb), 0.75)' }}
              >
                {a.step}
              </span>
              <h3 className="font-display text-lg font-bold mt-2 mb-2">{t(a.title)}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{t(a.body)}</p>
            </motion.div>
          ))}
        </div>
      </Section>

      {/* Honesty */}
      <Section label="BOUNDARIES">
        <h2 className="font-display text-3xl md:text-4xl font-bold mb-6">
          {t({ en: 'What it is not', ja: 'これは何ではないか' })}
        </h2>
        <ul className="space-y-3 text-slate-300 leading-relaxed max-w-3xl">
          <li className="flex gap-3">
            <span style={{ color: 'rgba(var(--accent-rgb), 0.7)' }}>▹</span>
            {t({
              en: 'Not an accuracy booster that turns 0.5B into a frontier model.',
              ja: '0.5B をフロンティアモデルに変える精度ブースターではありません。',
            })}
          </li>
          <li className="flex gap-3">
            <span style={{ color: 'rgba(var(--accent-rgb), 0.7)' }}>▹</span>
            {t({
              en: 'Not “structure equals more world knowledge.”',
              ja: '「構造＝より多くの世界知識」ではありません。',
            })}
          </li>
          <li className="flex gap-3">
            <span style={{ color: 'rgba(var(--accent-rgb), 0.7)' }}>▹</span>
            {t({
              en: 'Not a one-click cloud demo — real local setup, real tradeoffs.',
              ja: 'ワンクリックのクラウドデモではありません。本物のローカルセットアップとトレードオフがあります。',
            })}
          </li>
        </ul>
      </Section>

      {/* Releases / CTA */}
      <Section label="RELEASES">
        <div
          className="rounded-3xl p-8 md:p-12 relative overflow-hidden"
          style={{
            background: 'rgba(10,10,20,0.8)',
            border: '1px solid rgba(var(--accent-rgb), 0.22)',
          }}
        >
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                'radial-gradient(ellipse at top right, rgba(var(--accent-rgb), 0.1), transparent 55%)',
            }}
          />
          <div className="relative z-10">
            <h2 className="font-display text-3xl md:text-4xl font-bold mb-4">
              {t({ en: 'Ship with the community', ja: 'コミュニティと一緒に' })}
            </h2>
            <p className="text-slate-400 max-w-2xl mb-8 leading-relaxed">
              {t({
                en: 'Star the repo, try stable or v3.0.0-alpha, and read claim boundaries before quoting scores.',
                ja: 'リポジトリに Star を、stable か v3.0.0-alpha を試して、スコアを引用する前に主張の境界を読んでください。',
              })}
            </p>
            <div className="flex flex-wrap gap-4">
              <a
                href={GITHUB}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-accent rounded-xl px-7 py-3.5 text-sm font-semibold"
                style={{ textDecoration: 'none' }}
              >
                {t({ en: 'Open on GitHub', ja: 'GitHub で開く' })}
              </a>
              <a
                href={RELEASE}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-xl px-7 py-3.5 text-sm font-semibold text-slate-300"
                style={{ border: '1px solid rgba(148,163,184,0.28)', textDecoration: 'none' }}
              >
                v3.0.0-alpha
              </a>
            </div>
          </div>
        </div>
      </Section>

      <Footer />
    </main>
  );
}

function Section({
  id,
  label,
  children,
}: {
  id?: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="relative px-6 py-16 md:py-20">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          <div
            className="flex-1 h-px max-w-[80px]"
            style={{
              background: 'linear-gradient(90deg, rgba(var(--accent-rgb), 0.4), transparent)',
            }}
          />
          <span className="text-xs tracking-[0.35em] uppercase text-silver font-semibold">
            {label}
          </span>
        </div>
        {children}
      </div>
    </section>
  );
}
