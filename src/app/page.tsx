'use client';

/* cf-deploy-bump: 2026-07-15T20:45Z — conversion CTAs + 4xx hardening */

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef, useState } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import CliSpotlight from '@/components/CliSpotlight';
import StickyCliCta from '@/components/StickyCliCta';
import { useLanguage } from '@/lib/i18n';

const CLI_GITHUB = 'https://github.com/Ag3497120/verantyx-cli';

const COMMANDS = [
  'git clone https://github.com/Ag3497120/verantyx-cli.git',
  'cd verantyx-cli && git checkout stable',
  'python3 verantyx.py',
];

export default function Home() {
  const { lang } = useLanguage();
  const heroRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.97]);
  const heroBlur = useTransform(scrollYProgress, [0, 1], [0, 6]);

  return (
    <main className="relative text-white overflow-x-hidden min-h-screen">
      <Navbar />
      <StickyCliCta />

      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-[88vh] flex items-center justify-center px-6 pt-28 pb-20"
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 70% 45% at 50% 38%, rgba(var(--accent-rgb), 0.1) 0%, transparent 68%)',
          }}
        />

        <motion.div
          style={{ filter: useTransform(heroBlur, (v) => `blur(${v}px)`) }}
          className="max-w-5xl mx-auto text-center relative z-10"
        >
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto mb-8 h-px w-28"
            style={{
              background:
                'linear-gradient(90deg, transparent, rgba(var(--accent-rgb), 0.55), transparent)',
              transformOrigin: 'center',
            }}
          />

          <motion.h1
            initial={{ opacity: 0, y: 28, filter: 'blur(8px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ duration: 0.9, delay: 0.2, ease: 'easeOut' }}
            className="font-display text-7xl md:text-9xl font-extrabold mb-6 tracking-tight gradient-brand"
          >
            Verantyx
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.45 }}
            className="text-lg md:text-2xl font-light tracking-wide max-w-3xl mx-auto"
            style={{ color: 'rgba(226, 232, 240, 0.92)' }}
          >
            {lang === 'ja'
              ? '旗艦：Verantyx-CLI — 0.5B常駐ルーターと、必要なときだけ大型ローカルモデル'
              : 'Flagship: Verantyx-CLI — a resident 0.5B router that wakes larger local models only when needed'}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.65 }}
            className="mt-10 flex flex-col items-center gap-4"
          >
            <a
              href={CLI_GITHUB}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-accent rounded-xl px-8 py-3.5 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {lang === 'ja' ? 'GitHub で開く' : 'Open on GitHub'}
            </a>
            <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-sm">
              <a
                href="#demo"
                className="font-medium transition-colors"
                style={{ color: 'rgba(var(--accent-rgb), 0.95)', textDecoration: 'none' }}
              >
                {lang === 'ja' ? '30秒デモ' : '30s demo'}
              </a>
              <span className="text-slate-600" aria-hidden>
                ·
              </span>
              <a
                href="#start"
                className="font-medium text-slate-300 hover:text-white transition-colors"
                style={{ textDecoration: 'none' }}
              >
                {lang === 'ja' ? '3コマンドで開始' : 'Start in 3 commands'}
              </a>
              <span className="text-slate-600" aria-hidden>
                ·
              </span>
              <a
                href="#why"
                className="font-medium text-slate-400 hover:text-slate-200 transition-colors"
                style={{ textDecoration: 'none' }}
              >
                {lang === 'ja' ? '何が他と違うか' : 'What makes it different'}
              </a>
            </div>
          </motion.div>

          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1, delay: 0.85, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto mt-12 h-px w-28"
            style={{
              background:
                'linear-gradient(90deg, transparent, rgba(var(--accent-rgb), 0.55), transparent)',
              transformOrigin: 'center',
            }}
          />
        </motion.div>
      </motion.section>

      <DemoSection lang={lang} />
      <StartSection lang={lang} />
      <WhySection lang={lang} />

      <div className="relative px-6 py-2">
        <div className="max-w-6xl mx-auto">
          <SectionTitle text="VERANTYX-CLI" />
        </div>
      </div>
      <CliSpotlight />

      <div className="relative px-6 py-2">
        <div className="max-w-6xl mx-auto">
          <SectionTitle text="PROJECTS" />
        </div>
      </div>

      <section className="relative px-6 pb-16 pt-10">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ProjectCard
              icon="📦"
              title="Verantyx-CLI"
              description={
                lang === 'ja'
                  ? 'ローカル常駐ルーター＋評議会＋永遠の記憶（旗艦OSS）'
                  : 'Local resident router, council & eternal memory — flagship OSS'
              }
              subtitle="github.com/Ag3497120/verantyx-cli · v3.0.0-alpha"
              href="/verantyx-cli/"
              delay={0}
              lang={lang}
              featured
            />
            <ProjectCard
              icon="📚"
              title=".jcross Language"
              description={lang === 'ja' ? 'クロスワードパズルDSL' : 'Crossword puzzle DSL'}
              subtitle=""
              href="/jcross-language/"
              delay={0.1}
              lang={lang}
            />
            <ProjectCard
              icon="📱"
              title="Apps"
              description={
                lang === 'ja'
                  ? '口の動きで遊ぶ iOS ゲームと関連プロジェクト'
                  : 'Mouth-controlled iOS games and related projects'
              }
              subtitle={lang === 'ja' ? '一覧を見る' : 'Browse the catalog'}
              href="/apps/"
              delay={0.15}
              lang={lang}
            />
          </div>

          <OtherAppsCollapse lang={lang} />
        </div>
      </section>

      <Footer />
    </main>
  );
}

function DemoSection({ lang }: { lang: 'ja' | 'en' }) {
  const steps =
    lang === 'ja'
      ? [
          { n: '01', title: 'clone', body: 'リポジトリを取得' },
          { n: '02', title: 'stable', body: '訪問者向けブランチへ切替' },
          { n: '03', title: 'run', body: '常駐ルーターが起動し、プロンプトを待つ' },
        ]
      : [
          { n: '01', title: 'clone', body: 'Get the repository' },
          { n: '02', title: 'stable', body: 'Switch to the visitor-ready branch' },
          { n: '03', title: 'run', body: 'Resident router boots and waits for prompts' },
        ];

  return (
    <section id="demo" className="relative px-6 py-16 scroll-mt-24">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.7 }}
        >
          <p
            className="text-xs tracking-[0.3em] uppercase mb-3"
            style={{ color: 'rgba(var(--accent-rgb), 0.8)' }}
          >
            {lang === 'ja' ? '30秒デモ' : '30-second demo'}
          </p>
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-3">
            {lang === 'ja' ? '起動までの流れ' : 'From zero to resident router'}
          </h2>
          <p className="text-slate-400 mb-8 leading-relaxed">
            {lang === 'ja'
              ? '動画や asciinema はありません。代わりに、実際の3ステップをそのまま示します。'
              : 'No hosted video yet — here is the real three-step path, as it runs locally.'}
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.75, delay: 0.1 }}
          className="rounded-2xl overflow-hidden font-mono text-sm"
          style={{
            border: '1px solid rgba(var(--accent-rgb), 0.2)',
            background: 'rgba(0,0,0,0.55)',
          }}
        >
          <div
            className="flex items-center gap-2 px-4 py-2.5 text-xs text-slate-500"
            style={{ borderBottom: '1px solid rgba(148,163,184,0.12)' }}
          >
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-slate-700" />
            <span className="ml-2 tracking-wide">verantyx-cli · demo</span>
          </div>
          <div className="p-5 space-y-4 text-left">
            {COMMANDS.map((cmd, i) => (
              <motion.div
                key={cmd}
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.15 + i * 0.12 }}
              >
                <p className="text-slate-500 text-xs mb-1">
                  {steps[i].n} · {steps[i].title}
                </p>
                <p style={{ color: 'rgba(224, 242, 254, 0.92)' }}>
                  <span style={{ color: 'rgba(var(--accent-rgb), 0.9)' }}>$ </span>
                  {cmd}
                </p>
                <p className="text-slate-500 text-xs mt-1">{steps[i].body}</p>
              </motion.div>
            ))}
            <motion.p
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.55 }}
              className="pt-2"
              style={{ color: 'rgba(var(--accent-rgb), 0.85)' }}
            >
              {lang === 'ja'
                ? '→ ルーター常駐。大型モデルは必要なときだけ起床。'
                : '→ Router resident. Larger models wake only when needed.'}
            </motion.p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function StartSection({ lang }: { lang: 'ja' | 'en' }) {
  return (
    <section id="start" className="relative px-6 py-16 scroll-mt-24">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.7 }}
        >
          <p
            className="text-xs tracking-[0.3em] uppercase mb-3"
            style={{ color: 'rgba(var(--accent-rgb), 0.8)' }}
          >
            {lang === 'ja' ? '3コマンドで開始' : 'Start in 3 commands'}
          </p>
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-3">
            {lang === 'ja' ? 'コピーして実行' : 'Copy, paste, run'}
          </h2>
          <p className="text-slate-400 mb-8 leading-relaxed">
            {lang === 'ja'
              ? 'stable ブランチを推奨。main は研究用で変化が速いです。'
              : 'Prefer the stable branch. main moves fast as a research workbench.'}
          </p>
        </motion.div>

        <div className="space-y-3">
          {COMMANDS.map((cmd, i) => (
            <CopyCommand key={cmd} cmd={cmd} index={i} lang={lang} />
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mt-8 flex flex-wrap gap-3"
        >
          <a
            href={CLI_GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold"
            style={{ textDecoration: 'none' }}
          >
            {lang === 'ja' ? 'GitHub リポジトリ' : 'GitHub repository'} →
          </a>
          <a
            href="/verantyx-cli/"
            className="rounded-xl px-6 py-3 text-sm font-semibold text-slate-400"
            style={{ border: '1px solid rgba(148,163,184,0.22)', textDecoration: 'none' }}
          >
            {lang === 'ja' ? '製品ページ' : 'Product page'}
          </a>
        </motion.div>
      </div>
    </section>
  );
}

function CopyCommand({
  cmd,
  index,
  lang,
}: {
  cmd: string;
  index: number;
  lang: 'ja' | 'en';
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* ignore */
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className="flex items-stretch gap-2 rounded-xl overflow-hidden"
      style={{
        border: '1px solid rgba(var(--accent-rgb), 0.18)',
        background: 'rgba(0,0,0,0.45)',
      }}
    >
      <pre
        className="flex-1 overflow-x-auto px-4 py-3.5 text-sm font-mono text-left m-0"
        style={{ color: 'rgba(226,232,240,0.92)' }}
      >
        <span className="text-slate-500 mr-2">{index + 1}.</span>
        {cmd}
      </pre>
      <button
        type="button"
        onClick={copy}
        className="shrink-0 px-4 text-xs font-semibold tracking-wide transition-colors"
        style={{
          borderLeft: '1px solid rgba(148,163,184,0.15)',
          color: copied ? 'rgba(var(--accent-rgb), 1)' : '#94a3b8',
          background: 'transparent',
          cursor: 'pointer',
        }}
        aria-label={lang === 'ja' ? 'コピー' : 'Copy'}
      >
        {copied ? (lang === 'ja' ? 'コピー済' : 'Copied') : lang === 'ja' ? 'コピー' : 'Copy'}
      </button>
    </motion.div>
  );
}

function WhySection({ lang }: { lang: 'ja' | 'en' }) {
  const items =
    lang === 'ja'
      ? [
          {
            title: '分類専用ルーター',
            body: '0.5B は答えを捏造しない。分類して、必要なときだけ大型ローカルモデルを起こす。',
          },
          {
            title: 'ローカル優先',
            body: 'クラウドデモではない。実機セットアップとトレードオフを隠さない。',
          },
          {
            title: '永遠の記憶',
            body: '再起動のたびに忘れるのではなく、セッションをまたいで文脈を運ぶ。',
          },
          {
            title: '主張の境界が公開',
            body: '構造＝世界知識とは言わない。ベンチと claim boundaries はリポジトリにある。',
          },
        ]
      : [
          {
            title: 'Classify-only router',
            body: 'The 0.5B does not invent answers. It classifies, then wakes larger local models only when needed.',
          },
          {
            title: 'Local-first, honest',
            body: 'Not a one-click cloud demo. Real setup, real tradeoffs — no magic accuracy claims.',
          },
          {
            title: 'Eternal memory',
            body: 'Carry durable context across restarts instead of amnesia every boot.',
          },
          {
            title: 'Published claim boundaries',
            body: 'Structure ≠ world knowledge. Benchmarks and limits live in-repo.',
          },
        ];

  return (
    <section id="why" className="relative px-6 py-16 scroll-mt-24">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.7 }}
        >
          <p
            className="text-xs tracking-[0.3em] uppercase mb-3"
            style={{ color: 'rgba(var(--accent-rgb), 0.8)' }}
          >
            {lang === 'ja' ? '何が他と違うか' : 'What makes it different'}
          </p>
          <h2 className="font-display text-3xl md:text-4xl font-bold mb-8">
            {lang === 'ja' ? '小さな常駐。大きな起床。' : 'Stay small. Wake big.'}
          </h2>
        </motion.div>

        <ul className="space-y-6 list-none">
          {items.map((item, i) => (
            <motion.li
              key={item.title}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.55, delay: i * 0.07 }}
              className="text-left"
              style={{
                paddingBottom: 24,
                borderBottom:
                  i < items.length - 1 ? '1px solid rgba(148,163,184,0.12)' : 'none',
              }}
            >
              <h3
                className="font-display text-lg md:text-xl font-semibold mb-2"
                style={{ color: 'rgba(241,245,249,0.98)' }}
              >
                {item.title}
              </h3>
              <p className="text-slate-400 leading-relaxed">{item.body}</p>
            </motion.li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function OtherAppsCollapse({ lang }: { lang: 'ja' | 'en' }) {
  const [open, setOpen] = useState(false);

  return (
    <details
      className="other-apps-details mt-10 rounded-2xl overflow-hidden"
      style={{
        border: '1px solid rgba(var(--accent-rgb), 0.12)',
        background: 'rgba(10, 10, 20, 0.45)',
      }}
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary
        className="px-6 py-5 flex items-center justify-between gap-4"
        style={{ color: 'rgba(226, 232, 240, 0.85)' }}
      >
        <div>
          <p
            className="text-xs tracking-[0.28em] uppercase mb-1"
            style={{ color: 'rgba(var(--accent-rgb), 0.7)' }}
          >
            {lang === 'ja' ? 'その他のアプリ' : 'Other apps'}
          </p>
          <p className="text-base font-medium">
            {lang === 'ja'
              ? 'パクパク釣り · MouthEat — クリックで展開'
              : 'PakuPaku Fishing · MouthEat — expand to browse'}
          </p>
        </div>
        <span className="other-apps-chevron text-slate-400 text-lg" aria-hidden>
          ▾
        </span>
      </summary>
      <div className="px-6 pb-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
        <a
          href="/apps/pakupaku-fishing/"
          className="block rounded-xl p-5 transition-all duration-300"
          style={{
            border: '1px solid rgba(var(--accent-rgb), 0.1)',
            background: 'rgba(0,0,0,0.35)',
            textDecoration: 'none',
            color: 'inherit',
          }}
        >
          <span className="text-3xl">🎣</span>
          <h3 className="mt-3 font-semibold text-lg text-white">
            {lang === 'ja' ? 'パクパク釣り' : 'Paku Paku Fishing'}
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            {lang === 'ja' ? '口で釣るフィッシングゲーム' : 'Mouth-controlled fishing'}
          </p>
        </a>
        <a
          href="/apps/mouth-eat/"
          className="block rounded-xl p-5 transition-all duration-300"
          style={{
            border: '1px solid rgba(var(--accent-rgb), 0.1)',
            background: 'rgba(0,0,0,0.35)',
            textDecoration: 'none',
            color: 'inherit',
          }}
        >
          <span className="text-3xl">😋</span>
          <h3 className="mt-3 font-semibold text-lg text-white">MouthEat</h3>
          <p className="mt-1 text-sm text-slate-400">
            {lang === 'ja' ? '口を開けて食べるゲーム' : 'Open-mouth eating game'}
          </p>
        </a>
      </div>
    </details>
  );
}

function SectionTitle({ text }: { text: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.7 }}
      className="flex items-center justify-center gap-6 py-6"
    >
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        className="flex-1 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(var(--accent-rgb), 0.35))',
          transformOrigin: 'left',
        }}
      />
      <span className="font-display text-sm md:text-base font-semibold tracking-[0.4em] uppercase text-silver">
        {text}
      </span>
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        className="flex-1 h-px"
        style={{
          background: 'linear-gradient(90deg, rgba(var(--accent-rgb), 0.35), transparent)',
          transformOrigin: 'right',
        }}
      />
    </motion.div>
  );
}

function ProjectCard({
  icon,
  title,
  description,
  subtitle,
  href,
  delay,
  lang,
  featured = false,
}: {
  icon: string;
  title: string;
  description: string;
  subtitle: string;
  href: string;
  delay: number;
  lang: 'ja' | 'en';
  featured?: boolean;
}) {
  return (
    <motion.a
      href={href}
      initial={{ opacity: 0, y: 28, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.75, delay, ease: 'easeOut' }}
      whileHover={{ y: -6 }}
      className={`block group relative overflow-hidden ${featured ? 'md:col-span-2' : ''}`}
      style={{
        background: featured ? 'rgba(10, 10, 20, 0.82)' : 'rgba(10, 10, 20, 0.65)',
        border: featured
          ? '1px solid rgba(var(--accent-rgb), 0.28)'
          : '1px solid rgba(var(--accent-rgb), 0.1)',
        borderRadius: 20,
        padding: featured ? '2.75rem' : '2.25rem',
        textDecoration: 'none',
        transition: 'border-color 0.4s ease, box-shadow 0.4s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.45)';
        e.currentTarget.style.boxShadow =
          '0 0 50px rgba(var(--accent-rgb), 0.12), inset 0 1px 0 rgba(var(--accent-rgb), 0.08)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = featured
          ? 'rgba(var(--accent-rgb), 0.28)'
          : 'rgba(var(--accent-rgb), 0.1)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse at top left, rgba(var(--accent-rgb), 0.07), transparent 60%)',
          borderRadius: 20,
        }}
      />

      <div className="relative z-10 flex flex-col gap-3">
        <div className="text-4xl md:text-5xl">{icon}</div>
        <h2 className="font-display text-2xl md:text-3xl font-bold" style={{ color: '#f1f5f9' }}>
          {title}
        </h2>
        <p className="text-gray-400 text-base md:text-lg leading-relaxed">{description}</p>
        {subtitle && (
          <p
            className="text-sm font-medium tracking-wide"
            style={{ color: 'rgba(var(--accent-rgb), 0.85)' }}
          >
            {subtitle}
          </p>
        )}
        <div
          className="flex items-center gap-2 mt-2 text-sm font-semibold"
          style={{ color: 'rgba(var(--accent-rgb), 0.75)' }}
        >
          <span>{lang === 'ja' ? '詳しく見る' : 'Learn more'}</span>
          <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
        </div>
      </div>
    </motion.a>
  );
}
