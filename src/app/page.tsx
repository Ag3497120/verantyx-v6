'use client';

/* cf-deploy-bump: 2026-07-15T20:25Z — redesign: CLI-first, no splash, theme */

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef, useState } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import CliSpotlight from '@/components/CliSpotlight';
import StickyCliCta from '@/components/StickyCliCta';
import { useLanguage } from '@/lib/i18n';

const CLI_GITHUB = 'https://github.com/Ag3497120/verantyx-cli';

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
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <a
              href={CLI_GITHUB}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-accent rounded-xl px-7 py-3.5 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {lang === 'ja' ? 'GitHub で Verantyx-CLI を開く' : 'Open Verantyx-CLI on GitHub'}
            </a>
            <a
              href="/verantyx-cli/"
              className="rounded-xl px-7 py-3.5 text-sm font-semibold text-slate-300"
              style={{
                border: '1px solid rgba(148,163,184,0.28)',
                textDecoration: 'none',
                transition: 'border-color 0.3s ease, color 0.3s ease',
              }}
            >
              {lang === 'ja' ? 'CLI 製品ページ' : 'CLI product page'}
            </a>
            <a
              href="#verantyx-cli"
              className="rounded-xl px-5 py-3.5 text-sm font-medium text-slate-500 hover:text-slate-300 transition-colors"
              style={{ textDecoration: 'none' }}
            >
              {lang === 'ja' ? '概要へ ↓' : 'Overview ↓'}
            </a>
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
