'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import CinematicOpening from '@/components/CinematicOpening';
import CliSpotlight from '@/components/CliSpotlight';
import { useLanguage } from '@/lib/i18n';

export default function Home() {
  const { lang } = useLanguage();
  const heroRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.95]);
  const heroBlur = useTransform(scrollYProgress, [0, 1], [0, 8]);

  return (
    <main className="relative bg-black text-white overflow-x-hidden min-h-screen">
      <CinematicOpening />
      <Navbar />

      {/* ── Hero Section ── */}
      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-[85vh] flex items-center justify-center px-6 pt-32 pb-24"
      >
        {/* Ambient glow */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse 60% 40% at 50% 40%, rgba(14,165,233,0.06) 0%, transparent 70%)',
          }}
        />

        <motion.div
          style={{ filter: useTransform(heroBlur, (v) => `blur(${v}px)`) }}
          className="max-w-6xl mx-auto text-center relative z-10"
        >
          {/* Overline */}
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1.2, delay: 4.2, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto mb-8 h-px w-32"
            style={{
              background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.5), transparent)',
              transformOrigin: 'center',
            }}
          />

          <motion.h1
            initial={{ opacity: 0, y: 30, filter: 'blur(8px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ duration: 1, delay: 4.3, ease: 'easeOut' }}
            className="text-7xl md:text-9xl font-black mb-6 tracking-tight"
            style={{
              background: 'linear-gradient(135deg, #0EA5E9, #7C3AED, #06B6D4)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Verantyx
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20, filter: 'blur(4px)' }}
            animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
            transition={{ duration: 0.8, delay: 4.6 }}
            className="text-lg md:text-2xl font-light tracking-wide max-w-3xl mx-auto"
            style={{ color: 'rgba(226, 232, 240, 0.9)' }}
          >
            {lang === 'ja'
              ? '旗艦：Verantyx-CLI — 0.5B常駐ルーターと、必要なときだけ大型ローカルモデル'
              : 'Flagship: Verantyx-CLI — a resident 0.5B router that wakes larger local models only when needed'}
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 4.85 }}
            className="mt-4 text-sm md:text-base font-light tracking-widest uppercase"
            style={{ color: 'rgba(148, 163, 184, 0.65)' }}
          >
            {lang === 'ja' ? 'その下にアプリとその他プロジェクト' : 'Apps & other projects below'}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 5.0 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <a
              href="https://github.com/Ag3497120/verantyx-cli"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{
                background: 'rgba(14,165,233,0.18)',
                border: '1px solid rgba(14,165,233,0.45)',
                color: '#e0f2fe',
              }}
            >
              Open Verantyx-CLI on GitHub
            </a>
            <a
              href="#verantyx-cli"
              className="rounded-xl px-6 py-3 text-sm font-semibold text-slate-300"
              style={{ border: '1px solid rgba(148,163,184,0.3)' }}
            >
              {lang === 'ja' ? '詳細を見る' : 'Read the details'}
            </a>
          </motion.div>

          {/* Underline */}
          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1.2, delay: 5.1, ease: [0.22, 1, 0.36, 1] }}
            className="mx-auto mt-10 h-px w-32"
            style={{
              background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.5), transparent)',
              transformOrigin: 'center',
            }}
          />
        </motion.div>
      </motion.section>

      {/* ── Verantyx-CLI details (English) above Projects ── */}
      <div className="relative px-6 py-4">
        <div className="max-w-6xl mx-auto">
          <SectionTitle text="VERANTYX-CLI" delay={5.15} />
        </div>
      </div>
      <CliSpotlight />

      {/* ── Section Divider ── */}
      <div className="relative px-6 py-4">
        <div className="max-w-6xl mx-auto">
          <SectionTitle text="PROJECTS" delay={5.2} />
        </div>
      </div>

      {/* ── Project Cards Grid ── */}
      <section className="relative px-6 pb-32 pt-12">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <ProjectCard
              icon="📱"
              title="Apps"
              description={lang === 'ja' ? '口の動きで遊ぶ革新的なiOSゲーム' : 'Innovative iOS games controlled by mouth movements'}
              subtitle={lang === 'ja' ? 'パクパク釣り / MouthEat' : 'PakuPaku Fishing / MouthEat'}
              href="/apps"
              delay={0}
              lang={lang}
            />
            <ProjectCard
              icon="⚡"
              title="Verantyx Engine"
              description={lang === 'ja' ? 'LLMフリーのシンボリック推論' : 'LLM-free symbolic reasoning'}
              subtitle={lang === 'ja' ? 'ARC-AGI-2: 20.7% — ニューラルネットワークゼロ' : 'ARC-AGI-2: 20.7% — Zero Neural Networks'}
              href="/verantyx"
              delay={0.1}
              lang={lang}
            />
            <ProjectCard
              icon="📦"
              title="Verantyx-CLI"
              description={
                lang === 'ja'
                  ? 'ローカル常駐ルーター＋評議会＋永遠の記憶（旗艦OSS）'
                  : 'Local resident router, council & eternal memory — flagship OSS'
              }
              subtitle="github.com/Ag3497120/verantyx-cli · v3.0.0-alpha"
              href="/verantyx-cli"
              delay={0.2}
              lang={lang}
            />
            <ProjectCard
              icon="📚"
              title=".jcross Language"
              description={lang === 'ja' ? 'クロスワードパズルDSL' : 'Crossword puzzle DSL'}
              subtitle=""
              href="/jcross-language"
              delay={0.3}
              lang={lang}
            />
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}

/* ── Section Title with cinematic borders ── */
function SectionTitle({ text, delay }: { text: string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8, delay }}
      className="flex items-center justify-center gap-6 py-6"
    >
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        className="flex-1 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.3))',
          transformOrigin: 'left',
        }}
      />
      <span
        className="text-sm md:text-base font-semibold tracking-[0.4em] uppercase"
        style={{
          background: 'linear-gradient(45deg, #6b7280 0%, #d1d5db 45%, #f9fafb 70%, #d1d5db 85%, #6b7280 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}
      >
        {text}
      </span>
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        className="flex-1 h-px"
        style={{
          background: 'linear-gradient(90deg, rgba(14,165,233,0.3), transparent)',
          transformOrigin: 'right',
        }}
      />
    </motion.div>
  );
}

/* ── Project Card ── */
function ProjectCard({
  icon,
  title,
  description,
  subtitle,
  href,
  delay,
  lang,
}: {
  icon: string;
  title: string;
  description: string;
  subtitle: string;
  href: string;
  delay: number;
  lang: 'ja' | 'en';
}) {
  return (
    <motion.a
      href={href}
      initial={{ opacity: 0, y: 30, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.8, delay, ease: 'easeOut' }}
      whileHover={{ y: -6 }}
      className="block group relative overflow-hidden"
      style={{
        background: 'rgba(10, 10, 20, 0.7)',
        border: '1px solid rgba(14, 165, 233, 0.12)',
        borderRadius: 20,
        padding: '2.5rem',
        textDecoration: 'none',
        transition: 'border-color 0.4s ease, box-shadow 0.4s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.4)';
        e.currentTarget.style.boxShadow = '0 0 60px rgba(14, 165, 233, 0.12), inset 0 1px 0 rgba(14,165,233,0.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.12)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      {/* Hover gradient overlay */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at top left, rgba(14,165,233,0.05), transparent 60%)',
          borderRadius: 20,
        }}
      />

      <div className="relative z-10 flex flex-col gap-4">
        <div className="text-5xl">{icon}</div>
        <h2
          className="text-2xl md:text-3xl font-bold"
          style={{ color: '#f1f5f9' }}
        >
          {title}
        </h2>
        <p className="text-gray-400 text-lg leading-relaxed">{description}</p>
        {subtitle && (
          <p
            className="text-sm font-medium tracking-wide"
            style={{ color: 'rgba(14, 165, 233, 0.8)' }}
          >
            {subtitle}
          </p>
        )}
        <div className="flex items-center gap-2 mt-3 text-sm font-semibold" style={{ color: 'rgba(14, 165, 233, 0.7)' }}>
          <span>{lang === 'ja' ? '詳しく見る' : 'Learn more'}</span>
          <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
        </div>
      </div>
    </motion.a>
  );
}
