'use client';

import { motion } from 'framer-motion';
import { useLanguage } from '@/lib/i18n';

const GITHUB = 'https://github.com/Ag3497120/verantyx-cli';
const RELEASE = 'https://github.com/Ag3497120/verantyx-cli/releases/tag/v3.0.0-alpha';

export default function CliSpotlight() {
  const { lang } = useLanguage();

  return (
    <section id="verantyx-cli" className="relative px-6 pb-20 pt-8">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="relative overflow-hidden rounded-3xl p-8 md:p-12 theme-surface"
          style={{
            border: '1px solid rgba(var(--accent-rgb), 0.22)',
            background: 'rgba(10,10,20,0.82)',
          }}
        >
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                'radial-gradient(ellipse at top left, rgba(var(--accent-rgb), 0.1), transparent 55%)',
            }}
          />

          <div className="relative z-10 space-y-8">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className="rounded-full px-3 py-1 text-xs font-semibold tracking-[0.2em] uppercase"
                style={{
                  border: '1px solid rgba(var(--accent-rgb), 0.35)',
                  color: 'rgba(var(--accent-rgb), 0.95)',
                }}
              >
                Flagship · Open Source
              </span>
              <span className="text-xs tracking-wide text-slate-500">
                v3.0.0-alpha · MIT (code)
              </span>
            </div>

            <div className="space-y-4">
              <h2 className="font-display text-3xl md:text-5xl font-bold tracking-tight text-slate-50">
                Verantyx-CLI
              </h2>
              <p className="text-xl md:text-2xl font-light leading-snug text-slate-200 max-w-3xl">
                {lang === 'ja'
                  ? '小さなローカルルーターを常駐させ、必要なときだけ大型モデルを起こす。再起動をまたいで記憶を運ぶ。構造が世界知識の代わりになるとは言いません。'
                  : 'Keep a small local router resident. Wake larger local models only when needed. Carry memory across restarts — without pretending structure replaces world knowledge.'}
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2 text-left">
              <div className="space-y-3 text-slate-300 leading-relaxed">
                <h3
                  className="text-sm font-semibold tracking-[0.2em] uppercase"
                  style={{ color: 'rgba(var(--accent-rgb), 0.9)' }}
                >
                  {lang === 'ja' ? 'これは何か' : 'What it is'}
                </h3>
                <ul className="space-y-2 text-base list-disc list-inside marker:opacity-60">
                  {lang === 'ja' ? (
                    <>
                      <li>ローカル優先の AI ランタイム / ハーネス（Omni CLI）</li>
                      <li>分類専用ルーティング — ルーターは答えを捏造しない</li>
                      <li>ベクトル評議会、任意のパズル軸、エージェントツール、永遠の記憶</li>
                      <li>フィードバックとレビューループで一晩かけて自己修正</li>
                      <li>主張の境界をリポジトリに公開した誠実なベンチマーク</li>
                    </>
                  ) : (
                    <>
                      <li>Local-first AI runtime / harness (Omni CLI)</li>
                      <li>Classify-only routing — the router does not invent answers</li>
                      <li>Vector council, optional puzzle axes, agent tools, eternal memory</li>
                      <li>Feedback &amp; reviewer loops for overnight self-correction</li>
                      <li>Honest benchmarks with claim boundaries published in-repo</li>
                    </>
                  )}
                </ul>
              </div>
              <div className="space-y-3 text-slate-300 leading-relaxed">
                <h3
                  className="text-sm font-semibold tracking-[0.2em] uppercase"
                  style={{ color: 'rgba(var(--accent-rgb), 0.9)' }}
                >
                  {lang === 'ja' ? 'これは何かではない' : 'What it is not'}
                </h3>
                <ul className="space-y-2 text-base list-disc list-inside marker:text-slate-600">
                  {lang === 'ja' ? (
                    <>
                      <li>0.5B をフロンティアモデルにする精度ブースターではない</li>
                      <li>「構造＝より多くの世界知識」ではない</li>
                      <li>ワンクリックのクラウドデモではない — 実機セットアップとトレードオフ</li>
                    </>
                  ) : (
                    <>
                      <li>Not an accuracy booster that turns 0.5B into a frontier model</li>
                      <li>Not “structure equals more world knowledge”</li>
                      <li>Not a one-click cloud demo — real local setup, real tradeoffs</li>
                    </>
                  )}
                </ul>
                <p className="pt-2 text-sm text-slate-500 leading-relaxed">
                  {lang === 'ja' ? (
                    <>
                      訪問者向けは <code className="opacity-90">stable</code> か{' '}
                      <code className="opacity-90">v3.0.0-alpha</code> を推奨。{' '}
                      <code className="text-slate-400">main</code> は研究用ワークベンチです。
                    </>
                  ) : (
                    <>
                      Try the curated <code className="opacity-90">stable</code> branch or the{' '}
                      <code className="opacity-90">v3.0.0-alpha</code> release;{' '}
                      <code className="text-slate-400">main</code> moves fast as a research
                      workbench.
                    </>
                  )}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-4 pt-2">
              <a
                href={GITHUB}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold"
                style={{
                  background: 'rgba(var(--accent-rgb), 0.18)',
                  border: '1px solid rgba(var(--accent-rgb), 0.45)',
                  color: '#e0f2fe',
                }}
              >
                {lang === 'ja' ? 'GitHub リポジトリ' : 'GitHub repository'}
                <span aria-hidden>→</span>
              </a>
              <a
                href={RELEASE}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-slate-300"
                style={{ border: '1px solid rgba(148,163,184,0.25)' }}
              >
                {lang === 'ja' ? 'リリースノート' : 'Release notes'}
              </a>
              <a
                href="/verantyx-cli/"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-slate-400 hover:opacity-100"
                style={{ color: 'rgba(var(--accent-rgb), 0.85)' }}
              >
                {lang === 'ja' ? '製品ページ' : 'Full product page'}
                <span aria-hidden>→</span>
              </a>
            </div>

            <pre
              className="overflow-x-auto rounded-2xl border bg-black/50 p-4 text-left text-sm leading-relaxed font-mono"
              style={{
                borderColor: 'rgba(148,163,184,0.2)',
                color: 'rgba(226,232,240,0.9)',
              }}
            >
{`git clone https://github.com/Ag3497120/verantyx-cli.git
cd verantyx-cli
git checkout stable
python3 verantyx.py`}
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
