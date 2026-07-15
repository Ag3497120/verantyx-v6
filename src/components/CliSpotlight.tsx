'use client';

import { motion } from 'framer-motion';

const GITHUB = 'https://github.com/Ag3497120/verantyx-cli';
const RELEASE = 'https://github.com/Ag3497120/verantyx-cli/releases/tag/v3.0.0-alpha';

export default function CliSpotlight() {
  return (
    <section id="verantyx-cli" className="relative px-6 pb-20 pt-8">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="relative overflow-hidden rounded-3xl border border-sky-500/20 bg-[rgba(10,10,20,0.85)] p-8 md:p-12"
        >
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                'radial-gradient(ellipse at top left, rgba(14,165,233,0.08), transparent 55%)',
            }}
          />

          <div className="relative z-10 space-y-8">
            <div className="flex flex-wrap items-center gap-3">
              <span className="rounded-full border border-sky-400/30 px-3 py-1 text-xs font-semibold tracking-[0.2em] text-sky-300/90 uppercase">
                Flagship · Open Source
              </span>
              <span className="text-xs tracking-wide text-slate-500">v3.0.0-alpha · MIT (code)</span>
            </div>

            <div className="space-y-4">
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight text-slate-50">
                Verantyx-CLI
              </h2>
              <p className="text-xl md:text-2xl font-light leading-snug text-slate-200 max-w-3xl">
                Keep a small local router resident. Wake larger local models only when needed.
                Carry memory across restarts — without pretending structure replaces world knowledge.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2 text-left">
              <div className="space-y-3 text-slate-300 leading-relaxed">
                <h3 className="text-sm font-semibold tracking-[0.2em] uppercase text-sky-400/90">
                  What it is
                </h3>
                <ul className="space-y-2 text-base list-disc list-inside marker:text-sky-500/70">
                  <li>Local-first AI runtime / harness (Omni CLI)</li>
                  <li>Classify-only routing — the router does not invent answers</li>
                  <li>Vector council, optional puzzle axes, agent tools, eternal memory</li>
                  <li>Feedback &amp; reviewer loops for overnight self-correction</li>
                  <li>Honest benchmarks with claim boundaries published in-repo</li>
                </ul>
              </div>
              <div className="space-y-3 text-slate-300 leading-relaxed">
                <h3 className="text-sm font-semibold tracking-[0.2em] uppercase text-sky-400/90">
                  What it is not
                </h3>
                <ul className="space-y-2 text-base list-disc list-inside marker:text-slate-600">
                  <li>Not an accuracy booster that turns 0.5B into a frontier model</li>
                  <li>Not “structure equals more world knowledge”</li>
                  <li>Not a one-click cloud demo — real local setup, real tradeoffs</li>
                </ul>
                <p className="pt-2 text-sm text-slate-500 leading-relaxed">
                  Try the curated <code className="text-sky-300/80">stable</code> branch or the{' '}
                  <code className="text-sky-300/80">v3.0.0-alpha</code> release;{' '}
                  <code className="text-slate-400">main</code> moves fast as a research workbench.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-4 pt-2">
              <a
                href={GITHUB}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-colors"
                style={{
                  background: 'rgba(14,165,233,0.15)',
                  border: '1px solid rgba(14,165,233,0.45)',
                  color: '#e0f2fe',
                }}
              >
                GitHub repository
                <span aria-hidden>→</span>
              </a>
              <a
                href={RELEASE}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-slate-300 transition-colors hover:text-white"
                style={{ border: '1px solid rgba(148,163,184,0.25)' }}
              >
                Release notes
              </a>
              <a
                href="/verantyx-cli"
                className="inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-slate-400 transition-colors hover:text-sky-300"
              >
                On-site docs
                <span aria-hidden>→</span>
              </a>
            </div>

            <pre
              className="overflow-x-auto rounded-2xl border border-slate-700/60 bg-black/50 p-4 text-left text-sm leading-relaxed text-sky-100/90 font-mono"
            >
{`git clone https://github.com/Ag3497120/verantyx-cli.git
cd verantyx-cli
# Prefer: git checkout stable   # curated snapshot for visitors
python3 verantyx.py`}
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
