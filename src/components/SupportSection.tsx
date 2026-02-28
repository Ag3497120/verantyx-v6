'use client';

import { motion } from 'framer-motion';

export default function SupportSection() {
  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20">
      <div className="max-w-4xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-6xl font-bold text-center mb-8 gradient-text"
        >
          Support This Research
        </motion.h2>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-center mb-16"
        >
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Verantyx is built by a student in Kyoto with no GPU cluster — just a MacBook 
            and Claude as a development partner. The engine costs nothing to run, but 
            <span className="text-gray-200"> building it requires API credits</span> that 
            add up fast.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="border border-gray-800 rounded-2xl p-8 bg-gray-900/30 backdrop-blur-sm text-center hover:border-gray-600 transition-colors"
          >
            <div className="text-4xl mb-4">☕</div>
            <h3 className="text-xl font-semibold text-gray-200 mb-2">Supporter</h3>
            <p className="text-3xl font-bold text-electric mb-4">$5<span className="text-lg text-gray-500">/mo</span></p>
            <p className="text-gray-400 text-sm">Sponsors badge, early release notes, README shoutout</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="border border-purple-800/50 rounded-2xl p-8 bg-purple-900/10 backdrop-blur-sm text-center hover:border-purple-600/50 transition-colors relative"
          >
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-purple-600 rounded-full text-xs font-semibold">
              POPULAR
            </div>
            <div className="text-4xl mb-4">🔬</div>
            <h3 className="text-xl font-semibold text-gray-200 mb-2">Researcher</h3>
            <p className="text-3xl font-bold mb-4" style={{ color: '#A855F7' }}>$20<span className="text-lg text-gray-500">/mo</span></p>
            <p className="text-gray-400 text-sm">Inference logs for all 1,000 tasks, failure analysis, private Discord</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="border border-gray-800 rounded-2xl p-8 bg-gray-900/30 backdrop-blur-sm text-center hover:border-gray-600 transition-colors"
          >
            <div className="text-4xl mb-4">🏗️</div>
            <h3 className="text-xl font-semibold text-gray-200 mb-2">Architect</h3>
            <p className="text-3xl font-bold text-electric mb-4">$50<span className="text-lg text-gray-500">/mo</span></p>
            <p className="text-gray-400 text-sm">Experimental branches, DSL drafts, monthly roadmap, direct Q&A</p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="text-center"
        >
          <a
            href="https://github.com/sponsors/Ag3497120"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-10 py-4 rounded-xl font-semibold text-lg transition-all hover:scale-105"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #a855f7)', color: '#fff' }}
          >
            💜 Sponsor on GitHub
          </a>
          <a
            href="https://github.com/Ag3497120/verantyx-v6"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block ml-4 px-10 py-4 rounded-xl font-semibold text-lg border border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white transition-all hover:scale-105"
          >
            ⭐ Star on GitHub
          </a>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="text-center mt-8 text-gray-500 text-sm max-w-lg mx-auto"
        >
          You&apos;re not just funding a project — you&apos;re proving that a single researcher 
          with the right tools can compete with billion-dollar labs on the hardest AI benchmarks.
        </motion.p>
      </div>
    </section>
  );
}
