'use client';

import { motion } from 'framer-motion';

export default function Hero() {
  return (
    <section className="flex items-center justify-center px-6 relative">
      <div className="max-w-5xl mx-auto text-center">
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="text-8xl md:text-9xl font-bold mb-8 gradient-text"
        >
          Verantyx
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3, ease: 'easeOut' }}
          className="text-3xl md:text-4xl text-gray-300 mb-12"
        >
          82.6% on ARC-AGI-2
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
          className="space-y-4 text-lg md:text-xl text-gray-400"
        >
          <p>Hand-crafted solvers + Claude Sonnet 4.5 program synthesis</p>
          <p>Every solution is a verifiable Python program.</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.9 }}
          className="mt-16"
        >
          <div className="inline-block px-8 py-4 border-glow rounded-lg bg-gray-900/50 backdrop-blur-sm">
            <p className="text-2xl font-mono text-electric">
              826/1000 tasks solved — LLM writes code, system verifies
            </p>
          </div>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1.2 }}
          className="mt-8 text-sm text-gray-500"
        >
          Built by{' '}
          <a href="https://x.com/Koffdai" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">kofdai</a>
          {' '}×{' '}
          <a href="https://openclaw.ai" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">OpenClaw</a>
          {' '}— Human Logic + AI Implementation
        </motion.p>
      </div>
    </section>
  );
}
