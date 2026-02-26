'use client';

import { motion } from 'framer-motion';

const steps = [
  { label: 'Input Grid', icon: '▦' },
  { label: 'Piece Generation', icon: '⚙' },
  { label: 'Cross DSL', icon: '✦' },
  { label: 'Verification', icon: '✓' },
  { label: 'Output', icon: '▣' },
];

export default function HowItWorks() {
  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-6xl font-bold text-center mb-12 gradient-text"
        >
          How It Works
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl text-center text-gray-400 mb-16 max-w-3xl mx-auto"
        >
          Zero LLMs, zero neural networks — every solution is a verifiable program
        </motion.p>

        <div className="flex flex-wrap justify-center items-center gap-4 mb-16">
          {steps.map((step, index) => (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="flex items-center"
            >
              <div className="card-glow border-glow rounded-xl p-6 bg-gray-900/50 backdrop-blur-sm min-w-[180px] text-center">
                <div className="text-4xl mb-2 text-electric">{step.icon}</div>
                <p className="text-sm font-semibold text-gray-300">{step.label}</p>
              </div>
              {index < steps.length - 1 && (
                <div className="text-electric text-3xl mx-2 hidden md:block">→</div>
              )}
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="space-y-6 text-gray-300 text-lg max-w-4xl mx-auto"
        >
          <div className="card-glow border-glow rounded-xl p-8 bg-gray-900/50 backdrop-blur-sm">
            <h3 className="text-2xl font-semibold mb-4 text-electric">Cross DSL Approach</h3>
            <p className="leading-relaxed">
              Verantyx uses <span className="font-mono text-electric">neighborhood-rule</span> transformations
              to systematically explore the space of possible programs. Each transformation is a composable,
              verifiable operation — no probabilistic guessing, no gradient descent.
            </p>
          </div>

          <div className="card-glow border-glow rounded-xl p-8 bg-gray-900/50 backdrop-blur-sm">
            <h3 className="text-2xl font-semibold mb-4 text-electric">Verifiable by Design</h3>
            <p className="leading-relaxed">
              Every generated program can be inspected, understood, and verified. No black boxes.
              No unexplainable behavior. Just pure, deterministic symbolic reasoning.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
