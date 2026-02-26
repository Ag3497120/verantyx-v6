'use client';

import { motion } from 'framer-motion';

const phases = [
  {
    number: '1',
    title: 'Cross DSL',
    subtitle: 'Neighborhood Rules',
    icon: '◈',
  },
  {
    number: '2',
    title: 'Standalone Primitives',
    subtitle: 'Basic Operations',
    icon: '◆',
  },
  {
    number: '3',
    title: 'Stamp/Pattern Fill',
    subtitle: 'Repetition Detection',
    icon: '◉',
  },
  {
    number: '4',
    title: 'Composite Chains',
    subtitle: 'Multi-step Programs',
    icon: '⬡',
  },
  {
    number: '5',
    title: 'Iterative Cross',
    subtitle: 'Recursive Application',
    icon: '⟡',
  },
  {
    number: '6',
    title: 'Puzzle Reasoning',
    subtitle: 'Meta-level Strategies',
    icon: '⬢',
  },
  {
    number: '7',
    title: 'ProgramTree Synthesis',
    subtitle: 'Complete Solution',
    icon: '◎',
  },
];

export default function Architecture() {
  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20">
      <div className="max-w-7xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-6xl font-bold text-center mb-12 gradient-text"
        >
          7-Phase Architecture
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl text-center text-gray-400 mb-16 max-w-3xl mx-auto"
        >
          Hierarchical solving pipeline from simple transformations to complex synthesis
        </motion.p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {phases.map((phase, index) => (
            <motion.div
              key={phase.number}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
              className="card-glow border-glow rounded-xl p-6 bg-gray-900/50 backdrop-blur-sm relative overflow-hidden group"
            >
              <div className="absolute top-0 right-0 text-8xl font-bold text-electric/5 -mr-4 -mt-2">
                {phase.number}
              </div>

              <div className="relative z-10">
                <div className="text-5xl mb-4 text-electric">{phase.icon}</div>
                <div className="text-sm text-gray-500 mb-1">Phase {phase.number}</div>
                <h3 className="text-xl font-bold text-gray-200 mb-2">{phase.title}</h3>
                <p className="text-sm text-gray-400">{phase.subtitle}</p>
              </div>

              <div className="absolute inset-0 bg-gradient-to-br from-electric/5 to-purple-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mt-16 text-center"
        >
          <p className="text-xl text-gray-400">
            Each phase builds on the previous, creating increasingly sophisticated programs
          </p>
        </motion.div>
      </div>
    </section>
  );
}
