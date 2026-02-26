'use client';

import { motion } from 'framer-motion';

const stats = [
  { value: '304', label: 'files' },
  { value: '100K', label: 'lines' },
  { value: '0.48s', label: 'per task' },
  { value: '0', label: 'dependencies' },
];

export default function Stats() {
  return (
    <section className="py-20 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="card-glow border-glow rounded-2xl p-12 bg-gray-900/50 backdrop-blur-sm">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-electric mb-2 font-mono">
                  {stat.value}
                </div>
                <div className="text-sm md:text-base text-gray-400 uppercase tracking-wider">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="mt-10 pt-10 border-t border-gray-800 text-center"
          >
            <p className="text-xl text-gray-300">
              Pure symbolic reasoning. <span className="text-electric font-semibold">No neural networks.</span>
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
