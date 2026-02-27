'use client';

import { motion } from 'framer-motion';
import LiveMetrics from './LiveMetrics';

export default function Footer() {
  return (
    <footer className="py-16 px-6 border-t border-gray-800">
      <div className="max-w-6xl mx-auto">
        {/* Live Metrics from CI */}
        <LiveMetrics />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="flex flex-col md:flex-row items-center justify-between gap-8"
        >
          <div className="text-center md:text-left">
            <h3 className="text-3xl font-bold gradient-text mb-2">Verantyx</h3>
            <p className="text-gray-400">Symbolic Reasoning Engine</p>
          </div>

          <div className="flex flex-col items-center md:items-end gap-4">
            <a
              href="https://github.com/Ag3497120/verantyx-v6"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 border-glow rounded-lg bg-gray-900/50 backdrop-blur-sm hover:bg-gray-800/50 transition-colors"
            >
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path
                  fillRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  clipRule="evenodd"
                />
              </svg>
              <span className="text-gray-300 font-semibold">★ Star on GitHub</span>
            </a>

            <p className="text-gray-500 text-sm">
              Built by <a href="https://x.com/Koffdai" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">@Koffdai</a> × <a href="https://openclaw.ai" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-cyan-400 transition-colors">OpenClaw</a>
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-12 pt-8 border-t border-gray-800 text-center text-gray-500 text-sm"
        >
          <p>© 2025 Verantyx. Pure symbolic reasoning — zero neural networks, zero GPUs.</p>
          <p className="mt-2 text-gray-600 text-xs">
            Human architecture × AI implementation — kofdai designs the algorithms, <a href="https://openclaw.ai" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-cyan-400 transition-colors">OpenClaw (Claude)</a> helps implement and test them.
          </p>
        </motion.div>
      </div>
    </footer>
  );
}
