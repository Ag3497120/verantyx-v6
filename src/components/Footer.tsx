'use client';

import { motion } from 'framer-motion';

export default function Footer() {
  return (
    <footer className="relative py-20 px-6 overflow-hidden">
      {/* Top border — cinematic thin line */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.25) 30%, rgba(124,58,237,0.25) 70%, transparent)',
        }}
      />

      <div className="max-w-6xl mx-auto relative z-10">
        {/* Logo + GitHub row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="flex flex-col md:flex-row items-center justify-between gap-8"
        >
          <div className="text-center md:text-left">
            <h3
              className="text-4xl font-black tracking-tight mb-2"
              style={{
                background: 'linear-gradient(135deg, #0EA5E9, #7C3AED, #06B6D4)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Verantyx
            </h3>
            <p className="text-gray-500 text-sm tracking-wider uppercase">Symbolic Reasoning Engine</p>
          </div>

          <div className="flex flex-col items-center md:items-end gap-4">
            <a
              href="https://github.com/Ag3497120/verantyx-v6"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl transition-all duration-300"
              style={{
                background: 'rgba(14, 165, 233, 0.06)',
                border: '1px solid rgba(14, 165, 233, 0.15)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.4)';
                e.currentTarget.style.boxShadow = '0 0 30px rgba(14, 165, 233, 0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.15)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path
                  fillRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  clipRule="evenodd"
                />
              </svg>
              <span className="text-gray-300 font-semibold text-sm">★ Star on GitHub</span>
            </a>

            <p className="text-gray-600 text-xs">
              Built by{' '}
              <a href="https://x.com/Koffdai" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-cyan-400 transition-colors">@Koffdai</a>
              {' '}×{' '}
              <a href="https://openclaw.ai" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-cyan-400 transition-colors">OpenClaw</a>
            </p>
          </div>
        </motion.div>

        {/* Divider */}
        <div
          className="my-12 h-px"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(55,65,81,0.5) 30%, rgba(55,65,81,0.5) 70%, transparent)',
          }}
        />

        {/* Link Groups */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-10 text-sm"
        >
          {/* Projects */}
          <div>
            <h4
              className="font-semibold mb-4 text-xs tracking-[0.3em] uppercase"
              style={{ color: 'rgba(14, 165, 233, 0.7)' }}
            >
              Projects
            </h4>
            <div className="flex flex-col gap-3">
              <FooterLink href="/verantyx" label="Verantyx Engine" />
              <FooterLink href="/verantyx-cli" label="Verantyx-CLI" />
              <FooterLink href="/jcross-language" label=".jcross Language" />
            </div>
          </div>

          {/* Apps */}
          <div>
            <h4
              className="font-semibold mb-4 text-xs tracking-[0.3em] uppercase"
              style={{ color: 'rgba(14, 165, 233, 0.7)' }}
            >
              Apps
            </h4>
            <div className="flex flex-col gap-3">
              <FooterLink href="/apps/pakupaku-fishing" label="パクパク釣り" />
              <FooterLink href="/apps/mouth-eat" label="MouthEat" />
            </div>
          </div>

          {/* Legal */}
          <div>
            <h4
              className="font-semibold mb-4 text-xs tracking-[0.3em] uppercase"
              style={{ color: 'rgba(14, 165, 233, 0.7)' }}
            >
              Legal
            </h4>
            <div className="flex flex-col gap-3">
              <FooterLink href="/privacy" label="Privacy Policy" />
              <FooterLink href="/support" label="Support" />
              <FooterLink href="/terms" label="Terms" />
            </div>
          </div>
        </motion.div>

        {/* Divider */}
        <div
          className="my-12 h-px"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(55,65,81,0.3) 30%, rgba(55,65,81,0.3) 70%, transparent)',
          }}
        />

        {/* Copyright */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-center"
        >
          <p className="text-gray-600 text-xs tracking-wide">
            © 2025 Verantyx. Pure symbolic reasoning — zero neural networks, zero GPUs.
          </p>
          <p className="mt-2 text-gray-700 text-xs">
            Human architecture × AI implementation — kofdai designs the algorithms,{' '}
            <a href="https://openclaw.ai" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-cyan-400 transition-colors">
              OpenClaw (Claude)
            </a>{' '}
            helps implement and test them.
          </p>
        </motion.div>
      </div>
    </footer>
  );
}

function FooterLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      className="text-gray-500 transition-colors duration-300"
      style={{ textDecoration: 'none' }}
      onMouseEnter={(e) => { e.currentTarget.style.color = 'rgba(14, 165, 233, 0.8)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.color = '#6b7280'; }}
    >
      {label}
    </a>
  );
}
