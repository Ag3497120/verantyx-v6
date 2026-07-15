'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLanguage } from '@/lib/i18n';

const CLI_GITHUB = 'https://github.com/Ag3497120/verantyx-cli';

export default function StickyCliCta() {
  const { lang } = useLanguage();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > 420);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ y: 80, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 80, opacity: 0 }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          style={{
            position: 'fixed',
            bottom: 20,
            left: 16,
            right: 16,
            zIndex: 900,
            display: 'flex',
            justifyContent: 'center',
            pointerEvents: 'none',
          }}
        >
          <a
            href={CLI_GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent"
            style={{
              pointerEvents: 'auto',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 10,
              padding: '12px 20px',
              borderRadius: 999,
              textDecoration: 'none',
              fontWeight: 600,
              fontSize: '0.88em',
              backdropFilter: 'blur(12px)',
              boxShadow: '0 8px 40px rgba(0,0,0,0.45)',
            }}
          >
            <span aria-hidden>★</span>
            {lang === 'ja' ? 'Verantyx-CLI on GitHub' : 'Star Verantyx-CLI on GitHub'}
          </a>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
