'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function CinematicOpening() {
  const [phase, setPhase] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    document.body.style.overflowY = 'hidden';

    const timers = [
      setTimeout(() => setPhase(1), 300),   // "V" flash
      setTimeout(() => setPhase(2), 800),   // full word builds
      setTimeout(() => setPhase(3), 2200),  // tagline
      setTimeout(() => setPhase(4), 3600),  // fade out
      setTimeout(() => {
        setVisible(false);
        document.body.style.overflowY = 'visible';
      }, 4200),
    ];

    return () => timers.forEach(clearTimeout);
  }, []);

  if (!visible) return null;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: '#000',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: 24,
          }}
        >
          {/* Radial pulse behind text */}
          {phase >= 1 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.3 }}
              animate={{ opacity: [0, 0.4, 0.15], scale: [0.3, 1.5, 2] }}
              transition={{ duration: 2, ease: 'easeOut' }}
              style={{
                position: 'absolute',
                width: '60vw',
                height: '60vw',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(14,165,233,0.25) 0%, rgba(124,58,237,0.1) 40%, transparent 70%)',
                pointerEvents: 'none',
              }}
            />
          )}

          {/* Letter flash */}
          {phase >= 1 && phase < 2 && (
            <motion.span
              initial={{ opacity: 0, scale: 1.8, filter: 'blur(12px) brightness(3)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px) brightness(1)' }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              style={{
                fontSize: 'clamp(5rem, 15vw, 12rem)',
                fontWeight: 900,
                letterSpacing: '-0.03em',
                background: 'linear-gradient(135deg, #0EA5E9, #7C3AED)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              V
            </motion.span>
          )}

          {/* Full title */}
          {phase >= 2 && (
            <motion.h1
              initial={{ opacity: 0, scale: 1.3, filter: 'blur(8px) brightness(2)' }}
              animate={
                phase >= 4
                  ? { opacity: 0, scale: 0.95, filter: 'blur(6px) brightness(0.5)' }
                  : { opacity: 1, scale: 1, filter: 'blur(0px) brightness(1)' }
              }
              transition={{ duration: phase >= 4 ? 0.5 : 0.6, ease: 'easeOut' }}
              style={{
                fontSize: 'clamp(3rem, 10vw, 8rem)',
                fontWeight: 900,
                letterSpacing: '-0.03em',
                background: 'linear-gradient(135deg, #0EA5E9 0%, #7C3AED 50%, #06B6D4 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textShadow: 'none',
              }}
            >
              Verantyx
            </motion.h1>
          )}

          {/* Tagline */}
          {phase >= 3 && (
            <motion.p
              initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
              animate={
                phase >= 4
                  ? { opacity: 0, filter: 'blur(4px)' }
                  : { opacity: 1, y: 0, filter: 'blur(0px)' }
              }
              transition={{ duration: phase >= 4 ? 0.4 : 0.8, ease: 'easeOut' }}
              style={{
                fontSize: 'clamp(0.9rem, 2vw, 1.3rem)',
                color: 'rgba(148, 163, 184, 0.9)',
                letterSpacing: '0.3em',
                fontWeight: 300,
                textTransform: 'uppercase',
              }}
            >
              Explore Our Projects
            </motion.p>
          )}

          {/* Horizontal scan line */}
          {phase >= 1 && phase < 4 && (
            <motion.div
              initial={{ scaleX: 0, opacity: 0 }}
              animate={{ scaleX: 1, opacity: [0, 0.6, 0] }}
              transition={{ duration: 1.5, ease: 'easeInOut' }}
              style={{
                position: 'absolute',
                top: '50%',
                left: '10%',
                right: '10%',
                height: 1,
                background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.6), transparent)',
                transformOrigin: 'center',
              }}
            />
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
