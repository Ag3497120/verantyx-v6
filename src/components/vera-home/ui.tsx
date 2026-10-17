'use client';

import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

/* Small building blocks, same proportions as /vera/ so the two pages read as one site.
 * Colours come only from the globals.css tokens, so dark and light both work. */

export function Section({
  id,
  label,
  children,
  wide,
}: {
  id?: string;
  label: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section id={id} className="relative px-5 sm:px-6 py-12 sm:py-16 md:py-20 scroll-mt-16">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-40px' }}
        transition={{ duration: 0.55 }}
        className={`mx-auto w-full ${wide ? 'max-w-6xl' : 'max-w-4xl'}`}
      >
        <div className="flex items-center gap-4 mb-6 sm:mb-8">
          <span
            className="h-px w-8 sm:w-12 shrink-0"
            style={{
              background: 'linear-gradient(90deg, rgba(var(--accent-rgb), 0.5), transparent)',
            }}
          />
          <span
            className="uppercase font-semibold"
            style={{ fontSize: '0.66rem', letterSpacing: '0.26em', color: 'var(--ink-4)' }}
          >
            {label}
          </span>
        </div>
        {children}
      </motion.div>
    </section>
  );
}

export function H2({ ja, children }: { ja: boolean; children: ReactNode }) {
  return (
    <h2
      className="font-display font-bold"
      style={{
        color: 'var(--ink)',
        fontSize: ja ? 'clamp(1.4rem, 4.4vw, 2.1rem)' : 'clamp(1.55rem, 5vw, 2.35rem)',
        lineHeight: ja ? 1.45 : 1.15,
        letterSpacing: ja ? '0' : '-0.02em',
        marginBottom: '1.25rem',
      }}
    >
      {children}
    </h2>
  );
}

export function Body({
  ja,
  children,
  className = '',
  muted,
}: {
  ja: boolean;
  children: ReactNode;
  className?: string;
  muted?: boolean;
}) {
  return (
    <p
      className={`max-w-2xl ${className}`}
      style={{
        color: muted ? 'var(--ink-4)' : 'var(--ink-3)',
        fontSize: muted ? 'clamp(0.8rem, 2.1vw, 0.875rem)' : 'clamp(0.9rem, 2.3vw, 1rem)',
        lineHeight: ja ? 1.95 : 1.72,
      }}
    >
      {children}
    </p>
  );
}

export function Code({ children }: { children: string }) {
  return (
    <div
      className="rounded-xl border overflow-x-auto"
      style={{ borderColor: 'rgba(var(--accent-rgb), 0.18)', background: 'var(--surface-2)' }}
    >
      <pre
        className="px-4 py-4 sm:px-6 sm:py-5 font-mono"
        style={{
          fontSize: 'clamp(0.72rem, 2.1vw, 0.85rem)',
          lineHeight: 1.75,
          color: 'var(--ink-2)',
          margin: 0,
        }}
      >
        {children}
      </pre>
    </div>
  );
}
