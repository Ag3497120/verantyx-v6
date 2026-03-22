'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

/**
 * DEATHNOTE-inspired section wrapper with:
 * - Silver metallic section title (like DEATHNOTE's セクションタイトル)
 * - Thin gradient border lines above/below title
 * - Scroll-triggered blur→clear reveal for children
 * - Consistent spacing and typography
 */
export function CinematicSection({
  title,
  children,
  className = '',
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.section
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6 }}
      className={`relative px-6 py-20 ${className}`}
    >
      <div className="max-w-4xl mx-auto">
        {/* Section title with cinematic borders */}
        <div className="mb-12">
          <div className="flex items-center justify-center gap-6 py-5">
            <div
              className="flex-1 h-px"
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.2))',
              }}
            />
            <motion.h2
              initial={{ opacity: 0, filter: 'blur(4px)' }}
              whileInView={{ opacity: 1, filter: 'blur(0px)' }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="text-base md:text-lg font-semibold tracking-[0.35em] uppercase"
              style={{
                background: 'linear-gradient(45deg, #6b7280 0%, #d1d5db 45%, #f9fafb 70%, #d1d5db 85%, #6b7280 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              {title}
            </motion.h2>
            <div
              className="flex-1 h-px"
              style={{
                background: 'linear-gradient(90deg, rgba(14,165,233,0.2), transparent)',
              }}
            />
          </div>
        </div>

        {/* Content with stagger reveal */}
        <motion.div
          initial={{ opacity: 0, y: 20, filter: 'blur(4px)' }}
          whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {children}
        </motion.div>
      </div>
    </motion.section>
  );
}

/**
 * A card that looks like DEATHNOTE's dark glass panels.
 * Subtle border, hover glow, radial gradient on hover.
 */
export function GlassCard({
  children,
  className = '',
  hover = true,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}) {
  return (
    <div
      className={`relative overflow-hidden group ${className}`}
      style={{
        background: 'rgba(10, 10, 20, 0.6)',
        border: '1px solid rgba(14, 165, 233, 0.08)',
        borderRadius: 16,
        padding: '1.75rem',
        transition: hover ? 'border-color 0.4s ease, box-shadow 0.4s ease' : undefined,
      }}
      onMouseEnter={hover ? (e) => {
        e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.25)';
        e.currentTarget.style.boxShadow = '0 0 40px rgba(14, 165, 233, 0.06)';
      } : undefined}
      onMouseLeave={hover ? (e) => {
        e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.08)';
        e.currentTarget.style.boxShadow = 'none';
      } : undefined}
    >
      {/* Hover gradient */}
      {hover && (
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at top left, rgba(14,165,233,0.03), transparent 60%)',
            borderRadius: 16,
          }}
        />
      )}
      <div className="relative z-10">{children}</div>
    </div>
  );
}

/**
 * Inline feature check item (✓ style from DEATHNOTE's credit/cast style).
 */
export function FeatureCheck({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-3 text-gray-400">
      <span className="mt-0.5" style={{ color: 'rgba(14, 165, 233, 0.6)' }}>✓</span>
      <span className="leading-relaxed">{children}</span>
    </div>
  );
}

/**
 * Page hero with cinematic entry and DEATHNOTE-style blur flash.
 */
export function PageHero({
  icon,
  title,
  subtitle,
  subtitle2,
  gradient,
  children,
}: {
  icon?: string;
  title: string;
  subtitle: string;
  subtitle2?: string;
  gradient: string;
  children?: ReactNode;
}) {
  return (
    <section className="relative min-h-[50vh] flex items-center justify-center px-6 pt-32 pb-20">
      {/* Ambient glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse 50% 35% at 50% 40%, rgba(14,165,233,0.04) 0%, transparent 70%)',
        }}
      />

      <div className="max-w-5xl mx-auto text-center relative z-10">
        {icon && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, filter: 'blur(8px)' }}
            animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
            transition={{ duration: 0.6 }}
            className="text-7xl md:text-8xl mb-6"
          >
            {icon}
          </motion.div>
        )}

        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mb-6 h-px w-24"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.4), transparent)',
            transformOrigin: 'center',
          }}
        />

        <motion.h1
          initial={{ opacity: 0, y: 20, filter: 'blur(6px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-5xl md:text-7xl font-black tracking-tight mb-4"
          style={{
            background: gradient,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          {title}
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 15, filter: 'blur(3px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-lg md:text-xl text-gray-400 mb-2"
        >
          {subtitle}
        </motion.p>

        {subtitle2 && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-base text-gray-500 mb-8"
          >
            {subtitle2}
          </motion.p>
        )}

        {children && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            {children}
          </motion.div>
        )}

        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 1, delay: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto mt-8 h-px w-24"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.4), transparent)',
            transformOrigin: 'center',
          }}
        />
      </div>
    </section>
  );
}
