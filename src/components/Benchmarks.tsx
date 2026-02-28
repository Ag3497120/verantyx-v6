'use client';

import { motion, useInView, useMotionValue, useSpring } from 'framer-motion';
import { useEffect, useRef } from 'react';

function AnimatedCounter({ value, suffix = '' }: { value: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const motionValue = useMotionValue(0);
  const springValue = useSpring(motionValue, { damping: 50, stiffness: 100 });
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  useEffect(() => {
    if (isInView) {
      motionValue.set(value);
    }
  }, [isInView, motionValue, value]);

  useEffect(() => {
    return springValue.on('change', (latest) => {
      if (ref.current) {
        ref.current.textContent = `${latest.toFixed(1)}${suffix}`;
      }
    });
  }, [springValue, suffix]);

  return <span ref={ref}>0{suffix}</span>;
}

export default function Benchmarks() {
  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-6xl font-bold text-center mb-20 gradient-text"
        >
          Performance
        </motion.h2>

        <div className="grid md:grid-cols-2 gap-12">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="card-glow border-glow rounded-2xl p-10 bg-gray-900/50 backdrop-blur-sm"
          >
            <h3 className="text-2xl font-semibold text-gray-300 mb-4">ARC-AGI-2</h3>
            <div className="text-8xl font-bold mb-4 text-glow text-electric">
              <AnimatedCounter value={82.6} suffix="%" />
            </div>
            <p className="text-xl text-gray-400 mb-2">826/1000 tasks solved</p>
            <p className="text-lg text-gray-500">Hybrid: Hand-crafted solvers + Claude Sonnet 4.5 program synthesis</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="card-glow border-glow rounded-2xl p-10 bg-gray-900/50 backdrop-blur-sm"
          >
            <h3 className="text-2xl font-semibold text-gray-300 mb-4">Humanity&apos;s Last Exam</h3>
            <div className="text-8xl font-bold mb-4 text-glow-purple" style={{ color: '#A855F7' }}>
              <AnimatedCounter value={4.6} suffix="%" />
            </div>
            <p className="text-xl text-gray-400 mb-2">Bias-free structural verification</p>
            <p className="text-lg text-gray-500">Zero hallucination guaranteed</p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <p className="text-2xl text-gray-300 font-light">
            Every solution is a <span className="text-electric font-semibold">verifiable program</span>
          </p>
        </motion.div>
      </div>
    </section>
  );
}
