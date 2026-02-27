'use client';

import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';

const data = [
  { version: 'v19', score: 11.3 },
  { version: 'v25', score: 13.8 },
  { version: 'v32', score: 15.5 },
  { version: 'v40', score: 17.2 },
  { version: 'v47', score: 18.9 },
  { version: 'v53', score: 20.7 },
  { version: 'v59', score: 22.1 },
];

export default function ScoreChart() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  const maxScore = 30;
  const chartHeight = 300;
  const chartWidth = 600;
  const padding = 40;
  const barWidth = (chartWidth - padding * 2) / data.length - 20;

  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-5xl md:text-6xl font-bold text-center mb-12 gradient-text"
        >
          Evolution
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-xl text-center text-gray-400 mb-16"
        >
          From 11.3% to 22.1% — continuous improvement through symbolic refinement
        </motion.p>

        <div className="card-glow border-glow rounded-2xl p-10 bg-gray-900/50 backdrop-blur-sm overflow-x-auto">
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight + 80}`}
            className="w-full max-w-3xl mx-auto"
            style={{ minWidth: '500px' }}
          >
            {/* Grid lines */}
            {[0, 5, 10, 15, 20, 25].map((tick) => {
              const y = chartHeight - (tick / maxScore) * chartHeight + 30;
              return (
                <g key={tick}>
                  <line
                    x1={padding}
                    y1={y}
                    x2={chartWidth - padding}
                    y2={y}
                    stroke="rgba(75, 85, 99, 0.3)"
                    strokeWidth="1"
                  />
                  <text
                    x={padding - 10}
                    y={y + 5}
                    fill="#9CA3AF"
                    fontSize="12"
                    textAnchor="end"
                  >
                    {tick}%
                  </text>
                </g>
              );
            })}

            {/* Bars */}
            {data.map((item, index) => {
              const x = padding + index * ((chartWidth - padding * 2) / data.length) + 10;
              const barHeight = (item.score / maxScore) * chartHeight;
              const y = chartHeight - barHeight + 30;

              return (
                <g key={item.version}>
                  <motion.rect
                    x={x}
                    y={chartHeight + 30}
                    width={barWidth}
                    height={0}
                    fill="url(#barGradient)"
                    rx="4"
                    initial={{ height: 0, y: chartHeight + 30 }}
                    animate={isInView ? { height: barHeight, y } : {}}
                    transition={{ duration: 1, delay: index * 0.1, ease: 'easeOut' }}
                  />
                  <motion.text
                    x={x + barWidth / 2}
                    y={y - 10}
                    fill="#0EA5E9"
                    fontSize="14"
                    fontWeight="bold"
                    textAnchor="middle"
                    initial={{ opacity: 0 }}
                    animate={isInView ? { opacity: 1 } : {}}
                    transition={{ duration: 0.5, delay: index * 0.1 + 0.5 }}
                  >
                    {item.score}%
                  </motion.text>
                  <text
                    x={x + barWidth / 2}
                    y={chartHeight + 50}
                    fill="#9CA3AF"
                    fontSize="14"
                    textAnchor="middle"
                  >
                    {item.version}
                  </text>
                </g>
              );
            })}

            {/* Gradient definition */}
            <defs>
              <linearGradient id="barGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#0EA5E9" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#A855F7" stopOpacity="0.8" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mt-12 text-center"
        >
          <p className="text-lg text-gray-400">
            Each version introduces new symbolic transformations and reasoning strategies
          </p>
        </motion.div>
      </div>
    </section>
  );
}
