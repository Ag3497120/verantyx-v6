'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface Metrics {
  score: { solved: number; total: number; percentage: number };
  engine: { lines_of_code: number; files: number; avg_speed_sec: number; primitives: number };
  timestamp: string;
  commit: string;
}

const METRICS_URL = 'https://raw.githubusercontent.com/Ag3497120/verantyx-v6/main/metrics.json';

// Fallback values (updated manually until first CI run)
const FALLBACK: Metrics = {
  score: { solved: 227, total: 1000, percentage: 22.7 },
  engine: { lines_of_code: 28332, files: 47, avg_speed_sec: 0.61, primitives: 60 },
  timestamp: new Date().toISOString(),
  commit: 'c80e978',
};

export default function LiveMetrics() {
  const [metrics, setMetrics] = useState<Metrics>(FALLBACK);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    fetch(METRICS_URL, { cache: 'no-store' })
      .then(r => r.json())
      .then((data: Metrics) => { setMetrics(data); setIsLive(true); })
      .catch(() => {});
  }, []);

  const fmt = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n);

  const stats = [
    { label: 'ARC-AGI-2 Score', value: `${metrics.score.percentage}%`, color: 'text-cyan-400' },
    { label: 'Tasks Solved', value: `${metrics.score.solved}/${metrics.score.total}`, color: 'text-green-400' },
    { label: 'Lines of Code', value: `${fmt(metrics.engine.lines_of_code)}+`, color: 'text-purple-400' },
    { label: 'Avg Latency', value: `${metrics.engine.avg_speed_sec}s`, color: 'text-amber-400' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="mb-12"
    >
      <div className="flex items-center justify-center gap-2 mb-6">
        <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
        <span className="text-xs text-gray-500 uppercase tracking-wider">
          {isLive ? 'Live from CI' : 'Latest Score'}
        </span>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.1 }}
            className="p-5 rounded-xl bg-gray-900/60 border border-gray-800 hover:border-gray-700 transition-colors text-center"
          >
            <div className={`text-3xl font-bold ${stat.color}`}>{stat.value}</div>
            <div className="text-xs text-gray-500 mt-2">{stat.label}</div>
          </motion.div>
        ))}
      </div>
      
      {isLive && (
        <p className="text-center text-xs text-gray-600 mt-4">
          Last benchmarked: {new Date(metrics.timestamp).toLocaleDateString('en-US', { 
            month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit'
          })}
          {' · '}
          <a 
            href={`https://github.com/Ag3497120/verantyx-v6/commit/${metrics.commit}`}
            target="_blank" 
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-cyan-400 transition-colors"
          >
            {metrics.commit.slice(0, 7)}
          </a>
        </p>
      )}
    </motion.div>
  );
}
