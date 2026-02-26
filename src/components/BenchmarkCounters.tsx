'use client';

import { motion, useInView, useSpring, useTransform } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface CounterProps {
  end: number;
  suffix?: string;
  decimals?: number;
}

function AnimatedCounter({ end, suffix = '', decimals = 1 }: CounterProps) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });

  useEffect(() => {
    if (!isInView) return;

    let start = 0;
    const duration = 2000;
    const increment = end / (duration / 16);

    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(start);
      }
    }, 16);

    return () => clearInterval(timer);
  }, [isInView, end]);

  return (
    <span ref={ref}>
      {count.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export default function BenchmarkCounters() {
  return (
    <div className="benchmark-counters">
      <motion.div
        className="counters-container"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
      >
        <motion.div
          className="counter-card arc-counter"
          whileHover={{ scale: 1.05, rotateY: 5 }}
          transition={{ type: 'spring', stiffness: 300 }}
        >
          <div className="counter-label">ARC-AGI-2</div>
          <div className="counter-value">
            <AnimatedCounter end={20.7} suffix="%" decimals={1} />
          </div>
          <div className="counter-detail">207/1000 tasks</div>
          <div className="counter-glow arc-glow" />
        </motion.div>

        <motion.div
          className="counter-card hle-counter"
          whileHover={{ scale: 1.05, rotateY: -5 }}
          transition={{ type: 'spring', stiffness: 300 }}
        >
          <div className="counter-label">HLE</div>
          <div className="counter-value">
            <AnimatedCounter end={4.6} suffix="%" decimals={1} />
          </div>
          <div className="counter-detail">Hard Logic Evaluation</div>
          <div className="counter-glow hle-glow" />
        </motion.div>
      </motion.div>

      <motion.div
        className="benchmark-tagline"
        initial={{ opacity: 0, scale: 0.9 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.5, duration: 0.6 }}
      >
        <div className="tagline-icon">⚡</div>
        <div className="tagline-text">
          Zero neural networks. Every solution is a verifiable program.
        </div>
      </motion.div>

      <style jsx>{`
        .benchmark-counters {
          padding: 100px 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .counters-container {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 40px;
          max-width: 800px;
          margin: 0 auto;
        }

        .counter-card {
          position: relative;
          padding: 50px 40px;
          background: rgba(20, 20, 30, 0.6);
          border: 2px solid;
          border-radius: 20px;
          text-align: center;
          backdrop-filter: blur(10px);
          overflow: hidden;
          transform-style: preserve-3d;
          transition: all 0.3s ease;
        }

        .arc-counter {
          border-color: rgba(14, 165, 233, 0.5);
        }

        .hle-counter {
          border-color: rgba(168, 85, 247, 0.5);
        }

        .counter-card:hover {
          box-shadow: 0 20px 60px rgba(14, 165, 233, 0.3);
        }

        .counter-label {
          font-size: 18px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 2px;
          margin-bottom: 20px;
          opacity: 0.8;
        }

        .arc-counter .counter-label {
          color: #0EA5E9;
          text-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
        }

        .hle-counter .counter-label {
          color: #A855F7;
          text-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }

        .counter-value {
          font-size: 72px;
          font-weight: 900;
          line-height: 1;
          margin-bottom: 16px;
        }

        .arc-counter .counter-value {
          background: linear-gradient(135deg, #0EA5E9 0%, #60A5FA 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 40px rgba(14, 165, 233, 0.5);
        }

        .hle-counter .counter-value {
          background: linear-gradient(135deg, #A855F7 0%, #C084FC 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 40px rgba(168, 85, 247, 0.5);
        }

        .counter-detail {
          font-size: 14px;
          color: #9CA3AF;
          font-weight: 500;
        }

        .counter-glow {
          position: absolute;
          inset: -50%;
          opacity: 0.3;
          transition: opacity 0.3s ease;
          pointer-events: none;
        }

        .arc-glow {
          background: radial-gradient(circle, rgba(14, 165, 233, 0.3), transparent);
        }

        .hle-glow {
          background: radial-gradient(circle, rgba(168, 85, 247, 0.3), transparent);
        }

        .counter-card:hover .counter-glow {
          opacity: 0.6;
          animation: pulse 2s ease-in-out infinite;
        }

        .benchmark-tagline {
          margin-top: 60px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          padding: 24px 40px;
          background: rgba(14, 165, 233, 0.05);
          border: 1px solid rgba(14, 165, 233, 0.3);
          border-radius: 16px;
          backdrop-filter: blur(10px);
          max-width: 700px;
          margin-left: auto;
          margin-right: auto;
        }

        .tagline-icon {
          font-size: 32px;
          text-shadow: 0 0 20px #0EA5E9;
        }

        .tagline-text {
          font-size: 18px;
          font-weight: 600;
          color: #E5E7EB;
          line-height: 1.6;
        }

        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
            opacity: 0.6;
          }
          50% {
            transform: scale(1.1);
            opacity: 0.8;
          }
        }

        @media (max-width: 768px) {
          .counters-container {
            grid-template-columns: 1fr;
            gap: 30px;
          }

          .counter-card {
            padding: 40px 30px;
          }

          .counter-value {
            font-size: 56px;
          }

          .benchmark-tagline {
            flex-direction: column;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
          }

          .tagline-text {
            font-size: 16px;
          }
        }
      `}</style>
    </div>
  );
}
