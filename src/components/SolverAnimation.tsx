'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

const PHASES = [
  { name: 'Cross DSL', color: '#0EA5E9', desc: 'Neighborhood rules — 57% of solutions', icon: '✦' },
  { name: 'Standalone Primitives', color: '#2ECC40', desc: 'Flip, rotate, crop, scale', icon: '◆' },
  { name: 'Stamp / Pattern', color: '#F59E0B', desc: 'Object detection → pattern fill', icon: '▣' },
  { name: 'Composite Chains', color: '#FF851B', desc: '2-3 step compositions', icon: '⟐' },
  { name: 'Iterative Cross', color: '#FF4136', desc: 'Residual learning without gradients', icon: '⟳' },
  { name: 'Puzzle Reasoning', color: '#F012BE', desc: 'Declarative spatial predicates', icon: '⧉' },
  { name: 'ProgramTree', color: '#B10DC9', desc: 'CEGIS-based synthesis', icon: '⌬' },
];

const ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#B10DC9'];

const SAMPLE_INPUT = [
  [0,0,1,0,0],
  [0,1,1,1,0],
  [1,1,0,1,1],
  [0,1,1,1,0],
  [0,0,1,0,0],
];
const SAMPLE_OUTPUT = [
  [0,0,2,0,0],
  [0,2,2,2,0],
  [2,2,3,2,2],
  [0,2,2,2,0],
  [0,0,2,0,0],
];

function Grid({ grid, label, color = '#888' }: { grid: number[][]; label: string; color?: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ color, fontSize: '11px', marginBottom: '6px', letterSpacing: '2px', fontWeight: 600 }}>{label}</div>
      <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${grid[0].length}, 20px)`, gap: '1px' }}>
        {grid.flat().map((c, i) => (
          <div key={i} style={{ width: 20, height: 20, background: ARC_COLORS[c], borderRadius: 1, border: '1px solid #1a1a1a' }} />
        ))}
      </div>
    </div>
  );
}

function PhaseCard({ phase, index, isLast }: { phase: typeof PHASES[0]; index: number; isLast: boolean }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: false, margin: '-20% 0px -20% 0px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, x: index % 2 === 0 ? -40 : 40 }}
      animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0.15, x: index % 2 === 0 ? -20 : 20 }}
      transition={{ duration: 0.4 }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        padding: '16px 24px',
        background: isInView ? `${phase.color}11` : 'transparent',
        border: `1px solid ${isInView ? phase.color + '44' : '#222'}`,
        borderRadius: '12px',
        transition: 'border-color 0.3s, background 0.3s',
        maxWidth: '500px',
        margin: '0 auto',
      }}
    >
      {/* Connector line */}
      <div style={{ position: 'relative' }}>
        <div style={{
          width: 36, height: 36, borderRadius: '50%',
          border: `2px solid ${phase.color}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '16px',
          background: isInView ? `${phase.color}33` : 'transparent',
          boxShadow: isInView ? `0 0 20px ${phase.color}44` : 'none',
          transition: 'all 0.3s',
        }}>
          {phase.icon}
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ color: phase.color, fontWeight: 700, fontSize: '15px' }}>
          Phase {index + 1}: {phase.name}
        </div>
        <div style={{ color: '#888', fontSize: '12px', marginTop: '2px' }}>{phase.desc}</div>
      </div>
      {/* Electric dot */}
      {isInView && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: phase.color,
            boxShadow: `0 0 12px ${phase.color}`,
          }}
        />
      )}
    </motion.div>
  );
}

export default function SolverAnimation() {
  const inputRef = useRef(null);
  const outputRef = useRef(null);
  const inputInView = useInView(inputRef, { once: false, margin: '-10% 0px' });
  const outputInView = useInView(outputRef, { once: false, margin: '-10% 0px' });

  return (
    <div style={{ padding: '80px 20px 100px', maxWidth: '700px', margin: '0 auto' }}>
      {/* Title */}
      <motion.h2
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: false }}
        style={{
          textAlign: 'center', fontSize: '32px', fontWeight: 800, marginBottom: '48px',
          background: 'linear-gradient(135deg, #0EA5E9, #A855F7)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
        }}
      >
        How the Cross Engine Solves
      </motion.h2>

      {/* Input Grid */}
      <motion.div
        ref={inputRef}
        initial={{ opacity: 0, y: 20 }}
        animate={inputInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
        transition={{ duration: 0.4 }}
        style={{ textAlign: 'center', marginBottom: '32px' }}
      >
        <Grid grid={SAMPLE_INPUT} label="INPUT" color="#0EA5E9" />
      </motion.div>

      {/* Vertical connector */}
      <div style={{ width: '2px', height: '24px', background: 'linear-gradient(180deg, #0EA5E9, #0EA5E944)', margin: '0 auto 16px' }} />

      {/* Phases */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
        {PHASES.map((phase, i) => (
          <div key={i}>
            <PhaseCard phase={phase} index={i} isLast={i === PHASES.length - 1} />
            {i < PHASES.length - 1 && (
              <div style={{ width: '2px', height: '8px', background: `linear-gradient(180deg, ${phase.color}44, ${PHASES[i+1].color}44)`, margin: '0 auto' }} />
            )}
          </div>
        ))}
      </div>

      {/* Vertical connector */}
      <div style={{ width: '2px', height: '24px', background: 'linear-gradient(180deg, #B10DC944, #2ECC40)', margin: '0 auto 16px' }} />

      {/* Output Grid */}
      <motion.div
        ref={outputRef}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={outputInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.8 }}
        transition={{ duration: 0.4 }}
        style={{ textAlign: 'center' }}
      >
        <Grid grid={SAMPLE_OUTPUT} label="OUTPUT ✓" color="#2ECC40" />
        <motion.div
          animate={outputInView ? { opacity: 1 } : { opacity: 0 }}
          style={{ color: '#2ECC40', fontSize: '14px', marginTop: '12px', fontWeight: 600 }}
        >
          Solution verified on all training pairs
        </motion.div>
      </motion.div>
    </div>
  );
}
