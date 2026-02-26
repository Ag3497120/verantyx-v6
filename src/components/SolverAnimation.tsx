'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const PHASES = [
  { name: 'Cross DSL', color: '#0EA5E9', desc: 'Neighborhood rules — 57% of solutions' },
  { name: 'Standalone Primitives', color: '#2ECC40', desc: 'Flip, rotate, crop, scale' },
  { name: 'Stamp / Pattern', color: '#F59E0B', desc: 'Object detection → pattern fill' },
  { name: 'Composite Chains', color: '#FF851B', desc: '2-3 step compositions' },
  { name: 'Iterative Cross', color: '#FF4136', desc: 'Residual learning without gradients' },
  { name: 'Puzzle Reasoning', color: '#F012BE', desc: 'Declarative spatial predicates' },
  { name: 'ProgramTree', color: '#B10DC9', desc: 'CEGIS-based synthesis' },
];

// Simple ARC-like grid
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

const ARC_COLORS = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#B10DC9'];

export default function SolverAnimation() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  // Map scroll progress to animation phase (0-1)
  const progress = useTransform(scrollYProgress, [0, 1], [0, 1]);
  const [progressVal, setProgressVal] = useState(0);

  useEffect(() => {
    return progress.on('change', (v) => setProgressVal(Math.max(0, Math.min(1, v))));
  }, [progress]);

  const currentPhase = Math.min(Math.floor(progressVal * 8), 7); // 0-7 (7 = output)
  const phaseProgress = (progressVal * 8) % 1;

  // Canvas for electricity effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;
    let t = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
      ctx.scale(2, 2);
    };
    resize();
    window.addEventListener('resize', resize);

    function drawLightning(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, intensity: number) {
      if (intensity <= 0) return;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      const segments = 8;
      const dx = (x2 - x1) / segments;
      const dy = (y2 - y1) / segments;
      for (let i = 1; i < segments; i++) {
        const jitter = (Math.random() - 0.5) * 20 * intensity;
        ctx.lineTo(x1 + dx * i + jitter * (dy === 0 ? 0 : 1), y1 + dy * i + jitter * (dx === 0 ? 0 : 1));
      }
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * intensity;
      ctx.shadowColor = color;
      ctx.shadowBlur = 15 * intensity;
      ctx.globalAlpha = intensity;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    }

    function animate() {
      if (!canvas || !ctx) return;
      t += 0.02;
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;

      // Draw cross structure outline
      const armLen = Math.min(w, h) * 0.35;
      const armW = 4;
      ctx.strokeStyle = `rgba(14, 165, 233, ${0.15 + progressVal * 0.2})`;
      ctx.lineWidth = armW;

      // Vertical arm
      ctx.beginPath();
      ctx.moveTo(cx, cy - armLen);
      ctx.lineTo(cx, cy + armLen);
      ctx.stroke();
      // Horizontal arm
      ctx.beginPath();
      ctx.moveTo(cx - armLen, cy);
      ctx.lineTo(cx + armLen, cy);
      ctx.stroke();

      // Draw electricity flowing through based on progress
      if (progressVal > 0.05) {
        const elecProgress = Math.min(progressVal * 1.5, 1);
        // Top to center
        drawLightning(ctx, cx, cy - armLen, cx, cy, PHASES[Math.min(currentPhase, 6)]?.color || '#0EA5E9', elecProgress * (0.5 + 0.5 * Math.sin(t * 3)));
        // Center to right
        if (progressVal > 0.3)
          drawLightning(ctx, cx, cy, cx + armLen * (progressVal - 0.3) / 0.7, cy, PHASES[Math.min(currentPhase, 6)]?.color || '#0EA5E9', (progressVal - 0.3) / 0.7 * (0.5 + 0.5 * Math.sin(t * 4)));
        // Center to bottom (output)
        if (progressVal > 0.7)
          drawLightning(ctx, cx, cy, cx, cy + armLen * (progressVal - 0.7) / 0.3, '#2ECC40', (progressVal - 0.7) / 0.3);
      }

      // Pulsing center
      const centerGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 20);
      centerGlow.addColorStop(0, `rgba(14, 165, 233, ${0.3 + progressVal * 0.5})`);
      centerGlow.addColorStop(1, 'rgba(14, 165, 233, 0)');
      ctx.fillStyle = centerGlow;
      ctx.fillRect(cx - 20, cy - 20, 40, 40);

      frame = requestAnimationFrame(animate);
    }
    animate();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
    };
  }, [progressVal, currentPhase]);

  return (
    <div ref={containerRef} style={{ minHeight: '300vh', position: 'relative' }}>
      <div style={{
        position: 'sticky',
        top: 0,
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}>
        {/* Canvas for electricity */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        />

        {/* Content overlay */}
        <div style={{ position: 'relative', zIndex: 10, textAlign: 'center', maxWidth: '600px', padding: '0 20px' }}>
          {/* Input grid */}
          <motion.div
            animate={{ opacity: progressVal > 0.02 ? 1 : 0, y: progressVal > 0.02 ? 0 : 30 }}
            transition={{ duration: 0.3 }}
            style={{ marginBottom: '24px' }}
          >
            <div style={{ color: '#888', fontSize: '12px', marginBottom: '8px' }}>INPUT</div>
            <div style={{ display: 'inline-grid', gridTemplateColumns: 'repeat(5, 24px)', gap: '2px' }}>
              {SAMPLE_INPUT.flat().map((c, i) => (
                <div key={i} style={{ width: 24, height: 24, background: ARC_COLORS[c], borderRadius: 2, border: '1px solid #222' }} />
              ))}
            </div>
          </motion.div>

          {/* Phase indicator */}
          <div style={{ marginBottom: '24px' }}>
            {PHASES.map((p, i) => (
              <motion.div
                key={i}
                animate={{
                  opacity: currentPhase === i ? 1 : currentPhase > i ? 0.3 : 0.1,
                  scale: currentPhase === i ? 1.1 : 1,
                  x: currentPhase === i ? 0 : currentPhase > i ? -10 : 10,
                }}
                transition={{ duration: 0.3 }}
                style={{
                  padding: '6px 16px',
                  margin: '4px 0',
                  borderRadius: '8px',
                  border: `1px solid ${currentPhase === i ? p.color : 'transparent'}`,
                  background: currentPhase === i ? `${p.color}22` : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  fontSize: '14px',
                }}
              >
                <span style={{ color: p.color, fontWeight: 700, minWidth: '24px' }}>{currentPhase > i ? '✓' : `${i + 1}`}</span>
                <span style={{ color: currentPhase >= i ? '#fff' : '#444' }}>{p.name}</span>
                {currentPhase === i && (
                  <span style={{ color: '#888', fontSize: '11px', marginLeft: 'auto' }}>{p.desc}</span>
                )}
              </motion.div>
            ))}
          </div>

          {/* Output grid */}
          <motion.div
            animate={{ opacity: progressVal > 0.85 ? 1 : 0, y: progressVal > 0.85 ? 0 : 30, scale: progressVal > 0.85 ? 1 : 0.8 }}
            transition={{ duration: 0.4 }}
          >
            <div style={{ color: '#2ECC40', fontSize: '12px', marginBottom: '8px' }}>OUTPUT ✓</div>
            <div style={{ display: 'inline-grid', gridTemplateColumns: 'repeat(5, 24px)', gap: '2px' }}>
              {SAMPLE_OUTPUT.flat().map((c, i) => (
                <div key={i} style={{ width: 24, height: 24, background: ARC_COLORS[c], borderRadius: 2, border: '1px solid #222' }} />
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
