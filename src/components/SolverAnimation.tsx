'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  color: string;
}

const PHASES = [
  'Phase 1: Cross DSL',
  'Phase 2: Standalone Primitives',
  'Phase 3: Stamp',
  'Phase 4: Composite Chains',
  'Phase 5: Iterative Cross',
  'Phase 6: Puzzle Reasoning',
  'Phase 7: ProgramTree Synthesis',
];

export default function SolverAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [phase, setPhase] = useState(0);
  const [showGrid, setShowGrid] = useState(false);
  const [showLinks, setShowLinks] = useState(false);
  const animationRef = useRef<number>(0);
  const particlesRef = useRef<Particle[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Animation sequence
    setTimeout(() => setShowGrid(true), 500);

    // Start electricity animation
    let currentPhase = 0;
    const phaseInterval = setInterval(() => {
      if (currentPhase < PHASES.length) {
        setPhase(currentPhase);
        triggerElectricity(ctx, canvas, currentPhase);
        currentPhase++;
      } else {
        clearInterval(phaseInterval);
        setTimeout(() => {
          explodeToLinks(ctx, canvas);
        }, 1000);
      }
    }, 1500);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw and update particles
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.life -= 0.01;

        if (particle.life <= 0) return false;

        ctx.globalAlpha = particle.life;
        ctx.fillStyle = particle.color;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, 2, 0, Math.PI * 2);
        ctx.fill();

        return true;
      });

      ctx.globalAlpha = 1;
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      clearInterval(phaseInterval);
    };
  }, []);

  const triggerElectricity = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, phaseIndex: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Draw electricity bolts in cross pattern
    const directions = [
      { x: centerX, y: centerY - 200 }, // top
      { x: centerX + 200, y: centerY }, // right
      { x: centerX, y: centerY + 200 }, // bottom
      { x: centerX - 200, y: centerY }, // left
    ];

    directions.forEach(end => {
      drawElectricityBolt(ctx, centerX, centerY, end.x, end.y);

      // Add particles along the path
      for (let i = 0; i < 5; i++) {
        const t = Math.random();
        const x = centerX + (end.x - centerX) * t;
        const y = centerY + (end.y - centerY) * t;

        particlesRef.current.push({
          x,
          y,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          life: 1,
          color: `rgba(14, 165, 233, ${Math.random()})`,
        });
      }
    });
  };

  const drawElectricityBolt = (
    ctx: CanvasRenderingContext2D,
    startX: number,
    startY: number,
    endX: number,
    endY: number
  ) => {
    const segments = 10;
    const points = [];

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const x = startX + (endX - startX) * t + (Math.random() - 0.5) * 20;
      const y = startY + (endY - startY) * t + (Math.random() - 0.5) * 20;
      points.push({ x, y });
    }

    // Draw glow
    ctx.shadowBlur = 20;
    ctx.shadowColor = '#0EA5E9';
    ctx.strokeStyle = '#0EA5E9';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.stroke();

    // Draw core
    ctx.shadowBlur = 10;
    ctx.strokeStyle = '#60A5FA';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.stroke();
    ctx.shadowBlur = 0;
  };

  const explodeToLinks = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Create explosion particles
    for (let i = 0; i < 100; i++) {
      const angle = (Math.PI * 2 * i) / 100;
      const speed = 2 + Math.random() * 3;

      particlesRef.current.push({
        x: centerX,
        y: centerY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        color: `rgba(${Math.random() > 0.5 ? '14, 165, 233' : '168, 85, 247'}, ${Math.random()})`,
      });
    }

    setTimeout(() => {
      setShowLinks(true);
    }, 1000);
  };

  return (
    <div className="solver-animation-container">
      <canvas ref={canvasRef} className="solver-canvas" />

      <AnimatePresence>
        {showGrid && !showLinks && (
          <motion.div
            className="arc-grid"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            transition={{ duration: 0.5 }}
          >
            <div className="grid-title">ARC Input Grid</div>
            <div className="grid-cells">
              {[...Array(25)].map((_, i) => (
                <motion.div
                  key={i}
                  className="grid-cell"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.02 }}
                  style={{
                    backgroundColor: Math.random() > 0.5 ? '#0EA5E9' : '#A855F7',
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {phase < PHASES.length && (
        <motion.div
          className="phase-label"
          key={phase}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {PHASES[phase]}
        </motion.div>
      )}

      <AnimatePresence>
        {showLinks && (
          <motion.div
            className="solved-grid"
            initial={{ opacity: 0, scale: 0.5, y: -100 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.8, type: 'spring' }}
          >
            <div className="grid-title">Solved Output</div>
            <div className="grid-cells">
              {[...Array(25)].map((_, i) => (
                <motion.div
                  key={i}
                  className="grid-cell solved"
                  initial={{ scale: 0, rotate: 0 }}
                  animate={{ scale: 1, rotate: 360 }}
                  transition={{ delay: i * 0.02, duration: 0.5 }}
                  style={{
                    backgroundColor: Math.random() > 0.7 ? '#10B981' : '#0EA5E9',
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style jsx>{`
        .solver-animation-container {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          z-index: 10;
          pointer-events: none;
        }

        .solver-canvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
        }

        .arc-grid {
          position: absolute;
          top: 10%;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(10, 10, 15, 0.8);
          border: 2px solid rgba(14, 165, 233, 0.5);
          border-radius: 12px;
          padding: 20px;
          backdrop-filter: blur(10px);
        }

        .solved-grid {
          position: absolute;
          bottom: 10%;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(10, 10, 15, 0.9);
          border: 2px solid rgba(16, 185, 129, 0.5);
          border-radius: 12px;
          padding: 20px;
          backdrop-filter: blur(10px);
          box-shadow: 0 0 40px rgba(16, 185, 129, 0.3);
        }

        .grid-title {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 12px;
          text-align: center;
          color: #0EA5E9;
          text-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
        }

        .solved-grid .grid-title {
          color: #10B981;
          text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }

        .grid-cells {
          display: grid;
          grid-template-columns: repeat(5, 24px);
          grid-template-rows: repeat(5, 24px);
          gap: 4px;
        }

        .grid-cell {
          width: 24px;
          height: 24px;
          border-radius: 3px;
          box-shadow: 0 0 10px currentColor;
        }

        .grid-cell.solved {
          box-shadow: 0 0 15px rgba(16, 185, 129, 0.6);
        }

        .phase-label {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: 24px;
          font-weight: 700;
          color: #0EA5E9;
          text-shadow: 0 0 20px rgba(14, 165, 233, 0.8),
                       0 0 40px rgba(14, 165, 233, 0.5);
          background: rgba(10, 10, 15, 0.8);
          padding: 16px 32px;
          border-radius: 8px;
          border: 1px solid rgba(14, 165, 233, 0.5);
          backdrop-filter: blur(10px);
        }

        @media (max-width: 768px) {
          .phase-label {
            font-size: 16px;
            padding: 12px 24px;
          }

          .grid-cells {
            grid-template-columns: repeat(5, 16px);
            grid-template-rows: repeat(5, 16px);
            gap: 3px;
          }

          .grid-cell {
            width: 16px;
            height: 16px;
          }
        }
      `}</style>
    </div>
  );
}
