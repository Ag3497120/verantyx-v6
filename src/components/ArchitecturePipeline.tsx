'use client';

import { motion } from 'framer-motion';

const phases = [
  {
    number: 1,
    title: 'Cross DSL',
    subtitle: 'Neighborhood Rules',
    description: '57% of solutions',
    color: '#0EA5E9',
    icon: '十',
  },
  {
    number: 2,
    title: 'Standalone Primitives',
    subtitle: 'flip, rotate, crop',
    description: 'Pure transformations',
    color: '#10B981',
    icon: '⟲',
  },
  {
    number: 3,
    title: 'Stamp',
    subtitle: 'Pattern Fill',
    description: 'Template matching',
    color: '#F59E0B',
    icon: '▣',
  },
  {
    number: 4,
    title: 'Composite Chains',
    subtitle: 'Multi-step pipelines',
    description: 'Sequential operations',
    color: '#EF4444',
    icon: '⊕',
  },
  {
    number: 5,
    title: 'Iterative Cross',
    subtitle: 'Residual learning',
    description: 'Recursive refinement',
    color: '#A855F7',
    icon: '∞',
  },
  {
    number: 6,
    title: 'Puzzle Reasoning Language',
    subtitle: 'High-level abstractions',
    description: 'Symbolic logic',
    color: '#06B6D4',
    icon: '∴',
  },
  {
    number: 7,
    title: 'ProgramTree Synthesis',
    subtitle: 'CEGIS',
    description: 'Counter-example guided',
    color: '#EC4899',
    icon: '⊢',
  },
];

export default function ArchitecturePipeline() {
  return (
    <div className="architecture-pipeline">
      <motion.h2
        className="section-title"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        7-Phase Symbolic Pipeline
      </motion.h2>

      <div className="pipeline-container">
        {phases.map((phase, index) => (
          <motion.div
            key={phase.number}
            className="phase-card"
            initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50, rotateY: 45 }}
            whileInView={{ opacity: 1, x: 0, rotateY: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ delay: index * 0.1, duration: 0.6 }}
            whileHover={{ scale: 1.03, z: 50 }}
          >
            <div className="phase-number" style={{ borderColor: phase.color, color: phase.color }}>
              {phase.number}
            </div>

            <div className="phase-icon" style={{ color: phase.color }}>
              {phase.icon}
            </div>

            <div className="phase-content">
              <h3 className="phase-title" style={{ color: phase.color }}>
                {phase.title}
              </h3>
              <div className="phase-subtitle">{phase.subtitle}</div>
              <div className="phase-description">{phase.description}</div>
            </div>

            <div className="phase-glow" style={{ background: `radial-gradient(circle, ${phase.color}22, transparent)` }} />

            {index < phases.length - 1 && (
              <motion.div
                className="phase-connector"
                initial={{ scaleY: 0 }}
                whileInView={{ scaleY: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 + 0.3, duration: 0.4 }}
              >
                <div className="connector-line" style={{ borderColor: phase.color }} />
                <div className="connector-arrow" style={{ borderTopColor: phases[index + 1].color }} />
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      <motion.div
        className="pipeline-note"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.8 }}
      >
        <div className="note-icon">⚡</div>
        <div className="note-text">
          Each phase is a nested cross structure — symbolic reasoning all the way down
        </div>
      </motion.div>

      <style jsx>{`
        .architecture-pipeline {
          padding: 100px 20px;
          max-width: 900px;
          margin: 0 auto;
        }

        .section-title {
          font-size: 48px;
          font-weight: 800;
          text-align: center;
          margin-bottom: 80px;
          background: linear-gradient(135deg, #0EA5E9 0%, #A855F7 50%, #F59E0B 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 40px rgba(14, 165, 233, 0.3);
        }

        .pipeline-container {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 60px;
        }

        .phase-card {
          position: relative;
          display: flex;
          align-items: center;
          gap: 24px;
          padding: 32px;
          background: rgba(20, 20, 30, 0.6);
          border: 1px solid rgba(14, 165, 233, 0.2);
          border-radius: 16px;
          backdrop-filter: blur(10px);
          transition: all 0.3s ease;
          transform-style: preserve-3d;
          overflow: hidden;
        }

        .phase-card:hover {
          border-color: currentColor;
          box-shadow: 0 10px 40px rgba(14, 165, 233, 0.2);
        }

        .phase-number {
          width: 50px;
          height: 50px;
          border: 2px solid;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 24px;
          font-weight: 800;
          flex-shrink: 0;
          text-shadow: 0 0 10px currentColor;
        }

        .phase-icon {
          font-size: 48px;
          flex-shrink: 0;
          text-shadow: 0 0 20px currentColor;
        }

        .phase-content {
          flex: 1;
        }

        .phase-title {
          font-size: 24px;
          font-weight: 700;
          margin-bottom: 6px;
          text-shadow: 0 0 10px currentColor;
        }

        .phase-subtitle {
          font-size: 16px;
          color: #9CA3AF;
          margin-bottom: 4px;
        }

        .phase-description {
          font-size: 14px;
          color: #6B7280;
          font-style: italic;
        }

        .phase-glow {
          position: absolute;
          inset: 0;
          opacity: 0;
          transition: opacity 0.3s ease;
          pointer-events: none;
        }

        .phase-card:hover .phase-glow {
          opacity: 1;
        }

        .phase-connector {
          position: absolute;
          bottom: -60px;
          left: 50%;
          transform: translateX(-50%);
          width: 2px;
          height: 60px;
          transform-origin: top;
        }

        .connector-line {
          width: 100%;
          height: calc(100% - 10px);
          border-left: 2px dashed;
          opacity: 0.5;
        }

        .connector-arrow {
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-top: 10px solid;
          margin-left: -5px;
        }

        .pipeline-note {
          margin-top: 80px;
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 24px 32px;
          background: rgba(14, 165, 233, 0.05);
          border: 1px solid rgba(14, 165, 233, 0.3);
          border-radius: 12px;
          backdrop-filter: blur(10px);
        }

        .note-icon {
          font-size: 32px;
          text-shadow: 0 0 20px #0EA5E9;
        }

        .note-text {
          flex: 1;
          font-size: 16px;
          color: #9CA3AF;
          line-height: 1.6;
        }

        @media (max-width: 768px) {
          .section-title {
            font-size: 32px;
            margin-bottom: 50px;
          }

          .pipeline-container {
            gap: 50px;
          }

          .phase-card {
            flex-direction: column;
            text-align: center;
            padding: 24px;
          }

          .phase-number {
            width: 40px;
            height: 40px;
            font-size: 20px;
          }

          .phase-icon {
            font-size: 36px;
          }

          .phase-title {
            font-size: 20px;
          }

          .phase-connector {
            bottom: -50px;
            height: 50px;
          }

          .pipeline-note {
            flex-direction: column;
            text-align: center;
            padding: 20px;
          }
        }
      `}</style>
    </div>
  );
}
