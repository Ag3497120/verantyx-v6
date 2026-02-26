'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface CrossStructure3DProps {
  children?: React.ReactNode;
  className?: string;
}

export default function CrossStructure3D({ children, className = '' }: CrossStructure3DProps) {
  const [scrollY, setScrollY] = useState(0);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientY / window.innerHeight - 0.5) * 10;
      const y = (e.clientX / window.innerWidth - 0.5) * 10;
      setRotation({ x, y });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('mousemove', handleMouseMove, { passive: true });

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div className={`cross-structure-container ${className}`}>
      <div
        className="cross-structure"
        style={{
          transform: `perspective(1200px) rotateX(${rotation.x + scrollY * 0.05}deg) rotateY(${rotation.y}deg)`,
          transformStyle: 'preserve-3d',
        }}
      >
        {/* Vertical arm - top */}
        <motion.div
          className="cross-arm cross-arm-vertical-top"
          initial={{ opacity: 0, z: -200 }}
          animate={{ opacity: 1, z: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <div className="cross-arm-inner">
            <div className="cross-glow" />
          </div>
        </motion.div>

        {/* Horizontal arm - left */}
        <motion.div
          className="cross-arm cross-arm-horizontal-left"
          initial={{ opacity: 0, x: -200 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.4 }}
        >
          <div className="cross-arm-inner">
            <div className="cross-glow" />
          </div>
        </motion.div>

        {/* Center */}
        <motion.div
          className="cross-center"
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1, delay: 0.6 }}
        >
          {children}
        </motion.div>

        {/* Horizontal arm - right */}
        <motion.div
          className="cross-arm cross-arm-horizontal-right"
          initial={{ opacity: 0, x: 200 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.4 }}
        >
          <div className="cross-arm-inner">
            <div className="cross-glow" />
          </div>
        </motion.div>

        {/* Vertical arm - bottom */}
        <motion.div
          className="cross-arm cross-arm-vertical-bottom"
          initial={{ opacity: 0, y: 200 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <div className="cross-arm-inner">
            <div className="cross-glow" />
          </div>
        </motion.div>

        {/* Nested crosses */}
        <NestedCross position="top" delay={1.0} />
        <NestedCross position="left" delay={1.2} />
        <NestedCross position="right" delay={1.4} />
        <NestedCross position="bottom" delay={1.6} />
      </div>

      <style jsx>{`
        .cross-structure-container {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          pointer-events: none;
          z-index: 0;
          overflow: hidden;
        }

        .cross-structure {
          position: absolute;
          top: 50%;
          left: 50%;
          transform-origin: center center;
          transition: transform 0.1s ease-out;
        }

        .cross-arm {
          position: absolute;
          background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(168, 85, 247, 0.1));
          border: 1px solid rgba(14, 165, 233, 0.3);
          box-shadow: 0 0 30px rgba(14, 165, 233, 0.2),
                      inset 0 0 30px rgba(14, 165, 233, 0.1);
          backdrop-filter: blur(10px);
        }

        .cross-arm-vertical-top {
          width: 80px;
          height: 300px;
          left: 50%;
          bottom: 50%;
          margin-left: -40px;
          transform-origin: bottom center;
        }

        .cross-arm-vertical-bottom {
          width: 80px;
          height: 300px;
          left: 50%;
          top: 50%;
          margin-left: -40px;
          transform-origin: top center;
        }

        .cross-arm-horizontal-left {
          width: 300px;
          height: 80px;
          top: 50%;
          right: 50%;
          margin-top: -40px;
          transform-origin: right center;
        }

        .cross-arm-horizontal-right {
          width: 300px;
          height: 80px;
          top: 50%;
          left: 50%;
          margin-top: -40px;
          transform-origin: left center;
        }

        .cross-center {
          position: absolute;
          width: 120px;
          height: 120px;
          left: 50%;
          top: 50%;
          margin-left: -60px;
          margin-top: -60px;
          background: radial-gradient(circle, rgba(14, 165, 233, 0.2), transparent);
          border: 2px solid rgba(14, 165, 233, 0.5);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 0 60px rgba(14, 165, 233, 0.4);
        }

        .cross-glow {
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg,
            transparent 0%,
            rgba(14, 165, 233, 0.3) 50%,
            transparent 100%);
          animation: pulse 3s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }

        @media (max-width: 768px) {
          .cross-arm-vertical-top,
          .cross-arm-vertical-bottom {
            width: 40px;
            height: 150px;
            margin-left: -20px;
          }

          .cross-arm-horizontal-left,
          .cross-arm-horizontal-right {
            width: 150px;
            height: 40px;
            margin-top: -20px;
          }

          .cross-center {
            width: 80px;
            height: 80px;
            margin-left: -40px;
            margin-top: -40px;
          }
        }
      `}</style>
    </div>
  );
}

function NestedCross({ position, delay }: { position: 'top' | 'left' | 'right' | 'bottom', delay: number }) {
  const getPosition = () => {
    switch (position) {
      case 'top':
        return { left: '50%', bottom: '100%', marginLeft: '-30px', marginBottom: '100px' };
      case 'bottom':
        return { left: '50%', top: '100%', marginLeft: '-30px', marginTop: '100px' };
      case 'left':
        return { right: '100%', top: '50%', marginRight: '100px', marginTop: '-30px' };
      case 'right':
        return { left: '100%', top: '50%', marginLeft: '100px', marginTop: '-30px' };
    }
  };

  return (
    <motion.div
      className="nested-cross"
      style={{
        position: 'absolute',
        ...getPosition(),
        transformStyle: 'preserve-3d',
      }}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 0.6 }}
      transition={{ duration: 0.8, delay }}
    >
      <div className="nested-cross-vertical" />
      <div className="nested-cross-horizontal" />

      <style jsx>{`
        .nested-cross {
          width: 60px;
          height: 60px;
          transform: scale(0.5) translateZ(-50px);
        }

        .nested-cross-vertical {
          position: absolute;
          width: 20px;
          height: 60px;
          left: 50%;
          top: 0;
          margin-left: -10px;
          background: linear-gradient(180deg, rgba(168, 85, 247, 0.3), rgba(14, 165, 233, 0.3));
          border: 1px solid rgba(168, 85, 247, 0.4);
          box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
        }

        .nested-cross-horizontal {
          position: absolute;
          width: 60px;
          height: 20px;
          top: 50%;
          left: 0;
          margin-top: -10px;
          background: linear-gradient(90deg, rgba(168, 85, 247, 0.3), rgba(14, 165, 233, 0.3));
          border: 1px solid rgba(168, 85, 247, 0.4);
          box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
        }
      `}</style>
    </motion.div>
  );
}
