'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useEffect, useState } from 'react';

interface Section {
  id: string;
  title: string;
  direction: 'top' | 'right' | 'bottom' | 'left' | 'center';
  color: string;
}

const sections: Section[] = [
  { id: 'hero', title: 'Verantyx', direction: 'center', color: '#0EA5E9' },
  { id: 'how-it-works', title: 'How It Works', direction: 'top', color: '#A855F7' },
  { id: 'arc-score', title: 'ARC Score', direction: 'right', color: '#10B981' },
  { id: 'architecture', title: 'Architecture', direction: 'bottom', color: '#F59E0B' },
  { id: 'hle-score', title: 'HLE Score', direction: 'left', color: '#EF4444' },
];

export default function CrossNavigation() {
  const [activeSection, setActiveSection] = useState('hero');
  const { scrollYProgress } = useScroll();

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + window.innerHeight / 2;

      sections.forEach(section => {
        const element = document.getElementById(section.id);
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(section.id);
          }
        }
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="cross-navigation">
      <div className="nav-container">
        {sections.map((section, index) => {
          const isActive = activeSection === section.id;

          return (
            <motion.button
              key={section.id}
              className={`nav-button nav-${section.direction} ${isActive ? 'active' : ''}`}
              onClick={() => scrollToSection(section.id)}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <div
                className="nav-indicator"
                style={{
                  backgroundColor: isActive ? section.color : 'transparent',
                  borderColor: section.color,
                  boxShadow: isActive ? `0 0 20px ${section.color}` : 'none',
                }}
              />
              <span className="nav-label" style={{ color: section.color }}>
                {section.title}
              </span>
            </motion.button>
          );
        })}

        {/* Progress ring in center */}
        <motion.div className="progress-ring">
          <svg width="60" height="60" viewBox="0 0 60 60">
            <circle
              cx="30"
              cy="30"
              r="25"
              fill="none"
              stroke="rgba(14, 165, 233, 0.2)"
              strokeWidth="2"
            />
            <motion.circle
              cx="30"
              cy="30"
              r="25"
              fill="none"
              stroke="#0EA5E9"
              strokeWidth="2"
              strokeLinecap="round"
              style={{
                pathLength: scrollYProgress,
                rotate: -90,
                transformOrigin: 'center',
              }}
              strokeDasharray="0 1"
            />
          </svg>
        </motion.div>
      </div>

      <style jsx>{`
        .cross-navigation {
          position: fixed;
          bottom: 40px;
          right: 40px;
          z-index: 1000;
          pointer-events: none;
        }

        .nav-container {
          position: relative;
          width: 200px;
          height: 200px;
          pointer-events: all;
        }

        .nav-button {
          position: absolute;
          background: rgba(10, 10, 15, 0.8);
          border: none;
          border-radius: 8px;
          padding: 8px 12px;
          cursor: pointer;
          backdrop-filter: blur(10px);
          display: flex;
          align-items: center;
          gap: 8px;
          transition: all 0.3s ease;
        }

        .nav-button:hover {
          background: rgba(20, 20, 30, 0.9);
        }

        .nav-center {
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }

        .nav-top {
          top: 0;
          left: 50%;
          transform: translateX(-50%);
        }

        .nav-right {
          right: 0;
          top: 50%;
          transform: translateY(-50%);
        }

        .nav-bottom {
          bottom: 0;
          left: 50%;
          transform: translateX(-50%);
        }

        .nav-left {
          left: 0;
          top: 50%;
          transform: translateY(-50%);
        }

        .nav-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid;
          transition: all 0.3s ease;
        }

        .nav-button.active .nav-indicator {
          width: 16px;
          height: 16px;
        }

        .nav-label {
          font-size: 12px;
          font-weight: 600;
          white-space: nowrap;
          text-shadow: 0 0 10px currentColor;
        }

        .progress-ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          pointer-events: none;
        }

        @media (max-width: 768px) {
          .cross-navigation {
            bottom: 20px;
            right: 20px;
          }

          .nav-container {
            width: 120px;
            height: 120px;
            transform: scale(0.8);
          }

          .nav-label {
            display: none;
          }

          .progress-ring {
            transform: translate(-50%, -50%) scale(0.6);
          }
        }
      `}</style>
    </div>
  );
}
