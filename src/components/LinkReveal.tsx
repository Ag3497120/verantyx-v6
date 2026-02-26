'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

const links = [
  {
    title: 'View Source Code',
    url: 'https://github.com/Ag3497120/verantyx-v6',
    icon: 'github',
    color: '#0EA5E9',
    description: 'Explore the codebase',
  },
  {
    title: 'Models & Datasets',
    url: 'https://huggingface.co/kofdai',
    icon: 'huggingface',
    color: '#F59E0B',
    description: 'HuggingFace resources',
  },
  {
    title: 'Follow @kofdai',
    url: 'https://x.com/kofdai',
    icon: 'x',
    color: '#A855F7',
    description: 'Latest updates',
  },
];

export default function LinkReveal() {
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setRevealed(true);
    }, 12000); // After solver animation completes

    return () => clearTimeout(timer);
  }, []);

  return (
    <motion.div
      className="link-reveal-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: revealed ? 1 : 0 }}
      transition={{ duration: 1 }}
    >
      <motion.h2
        className="links-title"
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: revealed ? 0 : 50, opacity: revealed ? 1 : 0 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        Explore Verantyx
      </motion.h2>

      <div className="links-grid">
        {links.map((link, index) => (
          <motion.a
            key={link.url}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className="link-card"
            initial={{ scale: 0, rotate: -180, opacity: 0 }}
            animate={{
              scale: revealed ? 1 : 0,
              rotate: revealed ? 0 : -180,
              opacity: revealed ? 1 : 0,
            }}
            transition={{
              delay: 0.5 + index * 0.2,
              duration: 0.8,
              type: 'spring',
              stiffness: 200,
            }}
            whileHover={{ scale: 1.05, y: -10 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="link-icon" style={{ borderColor: link.color }}>
              {link.icon === 'github' && <GitHubIcon color={link.color} />}
              {link.icon === 'huggingface' && <HuggingFaceIcon color={link.color} />}
              {link.icon === 'x' && <XIcon color={link.color} />}
            </div>

            <div className="link-content">
              <h3 className="link-title" style={{ color: link.color }}>
                {link.title}
              </h3>
              <p className="link-description">{link.description}</p>
            </div>

            <div className="link-arrow" style={{ color: link.color }}>
              →
            </div>

            <div className="link-glow" style={{ background: `radial-gradient(circle, ${link.color}33, transparent)` }} />
          </motion.a>
        ))}
      </div>

      <style jsx>{`
        .link-reveal-container {
          padding: 80px 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .links-title {
          font-size: 48px;
          font-weight: 800;
          text-align: center;
          margin-bottom: 60px;
          background: linear-gradient(135deg, #0EA5E9 0%, #A855F7 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 40px rgba(14, 165, 233, 0.3);
        }

        .links-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 30px;
          max-width: 1000px;
          margin: 0 auto;
        }

        .link-card {
          position: relative;
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 30px;
          background: rgba(20, 20, 30, 0.6);
          border: 1px solid rgba(14, 165, 233, 0.3);
          border-radius: 16px;
          cursor: pointer;
          text-decoration: none;
          overflow: hidden;
          backdrop-filter: blur(10px);
          transition: all 0.3s ease;
        }

        .link-card:hover {
          border-color: currentColor;
          background: rgba(30, 30, 40, 0.8);
        }

        .link-icon {
          width: 60px;
          height: 60px;
          border: 2px solid;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          transition: all 0.3s ease;
        }

        .link-card:hover .link-icon {
          transform: rotate(10deg) scale(1.1);
        }

        .link-content {
          flex: 1;
        }

        .link-title {
          font-size: 20px;
          font-weight: 700;
          margin-bottom: 6px;
          text-shadow: 0 0 10px currentColor;
        }

        .link-description {
          font-size: 14px;
          color: #9CA3AF;
        }

        .link-arrow {
          font-size: 28px;
          font-weight: 700;
          transition: transform 0.3s ease;
        }

        .link-card:hover .link-arrow {
          transform: translateX(5px);
        }

        .link-glow {
          position: absolute;
          inset: 0;
          opacity: 0;
          transition: opacity 0.3s ease;
          pointer-events: none;
        }

        .link-card:hover .link-glow {
          opacity: 1;
        }

        @media (max-width: 768px) {
          .links-title {
            font-size: 32px;
            margin-bottom: 40px;
          }

          .links-grid {
            grid-template-columns: 1fr;
            gap: 20px;
          }

          .link-card {
            padding: 20px;
          }

          .link-icon {
            width: 50px;
            height: 50px;
          }

          .link-title {
            font-size: 18px;
          }
        }
      `}</style>
    </motion.div>
  );
}

function GitHubIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill={color}>
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
    </svg>
  );
}

function HuggingFaceIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill={color}>
      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 3c4.97 0 9 4.03 9 9s-4.03 9-9 9-9-4.03-9-9 4.03-9 9-9zm-2 6c-.552 0-1 .448-1 1s.448 1 1 1 1-.448 1-1-.448-1-1-1zm4 0c-.552 0-1 .448-1 1s.448 1 1 1 1-.448 1-1-.448-1-1-1zm-4 4c-.552 0-1 .448-1 1 0 1.657 1.343 3 3 3s3-1.343 3-3c0-.552-.448-1-1-1s-1 .448-1 1c0 .552-.448 1-1 1s-1-.448-1-1c0-.552-.448-1-1-1z" />
    </svg>
  );
}

function XIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill={color}>
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}
