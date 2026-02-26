'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

const links = [
  {
    title: 'View Source Code',
    url: 'https://github.com/Ag3497120/verantyx-v6',
    icon: 'github',
    color: '#0EA5E9',
    description: 'Explore the full codebase — 304 files, 100K lines',
  },
  {
    title: 'Models & Datasets',
    url: 'https://huggingface.co/kofdai',
    icon: 'huggingface',
    color: '#F59E0B',
    description: 'HuggingFace Spaces — live ARC-AGI-2 solver demo',
  },
  {
    title: 'Follow @kofdai',
    url: 'https://x.com/kofdai',
    icon: 'x',
    color: '#A855F7',
    description: 'Latest updates and research progress',
  },
  {
    title: 'Verantyx Logic',
    url: 'https://apps.apple.com/app/verantyx-logic/id6740806562',
    icon: 'appstore',
    color: '#2ECC40',
    description: 'iOS app — modal logic puzzles only',
    note: '※ Modal logic problems only. Not the full ARC solver.',
  },
];

export default function LinkReveal() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-100px' });

  return (
    <div ref={ref} className="link-reveal-container">
      <motion.h2
        className="links-title"
        initial={{ y: 50, opacity: 0 }}
        animate={isInView ? { y: 0, opacity: 1 } : {}}
        transition={{ duration: 0.6 }}
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
            initial={{ scale: 0, opacity: 0 }}
            animate={isInView ? { scale: 1, opacity: 1 } : {}}
            transition={{
              delay: 0.2 + index * 0.15,
              duration: 0.6,
              type: 'spring',
              stiffness: 200,
            }}
            whileHover={{ scale: 1.03, y: -5 }}
            whileTap={{ scale: 0.97 }}
            style={{ borderColor: `${link.color}44` }}
          >
            <div className="link-icon" style={{ borderColor: link.color }}>
              {link.icon === 'github' && <GitHubIcon color={link.color} />}
              {link.icon === 'huggingface' && <HuggingFaceIcon color={link.color} />}
              {link.icon === 'x' && <XIcon color={link.color} />}
              {link.icon === 'appstore' && <AppStoreIcon color={link.color} />}
            </div>

            <div className="link-content">
              <h3 style={{ color: link.color, fontSize: '18px', fontWeight: 700, marginBottom: '4px' }}>
                {link.title}
              </h3>
              <p style={{ fontSize: '13px', color: '#9CA3AF', margin: 0 }}>{link.description}</p>
              {link.note && (
                <p style={{ fontSize: '11px', color: '#EF4444', marginTop: '6px', opacity: 0.8 }}>
                  {link.note}
                </p>
              )}
            </div>

            <span style={{ color: link.color, fontSize: '24px', fontWeight: 700 }}>→</span>
          </motion.a>
        ))}
      </div>

      <style jsx>{`
        .link-reveal-container {
          padding: 100px 20px;
          max-width: 900px;
          margin: 0 auto;
          position: relative;
          z-index: 10;
        }

        .links-title {
          font-size: 42px;
          font-weight: 800;
          text-align: center;
          margin-bottom: 50px;
          background: linear-gradient(135deg, #0EA5E9 0%, #A855F7 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .links-grid {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .link-card {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 24px 28px;
          background: rgba(10, 10, 20, 0.8);
          border: 1px solid;
          border-radius: 14px;
          cursor: pointer;
          text-decoration: none;
          backdrop-filter: blur(10px);
          transition: background 0.3s, border-color 0.3s;
        }

        .link-card:hover {
          background: rgba(20, 20, 35, 0.9);
        }

        .link-icon {
          width: 52px;
          height: 52px;
          border: 2px solid;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .link-content {
          flex: 1;
        }

        @media (max-width: 768px) {
          .links-title { font-size: 28px; margin-bottom: 30px; }
          .link-card { padding: 16px 18px; gap: 14px; }
          .link-icon { width: 44px; height: 44px; }
        }
      `}</style>
    </div>
  );
}

function GitHubIcon({ color }: { color: string }) {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill={color}>
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
    </svg>
  );
}

function HuggingFaceIcon({ color }: { color: string }) {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill={color}>
      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm0 3c4.97 0 9 4.03 9 9s-4.03 9-9 9-9-4.03-9-9 4.03-9 9-9zm-2 6c-.552 0-1 .448-1 1s.448 1 1 1 1-.448 1-1-.448-1-1-1zm4 0c-.552 0-1 .448-1 1s.448 1 1 1 1-.448 1-1-.448-1-1-1zm-4 4c-.552 0-1 .448-1 1 0 1.657 1.343 3 3 3s3-1.343 3-3c0-.552-.448-1-1-1s-1 .448-1 1c0 .552-.448 1-1 1s-1-.448-1-1c0-.552-.448-1-1-1z" />
    </svg>
  );
}

function XIcon({ color }: { color: string }) {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill={color}>
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

function AppStoreIcon({ color }: { color: string }) {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill={color}>
      <path d="M8.809 14.92l6.11-11.037c.084-.152.076-.335-.021-.479C14.788 3.26 14.607 3.2 14.43 3.248l-.036.012c-.163.065-.288.21-.34.382l-2.531 5.525L8.809 14.92zm6.002 2.523H3.664c-.204 0-.392.112-.489.291-.097.179-.088.398.024.569l1.738 2.644c.106.161.284.258.475.258h9.399l-2.003-3.762h2.003zM21.264 17.443l-4.584-8.283-.002-.002-1.593 2.879 3.178 5.741c.1.18.09.399-.025.571-.116.172-.31.277-.52.277h3.052c.21 0 .404-.105.52-.277.115-.172.125-.391.025-.571l.949.335-1-.67z" />
    </svg>
  );
}
