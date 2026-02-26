'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';

const INSTALL_COMMAND = 'git clone https://github.com/Ag3497120/verantyx-v6 && cd verantyx-v6 && pip install -r requirements.txt';

export default function InstallCommand() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL_COMMAND);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <motion.div
      className="install-command-container"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
    >
      <div className="install-label">Quick Install</div>

      <div className="command-box">
        <code className="command-text">{INSTALL_COMMAND}</code>

        <motion.button
          className="copy-button"
          onClick={handleCopy}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {copied ? (
            <>
              <CheckIcon />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <CopyIcon />
              <span>Copy</span>
            </>
          )}
        </motion.button>
      </div>

      <style jsx>{`
        .install-command-container {
          max-width: 900px;
          margin: 40px auto;
        }

        .install-label {
          font-size: 14px;
          font-weight: 600;
          color: #0EA5E9;
          margin-bottom: 12px;
          text-transform: uppercase;
          letter-spacing: 1px;
          text-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
        }

        .command-box {
          display: flex;
          align-items: center;
          gap: 16px;
          background: rgba(20, 20, 30, 0.8);
          border: 1px solid rgba(14, 165, 233, 0.3);
          border-radius: 12px;
          padding: 20px 24px;
          backdrop-filter: blur(10px);
          box-shadow: 0 0 30px rgba(14, 165, 233, 0.1);
        }

        .command-text {
          flex: 1;
          font-family: 'Courier New', monospace;
          font-size: 14px;
          color: #60A5FA;
          word-break: break-all;
          line-height: 1.6;
        }

        .copy-button {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 20px;
          background: rgba(14, 165, 233, 0.1);
          border: 1px solid rgba(14, 165, 233, 0.5);
          border-radius: 8px;
          color: #0EA5E9;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.3s ease;
        }

        .copy-button:hover {
          background: rgba(14, 165, 233, 0.2);
          box-shadow: 0 0 20px rgba(14, 165, 233, 0.3);
        }

        @media (max-width: 768px) {
          .command-box {
            flex-direction: column;
            align-items: stretch;
            gap: 12px;
          }

          .command-text {
            font-size: 12px;
          }

          .copy-button {
            justify-content: center;
          }
        }
      `}</style>
    </motion.div>
  );
}

function CopyIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
