'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLanguage } from '@/lib/i18n';
import ThemePicker from '@/components/ThemePicker';

const CLI_GITHUB = 'https://github.com/Ag3497120/verantyx-cli';

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { lang, setLang } = useLanguage();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        background: scrolled ? 'rgba(5, 5, 8, 0.94)' : 'rgba(5, 5, 8, 0.55)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: scrolled
          ? '1px solid rgba(var(--accent-rgb), 0.14)'
          : '1px solid transparent',
        transition: 'background 0.4s ease, border-color 0.4s ease',
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: '0 auto',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: 60,
          gap: 12,
        }}
      >
        <a
          href="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            textDecoration: 'none',
            color: '#fff',
            fontWeight: 800,
            fontSize: '1.15em',
            letterSpacing: '-0.5px',
            fontFamily: 'var(--font-display)',
          }}
        >
          <span className="gradient-brand" style={{ fontSize: '1.05em' }}>
            ◆
          </span>
          <span>Verantyx</span>
        </a>

        <div
          style={{ display: 'flex', alignItems: 'center', gap: 4 }}
          className="navbar-desktop"
        >
          <NavLink href="/" label="Home" />
          <NavLink href="/apps/" label="Apps" />
          <NavLink href="/verantyx-cli/" label="CLI" />
          <NavLink href="/jcross-language/" label=".jcross" />

          <div style={{ marginLeft: 10, marginRight: 4 }}>
            <ThemePicker compact />
          </div>

          <button
            onClick={() => setLang(lang === 'en' ? 'ja' : 'en')}
            style={{
              padding: '6px 12px',
              marginLeft: 4,
              borderRadius: 8,
              border: '1px solid rgba(107, 114, 128, 0.2)',
              background: 'transparent',
              color: '#9ca3af',
              fontWeight: 500,
              fontSize: '0.8em',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              letterSpacing: '0.05em',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.35)';
              e.currentTarget.style.color = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(107, 114, 128, 0.2)';
              e.currentTarget.style.color = '#9ca3af';
            }}
          >
            {lang === 'en' ? 'JP' : 'EN'}
          </button>

          <a
            href={CLI_GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              marginLeft: 8,
              padding: '7px 14px',
              borderRadius: 8,
              fontWeight: 600,
              fontSize: '0.8em',
              textDecoration: 'none',
            }}
          >
            GitHub · CLI
          </a>
        </div>

        <div
          className="navbar-mobile-cluster"
          style={{ display: 'none', alignItems: 'center', gap: 12 }}
        >
          <ThemePicker compact />
          <a
            href={CLI_GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent"
            style={{
              padding: '6px 10px',
              borderRadius: 8,
              fontSize: '0.72em',
              fontWeight: 600,
              textDecoration: 'none',
            }}
          >
            CLI
          </a>
          <button
            className="navbar-mobile-btn"
            onClick={() => setMobileOpen(!mobileOpen)}
            style={{
              background: 'none',
              border: 'none',
              color: '#e2e8f0',
              fontSize: '1.3em',
              cursor: 'pointer',
              padding: 4,
              letterSpacing: '0.1em',
            }}
            aria-label="Menu"
          >
            {mobileOpen ? '✕' : '☰'}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            style={{
              overflow: 'hidden',
              background: 'rgba(5, 5, 8, 0.98)',
              borderTop: '1px solid rgba(var(--accent-rgb), 0.1)',
            }}
            className="navbar-mobile-menu"
          >
            <div
              style={{
                padding: '16px 24px 24px',
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              <MobileNavLink href="/" label="Home" onClick={() => setMobileOpen(false)} />
              <MobileNavLink href="/apps/" label="Apps" onClick={() => setMobileOpen(false)} />
              <MobileNavLink
                href="/verantyx-cli/"
                label="CLI"
                onClick={() => setMobileOpen(false)}
              />
              <MobileNavLink
                href="/jcross-language/"
                label=".jcross"
                onClick={() => setMobileOpen(false)}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  onClick={() => setLang(lang === 'en' ? 'ja' : 'en')}
                  style={{
                    flex: 1,
                    padding: '10px 16px',
                    borderRadius: 10,
                    border: '1px solid rgba(107, 114, 128, 0.2)',
                    background: 'transparent',
                    color: '#9ca3af',
                    fontWeight: 500,
                    fontSize: '0.9em',
                    cursor: 'pointer',
                  }}
                >
                  {lang === 'en' ? '日本語' : 'English'}
                </button>
                <a
                  href={CLI_GITHUB}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-accent"
                  style={{
                    flex: 1,
                    display: 'block',
                    padding: '10px 16px',
                    borderRadius: 10,
                    fontWeight: 600,
                    fontSize: '0.9em',
                    textDecoration: 'none',
                    textAlign: 'center',
                  }}
                >
                  GitHub · CLI
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @media (max-width: 768px) {
          .navbar-desktop { display: none !important; }
          .navbar-mobile-cluster { display: flex !important; }
        }
        @media (min-width: 769px) {
          .navbar-mobile-menu { display: none !important; }
        }
      `}</style>
    </nav>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      style={{
        padding: '6px 12px',
        borderRadius: 8,
        color: '#9ca3af',
        fontWeight: 500,
        fontSize: '0.85em',
        textDecoration: 'none',
        letterSpacing: '0.02em',
        transition: 'color 0.3s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = '#e2e8f0';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = '#9ca3af';
      }}
    >
      {label}
    </a>
  );
}

function MobileNavLink({
  href,
  label,
  onClick,
}: {
  href: string;
  label: string;
  onClick: () => void;
}) {
  return (
    <a
      href={href}
      onClick={onClick}
      style={{
        display: 'block',
        padding: '12px 16px',
        borderRadius: 10,
        color: '#d1d5db',
        fontWeight: 500,
        fontSize: '0.95em',
        textDecoration: 'none',
      }}
    >
      {label}
    </a>
  );
}
