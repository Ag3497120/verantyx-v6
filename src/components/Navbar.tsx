'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLanguage } from '@/lib/i18n';

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
        background: scrolled ? 'rgba(5, 5, 8, 0.92)' : 'rgba(5, 5, 8, 0.6)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: scrolled
          ? '1px solid rgba(14, 165, 233, 0.1)'
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
        }}
      >
        {/* Logo */}
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
          }}
        >
          <span
            style={{
              fontSize: '1.1em',
              background: 'linear-gradient(135deg, #0EA5E9, #7C3AED)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            ⚡
          </span>
          <span>Verantyx</span>
        </a>

        {/* Desktop links */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
          className="navbar-desktop"
        >
          <NavLink href="/" label="Home" />
          <NavLink href="/apps" label="Apps" />
          <NavLink href="/verantyx" label="Engine" />
          <NavLink href="/verantyx-cli" label="CLI" />
          <NavLink href="/jcross-language" label=".jcross" />

          {/* Language toggle */}
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
              e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.3)';
              e.currentTarget.style.color = '#e2e8f0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(107, 114, 128, 0.2)';
              e.currentTarget.style.color = '#9ca3af';
            }}
          >
            {lang === 'en' ? 'JP' : 'EN'}
          </button>

          {/* GitHub */}
          <a
            href="https://github.com/Ag3497120/verantyx-v6"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              marginLeft: 8,
              padding: '7px 14px',
              borderRadius: 8,
              background: 'rgba(14, 165, 233, 0.08)',
              border: '1px solid rgba(14, 165, 233, 0.2)',
              color: '#e2e8f0',
              fontWeight: 600,
              fontSize: '0.8em',
              textDecoration: 'none',
              transition: 'all 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.5)';
              e.currentTarget.style.boxShadow = '0 0 20px rgba(14, 165, 233, 0.15)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.2)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            ⭐ GitHub
          </a>
        </div>

        {/* Mobile hamburger */}
        <button
          className="navbar-mobile-btn"
          onClick={() => setMobileOpen(!mobileOpen)}
          style={{
            display: 'none',
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

      {/* Mobile menu */}
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
              borderTop: '1px solid rgba(14, 165, 233, 0.08)',
            }}
            className="navbar-mobile-menu"
          >
            <div style={{ padding: '16px 24px 24px', display: 'flex', flexDirection: 'column', gap: 4 }}>
              <MobileNavLink href="/" label="Home" onClick={() => setMobileOpen(false)} />
              <MobileNavLink href="/apps" label="Apps" onClick={() => setMobileOpen(false)} />
              <MobileNavLink href="/verantyx" label="Engine" onClick={() => setMobileOpen(false)} />
              <MobileNavLink href="/verantyx-cli" label="CLI" onClick={() => setMobileOpen(false)} />
              <MobileNavLink href="/jcross-language" label=".jcross" onClick={() => setMobileOpen(false)} />
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
                  {lang === 'en' ? '🇯🇵 日本語' : '🇺🇸 English'}
                </button>
                <a
                  href="https://github.com/Ag3497120/verantyx-v6"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    flex: 1,
                    display: 'block',
                    padding: '10px 16px',
                    borderRadius: 10,
                    background: 'rgba(14, 165, 233, 0.08)',
                    border: '1px solid rgba(14, 165, 233, 0.2)',
                    color: '#e2e8f0',
                    fontWeight: 600,
                    fontSize: '0.9em',
                    textDecoration: 'none',
                    textAlign: 'center',
                  }}
                >
                  ⭐ GitHub
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        @media (max-width: 768px) {
          .navbar-desktop { display: none !important; }
          .navbar-mobile-btn { display: block !important; }
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

function MobileNavLink({ href, label, onClick }: { href: string; label: string; onClick: () => void }) {
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
