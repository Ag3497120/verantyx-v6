'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLanguage } from '@/lib/i18n';
import { useTheme } from '@/lib/theme';
import Logo from '@/components/Logo';
import ThemePicker from '@/components/ThemePicker';

const CLI_GITHUB = 'https://github.com/Ag3497120/Verantyx';

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { lang, setLang } = useLanguage();
  const { mode, toggleMode } = useTheme();

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
        background: scrolled ? 'var(--chrome-solid)' : 'var(--chrome)',
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
          // Was a flat 24px. On a 375px phone that left the row too narrow
          // for the wordmark, so "Verantyx" broke across two lines and the
          // CLI pill stacked one letter per line.
          padding: '0 clamp(14px, 4vw, 24px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: 60,
          gap: 10,
          flexWrap: 'nowrap',
        }}
      >
        <a
          href="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            textDecoration: 'none',
            color: 'var(--ink)',
            fontWeight: 800,
            fontSize: 'clamp(1rem, 3.6vw, 1.15em)',
            letterSpacing: '-0.5px',
            fontFamily: 'var(--font-display)',
            whiteSpace: 'nowrap',
            flexShrink: 0,
          }}
        >
          <Logo size={24} />
          <span>Verantyx</span>
        </a>

        <div
          style={{ display: 'flex', alignItems: 'center', gap: 4 }}
          className="navbar-desktop"
        >
          <NavLink href="/" label="Home" />
          <NavLink href="/apps/" label="Apps" />
          <NavLink href="/vera/" label="Vera" />
          <NavLink href="/vera/demo/" label="Demo" />
          <NavLink href="/catalogue/" label="Catalogue" />
          <NavLink href="/verantyx-cli/" label="CLI" />
          <NavLink href="/jcross-language/" label=".jcross" />
          <NavLink href="/apple-music-api/" label="API" />

          <div style={{ marginLeft: 10, marginRight: 4 }}>
            <ThemePicker compact />
          </div>

          <ModeToggle mode={mode} onToggle={toggleMode} />

          <button
            onClick={() => setLang(lang === 'en' ? 'ja' : 'en')}
            style={{
              padding: '6px 12px',
              marginLeft: 4,
              borderRadius: 8,
              border: '1px solid var(--line-strong)',
              background: 'transparent',
              color: 'var(--ink-3)',
              fontWeight: 500,
              fontSize: '0.8em',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              letterSpacing: '0.05em',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.35)';
              e.currentTarget.style.color = 'var(--ink)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--line-strong)';
              e.currentTarget.style.color = 'var(--ink-3)';
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
          style={{
            display: 'none',
            alignItems: 'center',
            gap: 10,
            flexShrink: 0,
          }}
        >
          <ModeToggle mode={mode} onToggle={toggleMode} />
          <span className="navbar-narrow-hide">
            <ThemePicker compact />
          </span>
          <a
            href={CLI_GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-accent navbar-narrow-hide"
            style={{
              padding: '6px 10px',
              borderRadius: 8,
              fontSize: '0.72em',
              fontWeight: 600,
              textDecoration: 'none',
              whiteSpace: 'nowrap',
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
              color: 'var(--ink-2)',
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
              background: 'var(--chrome-solid)',
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
                href="/vera/"
                label="Vera"
                onClick={() => setMobileOpen(false)}
              />
              <MobileNavLink
                href="/vera/demo/"
                label="Demo"
                onClick={() => setMobileOpen(false)}
              />
              <MobileNavLink
                href="/catalogue/"
                label="Catalogue"
                onClick={() => setMobileOpen(false)}
              />
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
              <MobileNavLink
                href="/apple-music-api/"
                label="Apple Music API"
                onClick={() => setMobileOpen(false)}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  onClick={toggleMode}
                  style={{
                    flex: 1,
                    padding: '10px 16px',
                    borderRadius: 10,
                    border: '1px solid var(--line-strong)',
                    background: 'transparent',
                    color: 'var(--ink-3)',
                    fontWeight: 500,
                    fontSize: '0.9em',
                    cursor: 'pointer',
                  }}
                >
                  {mode === 'dark'
                    ? lang === 'ja' ? '☀ ライト' : '☀ Light'
                    : lang === 'ja' ? '☾ ダーク' : '☾ Dark'}
                </button>
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  onClick={() => setLang(lang === 'en' ? 'ja' : 'en')}
                  style={{
                    flex: 1,
                    padding: '10px 16px',
                    borderRadius: 10,
                    border: '1px solid var(--line-strong)',
                    background: 'transparent',
                    color: 'var(--ink-3)',
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
        /* Below this the bar has room for the wordmark and the menu button
           and nothing else. Both hidden items are in the drawer. */
        @media (max-width: 420px) {
          .navbar-narrow-hide { display: none !important; }
        }
        @media (min-width: 769px) {
          .navbar-mobile-menu { display: none !important; }
        }
      `}</style>
    </nav>
  );
}

function ModeToggle({
  mode,
  onToggle,
}: {
  mode: 'dark' | 'light';
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      title={mode === 'dark' ? 'Light mode' : 'Dark mode'}
      style={{
        width: 32,
        height: 32,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 9,
        border: '1px solid var(--line-strong)',
        background: 'transparent',
        color: 'var(--ink-3)',
        cursor: 'pointer',
        fontSize: '0.95em',
        lineHeight: 1,
        transition: 'color 0.25s ease, border-color 0.25s ease',
        flexShrink: 0,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--accent)';
        e.currentTarget.style.borderColor = 'rgba(var(--accent-rgb), 0.5)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--ink-3)';
        e.currentTarget.style.borderColor = 'var(--line-strong)';
      }}
    >
      {mode === 'dark' ? '☀' : '☾'}
    </button>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      style={{
        padding: '6px 12px',
        borderRadius: 8,
        color: 'var(--ink-3)',
        fontWeight: 500,
        fontSize: '0.85em',
        textDecoration: 'none',
        letterSpacing: '0.02em',
        transition: 'color 0.3s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--ink)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--ink-3)';
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
