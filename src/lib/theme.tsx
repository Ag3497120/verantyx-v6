'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';

export type ThemeId = 'cyan' | 'teal' | 'amber' | 'coral' | 'ice';

export type ThemeTokens = {
  id: ThemeId;
  label: string;
  swatch: string;
  accent: string;
  accent2: string;
  accentRgb: string;
  glow: string;
};

export const THEMES: ThemeTokens[] = [
  {
    id: 'cyan',
    label: 'Cyan',
    swatch: '#0EA5E9',
    accent: '#0EA5E9',
    accent2: '#06B6D4',
    accentRgb: '14, 165, 233',
    glow: 'rgba(14, 165, 233, 0.35)',
  },
  {
    id: 'teal',
    label: 'Teal',
    swatch: '#14B8A6',
    accent: '#14B8A6',
    accent2: '#2DD4BF',
    accentRgb: '20, 184, 166',
    glow: 'rgba(20, 184, 166, 0.35)',
  },
  {
    id: 'amber',
    label: 'Amber',
    swatch: '#F59E0B',
    accent: '#F59E0B',
    accent2: '#FBBF24',
    accentRgb: '245, 158, 11',
    glow: 'rgba(245, 158, 11, 0.35)',
  },
  {
    id: 'coral',
    label: 'Coral',
    swatch: '#F43F5E',
    accent: '#F43F5E',
    accent2: '#FB7185',
    accentRgb: '244, 63, 94',
    glow: 'rgba(244, 63, 94, 0.35)',
  },
  {
    id: 'ice',
    label: 'Ice',
    swatch: '#94A3B8',
    accent: '#94A3B8',
    accent2: '#E2E8F0',
    accentRgb: '148, 163, 184',
    glow: 'rgba(148, 163, 184, 0.3)',
  },
];

const STORAGE_KEY = 'verantyx-theme';
const MODE_KEY = 'verantyx-mode';

export type Mode = 'dark' | 'light';

/* Accent only. The surface tokens are NOT written here any more: they used to
 * be set as inline styles on <html>, which beats any stylesheet rule, so a
 * light-mode block could never take effect. They live in globals.css now,
 * where `html[data-mode='light']` can override them. */
function applyTheme(theme: ThemeTokens) {
  const root = document.documentElement;
  root.dataset.theme = theme.id;
  root.style.setProperty('--accent', theme.accent);
  root.style.setProperty('--accent-2', theme.accent2);
  root.style.setProperty('--accent-rgb', theme.accentRgb);
  root.style.setProperty('--accent-glow', theme.glow);
}

function applyMode(mode: Mode) {
  document.documentElement.dataset.mode = mode;
}

type ThemeContextType = {
  themeId: ThemeId;
  theme: ThemeTokens;
  setThemeId: (id: ThemeId) => void;
  mode: Mode;
  setMode: (mode: Mode) => void;
  toggleMode: () => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [themeId, setThemeIdState] = useState<ThemeId>('cyan');
  const [mode, setModeState] = useState<Mode>('dark');
  const theme = THEMES.find((t) => t.id === themeId) ?? THEMES[0];

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as ThemeId | null;
      if (stored && THEMES.some((t) => t.id === stored)) {
        setThemeIdState(stored);
        applyTheme(THEMES.find((t) => t.id === stored)!);
        return;
      }
    } catch {
      /* ignore */
    }
    applyTheme(THEMES[0]);
  }, []);

  // A stored choice wins; otherwise follow the operating system, which is
  // what a visitor who has set a preference there already expects.
  useEffect(() => {
    let next: Mode = 'dark';
    try {
      const stored = localStorage.getItem(MODE_KEY) as Mode | null;
      if (stored === 'light' || stored === 'dark') {
        next = stored;
      } else if (
        typeof window !== 'undefined' &&
        window.matchMedia('(prefers-color-scheme: light)').matches
      ) {
        next = 'light';
      }
    } catch {
      /* ignore */
    }
    setModeState(next);
    applyMode(next);
  }, []);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setThemeId = useCallback((id: ThemeId) => {
    setThemeIdState(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* ignore */
    }
  }, []);

  const setMode = useCallback((next: Mode) => {
    setModeState(next);
    applyMode(next);
    try {
      localStorage.setItem(MODE_KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  const toggleMode = useCallback(() => {
    setMode(mode === 'dark' ? 'light' : 'dark');
  }, [mode, setMode]);

  return (
    <ThemeContext.Provider
      value={{ themeId, theme, setThemeId, mode, setMode, toggleMode }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
