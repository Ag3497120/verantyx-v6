'use client';
import { useTheme } from '@/lib/theme';

export default function CleanroomThemeToggle() {
  const { mode, toggleMode } = useTheme();
  return <button className="cr-theme-toggle" onClick={toggleMode}
    aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
    {mode === 'dark' ? 'Light' : 'Dark'}
  </button>;
}
