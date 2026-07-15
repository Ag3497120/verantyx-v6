'use client';

import { THEMES, useTheme, type ThemeId } from '@/lib/theme';

export default function ThemePicker({ compact = false }: { compact?: boolean }) {
  const { themeId, setThemeId } = useTheme();

  return (
    <div
      role="group"
      aria-label="Theme color"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: compact ? 6 : 8,
      }}
    >
      {THEMES.map((t) => {
        const active = themeId === t.id;
        return (
          <button
            key={t.id}
            type="button"
            title={t.label}
            aria-label={`Theme ${t.label}`}
            aria-pressed={active}
            onClick={() => setThemeId(t.id as ThemeId)}
            className="theme-swatch"
            style={{
              width: compact ? 14 : 16,
              height: compact ? 14 : 16,
              borderRadius: '50%',
              padding: 0,
              cursor: 'pointer',
              background: t.swatch,
              border: active
                ? '2px solid #fff'
                : '2px solid transparent',
              boxShadow: active
                ? `0 0 0 1px rgba(var(--accent-rgb), 0.5), 0 0 12px ${t.glow}`
                : '0 0 0 1px rgba(255,255,255,0.15)',
              transition:
                'transform 0.25s ease, box-shadow 0.3s ease, border-color 0.25s ease',
              transform: active ? 'scale(1.15)' : 'scale(1)',
              flexShrink: 0,
            }}
          />
        );
      })}
    </div>
  );
}
