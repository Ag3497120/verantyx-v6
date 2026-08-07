'use client';

import { LanguageProvider } from '@/lib/i18n';
import { ThemeProvider } from '@/lib/theme';
import SiteBot from '@/components/SiteBot';
import { ReactNode } from 'react';

/* The bot is mounted here rather than per page, so it is resident: it stays
 * on screen on every route, including ones added later. */
export default function ClientProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <LanguageProvider>
        {children}
        <SiteBot />
      </LanguageProvider>
    </ThemeProvider>
  );
}
