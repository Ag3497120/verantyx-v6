'use client';

import { useEffect, useState } from 'react';
import { useLanguage } from '@/lib/i18n';

export type HomeLang = 'ja' | 'en';
export type L = { ja: string; en: string };

/* The shared LanguageProvider always renders English during the static export and then
 * guesses from the browser. This home is written for Japanese readers first, so it keeps
 * its own view: Japanese unless the visitor has explicitly chosen English (the stored
 * value that the Navbar's JP/EN button writes). */
export function useHomeLang(): HomeLang {
  const { lang, resolving } = useLanguage();
  const [view, setView] = useState<HomeLang>('ja');
  useEffect(() => {
    if (resolving) return;
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem('verantyx.lang');
    } catch {
      /* storage denied: stay Japanese */
    }
    setView(stored === 'en' ? 'en' : 'ja');
  }, [lang, resolving]);
  return view;
}

export function pick(view: HomeLang, o: L): string {
  return o[view];
}
