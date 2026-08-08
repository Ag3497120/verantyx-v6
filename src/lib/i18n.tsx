'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from 'react';

type Language = 'ja' | 'en';

const KEY = 'verantyx.lang';

/* Where the region guess comes from, and what it deliberately is not.
 *
 * Not an IP lookup. A geo-IP call would mean this page reports every visitor's
 * address to a third party in order to pick a font, and it would also make the
 * site stop being a static export — there would be a server, and a server is
 * the thing this project keeps refusing to have. Both signals below are
 * already in the browser, cost nothing, and leave nothing.
 *
 *   navigator.languages   what the visitor told their own browser to prefer
 *   time zone             Asia/Tokyo, for a Japanese reader whose OS is in
 *                         English — common enough among the municipal and
 *                         engineering audiences this site is written for
 *
 * The language list wins when it says anything at all, because a stated
 * preference beats an inferred one. The time zone only breaks a tie.
 */
function detect(): Language {
  if (typeof navigator === 'undefined') return 'en';

  const stated = (navigator.languages && navigator.languages.length
    ? navigator.languages
    : [navigator.language]
  ).filter(Boolean);

  for (const tag of stated) {
    const base = String(tag).toLowerCase().split('-')[0];
    if (base === 'ja') return 'ja';
    if (base) return 'en';
  }

  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    if (tz === 'Asia/Tokyo') return 'ja';
  } catch {
    /* Intl without a resolvable zone: fall through to English. */
  }
  return 'en';
}

interface LanguageContextType {
  lang: Language;
  setLang: (lang: Language) => void;
  /* True until the first client pass has run. A page can use this to avoid
   * announcing a language it is about to change. */
  resolving: boolean;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  /* Starts English to match the prerendered HTML. Detection cannot run during
   * the static export — there is no visitor at build time — so hydrating with
   * anything else would mismatch the server markup on the very first paint. */
  const [lang, setLangState] = useState<Language>('en');
  const [resolving, setResolving] = useState(true);

  useEffect(() => {
    let chosen: Language;
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(KEY);
    } catch {
      /* Storage can be denied outright; detection still works without it. */
    }
    /* An explicit choice outranks the guess, permanently. Someone who
     * switched to English in Tokyo meant it, and re-guessing on every visit
     * would quietly overrule them. */
    chosen = stored === 'ja' || stored === 'en' ? stored : detect();
    setLangState(chosen);
    setResolving(false);
  }, []);

  /* Keep the document in step. The <html lang> attribute is what a screen
   * reader picks its voice from and what the browser offers to translate, and
   * a page whose text is Japanese under lang="en" gets read aloud wrong. */
  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = lang;
    }
  }, [lang]);

  const setLang = useCallback((next: Language) => {
    setLangState(next);
    try {
      window.localStorage.setItem(KEY, next);
    } catch {
      /* A refused write costs the preference on the next visit, nothing more. */
    }
  }, []);

  return (
    <LanguageContext.Provider value={{ lang, setLang, resolving }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
