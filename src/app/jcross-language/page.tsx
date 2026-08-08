'use client';

import { useLanguage } from '@/lib/i18n';

/* The guide is a standalone HTML document in public/, not a React page, so the
 * language cannot be passed through context. It goes in the URL fragment
 * instead of a query string: a fragment never reaches a server, and this page
 * has spent its whole life not sending anything anywhere. The guide reads it
 * if it knows how, and ignores it harmlessly if it does not. */
export default function JCrossLanguage() {
  const { lang } = useLanguage();

  return (
    <iframe
      src={`/jcross-language-guide.html#lang=${lang}`}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        border: 'none',
        margin: 0,
        padding: 0,
        overflow: 'hidden',
      }}
      title={lang === 'ja' ? '.jcross 言語ガイド' : '.jcross Language Guide'}
    />
  );
}
