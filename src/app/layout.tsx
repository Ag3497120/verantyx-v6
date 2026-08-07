import type { Metadata } from 'next';
import { Syne, DM_Sans, Noto_Sans_JP } from 'next/font/google';
import './globals.css';
import ClientProviders from '@/components/ClientProviders';

const syne = Syne({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-syne',
  weight: ['500', '600', '700', '800'],
});

const dmSans = DM_Sans({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-dm-sans',
  weight: ['300', '400', '500', '600', '700'],
});

// Neither Syne nor DM Sans carries a single Japanese glyph, so every kana and
// kanji on the site was falling through to whatever the visitor's OS happened
// to supply — a different face, a different weight, and a different optical
// size from the Latin around it, sentence by sentence. Noto Sans JP sits
// AFTER the Latin faces in the stack, so Latin still renders in Syne/DM Sans
// and only the characters they lack reach it.
const notoJP = Noto_Sans_JP({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-noto-jp',
  weight: ['400', '500', '700'],
});

export const metadata: Metadata = {
  title: 'Verantyx — Local AI CLI & Apps',
  description:
    'Verantyx-CLI: resident local router, council & eternal memory. Plus mouth-controlled iOS games.',
  keywords: [
    'verantyx-cli',
    'local AI',
    'router',
    'iOS game',
    'face tracking',
    'PakuPaku Fishing',
    'MouthEat',
  ],
  metadataBase: new URL('https://verantyx.ai'),
  icons: {
    icon: [
      // SVG first: a browser that understands it gets the mark sharp at any
      // size, and the .ico stays as the fallback for the ones that do not.
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon.ico', sizes: '32x32' },
    ],
    apple: [{ url: '/apple-touch-icon.png', sizes: '180x180' }],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${syne.variable} ${dmSans.variable} ${notoJP.variable}`}>
      <body className="antialiased" style={{ fontFamily: 'var(--font-body)' }}>
        <ClientProviders>{children}</ClientProviders>
      </body>
    </html>
  );
}
