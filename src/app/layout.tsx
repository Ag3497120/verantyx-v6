import type { Metadata } from 'next';
import { Syne, DM_Sans } from 'next/font/google';
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
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${syne.variable} ${dmSans.variable}`}>
      <body className="antialiased" style={{ fontFamily: 'var(--font-body)' }}>
        <ClientProviders>{children}</ClientProviders>
      </body>
    </html>
  );
}
