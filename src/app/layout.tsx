import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import ClientProviders from '@/components/ClientProviders';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'Verantyx - Symbolic Reasoning & iOS Apps',
  description: 'Symbolic reasoning engine and innovative iOS games using facial recognition. PakuPaku Fishing, MouthEat.',
  keywords: ['symbolic reasoning', 'ARC-AGI', 'iOS game', 'face tracking', 'fishing game', 'mouth game'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">
        <ClientProviders>
          {children}
        </ClientProviders>
      </body>
    </html>
  );
}
