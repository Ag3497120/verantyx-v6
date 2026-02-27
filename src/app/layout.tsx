import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';


const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'Verantyx - Symbolic Reasoning. Zero Neural Networks.',
  description: 'LLM-free symbolic reasoning engine achieving 22.1% on ARC-AGI-2 through pure program synthesis.',
  keywords: ['symbolic reasoning', 'ARC-AGI', 'program synthesis', 'no neural networks'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
