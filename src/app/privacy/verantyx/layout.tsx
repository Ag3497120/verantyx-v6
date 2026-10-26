import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'VeRantyx Privacy Policy',
  description: 'VeRantyx (the Mac app and the Apple Vision Pro app) collects no data.',
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
