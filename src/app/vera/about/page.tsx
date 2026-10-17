import type { Metadata } from 'next';
import VeraHome from '@/components/vera-home/VeraHome';

/* Vera's own page; it was the site's home until the home became the portal (VeRantyx first). */
export const metadata: Metadata = {
  title: 'Vera — 嘘をつけない枠をつくる AI | Verantyx',
  description:
    'Vera: 答えには出所を付け、分からなければ型つきで黙る AI。実測の値、できないこと、記録済みの実行結果での体験。',
};

export default function VeraAbout() {
  return <VeraHome />;
}
