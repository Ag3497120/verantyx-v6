import type { Metadata } from 'next';
import TryPage from '@/components/vera-home/TryPage';

export const metadata: Metadata = {
  title: 'Vera を体験する | Verantyx',
  description: '固定の文書に問いを投げて、Vera の答えと型つきの沈黙を見る（記録済みの実行結果）。',
};

export default function Try() {
  return <TryPage />;
}
