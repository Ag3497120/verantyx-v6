import ParticleBackground from '@/components/ParticleBackground';
import Hero from '@/components/Hero';
import Benchmarks from '@/components/Benchmarks';
import HowItWorks from '@/components/HowItWorks';
import Architecture from '@/components/Architecture';
import ScoreChart from '@/components/ScoreChart';
import Stats from '@/components/Stats';
import Footer from '@/components/Footer';


export default function Home() {
  return (
    <main className="relative">
      <ParticleBackground />
      <Hero />
      <Benchmarks />
      <HowItWorks />
      <Architecture />
      <ScoreChart />
      <Stats />
      <Footer />
    </main>
  );
}
