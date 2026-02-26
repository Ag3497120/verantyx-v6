'use client';

import CrossStructure3D from '@/components/CrossStructure3D';
import CrossNavigation from '@/components/CrossNavigation';
import SolverAnimation from '@/components/SolverAnimation';
import Hero from '@/components/Hero';
import BenchmarkCounters from '@/components/BenchmarkCounters';
import InstallCommand from '@/components/InstallCommand';
import HowItWorks from '@/components/HowItWorks';
import ArchitecturePipeline from '@/components/ArchitecturePipeline';
import ScoreChart from '@/components/ScoreChart';
import LinkReveal from '@/components/LinkReveal';
import Stats from '@/components/Stats';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      {/* Section 1: 3D Cross Hero */}
      <section className="relative min-h-screen flex items-center justify-center">
        <CrossStructure3D />
        <div className="relative z-10 text-center">
          <Hero />
          <InstallCommand />
        </div>
      </section>

      {/* Section 2: Benchmarks */}
      <section className="relative min-h-screen flex items-center justify-center">
        <CrossNavigation />
        <BenchmarkCounters />
      </section>

      {/* Section 3: Solver Animation */}
      <section className="relative">
        <SolverAnimation />
      </section>

      {/* Section 4: How It Works */}
      <section className="relative min-h-screen flex items-center justify-center px-8">
        <HowItWorks />
      </section>

      {/* Section 5: Architecture Pipeline */}
      <section className="relative min-h-screen flex items-center justify-center px-8">
        <ArchitecturePipeline />
      </section>

      {/* Section 6: Score Evolution */}
      <section className="relative min-h-[80vh] flex items-center justify-center px-8">
        <ScoreChart />
      </section>

      {/* Section 7: Links revealed from cross */}
      <section className="relative min-h-screen flex items-center justify-center">
        <LinkReveal />
      </section>

      {/* Stats + Footer */}
      <Stats />
      <Footer />
    </main>
  );
}
