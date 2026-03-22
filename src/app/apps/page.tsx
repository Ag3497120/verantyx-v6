'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { PageHero, GlassCard, FeatureCheck } from '@/components/CinematicSection';
import { useLanguage } from '@/lib/i18n';

export default function AppsPage() {
  const { lang } = useLanguage();

  return (
    <main className="relative bg-black text-white overflow-x-hidden min-h-screen">
      <Navbar />

      <PageHero
        icon="📱"
        title={lang === 'ja' ? 'アプリ' : 'Apps'}
        subtitle={lang === 'ja' ? '口の動きで遊ぶ革新的なiOSゲーム' : 'Innovative iOS games controlled by mouth movements'}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />

      {/* App Cards */}
      <section className="relative px-6 pb-32">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Paku Paku Fishing */}
          <AppCard
            icon="🎣"
            title={lang === 'ja' ? 'パクパク釣り' : 'Paku Paku Fishing'}
            subtitle={lang === 'ja' ? 'Paku Paku Fishing' : 'パクパク釣り'}
            description={lang === 'ja' ? '口で釣る、新感覚フィッシングゲーム' : 'Revolutionary fishing game controlled by mouth movements'}
            features={lang === 'ja' ? [
              '20種の魚 × 4段階の難易度',
              '13種の顔ジェスチャー',
              'カスタムコンボ作成',
              '3つのゲームモード',
            ] : [
              '20 fish × 4 difficulty levels',
              '13 facial gestures',
              'Custom combo creation',
              '3 game modes',
            ]}
            pricing="Free + ¥250 PASS"
            href="/apps/pakupaku-fishing"
            delay={0}
            lang={lang}
          />

          {/* MouthEat */}
          <AppCard
            icon="😋"
            title="MouthEat"
            subtitle=""
            description={lang === 'ja' ? '口を開けて食べまくれ！リアルタイム食べゲーム' : 'Open your mouth and eat! Real-time eating game'}
            features={lang === 'ja' ? [
              '100種類以上のアイテム',
              '4つのゲームモード',
              'アバターカスタマイズ',
              'ゲームプレイ録画',
            ] : [
              '100+ items',
              '4 game modes',
              'Avatar customization',
              'Gameplay recording',
            ]}
            pricing="Free + ¥250 PASS"
            href="/apps/mouth-eat"
            delay={0.15}
            lang={lang}
          />

          {/* World Rankings */}
          <RankingCard
            icon="🏆"
            title={lang === 'ja' ? 'ワールドランキング' : 'World Rankings'}
            subtitle={lang === 'ja' ? 'World Rankings' : 'ワールドランキング'}
            description={lang === 'ja' ? 'リアルタイムのワールドランキング' : 'Real-time global leaderboards'}
            features={lang === 'ja' ? [
              'パクパク釣り タイムアタック',
              'MouthEat 全4モード',
              'リアルタイム更新',
              'ニックネーム検索',
            ] : [
              'Paku Paku Fishing Time Attack',
              'MouthEat All 4 Modes',
              'Real-time updates',
              'Nickname search',
            ]}
            href="/apps/ranking"
            delay={0.3}
            lang={lang}
          />
        </div>
      </section>

      <Footer />
    </main>
  );
}

function AppCard({
  icon,
  title,
  subtitle,
  description,
  features,
  pricing,
  href,
  delay,
  lang,
}: {
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  features: string[];
  pricing: string;
  href: string;
  delay: number;
  lang: 'ja' | 'en';
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true }}
      transition={{ duration: 0.8, delay }}
    >
      <GlassCard>
        <div className="flex flex-col gap-4">
          <div className="text-6xl">{icon}</div>
          <div>
            <h2 className="text-3xl font-bold text-white mb-1">{title}</h2>
            {subtitle && <p className="text-lg text-gray-500">{subtitle}</p>}
          </div>
          <p className="text-gray-400 text-lg leading-relaxed">{description}</p>

          <div className="space-y-2.5 my-4">
            {features.map((feature, idx) => (
              <FeatureCheck key={idx}>{feature}</FeatureCheck>
            ))}
          </div>

          <p className="text-sm font-medium" style={{ color: 'rgba(14, 165, 233, 0.7)' }}>{pricing}</p>

          {/* App Store Badge */}
          <a
            href="#"
            className="inline-block my-2"
            style={{ opacity: 0.7, transition: 'opacity 0.3s' }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.7')}
          >
            <img
              src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='40' viewBox='0 0 120 40'%3E%3Cg fill='none'%3E%3Crect width='120' height='40' rx='5' fill='%23000'/%3E%3Ctext x='60' y='24' font-family='Arial' font-size='10' fill='%23FFF' text-anchor='middle'%3EApp Store%3C/text%3E%3C/g%3E%3C/svg%3E"
              alt="Download on the App Store"
              width="120"
              height="40"
            />
          </a>

          <a
            href={href}
            className="inline-flex items-center gap-2 text-sm font-semibold group"
            style={{ color: 'rgba(14, 165, 233, 0.7)', textDecoration: 'none' }}
          >
            <span>{lang === 'ja' ? '詳しく見る' : 'Learn more'}</span>
            <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
          </a>
        </div>
      </GlassCard>
    </motion.div>
  );
}

function RankingCard({
  icon,
  title,
  subtitle,
  description,
  features,
  href,
  delay,
  lang,
}: {
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  features: string[];
  href: string;
  delay: number;
  lang: 'ja' | 'en';
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true }}
      transition={{ duration: 0.8, delay }}
    >
      <GlassCard>
        <div className="flex flex-col gap-4">
          <div className="text-6xl">{icon}</div>
          <div>
            <h2 className="text-3xl font-bold text-white mb-1">{title}</h2>
            {subtitle && <p className="text-lg text-gray-500">{subtitle}</p>}
          </div>
          <p className="text-gray-400 text-lg leading-relaxed">{description}</p>

          <div className="space-y-2.5 my-4">
            {features.map((feature, idx) => (
              <FeatureCheck key={idx}>{feature}</FeatureCheck>
            ))}
          </div>

          <a
            href={href}
            className="inline-flex items-center gap-2 text-sm font-semibold mt-2 group"
            style={{ color: 'rgba(14, 165, 233, 0.7)', textDecoration: 'none' }}
          >
            <span>{lang === 'ja' ? 'ランキングを見る' : 'View rankings'}</span>
            <span className="group-hover:translate-x-2 transition-transform duration-300">→</span>
          </a>
        </div>
      </GlassCard>
    </motion.div>
  );
}
