'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { CinematicSection, GlassCard, FeatureCheck, PageHero } from '@/components/CinematicSection';

export default function MouthEatPage() {
  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      <PageHero
        icon="😋"
        title="MouthEat"
        subtitle="口を開けて食べまくれ！リアルタイム食べゲーム"
        subtitle2="Open wide and eat! A real-time mouth-controlled eating game"
        gradient="linear-gradient(135deg, #F59E0B, #EF4444)"
      >
        <a href="#" className="inline-block">
          <img
            src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='48' viewBox='0 0 140 48'%3E%3Cg fill='none'%3E%3Crect width='140' height='48' rx='8' fill='%23000'/%3E%3Ctext x='70' y='28' font-family='Arial' font-size='12' fill='%23FFF' text-anchor='middle' font-weight='600'%3EDownload on the%3C/text%3E%3Ctext x='70' y='40' font-family='Arial' font-size='16' fill='%23FFF' text-anchor='middle' font-weight='700'%3EApp Store%3C/text%3E%3C/g%3E%3C/svg%3E"
            alt="Download on the App Store"
            width="140"
            height="48"
          />
        </a>
      </PageHero>

      {/* Game Overview */}
      <CinematicSection title="Game Overview">
        <p className="text-gray-400 text-lg leading-[2em]">
          口の動きをリアルタイム検出して食べ物をキャッチする新感覚ゲーム。画面を流れてくる100種類以上の食べ物を、実際に口を開けてパクパク食べよう！食べられないものには気をつけて。
        </p>
      </CinematicSection>

      {/* Game Modes */}
      <CinematicSection title="Game Modes">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <GameModeCard
            icon="⏱️"
            title="タイムアタック"
            color="#EF4444, #F97316"
            features={[
              '制限時間60秒',
              '1レーン',
              '時間とともにスピードアップ',
            ]}
          />
          <GameModeCard
            icon="❤️"
            title="サバイバル"
            color="#DC2626, #991B1B"
            features={[
              'HP制（初期HP: 100）',
              '3レーン（高難易度）',
              '食べられない物でダメージ',
            ]}
          />
          <GameModeCard
            icon="🔥"
            title="コンボチャレンジ"
            color="#A855F7, #EC4899"
            features={[
              'コンボ継続がカギ',
              'なし → Warming → Hot → Blazing → Legendary',
              'コンボが途切れたらゲームオーバー',
            ]}
          />
          <GameModeCard
            icon="🧠"
            title="ジャッジメント"
            color="#14B8A6, #06B6D4"
            features={[
              'ベルトコンベア方式',
              '全35アイテム（75%食用, 25%非食用）',
              '正しく判断して食べよう',
            ]}
          />
        </div>
      </CinematicSection>

      {/* Food Encyclopedia */}
      <CinematicSection title="Food Encyclopedia">
        <p className="text-gray-500 text-sm tracking-wider uppercase mb-8">100種類以上のアイテム</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-10">
          <FoodCategory title="🍎 フルーツ（16種）" items="りんご、バナナ、ぶどう、みかん、すいか..." />
          <FoodCategory title="🥩 肉類（6種）" items="ステーキ、チキン、ハンバーガー..." />
          <FoodCategory title="🦐 海鮮（5種）" items="寿司、エビ、たこ焼き..." />
          <FoodCategory title="🍰 スイーツ（12種）" items="ケーキ、ドーナツ、アイスクリーム..." />
          <FoodCategory title="🥕 野菜（11種）" items="にんじん、ブロッコリー、とうもろこし..." />
          <FoodCategory title="🍕 料理（24種）" items="ピザ、ラーメン、おにぎり、カレー..." />
          <FoodCategory title="🥤 ドリンク（8種）" items="ジュース、牛乳、お茶、コーヒー..." />
        </div>

        <h3
          className="text-lg font-semibold mb-4 tracking-wider"
          style={{ color: 'rgba(250, 204, 21, 0.8)' }}
        >
          スペシャルアイテム
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-10">
          <SpecialItem icon="✨" name="金のりんご" description="レア出現、200pt" />
          <SpecialItem icon="🌈" name="レインボー" description="超レア、500pt" />
          <SpecialItem icon="🌶️" name="激辛唐辛子" description="ダメージ" />
          <SpecialItem icon="💣" name="爆弾" description="マイナススコア" />
        </div>

        <h3
          className="text-lg font-semibold mb-4 tracking-wider"
          style={{ color: 'rgba(239, 68, 68, 0.8)' }}
        >
          食べられないもの（60種以上）
        </h3>
        <p className="text-gray-500 leading-relaxed">
          車、バス、ロケット、家、病院、ハンマー、パソコン、靴、サッカーボール...
        </p>
      </CinematicSection>

      {/* Avatar System */}
      <CinematicSection title="Avatar System">
        <GlassCard>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">複数のアバタースタイル</h3>
              <p className="text-gray-500">写真からカスタムアバター作成が可能</p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-3">10種類の表情</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 text-gray-400 text-sm">
                <div>😐 アイドル</div>
                <div>😮 口開け</div>
                <div>😋 もぐもぐ</div>
                <div>😊 幸せ</div>
                <div>🤤 満腹</div>
                <div>🤢 気持ち悪い</div>
                <div>🥵 辛い</div>
                <div>🤩 超幸せ</div>
                <div>😵 瀕死</div>
                <div>😍 ハート目</div>
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">テーマカラーのカスタマイズ</h3>
              <p className="text-gray-500">ゲームの雰囲気を自分好みに変更可能</p>
            </div>
          </div>
        </GlassCard>
      </CinematicSection>

      {/* Gameplay Recording */}
      <CinematicSection title="Recording">
        <GlassCard>
          <div className="space-y-3">
            <FeatureCheck>ReplayKitで15秒クリップ録画</FeatureCheck>
            <FeatureCheck>フォトライブラリに保存</FeatureCheck>
            <FeatureCheck>ベストプレイを友達とシェア</FeatureCheck>
          </div>
        </GlassCard>
      </CinematicSection>

      {/* Pricing */}
      <CinematicSection title="Pricing">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(55,65,81,0.5)' }}>
                <th className="text-left py-4 px-4 text-gray-500 font-medium text-sm tracking-wider uppercase">機能</th>
                <th className="text-center py-4 px-4 font-medium text-sm tracking-wider" style={{ color: 'rgba(14, 165, 233, 0.7)' }}>Free</th>
                <th className="text-center py-4 px-4 font-medium text-sm tracking-wider" style={{ color: 'rgba(168, 85, 247, 0.7)' }}>PASS (¥250)</th>
              </tr>
            </thead>
            <tbody className="text-gray-400">
              <PricingRow feature="全ゲームモード" free={true} pass={true} />
              <PricingRow feature="基本プレイ" free={true} pass={true} />
              <PricingRow feature="広告なし" free={false} pass={true} />
              <PricingRow feature="スピード調整（0.5x〜2.0x）" free={false} pass={true} />
              <PricingRow feature="出現間隔調整" free={false} pass={true} />
            </tbody>
          </table>
        </div>
        <p className="text-center text-gray-600 text-xs mt-6 tracking-wide">
          PASS = ¥250の買い切り（サブスクリプションではない）
        </p>
      </CinematicSection>

      {/* Privacy */}
      <CinematicSection title="Privacy">
        <GlassCard>
          <div className="space-y-3">
            <FeatureCheck>カメラ: 口の動き検出（デバイス上処理のみ、外部送信なし）</FeatureCheck>
            <FeatureCheck>フォトライブラリ: 録画クリップの保存</FeatureCheck>
            <FeatureCheck>ランキング: スコアデータのみ送信</FeatureCheck>
            <FeatureCheck>顔データの保存・収集なし</FeatureCheck>
          </div>
        </GlassCard>
        <p className="text-center text-gray-600 text-xs mt-6">
          詳細は<a href="/privacy" className="transition-colors duration-300" style={{ color: 'rgba(14, 165, 233, 0.6)' }} onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.9)'} onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.6)'}>プライバシーポリシー</a>をご覧ください
        </p>
      </CinematicSection>

      {/* Other App */}
      <section className="relative px-6 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-gray-600 text-xs tracking-[0.3em] uppercase mb-6">Other App</p>
          <motion.a
            href="/apps/pakupaku-fishing"
            whileHover={{ y: -3 }}
            className="inline-flex items-center gap-4 px-8 py-5 rounded-2xl transition-all duration-300"
            style={{
              background: 'rgba(10, 10, 20, 0.6)',
              border: '1px solid rgba(14, 165, 233, 0.1)',
              textDecoration: 'none',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.3)';
              e.currentTarget.style.boxShadow = '0 0 30px rgba(14, 165, 233, 0.06)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.1)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <span className="text-4xl">🎣</span>
            <div className="text-left">
              <div className="font-bold text-white">パクパク釣り</div>
              <div className="text-sm text-gray-500">口で釣る、新感覚フィッシングゲーム</div>
            </div>
          </motion.a>
        </div>
      </section>

      <Footer />
    </main>
  );
}

function GameModeCard({
  icon,
  title,
  color,
  features,
}: {
  icon: string;
  title: string;
  color: string;
  features: string[];
}) {
  return (
    <GlassCard>
      <div className="text-4xl mb-4">{icon}</div>
      <h3
        className="text-xl font-bold mb-4"
        style={{
          background: `linear-gradient(135deg, ${color})`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}
      >
        {title}
      </h3>
      <ul className="space-y-2">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start gap-2.5 text-gray-400 text-sm">
            <span className="mt-0.5" style={{ color: 'rgba(14, 165, 233, 0.5)' }}>•</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

function FoodCategory({ title, items }: { title: string; items: string }) {
  return (
    <GlassCard hover={false}>
      <h3 className="font-semibold text-white text-sm mb-2">{title}</h3>
      <p className="text-gray-500 text-sm">{items}</p>
    </GlassCard>
  );
}

function SpecialItem({ icon, name, description }: { icon: string; name: string; description: string }) {
  return (
    <GlassCard hover={false}>
      <div className="flex items-center gap-3">
        <span className="text-2xl">{icon}</span>
        <div>
          <div className="font-semibold text-white text-sm">{name}</div>
          <div className="text-xs text-gray-500">{description}</div>
        </div>
      </div>
    </GlassCard>
  );
}

function PricingRow({ feature, free, pass }: { feature: string; free: boolean; pass: boolean }) {
  return (
    <tr style={{ borderBottom: '1px solid rgba(55,65,81,0.3)' }}>
      <td className="py-3.5 px-4 text-sm">{feature}</td>
      <td className="py-3.5 px-4 text-center text-sm">{free ? '✅' : '❌'}</td>
      <td className="py-3.5 px-4 text-center text-sm">{pass ? '✅' : '❌'}</td>
    </tr>
  );
}
