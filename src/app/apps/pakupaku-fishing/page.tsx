'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

type Lang = 'ja' | 'en';

type BilingualText = {
  ja: string;
  en: string;
};

export default function PakuPakuFishingPage() {
  const [lang, setLang] = useState<Lang>('ja');

  const t = (obj: BilingualText) => obj[lang];

  // Content
  const content = {
    hero: {
      title: { ja: 'パクパク釣り', en: 'Paku Paku Fishing' },
      tagline: { ja: '口で釣る、新感覚フィッシングゲーム', en: 'Catch fish using only your mouth — a revolutionary fishing experience' },
    },
    sections: {
      overview: { ja: 'ゲーム概要', en: 'Game Overview' },
      gameModes: { ja: 'ゲームモード', en: 'Game Modes' },
      fishEncyclopedia: { ja: '魚図鑑', en: 'Fish Encyclopedia' },
      gestureSystem: { ja: '顔ジェスチャーシステム', en: 'Gesture System' },
      comboSystem: { ja: 'コンボシステム', en: 'Combo System' },
      resistanceSystem: { ja: '魚の抵抗システム', en: 'Fish Resistance System' },
      advancedTechniques: { ja: '応用テクニック', en: 'Advanced Techniques' },
      customization: { ja: 'カスタマイズ', en: 'Customization' },
      pricing: { ja: '料金プラン', en: 'Pricing Plans' },
      privacy: { ja: 'プライバシー', en: 'Privacy' },
      otherApps: { ja: '他のアプリもチェック', en: 'Check Out Other Apps' },
    },
    overview: {
      description: {
        ja: '顔の動きで釣りを楽しむ革新的なiOSゲーム。デバイスのカメラで口の動きと顔のジェスチャーをリアルタイム検出し、バーチャルな釣りを体験できます。',
        en: 'An innovative iOS game where you fish using facial movements. The device camera detects your mouth movements and facial gestures in real-time for an immersive virtual fishing experience.',
      },
      phases: [
        { icon: '🎯', title: { ja: '観察', en: 'Observe' }, description: { ja: '池を見渡し、キャスティングポイントを決める', en: 'Survey the pond and choose your casting point' } },
        { icon: '🎣', title: { ja: 'キャスティング', en: 'Cast' }, description: { ja: 'タップして釣り糸を投げる', en: 'Tap to cast your line' } },
        { icon: '⏳', title: { ja: '待機', en: 'Wait' }, description: { ja: '口を閉じてじっと待つ', en: 'Keep your mouth closed and wait' } },
        { icon: '🐟', title: { ja: 'アタリ', en: 'Bite' }, description: { ja: '魚が食いつく', en: 'Fish takes the bait' } },
        { icon: '⚡', title: { ja: 'フッキング', en: 'Hook' }, description: { ja: 'タイミングよく口を閉じて針を合わせる', en: 'Close mouth at the right moment to set the hook' } },
        { icon: '🔄', title: { ja: 'リーリング', en: 'Reel' }, description: { ja: '口を素早く開閉して魚を巻き上げる。魚の抵抗に注意！', en: 'Open and close mouth rapidly to reel in. Watch out for resistance!' } },
      ],
    },
    gameModes: [
      {
        icon: '🎣',
        title: { ja: 'フリーモード', en: 'Free Mode' },
        features: [
          { ja: '無制限に釣りを楽しめる', en: 'Unlimited fishing enjoyment' },
          { ja: 'スコア蓄積 & コンボシステム', en: 'Score accumulation & combo system' },
          { ja: '深度ゾーンによる魚の分布', en: 'Fish distribution by depth zones' },
        ],
      },
      {
        icon: '⏱️',
        title: { ja: 'タイムアタック', en: 'Time Attack' },
        features: [
          { ja: '60秒の時間制限チャレンジ', en: '60-second time limit challenge' },
          { ja: '3-2-1 カウントダウンでスタート', en: 'Start with 3-2-1 countdown' },
          { ja: 'ランキング対応', en: 'Ranking support' },
        ],
      },
      {
        icon: '🏆',
        title: { ja: '一本釣りチャレンジ', en: 'Boss Battle Challenge' },
        features: [
          { ja: '巨大な一匹を狙う', en: 'Target one massive catch' },
          { ja: '持続的な集中力が必要', en: 'Requires sustained focus' },
          { ja: '5〜120秒以上の長時間ファイト', en: '5-120+ second epic battles' },
          { ja: '伝説級の魚との真剣勝負', en: 'Serious showdown with legendary fish' },
        ],
      },
    ],
    fishEncyclopedia: {
      subtitle: { ja: '20種の魚 × 4段階の難易度', en: '20 species × 4 difficulty tiers' },
      depthNote: { ja: '深さ4ゾーン（浅瀬・中層・深海・深淵）で出現分布が変化', en: '4 depth zones (Shallow, Mid, Deep, Abyss) affect spawn distribution' },
      difficulties: [
        {
          level: 'Easy',
          color: 'bg-green-500/20 text-green-400 border-green-400/50',
          fish: { ja: ['メダカ', 'ハゼ', 'イワシ', 'フナ', 'アユ'], en: ['Killifish', 'Goby', 'Sardine', 'Crucian Carp', 'Sweetfish'] },
          score: '50〜150pt',
        },
        {
          level: 'Medium',
          color: 'bg-yellow-500/20 text-yellow-400 border-yellow-400/50',
          fish: { ja: ['タイ', 'サバ', 'イカ', 'ヒラメ', 'ブリ'], en: ['Sea Bream', 'Mackerel', 'Squid', 'Flounder', 'Yellowtail'] },
          score: '200〜400pt',
        },
        {
          level: 'Hard',
          color: 'bg-orange-500/20 text-orange-400 border-orange-400/50',
          fish: { ja: ['マグロ', 'カツオ', 'タコ', 'スズキ', 'フグ'], en: ['Tuna', 'Bonito', 'Octopus', 'Sea Bass', 'Pufferfish'] },
          score: '450〜800pt',
        },
        {
          level: 'Legendary',
          color: 'bg-purple-500/20 text-purple-400 border-purple-400/50',
          fish: { ja: ['クジラ', 'サメ', 'ダイオウイカ', 'シーラカンス', '金の鯉'], en: ['Whale', 'Shark', 'Giant Squid', 'Coelacanth', 'Golden Carp'] },
          score: '1,000〜5,000pt',
        },
      ],
    },
    gestureSystem: {
      basicTitle: { ja: '基本（無料）', en: 'Basic (Free)' },
      gestures: [
        { ja: '口を開ける', en: 'Open mouth' },
        { ja: '口を閉じる', en: 'Close mouth' },
      ],
    },
    comboSystem: {
      subtitle: { ja: '5つのビルトインコンボ（全ユーザー利用可能）', en: '5 built-in combos (available to all users)' },
      combos: [
        { icon: '💥', name: { ja: 'ラピッドマウスフラップ', en: 'Rapid Mouth Flap' }, effect: { ja: '超高速口パクパクで1.5倍スコア', en: 'Ultra-fast chomping for 1.5× score' } },
        { icon: '🔥', name: { ja: 'パワーリフト', en: 'Power Lift' }, effect: { ja: '上を向いてホールド+口閉じで2倍ダメージ', en: 'Look up + hold + close mouth for 2× damage' } },
        { icon: '🎯', name: { ja: 'ダイブキャスト', en: 'Dive Cast' }, effect: { ja: '下を向いて口を開けると1.5倍射程', en: 'Look down + open mouth for 1.5× cast range' } },
        { icon: '🌊', name: { ja: 'ドッジカウンター', en: 'Dodge Counter' }, effect: { ja: '左右に首振りで1.5倍ダメージ+魚フリーズ', en: 'Turn head left/right for 1.5× damage + fish freeze' } },
        { icon: '🧘', name: { ja: 'フォーカスフィッシング', en: 'Focus Fishing' }, effect: { ja: '眉上げ3秒キープでレア魚70%出現', en: 'Raise eyebrows for 3s → 70% rare fish spawn' } },
      ],
      customCombo: {
        title: { ja: 'カスタムコンボ作成（PASS限定）', en: 'Custom Combo Creation (PASS only)' },
        features: [
          { ja: '最大5個のオリジナルコンボを作成可能', en: 'Create up to 5 custom combos' },
          { ja: 'ジェスチャーの組み合わせ・エフェクト・クールダウンを自由にカスタマイズ', en: 'Customize gesture combos, effects, and cooldowns freely' },
        ],
      },
    },
    resistanceSystem: {
      description: {
        ja: '大物ほど激しく抵抗する！リール中に魚が暴れてリールが引き戻されます。レア度が高い魚ほど抵抗力が強く、戦略的な対処が求められます。',
        en: 'Bigger fish fight back harder! Fish will struggle and pull the reel back. Rarer fish have stronger resistance, demanding strategic handling.',
      },
      mechanics: [
        {
          icon: '💪',
          title: { ja: '引き戻し', en: 'Pullback' },
          description: {
            ja: '魚は常にリールを引き戻す力を加えています。深い場所ほど、レアな魚ほど引き戻しが強くなります。パクパクを止めるとリールが戻ってしまいます。',
            en: 'Fish constantly pull back on the reel. Deeper and rarer fish pull harder. Stop chomping and the reel retracts.',
          },
          color: 'border-orange-500/30',
        },
        {
          icon: '⚡',
          title: { ja: '突発的な抵抗', en: 'Sudden Resistance' },
          description: {
            ja: 'リール中にランダムで魚が激しく暴れます。通常の3倍の速度でリールが引き戻され、画面が揺れて警告が表示されます。',
            en: 'Fish randomly thrash while reeling. Reel retracts 3× faster, screen shakes, and warnings appear.',
          },
          color: 'border-red-500/30',
        },
      ],
    },
    advancedTechniques: {
      subtitle: { ja: '大物を釣り上げるためのコンボ技を使いこなそう', en: 'Master combo techniques to catch big fish' },
      techniques: [
        {
          icon: '🌊',
          name: { ja: 'いなし（ドッジカウンター）', en: 'Dodge Counter' },
          timing: { ja: '魚が抵抗した時', en: 'When fish resists' },
          operation: { ja: '首を左 → 右に振る', en: 'Turn head left → right' },
          effect: { ja: '魚が2秒間フリーズし、リールの引き戻しが止まる', en: 'Fish freezes for 2s, pullback stops' },
          color: 'border-cyan-500/30',
        },
        {
          icon: '🔥',
          name: { ja: '力強い引き上げ（パワーリフト）', en: 'Power Lift' },
          timing: { ja: 'リーリング中', en: 'While reeling' },
          operation: { ja: '上を向いて0.5秒ホールド → 口を閉じる', en: 'Look up + hold 0.5s → close mouth' },
          effect: { ja: 'ダメージ2倍で一気にリールを巻き上げ', en: '2× damage for rapid reeling' },
          color: 'border-red-500/30',
        },
        {
          icon: '🎯',
          name: { ja: 'ダイブ投げ（ダイブキャスト）', en: 'Dive Cast' },
          timing: { ja: '観察フェーズ', en: 'Observation phase' },
          operation: { ja: '下を向いて口を開ける', en: 'Look down + open mouth' },
          effect: { ja: 'キャスト飛距離1.5倍で深い場所に届く', en: '1.5× cast distance to reach deeper zones' },
          color: 'border-blue-500/30',
        },
        {
          icon: '🧘',
          name: { ja: '集中釣り（フォーカスフィッシング）', en: 'Focus Fishing' },
          timing: { ja: '観察フェーズ', en: 'Observation phase' },
          operation: { ja: '眉を上げて3秒キープ', en: 'Raise eyebrows for 3s' },
          effect: { ja: 'レア魚の出現率が70%アップ', en: '70% rare fish spawn rate boost' },
          color: 'border-purple-500/30',
        },
        {
          icon: '💥',
          name: { ja: '連続口パク（ラピッドマウスフラップ）', en: 'Rapid Mouth Flap' },
          timing: { ja: 'いつでも', en: 'Anytime' },
          operation: { ja: '口を素早く5回パクパク', en: 'Rapidly chomp 5 times' },
          effect: { ja: 'スコア1.5倍ボーナス', en: '1.5× score bonus' },
          color: 'border-yellow-500/30',
        },
      ],
    },
    customization: {
      cards: [
        {
          title: { ja: 'カスタム魚パック', en: 'Custom Fish Packs' },
          features: [
            { ja: 'オリジナルの魚を作成（写真 or 絵文字）', en: 'Create custom fish (photo or emoji)' },
            { ja: '難易度・名前・著作権情報を設定', en: 'Set difficulty, name, and copyright info' },
            { ja: 'ディープリンクで友達と共有可能', en: 'Share with friends via deep link' },
          ],
        },
        {
          title: { ja: '池キャプチャ（PASS限定）', en: 'Pond Capture (PASS only)' },
          features: [
            { ja: 'カメラで実際の水辺を撮影', en: 'Capture real water bodies with camera' },
            { ja: '自分だけの釣り場を作成', en: 'Create your own fishing spot' },
          ],
        },
      ],
    },
    pricing: {
      passNote: { ja: 'PASS = ¥250の買い切り（サブスクリプションではない）', en: 'PASS = ¥250 one-time purchase (not a subscription)' },
      headers: {
        feature: { ja: '機能', en: 'Feature' },
        free: { ja: 'Free', en: 'Free' },
        pass: { ja: 'PASS (¥250)', en: 'PASS (¥250)' },
      },
      rows: [
        { feature: { ja: '基本ゲームプレイ', en: 'Basic gameplay' }, free: true, pass: true },
        { feature: { ja: 'ビルトインコンボ', en: 'Built-in combos' }, free: true, pass: true },
        { feature: { ja: '広告なし', en: 'Ad-free' }, free: false, pass: true },
        { feature: { ja: '感度調整', en: 'Sensitivity adjustment' }, free: false, pass: true },
        { feature: { ja: 'カスタムコンボ作成', en: 'Custom combo creation' }, free: false, pass: true },
        { feature: { ja: '高度なジェスチャー（5種）', en: 'Advanced gestures (5 types)' }, free: true, pass: true },
        { feature: { ja: 'カスタム魚パック', en: 'Custom fish packs' }, free: false, pass: true },
        { feature: { ja: '池キャプチャ', en: 'Pond capture' }, free: false, pass: true },
      ],
    },
    privacy: {
      points: [
        { ja: 'カメラ映像はデバイス上でのみ処理', en: 'Camera feed processed on-device only' },
        { ja: '外部サーバーへの映像送信なし', en: 'No video sent to external servers' },
        { ja: '顔データの保存・収集なし', en: 'No facial data stored or collected' },
        { ja: '広告: Google AdMob（Free版のみ）', en: 'Ads: Google AdMob (Free version only)' },
      ],
      linkText: { ja: 'プライバシーポリシー', en: 'Privacy Policy' },
      detailsText: { ja: '詳細は', en: 'See' },
      detailsTextEnd: { ja: 'をご覧ください', en: 'for details' },
    },
    otherApp: {
      name: 'MouthEat',
      description: { ja: '口を開けて食べまくれ！', en: 'Open your mouth and eat everything!' },
    },
  };

  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      {/* Radial gradient overlay */}
      <div className="fixed inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse at 50% 0%, rgba(10,22,40,0.5) 0%, #050508 70%)' }} />

      <Navbar />

      {/* Language Toggle */}
      <div className="fixed top-20 right-6 z-50 backdrop-blur-md rounded-full px-1 py-1 flex gap-1" style={{ background: 'rgba(5,5,8,0.7)', border: '1px solid rgba(14,165,233,0.1)' }}>
        <button onClick={() => setLang('ja')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'ja' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'ja' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>JP</button>
        <button onClick={() => setLang('en')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'en' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'en' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>EN</button>
      </div>

      {/* Hero Section - Full viewport redesign */}
      <section className="relative min-h-screen flex items-center justify-center px-6 pt-32 pb-20">
        {/* Floating particles background */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(15)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-cyan-400/30 rounded-full animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${8 + Math.random() * 4}s`,
              }}
            />
          ))}
        </div>

        {/* Ambient glow blob behind fish */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="max-w-6xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.5, rotateY: 180 }}
            animate={{ opacity: 1, scale: 1, rotateY: 0 }}
            transition={{ duration: 0.8, type: 'spring' }}
            className="text-9xl mb-8"
          >
            🎣
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-8xl md:text-[10rem] font-black mb-6 tracking-wider"
            style={{
              background: 'linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #06B6D4 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              filter: 'drop-shadow(0 0 60px rgba(6,182,212,0.6))',
            }}
          >
            {t(content.hero.title)}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-xl text-cyan-300/70 italic tracking-widest mb-12 max-w-3xl mx-auto"
          >
            {t(content.hero.tagline)}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            <a
              href="#"
              className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl font-bold text-lg hover:scale-105 transition-transform shadow-lg shadow-cyan-400/30"
            >
              <span>📱</span>
              <span>Download on the App Store</span>
            </a>
          </motion.div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="absolute bottom-10 left-1/2 -translate-x-1/2"
          >
            <div className="w-6 h-10 border-2 border-cyan-400/30 rounded-full flex items-start justify-center p-2 animate-bounce">
              <div className="w-1.5 h-2 bg-cyan-400 rounded-full" />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Game Overview */}
      <Section title={t(content.sections.overview)} number="01" delay={0.1}>
        <p className="text-gray-300 text-lg leading-relaxed mb-10">
          {t(content.overview.description)}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {content.overview.phases.map((phase, idx) => (
            <PhaseCard key={idx} icon={phase.icon} title={t(phase.title)} description={t(phase.description)} />
          ))}
        </div>
      </Section>

      {/* Game Modes */}
      <Section title={t(content.sections.gameModes)} number="02" delay={0.2}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {content.gameModes.map((mode, idx) => (
            <GameModeCard
              key={idx}
              icon={mode.icon}
              title={t(mode.title)}
              features={mode.features.map((f) => t(f))}
              gradientColor={idx === 0 ? 'from-cyan-400 to-blue-500' : idx === 1 ? 'from-purple-400 to-pink-500' : 'from-orange-400 to-red-500'}
            />
          ))}
        </div>
      </Section>

      {/* Fish Encyclopedia */}
      <Section title={t(content.sections.fishEncyclopedia)} number="03" delay={0.3}>
        <p className="text-cyan-300/60 mb-10 text-lg">{t(content.fishEncyclopedia.subtitle)}</p>
        <div className="space-y-5">
          {content.fishEncyclopedia.difficulties.map((difficulty, idx) => (
            <DifficultyRow
              key={idx}
              level={difficulty.level}
              color={difficulty.color}
              fish={difficulty.fish[lang]}
              score={difficulty.score}
              lang={lang}
            />
          ))}
        </div>
        <p className="text-gray-500 text-sm mt-10 text-center">
          {t(content.fishEncyclopedia.depthNote)}
        </p>
      </Section>

      {/* Gesture System */}
      <Section title={t(content.sections.gestureSystem)} number="04" delay={0.4}>
        <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg p-8 hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all">
          <h3 className="text-2xl font-black tracking-wider text-cyan-400 mb-6">{t(content.gestureSystem.basicTitle)}</h3>
          <ul className="space-y-3 text-gray-300 text-lg">
            {content.gestureSystem.gestures.map((gesture, idx) => (
              <li key={idx} className="flex items-center gap-3">
                <span className="w-2 h-2 bg-cyan-400 rounded-full" />
                {t(gesture)}
              </li>
            ))}
          </ul>
        </div>
      </Section>

      {/* Combo System */}
      <Section title={t(content.sections.comboSystem)} number="05" delay={0.5}>
        <p className="text-cyan-300/60 mb-8 text-lg">{t(content.comboSystem.subtitle)}</p>
        <div className="space-y-4">
          {content.comboSystem.combos.map((combo, idx) => (
            <ComboCard key={idx} icon={combo.icon} name={t(combo.name)} effect={t(combo.effect)} />
          ))}
        </div>
        <div className="mt-10 p-8 bg-gradient-to-br from-purple-500/15 to-purple-900/10 backdrop-blur-md border border-purple-400/30 rounded-lg hover:border-purple-400/50 hover:shadow-2xl hover:shadow-purple-400/15 hover:-translate-y-2 transition-all">
          <div className="absolute -top-3 left-6 px-4 py-1 bg-purple-500 text-white text-xs font-black tracking-wider rounded-full">
            PASS EXCLUSIVE
          </div>
          <h3 className="text-2xl font-black tracking-wider text-purple-400 mb-4">{t(content.comboSystem.customCombo.title)}</h3>
          <ul className="text-gray-300 space-y-3">
            {content.comboSystem.customCombo.features.map((feature, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-purple-400 mt-1">✦</span>
                <span>{t(feature)}</span>
              </li>
            ))}
          </ul>
        </div>
      </Section>

      {/* Resistance System */}
      <Section title={t(content.sections.resistanceSystem)} number="06" delay={0.55}>
        <p className="text-gray-300 text-lg leading-relaxed mb-10">
          {t(content.resistanceSystem.description)}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {content.resistanceSystem.mechanics.map((mechanic, idx) => (
            <div
              key={idx}
              className={`group bg-gradient-to-br from-white/8 to-white/3 backdrop-blur-md border ${mechanic.color} rounded-lg p-7 hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all`}
            >
              <div className="w-16 h-16 rounded-full bg-cyan-400/10 flex items-center justify-center text-4xl mb-4">{mechanic.icon}</div>
              <h3 className="text-2xl font-black tracking-wider mb-3">{t(mechanic.title)}</h3>
              <p className="text-gray-400 leading-relaxed">{t(mechanic.description)}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Advanced Techniques */}
      <Section title={t(content.sections.advancedTechniques)} number="07" delay={0.58}>
        <p className="text-cyan-300/60 mb-8 text-lg">{t(content.advancedTechniques.subtitle)}</p>
        <div className="space-y-4">
          {content.advancedTechniques.techniques.map((technique, idx) => (
            <TechniqueCard
              key={idx}
              number={`0${idx + 1}`}
              icon={technique.icon}
              name={t(technique.name)}
              timing={t(technique.timing)}
              operation={t(technique.operation)}
              effect={t(technique.effect)}
              color={technique.color}
              lang={lang}
            />
          ))}
        </div>
      </Section>

      {/* Customization */}
      <Section title={t(content.sections.customization)} number="08" delay={0.6}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {content.customization.cards.map((card, idx) => (
            <CustomizationCard
              key={idx}
              title={t(card.title)}
              features={card.features.map((f) => t(f))}
            />
          ))}
        </div>
      </Section>

      {/* Pricing */}
      <Section title={t(content.sections.pricing)} number="09" delay={0.7}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {/* Free Plan */}
          <div className="bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-xl p-8 hover:-translate-y-2 transition-all">
            <h3 className="text-3xl font-black tracking-wider mb-6">{t(content.pricing.headers.free)}</h3>
            <div className="space-y-3 mb-8">
              {content.pricing.rows.map((row, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">{t(row.feature)}</span>
                  <span>{row.free ? '✅' : '❌'}</span>
                </div>
              ))}
            </div>
          </div>

          {/* PASS Plan */}
          <div className="relative bg-gradient-to-br from-cyan-500/15 to-blue-900/10 backdrop-blur-md border border-cyan-400/50 rounded-xl p-8 shadow-lg shadow-cyan-400/10 hover:-translate-y-2 transition-all">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-6 py-1 bg-gradient-to-r from-cyan-400 to-blue-500 text-white text-xs font-black tracking-widest rounded-full">
              RECOMMENDED
            </div>
            <h3 className="text-3xl font-black tracking-wider text-cyan-400 mb-2">{t(content.pricing.headers.pass)}</h3>
            <p className="text-cyan-300/60 text-sm mb-6">{t(content.pricing.passNote)}</p>
            <div className="space-y-3 mb-8">
              {content.pricing.rows.map((row, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">{t(row.feature)}</span>
                  <span>{row.pass ? '✅' : '❌'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* Privacy */}
      <Section title={t(content.sections.privacy)} number="10" delay={0.8}>
        <div className="bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg p-8 space-y-4 text-gray-300 hover:-translate-y-1 transition-all">
          {content.privacy.points.map((point, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <span className="text-green-400 mt-1">✓</span>
              <span>{t(point)}</span>
            </div>
          ))}
        </div>
        <p className="text-center text-gray-500 text-sm mt-8">
          {t(content.privacy.detailsText)}{' '}
          <a href="/privacy" className="text-cyan-400 hover:underline hover:text-cyan-300 transition-colors">
            {t(content.privacy.linkText)}
          </a>
          {lang === 'ja' && ` ${t(content.privacy.detailsTextEnd)}`}
        </p>
      </Section>

      {/* Other App - Full-width banner */}
      <section className="relative px-6 py-20">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-black tracking-wider mb-8 text-cyan-300/70 text-center">{t(content.sections.otherApps)}</h2>
          <a
            href="/apps/mouth-eat"
            className="block w-full p-8 bg-gradient-to-r from-cyan-900/20 to-blue-900/20 backdrop-blur-md border border-cyan-400/30 rounded-xl hover:border-cyan-400/60 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all"
          >
            <div className="flex items-center gap-6">
              <div className="text-7xl">😋</div>
              <div className="flex-1">
                <div className="font-black text-white text-3xl tracking-wider mb-2">{content.otherApp.name}</div>
                <div className="text-lg text-gray-400">{t(content.otherApp.description)}</div>
              </div>
              <div className="text-cyan-400 text-2xl">→</div>
            </div>
          </a>
        </div>
      </section>

      <Footer />

      {/* CSS for floating animation */}
      <style jsx>{`
        @keyframes float {
          0%, 100% {
            transform: translateY(0) translateX(0);
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          90% {
            opacity: 1;
          }
          100% {
            transform: translateY(-100vh) translateX(20px);
            opacity: 0;
          }
        }
        .animate-float {
          animation: float linear infinite;
        }
      `}</style>
    </main>
  );
}

function Section({ title, number, delay, children }: { title: string; number: string; delay: number; children: React.ReactNode }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.8, delay }}
      className="relative px-6 py-20"
    >
      <div className="max-w-5xl mx-auto">
        <div className="relative mb-16">
          <div className="text-8xl font-black text-white/[0.03] absolute -top-8 -left-4 select-none">{number}</div>
          <div className="relative z-10">
            <div className="flex items-center gap-5 mb-3">
              <div className="h-px flex-1" style={{ background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.15))' }} />
              <h2 className="text-3xl md:text-4xl font-black tracking-wider text-white">{title}</h2>
              <div className="h-px flex-1" style={{ background: 'linear-gradient(90deg, rgba(14,165,233,0.15), transparent)' }} />
            </div>
          </div>
        </div>
        {children}
      </div>
    </motion.section>
  );
}

function PhaseCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg p-6 text-center hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all">
      <div className="w-16 h-16 rounded-full bg-cyan-400/10 flex items-center justify-center text-4xl mx-auto mb-4">{icon}</div>
      <h3 className="font-black text-white mb-2 text-lg tracking-wide">{title}</h3>
      <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
    </div>
  );
}

function GameModeCard({ icon, title, features, gradientColor }: { icon: string; title: string; features: string[]; gradientColor: string }) {
  return (
    <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg overflow-hidden hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all">
      <div className={`h-1 bg-gradient-to-r ${gradientColor}`} />
      <div className="p-7">
        <div className="w-16 h-16 rounded-full bg-cyan-400/10 flex items-center justify-center text-3xl mb-5">{icon}</div>
        <h3 className="text-2xl font-black text-white mb-5 tracking-wide">{title}</h3>
        <ul className="space-y-3 text-gray-300">
          {features.map((feature, idx) => (
            <li key={idx} className="flex items-start gap-3">
              <span className="text-cyan-400 mt-1">•</span>
              <span className="leading-relaxed">{feature}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function DifficultyRow({ level, color, fish, score, lang }: { level: string; color: string; fish: string[]; score: string; lang: 'ja' | 'en' }) {
  return (
    <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg p-6 hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-1 transition-all">
      <div className="flex flex-col md:flex-row md:items-center gap-5">
        <div className={`font-bold px-5 py-2 rounded-full border inline-block ${color} min-w-[140px] text-center shadow-sm`}>
          {level}
        </div>
        <div className="flex-1 flex flex-wrap gap-2">
          {fish.map((f, idx) => (
            <span key={idx} className="px-3 py-1 bg-white/5 border border-white/10 rounded-full text-sm text-gray-300">
              {f}
            </span>
          ))}
        </div>
        <div className="text-cyan-400 text-sm min-w-[140px] text-right font-mono flex items-center justify-end gap-2">
          <span>💰</span>
          <span>{score}</span>
        </div>
      </div>
    </div>
  );
}

function ComboCard({ icon, name, effect }: { icon: string; name: string; effect: string }) {
  return (
    <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg overflow-hidden hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-1 transition-all">
      <div className="w-1 h-full bg-cyan-400 absolute left-0" />
      <div className="pl-6 pr-5 py-5 flex items-start gap-5">
        <div className="w-14 h-14 rounded-full bg-cyan-400/10 flex items-center justify-center text-3xl flex-shrink-0">{icon}</div>
        <div className="flex-1">
          <h3 className="font-black text-white mb-2 text-lg tracking-wide">{name}</h3>
          <p className="text-gray-400 text-sm leading-relaxed">{effect}</p>
        </div>
      </div>
    </div>
  );
}

function CustomizationCard({ title, features }: { title: string; features: string[] }) {
  return (
    <div className="group bg-gradient-to-br from-white/[0.04] to-white/[0.01] border border-white/[0.06] rounded-lg p-8 hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-2 transition-all">
      <h3 className="text-2xl font-black text-white mb-6 tracking-wide">{title}</h3>
      <ul className="space-y-4 text-gray-300">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start gap-3">
            <span className="text-cyan-400 mt-1">✦</span>
            <span className="leading-relaxed">{feature}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TechniqueCard({ number, icon, name, timing, operation, effect, color, lang }: {
  number: string; icon: string; name: string; timing: string; operation: string; effect: string; color: string; lang: 'ja' | 'en';
}) {
  return (
    <div className={`group bg-gradient-to-br from-white/8 to-white/3 backdrop-blur-md border ${color} rounded-lg p-6 hover:border-cyan-400/40 hover:shadow-2xl hover:shadow-cyan-400/15 hover:-translate-y-1 transition-all`}>
      <div className="flex items-start gap-6">
        <div className="text-6xl font-black text-white/5 leading-none">{number}</div>
        <div className="w-14 h-14 rounded-full bg-cyan-400/10 flex items-center justify-center text-3xl flex-shrink-0">{icon}</div>
        <div className="flex-1">
          <h3 className="font-black text-white mb-4 text-xl tracking-wide">{name}</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">⏰</span>
                <span className="text-gray-500 font-bold uppercase tracking-wider text-xs">{lang === 'ja' ? 'タイミング' : 'Timing'}</span>
              </div>
              <span className="text-gray-300">{timing}</span>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🎮</span>
                <span className="text-gray-500 font-bold uppercase tracking-wider text-xs">{lang === 'ja' ? '操作' : 'Action'}</span>
              </div>
              <span className="text-cyan-400 font-semibold">{operation}</span>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">✨</span>
                <span className="text-gray-500 font-bold uppercase tracking-wider text-xs">{lang === 'ja' ? '効果' : 'Effect'}</span>
              </div>
              <span className="text-green-400 font-semibold">{effect}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
