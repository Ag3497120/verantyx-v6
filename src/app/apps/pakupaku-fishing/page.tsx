'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';


export default function PakuPakuFishingPage() {
  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      {/* Hero Section */}
      <section className="relative min-h-[50vh] flex items-center justify-center px-6 pt-32 pb-16">
        <div className="max-w-5xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="text-8xl mb-6"
          >
            🎣
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-7xl font-bold mb-4"
            style={{
              background: 'linear-gradient(135deg, #06B6D4, #0EA5E9)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            パクパク釣り
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-xl md:text-2xl text-gray-400 mb-2"
          >
            口で釣る、新感覚フィッシングゲーム
          </motion.p>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-lg text-gray-500 mb-8"
          >
            Catch fish using only your mouth — a revolutionary fishing experience
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            <a href="#" className="inline-block">
              <img
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='48' viewBox='0 0 140 48'%3E%3Cg fill='none'%3E%3Crect width='140' height='48' rx='8' fill='%23000'/%3E%3Ctext x='70' y='28' font-family='Arial' font-size='12' fill='%23FFF' text-anchor='middle' font-weight='600'%3EDownload on the%3C/text%3E%3Ctext x='70' y='40' font-family='Arial' font-size='16' fill='%23FFF' text-anchor='middle' font-weight='700'%3EApp Store%3C/text%3E%3C/g%3E%3C/svg%3E"
                alt="Download on the App Store"
                width="140"
                height="48"
              />
            </a>
          </motion.div>
        </div>
      </section>

      {/* Game Overview */}
      <Section title="ゲーム概要" delay={0.1}>
        <p className="text-gray-300 text-lg leading-relaxed mb-8">
          顔の動きで釣りを楽しむ革新的なiOSゲーム。デバイスのカメラで口の動きと顔のジェスチャーをリアルタイム検出し、バーチャルな釣りを体験できます。
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <PhaseCard icon="🎯" title="観察" description="池を見渡し、キャスティングポイントを決める" />
          <PhaseCard icon="🎣" title="キャスティング" description="タップして釣り糸を投げる" />
          <PhaseCard icon="⏳" title="待機" description="口を閉じてじっと待つ" />
          <PhaseCard icon="🐟" title="アタリ" description="魚が食いつく" />
          <PhaseCard icon="⚡" title="フッキング" description="タイミングよく口を閉じて針を合わせる" />
          <PhaseCard icon="🔄" title="リーリング" description="口を素早く開閉して魚を巻き上げる。魚の抵抗に注意！" />
        </div>
      </Section>

      {/* Game Modes */}
      <Section title="ゲームモード" delay={0.2}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <GameModeCard
            icon="🎣"
            title="フリーモード"
            features={[
              '無制限に釣りを楽しめる',
              'スコア蓄積 & コンボシステム',
              '深度ゾーンによる魚の分布',
            ]}
          />
          <GameModeCard
            icon="⏱️"
            title="タイムアタック"
            features={[
              '60秒の時間制限チャレンジ',
              '3-2-1 カウントダウンでスタート',
              'ランキング対応',
            ]}
          />
          <GameModeCard
            icon="🏆"
            title="一本釣りチャレンジ"
            features={[
              '巨大な一匹を狙う',
              '持続的な集中力が必要',
              '5〜120秒以上の長時間ファイト',
              '伝説級の魚との真剣勝負',
            ]}
          />
        </div>
      </Section>

      {/* Fish Encyclopedia */}
      <Section title="魚図鑑" delay={0.3}>
        <p className="text-gray-400 mb-8">20種の魚 × 4段階の難易度</p>
        <div className="space-y-6">
          <DifficultyRow
            level="Easy"
            color="text-green-400"
            fish={['メダカ', 'ハゼ', 'イワシ', 'フナ', 'アユ']}
            score="50〜150pt"
          />
          <DifficultyRow
            level="Medium"
            color="text-yellow-400"
            fish={['タイ', 'サバ', 'イカ', 'ヒラメ', 'ブリ']}
            score="200〜400pt"
          />
          <DifficultyRow
            level="Hard"
            color="text-orange-400"
            fish={['マグロ', 'カツオ', 'タコ', 'スズキ', 'フグ']}
            score="450〜800pt"
          />
          <DifficultyRow
            level="Legendary"
            color="text-purple-400"
            fish={['クジラ', 'サメ', 'ダイオウイカ', 'シーラカンス', '金の鯉']}
            score="1,000〜5,000pt"
          />
        </div>
        <p className="text-gray-500 text-sm mt-8">
          深さ4ゾーン（浅瀬・中層・深海・深淵）で出現分布が変化
        </p>
      </Section>

      {/* Gesture System */}
      <Section title="顔ジェスチャーシステム" delay={0.4}>
        <div>
          <h3 className="text-xl font-bold text-cyan-400 mb-4">基本（無料）</h3>
          <ul className="space-y-2 text-gray-300">
            <li>✓ 口を開ける</li>
            <li>✓ 口を閉じる</li>
          </ul>
        </div>
      </Section>

      {/* Combo System */}
      <Section title="コンボシステム" delay={0.5}>
        <p className="text-gray-400 mb-6">5つのビルトインコンボ（全ユーザー利用可能）</p>
        <div className="space-y-4">
          <ComboCard icon="💥" name="ラピッドマウスフラップ" effect="超高速口パクパクで1.5倍スコア" />
          <ComboCard icon="🔥" name="パワーリフト" effect="上を向いてホールド+口閉じで2倍ダメージ" />
          <ComboCard icon="🎯" name="ダイブキャスト" effect="下を向いて口を開けると1.5倍射程" />
          <ComboCard icon="🌊" name="ドッジカウンター" effect="左右に首振りで1.5倍ダメージ+魚フリーズ" />
          <ComboCard icon="🧘" name="フォーカスフィッシング" effect="眉上げ3秒キープでレア魚70%出現" />
        </div>
        <div className="mt-8 p-6 bg-purple-900/20 border border-purple-500/30 rounded-lg">
          <h3 className="text-xl font-bold text-purple-400 mb-2">カスタムコンボ作成（PASS限定）</h3>
          <ul className="text-gray-300 space-y-2">
            <li>• 最大5個のオリジナルコンボを作成可能</li>
            <li>• ジェスチャーの組み合わせ・エフェクト・クールダウンを自由にカスタマイズ</li>
          </ul>
        </div>
      </Section>

      {/* Resistance System */}
      <Section title="魚の抵抗システム" delay={0.55}>
        <p className="text-gray-300 text-lg leading-relaxed mb-8">
          大物ほど激しく抵抗する！リール中に魚が暴れてリールが引き戻されます。レア度が高い魚ほど抵抗力が強く、戦略的な対処が求められます。
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-900/50 border border-orange-500/30 rounded-lg p-6">
            <div className="text-4xl mb-3">💪</div>
            <h3 className="text-xl font-bold text-orange-400 mb-2">引き戻し</h3>
            <p className="text-gray-400">魚は常にリールを引き戻す力を加えています。深い場所ほど、レアな魚ほど引き戻しが強くなります。パクパクを止めるとリールが戻ってしまいます。</p>
          </div>
          <div className="bg-gray-900/50 border border-red-500/30 rounded-lg p-6">
            <div className="text-4xl mb-3">⚡</div>
            <h3 className="text-xl font-bold text-red-400 mb-2">突発的な抵抗</h3>
            <p className="text-gray-400">リール中にランダムで魚が激しく暴れます。通常の3倍の速度でリールが引き戻され、画面が揺れて警告が表示されます。</p>
          </div>
        </div>
      </Section>

      {/* Advanced Techniques */}
      <Section title="応用テクニック" delay={0.58}>
        <p className="text-gray-400 mb-6">大物を釣り上げるためのコンボ技を使いこなそう</p>
        <div className="space-y-4">
          <TechniqueCard
            icon="🌊"
            name="いなし（ドッジカウンター）"
            timing="魚が抵抗した時"
            operation="首を左 → 右に振る"
            effect="魚が2秒間フリーズし、リールの引き戻しが止まる"
            color="border-cyan-500/30"
          />
          <TechniqueCard
            icon="🔥"
            name="力強い引き上げ（パワーリフト）"
            timing="リーリング中"
            operation="上を向いて0.5秒ホールド → 口を閉じる"
            effect="ダメージ2倍で一気にリールを巻き上げ"
            color="border-red-500/30"
          />
          <TechniqueCard
            icon="🎯"
            name="ダイブ投げ（ダイブキャスト）"
            timing="観察フェーズ"
            operation="下を向いて口を開ける"
            effect="キャスト飛距離1.5倍で深い場所に届く"
            color="border-blue-500/30"
          />
          <TechniqueCard
            icon="🧘"
            name="集中釣り（フォーカスフィッシング）"
            timing="観察フェーズ"
            operation="眉を上げて3秒キープ"
            effect="レア魚の出現率が70%アップ"
            color="border-purple-500/30"
          />
          <TechniqueCard
            icon="💥"
            name="連続口パク（ラピッドマウスフラップ）"
            timing="いつでも"
            operation="口を素早く5回パクパク"
            effect="スコア1.5倍ボーナス"
            color="border-yellow-500/30"
          />
        </div>
      </Section>

      {/* Customization */}
      <Section title="カスタマイズ" delay={0.6}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <CustomizationCard
            title="カスタム魚パック"
            features={[
              'オリジナルの魚を作成（写真 or 絵文字）',
              '難易度・名前・著作権情報を設定',
              'ディープリンクで友達と共有可能',
            ]}
          />
          <CustomizationCard
            title="池キャプチャ（PASS限定）"
            features={[
              'カメラで実際の水辺を撮影',
              '自分だけの釣り場を作成',
            ]}
          />
        </div>
      </Section>

      {/* Pricing */}
      <Section title="料金プラン" delay={0.7}>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">機能</th>
                <th className="text-center py-4 px-4 text-cyan-400 font-semibold">Free</th>
                <th className="text-center py-4 px-4 text-purple-400 font-semibold">PASS (¥250)</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <PricingRow feature="基本ゲームプレイ" free={true} pass={true} />
              <PricingRow feature="ビルトインコンボ" free={true} pass={true} />
              <PricingRow feature="広告なし" free={false} pass={true} />
              <PricingRow feature="感度調整" free={false} pass={true} />
              <PricingRow feature="カスタムコンボ作成" free={false} pass={true} />
              <PricingRow feature="高度なジェスチャー（11種）" free={false} pass={true} />
              <PricingRow feature="カスタム魚パック" free={false} pass={true} />
              <PricingRow feature="池キャプチャ" free={false} pass={true} />
            </tbody>
          </table>
        </div>
        <p className="text-center text-gray-500 text-sm mt-6">
          PASS = ¥250の買い切り（サブスクリプションではない）
        </p>
      </Section>

      {/* Privacy */}
      <Section title="プライバシー" delay={0.8}>
        <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6 space-y-4 text-gray-300">
          <p>✓ カメラ映像はデバイス上でのみ処理</p>
          <p>✓ 外部サーバーへの映像送信なし</p>
          <p>✓ 顔データの保存・収集なし</p>
          <p>✓ 広告: Google AdMob（Free版のみ）</p>
        </div>
        <p className="text-center text-gray-500 text-sm mt-6">
          詳細は<a href="/privacy" className="text-cyan-400 hover:underline">プライバシーポリシー</a>をご覧ください
        </p>
      </Section>

      {/* Other App */}
      <section className="relative px-6 py-16">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-2xl font-bold mb-4 text-gray-400">他のアプリもチェック</h2>
          <a
            href="/apps/mouth-eat"
            className="inline-flex items-center gap-3 px-6 py-4 bg-gray-900/50 border border-cyan-500/30 rounded-lg hover:border-cyan-500/60 transition-colors"
          >
            <span className="text-4xl">😋</span>
            <div className="text-left">
              <div className="font-bold text-white">MouthEat</div>
              <div className="text-sm text-gray-400">口を開けて食べまくれ！</div>
            </div>
          </a>
        </div>
      </section>

      <Footer />
    </main>
  );
}

function Section({ title, delay, children }: { title: string; delay: number; children: React.ReactNode }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.8, delay }}
      className="relative px-6 py-16"
    >
      <div className="max-w-4xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-8 text-cyan-400">{title}</h2>
        {children}
      </div>
    </motion.section>
  );
}

function PhaseCard({ icon, title, description }: { icon: string; title: string; description: string }) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 text-center">
      <div className="text-3xl mb-2">{icon}</div>
      <h3 className="font-bold text-white mb-1">{title}</h3>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );
}

function GameModeCard({ icon, title, features }: { icon: string; title: string; features: string[] }) {
  return (
    <div className="bg-gray-900/50 border border-cyan-500/20 rounded-lg p-6">
      <div className="text-5xl mb-4">{icon}</div>
      <h3 className="text-2xl font-bold text-white mb-4">{title}</h3>
      <ul className="space-y-2 text-gray-300">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start gap-2">
            <span className="text-cyan-400 mt-1">•</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DifficultyRow({ level, color, fish, score }: { level: string; color: string; fish: string[]; score: string }) {
  return (
    <div className="bg-gray-900/30 border border-gray-800 rounded-lg p-4">
      <div className="flex flex-col md:flex-row md:items-center gap-4">
        <div className={`font-bold ${color} min-w-[100px]`}>{level}</div>
        <div className="flex-1 text-gray-300">{fish.join('、')}</div>
        <div className="text-gray-400 text-sm min-w-[120px]">{score}</div>
      </div>
    </div>
  );
}

function ComboCard({ icon, name, effect }: { icon: string; name: string; effect: string }) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 flex items-start gap-4">
      <div className="text-3xl">{icon}</div>
      <div>
        <h3 className="font-bold text-white mb-1">{name}</h3>
        <p className="text-gray-400 text-sm">{effect}</p>
      </div>
    </div>
  );
}

function CustomizationCard({ title, features }: { title: string; features: string[] }) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-bold text-white mb-4">{title}</h3>
      <ul className="space-y-2 text-gray-300">
        {features.map((feature, idx) => (
          <li key={idx} className="flex items-start gap-2">
            <span className="text-cyan-400">•</span>
            <span>{feature}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function PricingRow({ feature, free, pass }: { feature: string; free: boolean; pass: boolean }) {
  return (
    <tr className="border-b border-gray-800">
      <td className="py-3 px-4">{feature}</td>
      <td className="py-3 px-4 text-center">{free ? '✅' : '❌'}</td>
      <td className="py-3 px-4 text-center">{pass ? '✅' : '❌'}</td>
    </tr>
  );
}

function TechniqueCard({ icon, name, timing, operation, effect, color }: {
  icon: string; name: string; timing: string; operation: string; effect: string; color: string;
}) {
  return (
    <div className={`bg-gray-900/50 border ${color} rounded-lg p-5`}>
      <div className="flex items-start gap-4">
        <div className="text-3xl">{icon}</div>
        <div className="flex-1">
          <h3 className="font-bold text-white mb-2">{name}</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
            <div>
              <span className="text-gray-500">タイミング:</span>
              <span className="text-gray-300 ml-1">{timing}</span>
            </div>
            <div>
              <span className="text-gray-500">操作:</span>
              <span className="text-cyan-400 ml-1">{operation}</span>
            </div>
            <div>
              <span className="text-gray-500">効果:</span>
              <span className="text-green-400 ml-1">{effect}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
