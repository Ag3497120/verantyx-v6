'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { PageHero, CinematicSection, GlassCard, FeatureCheck } from '@/components/CinematicSection';

type Lang = 'ja' | 'en';
type T = { ja: string; en: string };
const t = (obj: T, lang: Lang) => obj[lang];

const content = {
  hero: {
    title: { ja: 'サポート', en: 'Support' },
    subtitle: { ja: 'よくある質問・お問い合わせ', en: 'FAQ & Contact' },
  },
  faq: {
    title: { ja: 'よくある質問', en: 'Frequently Asked Questions' },
    items: [
      {
        q: { ja: 'Q: どのiPhoneで使えますか？', en: 'Q: Which iPhones are supported?' },
        a: { ja: 'A: TrueDepthカメラ搭載のiPhone（iPhone X以降）、iOS 17.0以上が必要です。', en: 'A: iPhone X or later with a TrueDepth camera and iOS 17.0+ is required.' },
      },
      {
        q: { ja: 'Q: カメラの映像は保存されますか？', en: 'Q: Is camera footage saved?' },
        a: { ja: 'A: いいえ。カメラ映像はデバイス上でリアルタイム処理のみ行われ、保存・送信されることは一切ありません。', en: 'A: No. Camera footage is processed in real time on-device only and is never saved or transmitted.' },
      },
      {
        q: { ja: 'Q: PASSを購入したら広告は消えますか？', en: 'Q: Will ads disappear after purchasing PASS?' },
        a: { ja: 'A: はい。PASS購入後は全ての広告が非表示になります。', en: 'A: Yes. All ads are removed after purchasing PASS.' },
      },
      {
        q: { ja: 'Q: PASSの購入はサブスクリプションですか？', en: 'Q: Is PASS a subscription?' },
        a: { ja: 'A: いいえ。¥250の買い切りです。一度購入すれば永久にご利用いただけます。', en: 'A: No. It is a one-time purchase of ¥250. Once purchased, you have lifetime access.' },
      },
      {
        q: { ja: 'Q: 購入を復元するには？', en: 'Q: How do I restore my purchase?' },
        a: { ja: 'A: アプリ内の設定画面から「購入を復元」をタップしてください。', en: 'A: Tap "Restore Purchase" in the in-app settings screen.' },
      },
      {
        q: { ja: 'Q: 口の検出がうまくいきません', en: 'Q: Mouth detection is not working well' },
        a: { ja: 'A: 明るい場所で、カメラに顔全体が映るように端末を持ってください。キャリブレーション機能で感度を調整することもできます。', en: 'A: Make sure you are in a well-lit area with your face fully visible to the camera. You can also adjust sensitivity using the calibration feature.' },
      },
    ],
  },
  contact: {
    title: { ja: 'お問い合わせ', en: 'Contact' },
    description: {
      ja: 'バグの報告・機能のご要望は、各アプリのGitHub Issuesからお送りください。',
      en: 'For bug reports and feature requests, please submit via GitHub Issues for the relevant app.',
    },
    apps: [
      {
        icon: '🎣',
        name: 'パクパク釣り / Paku Paku Fishing',
        issueUrl: 'https://github.com/Ag3497120/paku-paku-fishing/issues',
        issueLabel: { ja: 'Issueを作成', en: 'Open an Issue' },
      },
      {
        icon: '😋',
        name: 'MouthEat',
        issueUrl: 'https://github.com/Ag3497120/paku-paku-eating/issues',
        issueLabel: { ja: 'Issueを作成', en: 'Open an Issue' },
      },
    ],
  },
  sysReq: {
    title: { ja: '動作環境', en: 'System Requirements' },
    items: [
      { ja: 'iOS 17.0以上', en: 'iOS 17.0 or later' },
      { ja: 'iPhone X 以降（TrueDepthカメラ搭載機種）', en: 'iPhone X or later (TrueDepth camera required)' },
      { ja: '約100 MBのストレージ空間', en: 'Approx. 100 MB of storage space' },
    ],
  },
  apps: {
    title: { ja: 'アプリ', en: 'Apps' },
    fishing: { ja: '口で釣る、新感覚フィッシングゲーム', en: 'A revolutionary mouth-controlled fishing game' },
    eating: { ja: '口を開けて食べまくれ！', en: 'Open wide and eat everything!' },
  },
};

export default function SupportPage() {
  const [lang, setLang] = useState<Lang>('ja');

  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      {/* Language Toggle */}
      <div className="fixed top-20 right-6 z-50 backdrop-blur-md rounded-full px-1 py-1 flex gap-1" style={{ background: 'rgba(5,5,8,0.7)', border: '1px solid rgba(14,165,233,0.1)' }}>
        <button onClick={() => setLang('ja')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'ja' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'ja' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>JP</button>
        <button onClick={() => setLang('en')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'en' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'en' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>EN</button>
      </div>

      <PageHero
        title={t(content.hero.title, lang)}
        subtitle={t(content.hero.subtitle, lang)}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />

      {/* FAQ */}
      <CinematicSection title={t(content.faq.title, lang)}>
        <div className="space-y-4">
          {content.faq.items.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 15, filter: 'blur(4px)' }}
              whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: idx * 0.05 }}
            >
              <GlassCard>
                <h3 className="text-base font-semibold text-white mb-2">{t(item.q, lang)}</h3>
                <p className="text-gray-400 leading-relaxed text-sm">{t(item.a, lang)}</p>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      </CinematicSection>

      {/* Contact */}
      <CinematicSection title={t(content.contact.title, lang)}>
        <p className="text-gray-500 mb-8 leading-relaxed">{t(content.contact.description, lang)}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {content.contact.apps.map((app, idx) => (
            <GlassCard key={idx}>
              <div className="flex items-center gap-3 mb-5">
                <span className="text-4xl">{app.icon}</span>
                <span className="font-semibold text-white text-sm">{app.name}</span>
              </div>
              <a
                href={app.issueUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-300"
                style={{
                  background: 'rgba(14, 165, 233, 0.06)',
                  border: '1px solid rgba(14, 165, 233, 0.2)',
                  color: 'rgba(14, 165, 233, 0.8)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.5)';
                  e.currentTarget.style.boxShadow = '0 0 20px rgba(14,165,233,0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.2)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                {t(app.issueLabel, lang)}
              </a>
            </GlassCard>
          ))}
        </div>
      </CinematicSection>

      {/* System Requirements */}
      <CinematicSection title={t(content.sysReq.title, lang)}>
        <GlassCard>
          <div className="space-y-3">
            {content.sysReq.items.map((item, idx) => (
              <FeatureCheck key={idx}>{t(item, lang)}</FeatureCheck>
            ))}
          </div>
        </GlassCard>
      </CinematicSection>

      {/* App Links */}
      <CinematicSection title={t(content.apps.title, lang)}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <motion.a
            href="/apps/pakupaku-fishing"
            whileHover={{ y: -3 }}
            className="flex items-center gap-4 transition-all duration-300"
            style={{ textDecoration: 'none' }}
          >
            <GlassCard className="w-full">
              <div className="flex items-center gap-4">
                <span className="text-5xl">🎣</span>
                <div>
                  <div className="font-semibold text-white">{lang === 'ja' ? 'パクパク釣り' : 'Paku Paku Fishing'}</div>
                  <div className="text-sm text-gray-500">{t(content.apps.fishing, lang)}</div>
                </div>
              </div>
            </GlassCard>
          </motion.a>
          <motion.a
            href="/apps/mouth-eat"
            whileHover={{ y: -3 }}
            className="flex items-center gap-4 transition-all duration-300"
            style={{ textDecoration: 'none' }}
          >
            <GlassCard className="w-full">
              <div className="flex items-center gap-4">
                <span className="text-5xl">😋</span>
                <div>
                  <div className="font-semibold text-white">MouthEat</div>
                  <div className="text-sm text-gray-500">{t(content.apps.eating, lang)}</div>
                </div>
              </div>
            </GlassCard>
          </motion.a>
        </div>
      </CinematicSection>

      <Footer />
    </main>
  );
}
