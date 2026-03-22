'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { PageHero, CinematicSection, GlassCard, FeatureCheck } from '@/components/CinematicSection';

type Lang = 'ja' | 'en';

export default function PrivacyPage() {
  const [lang, setLang] = useState<Lang>('ja');

  const sections = [
    {
      id: 'intro',
      title: { ja: 'はじめに', en: 'Introduction' },
      content: (
        <p className="text-gray-400 leading-[2em]">
          {lang === 'ja'
            ? 'Verantyx（以下「当社」）が提供するiOSアプリケーション（パクパク釣り、MouthEat）に関するプライバシーポリシーです。本ポリシーは、当社がどのような情報を収集し、どのように利用するかを説明します。'
            : 'This is the Privacy Policy for iOS applications (Paku Paku Fishing, MouthEat) provided by Verantyx ("we"). This policy explains what information we collect and how we use it.'}
        </p>
      ),
    },
    {
      id: 'data',
      title: { ja: '収集するデータ', en: 'Data We Collect' },
      content: (
        <div className="space-y-8">
          <div>
            <h3 className="text-lg font-semibold mb-3" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>
              {lang === 'ja' ? 'カメラ映像' : 'Camera'}
            </h3>
            <div className="space-y-2.5">
              {(lang === 'ja'
                ? ['顔の動き・口の動きを検出するために使用', '映像データはすべてデバイス上でリアルタイム処理', 'サーバーへの送信・保存は一切行いません', 'ARKit（ARFaceAnchor）を使用した顔トラッキング']
                : ['Used to detect face and mouth movements', 'All footage is processed in real time on-device only', 'Never transmitted to or stored on any server', 'Face tracking via ARKit (ARFaceAnchor)']
              ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>
              {lang === 'ja' ? 'フォトライブラリ' : 'Photo Library'}
            </h3>
            <div className="space-y-2.5">
              {(lang === 'ja'
                ? ['MouthEat: ゲームプレイ録画クリップの保存', 'パクパク釣り: カスタム魚パックの画像追加、池キャプチャ', 'ユーザーの明示的な許可を得た場合のみアクセス']
                : ['MouthEat: Saving gameplay recording clips', 'Paku Paku Fishing: Adding images for custom fish packs, pond capture', 'Accessed only with explicit user permission']
              ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>
              {lang === 'ja' ? 'スコアデータ' : 'Score Data'}
            </h3>
            <div className="space-y-2.5">
              {(lang === 'ja'
                ? ['ランキング機能のためにスコア・ユーザー名をサーバー送信', '個人を特定する情報は含まれません']
                : ['Score and username sent to server for leaderboard functionality', 'No personally identifiable information is included']
              ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 'ads',
      title: { ja: '広告について', en: 'Advertising' },
      content: (
        <GlassCard>
          <h3 className="text-lg font-semibold mb-4" style={{ color: 'rgba(250, 204, 21, 0.8)' }}>
            {lang === 'ja' ? 'Google AdMob（無料版のみ）' : 'Google AdMob (Free version only)'}
          </h3>
          <div className="space-y-2.5">
            {(lang === 'ja'
              ? ['Google Mobile Adsを利用してバナー広告・インタースティシャル広告を表示', 'AdMobは広告配信のために広告識別子（IDFA）を使用する場合があります', 'PASS購入後は広告は表示されません']
              : ['Banner and interstitial ads displayed via Google Mobile Ads', 'AdMob may use advertising identifiers (IDFA) for ad delivery', 'Ads are not shown after purchasing PASS']
            ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
            <div className="flex items-start gap-3 text-gray-400 mt-1">
              <span className="mt-0.5" style={{ color: 'rgba(14, 165, 233, 0.6)' }}>✓</span>
              <span className="leading-relaxed">
                {lang === 'ja' ? '詳細: ' : 'Details: '}
                <a href="https://policies.google.com/privacy" target="_blank" rel="noopener noreferrer" className="transition-colors duration-300" style={{ color: 'rgba(14, 165, 233, 0.6)' }} onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.9)'} onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.6)'}>
                  policies.google.com/privacy
                </a>
              </span>
            </div>
          </div>
        </GlassCard>
      ),
    },
    {
      id: 'iap',
      title: { ja: 'アプリ内課金', en: 'In-App Purchases' },
      content: (
        <div className="space-y-2.5">
          {(lang === 'ja'
            ? ['Apple StoreKit 2 を使用', '決済はAppleが処理し、当社はクレジットカード情報等を一切保持しません']
            : ['Uses Apple StoreKit 2', 'Payments are processed by Apple; we never store credit card or payment information']
          ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
        </div>
      ),
    },
    {
      id: 'storage',
      title: { ja: 'データの保存', en: 'Data Storage' },
      content: (
        <div className="space-y-2.5">
          {(lang === 'ja'
            ? ['ゲーム設定・進行状況はデバイスローカル（UserDefaults）に保存', 'iCloudやクラウドサービスへの自動同期はありません']
            : ['Game settings and progress are stored locally on-device (UserDefaults)', 'No automatic sync to iCloud or any cloud service']
          ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
        </div>
      ),
    },
    {
      id: 'thirdparty',
      title: { ja: '第三者への提供', en: 'Third-Party Sharing' },
      content: (
        <div className="space-y-2.5">
          {(lang === 'ja'
            ? ['Google AdMob以外の第三者へのデータ提供はありません', 'Apple App Storeの規約に準拠']
            : ['No data is shared with any third party other than Google AdMob', 'Compliant with Apple App Store guidelines']
          ).map((item, i) => <FeatureCheck key={i}>{item}</FeatureCheck>)}
        </div>
      ),
    },
    {
      id: 'children',
      title: { ja: '児童のプライバシー', en: "Children's Privacy" },
      content: (
        <p className="text-gray-400 leading-[2em]">
          {lang === 'ja'
            ? '13歳未満の児童から意図的に個人情報を収集することはありません。'
            : 'We do not knowingly collect personal information from children under the age of 13.'}
        </p>
      ),
    },
    {
      id: 'contact',
      title: { ja: 'お問い合わせ', en: 'Contact' },
      content: (
        <p className="text-gray-400">
          {lang === 'ja' ? 'ウェブサイト: ' : 'Website: '}
          <a href="/support" className="transition-colors duration-300" style={{ color: 'rgba(14, 165, 233, 0.6)' }} onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.9)'} onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.6)'}>
            https://verantyx.ai/support
          </a>
        </p>
      ),
    },
    {
      id: 'updates',
      title: { ja: '改定', en: 'Updates' },
      content: (
        <>
          <p className="text-gray-400 mb-4 leading-[2em]">
            {lang === 'ja'
              ? '本ポリシーは予告なく変更される場合があります。'
              : 'This policy may be updated without prior notice.'}
          </p>
          <p className="text-gray-600 text-xs tracking-wide">
            {lang === 'ja' ? '最終更新日: 2025年3月' : 'Last updated: March 2025'}
          </p>
        </>
      ),
    },
  ];

  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      {/* Language Toggle */}
      <div className="fixed top-20 right-6 z-50 backdrop-blur-md rounded-full px-1 py-1 flex gap-1" style={{ background: 'rgba(5,5,8,0.7)', border: '1px solid rgba(14,165,233,0.1)' }}>
        <button onClick={() => setLang('ja')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'ja' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'ja' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>JP</button>
        <button onClick={() => setLang('en')} className={`px-3 py-1 text-xs font-semibold rounded-full transition-all ${lang === 'en' ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`} style={lang === 'en' ? { background: 'rgba(14,165,233,0.2)', border: '1px solid rgba(14,165,233,0.3)' } : {}}>EN</button>
      </div>

      <PageHero
        title={lang === 'ja' ? 'プライバシーポリシー' : 'Privacy Policy'}
        subtitle={lang === 'ja' ? 'Privacy Policy' : 'プライバシーポリシー'}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />

      {sections.map((section, idx) => (
        <CinematicSection key={`${section.id}-${lang}`} title={section.title[lang]}>
          {section.content}
        </CinematicSection>
      ))}

      <Footer />
    </main>
  );
}
