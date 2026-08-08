'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { PageHero, CinematicSection, GlassCard } from '@/components/CinematicSection';
import { useLanguage } from '@/lib/i18n';

type L = { ja: string; en: string };

/* The Japanese is the operative text and the English is a reading aid, said
 * plainly at the bottom of the page rather than left for someone to discover
 * during a dispute. Article 9 already puts this under Japanese law; a
 * translation that quietly presented itself as equally binding would be
 * making a promise the governing-law clause does not. */
const ARTICLES: { title: L; intro?: L; items?: L[]; paras?: L[] }[] = [
  {
    title: { ja: '第1条（適用範囲）', en: 'Article 1 (Scope)' },
    paras: [
      {
        ja: '本利用規約（以下「本規約」）は、Verantyx（以下「当社」）が提供するiOSアプリケーション（パクパク釣り、MouthEat、以下「本アプリ」）の利用に関する条件を定めるものです。本アプリをご利用いただく全てのユーザー（以下「ユーザー」）に適用されます。',
        en: 'These Terms of Service (the "Terms") set out the conditions for using the iOS applications provided by Verantyx ("we", "us") — Paku Paku Fishing and MouthEat (the "Apps"). They apply to every user of the Apps (each, a "User").',
      },
    ],
  },
  {
    title: { ja: '第2条（利用条件）', en: 'Article 2 (Conditions of Use)' },
    intro: {
      ja: 'ユーザーは、本規約に同意した上で本アプリを利用するものとします。',
      en: 'Users shall use the Apps having agreed to these Terms.',
    },
    items: [
      {
        ja: '1. 本アプリの利用には、iOS 17.0以上およびTrueDepthカメラ搭載のiPhoneが必要です',
        en: '1. Use of the Apps requires iOS 17.0 or later and an iPhone with a TrueDepth camera',
      },
      {
        ja: '2. ユーザーは、本アプリを適切に利用する責任を負います',
        en: '2. Users are responsible for using the Apps appropriately',
      },
      {
        ja: '3. 本アプリの利用により生じた一切の損害について、当社は責任を負いません',
        en: '3. We accept no liability for any damage arising from use of the Apps',
      },
    ],
  },
  {
    title: { ja: '第3条（禁止事項）', en: 'Article 3 (Prohibited Conduct)' },
    intro: {
      ja: 'ユーザーは、本アプリの利用にあたり、以下の行為を行ってはなりません：',
      en: 'When using the Apps, Users must not:',
    },
    items: [
      { ja: '1. 法令または公序良俗に違反する行為', en: '1. Violate laws, regulations, or public order and morals' },
      { ja: '2. 犯罪行為に関連する行為', en: '2. Engage in conduct connected to criminal activity' },
      { ja: '3. 本アプリの運営を妨害する行為', en: '3. Interfere with the operation of the Apps' },
      {
        ja: '4. 本アプリの不正利用、リバースエンジニアリング、逆コンパイル',
        en: '4. Misuse, reverse-engineer, or decompile the Apps',
      },
      {
        ja: '5. 他のユーザーまたは第三者の権利を侵害する行為',
        en: '5. Infringe the rights of other Users or third parties',
      },
      { ja: '6. 本アプリのセキュリティを脅かす行為', en: '6. Compromise the security of the Apps' },
      {
        ja: '7. その他、当社が不適切と判断する行為',
        en: '7. Engage in any other conduct we deem inappropriate',
      },
    ],
  },
  {
    title: { ja: '第4条（アプリ内課金）', en: 'Article 4 (In-App Purchases)' },
    items: [
      {
        ja: '1. 本アプリ内でPASS（買い切り課金）を購入することができます',
        en: '1. A PASS (one-time purchase) may be bought within the Apps',
      },
      {
        ja: '2. 購入はApple App Storeを通じて行われ、Appleの利用規約が適用されます',
        en: "2. Purchases are made through the Apple App Store and Apple's terms apply",
      },
      {
        ja: '3. 購入後の返金は、Appleの返金ポリシーに従います',
        en: "3. Refunds after purchase follow Apple's refund policy",
      },
      {
        ja: '4. 購入した機能は、同一Apple IDでサインインしている端末で復元可能です',
        en: '4. Purchased features can be restored on devices signed in with the same Apple ID',
      },
    ],
  },
  {
    title: { ja: '第5条（知的財産権）', en: 'Article 5 (Intellectual Property)' },
    paras: [
      {
        ja: '本アプリに関する著作権、商標権、その他の知的財産権は、当社または正当な権利者に帰属します。ユーザーは、これらの権利を侵害する行為を行ってはなりません。',
        en: 'Copyright, trademark, and other intellectual property rights in the Apps belong to us or to their rightful holders. Users must not infringe those rights.',
      },
    ],
  },
  {
    title: { ja: '第6条（免責事項）', en: 'Article 6 (Disclaimer)' },
    items: [
      {
        ja: '1. 当社は、本アプリの動作について一切の保証を行いません',
        en: '1. We give no warranty whatsoever as to the operation of the Apps',
      },
      {
        ja: '2. 本アプリの利用により生じた損害について、当社は責任を負いません',
        en: '2. We accept no liability for damage arising from use of the Apps',
      },
      {
        ja: '3. 本アプリは予告なく変更、終了する場合があります',
        en: '3. The Apps may be changed or discontinued without notice',
      },
      {
        ja: '4. カメラ使用時の安全性については、ユーザー自身の責任で管理してください',
        en: '4. Users are responsible for their own safety while using the camera',
      },
    ],
  },
  {
    title: {
      ja: '第7条（サービスの変更・終了）',
      en: 'Article 7 (Changes to or Discontinuation of the Service)',
    },
    paras: [
      {
        ja: '当社は、ユーザーへの事前通知なく、本アプリの内容を変更、またはサービスを終了することができます。これにより生じた損害について、当社は責任を負いません。',
        en: 'We may change the Apps or discontinue the service without prior notice to Users, and accept no liability for damage arising from doing so.',
      },
    ],
  },
  {
    title: { ja: '第8条（規約の変更）', en: 'Article 8 (Changes to these Terms)' },
    paras: [
      {
        ja: '当社は、必要に応じて本規約を変更することができます。変更後の規約は、本アプリ内またはウェブサイトに掲載した時点で効力を生じます。',
        en: 'We may amend these Terms as necessary. Amended Terms take effect when posted within the Apps or on the website.',
      },
    ],
  },
  {
    title: {
      ja: '第9条（準拠法・管轄裁判所）',
      en: 'Article 9 (Governing Law and Jurisdiction)',
    },
    paras: [
      {
        ja: '本規約の解釈および適用については、日本法に準拠します。',
        en: 'These Terms are governed by and construed in accordance with the laws of Japan.',
      },
      {
        ja: '本規約に関する紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とします。',
        en: 'The Tokyo District Court shall have exclusive jurisdiction as the court of first instance for disputes concerning these Terms.',
      },
    ],
  },
];

export default function TermsPage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];

  return (
    <main lang={lang} className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      <PageHero
        title={t({ ja: '利用規約', en: 'Terms of Service' })}
        subtitle={t({ ja: 'Terms of Service', en: '利用規約' })}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />

      {ARTICLES.map((a) => (
        <CinematicSection key={a.title.en} title={t(a.title)}>
          <div className="space-y-4 text-gray-400 leading-[2em]">
            {a.intro && <p>{t(a.intro)}</p>}
            {a.paras?.map((p) => (
              <p key={p.en}>{t(p)}</p>
            ))}
            {a.items && (
              <ul className="space-y-2 ml-4">
                {a.items.map((item) => (
                  <li key={item.en}>{t(item)}</li>
                ))}
              </ul>
            )}
          </div>
        </CinematicSection>
      ))}

      <CinematicSection title={t({ ja: '第10条（お問い合わせ）', en: 'Article 10 (Contact)' })}>
        <GlassCard>
          <p className="text-gray-400 mb-4 leading-[2em]">
            {t({
              ja: '本規約に関するお問い合わせは、以下までご連絡ください。',
              en: 'For enquiries about these Terms, please use the contact below.',
            })}
          </p>
          <p className="text-gray-600 text-sm">
            {t({ ja: 'メールアドレス: ', en: 'Email: ' })}
            <span className="text-gray-500">
              {t({
                ja: '[お問い合わせメールアドレス設定予定]',
                en: '[contact address to be published]',
              })}
            </span>
          </p>
          <p className="text-gray-600 text-sm mt-2">
            {t({ ja: 'サポートページ: ', en: 'Support page: ' })}
            <a
              href="/support/"
              className="transition-colors duration-300"
              style={{ color: 'rgba(14, 165, 233, 0.6)' }}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'rgba(14, 165, 233, 0.9)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'rgba(14, 165, 233, 0.6)')}
            >
              https://verantyx.ai/support
            </a>
          </p>
        </GlassCard>
      </CinematicSection>

      {/* Effective Date */}
      <section className="relative px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center pt-8"
            style={{ borderTop: '1px solid rgba(55,65,81,0.3)' }}
          >
            <p className="text-gray-600 text-xs tracking-wide">
              {t({ ja: '制定日: 2025年3月', en: 'Effective: March 2025' })}
            </p>
            <p className="mt-1 text-gray-600 text-xs tracking-wide">
              {t({ ja: '最終更新日: 2025年3月', en: 'Last updated: March 2025' })}
            </p>
            {lang === 'en' && (
              <p className="mt-4 text-gray-600 text-xs leading-relaxed max-w-xl mx-auto">
                The Japanese text of these Terms is the operative version. This
                English text is provided for convenience; where the two differ,
                the Japanese governs — consistent with Article 9.
              </p>
            )}
          </motion.div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
