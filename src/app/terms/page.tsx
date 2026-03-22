'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { PageHero, CinematicSection, GlassCard } from '@/components/CinematicSection';

export default function TermsPage() {
  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />

      <PageHero
        title="利用規約"
        subtitle="Terms of Service"
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />

      <CinematicSection title="第1条（適用範囲）">
        <p className="text-gray-400 leading-[2em]">
          本利用規約（以下「本規約」）は、Verantyx（以下「当社」）が提供するiOSアプリケーション（パクパク釣り、MouthEat、以下「本アプリ」）の利用に関する条件を定めるものです。本アプリをご利用いただく全てのユーザー（以下「ユーザー」）に適用されます。
        </p>
      </CinematicSection>

      <CinematicSection title="第2条（利用条件）">
        <div className="space-y-4 text-gray-400 leading-[2em]">
          <p>ユーザーは、本規約に同意した上で本アプリを利用するものとします。</p>
          <ul className="space-y-2 ml-4">
            <li>1. 本アプリの利用には、iOS 17.0以上およびTrueDepthカメラ搭載のiPhoneが必要です</li>
            <li>2. ユーザーは、本アプリを適切に利用する責任を負います</li>
            <li>3. 本アプリの利用により生じた一切の損害について、当社は責任を負いません</li>
          </ul>
        </div>
      </CinematicSection>

      <CinematicSection title="第3条（禁止事項）">
        <div className="space-y-4 text-gray-400 leading-[2em]">
          <p>ユーザーは、本アプリの利用にあたり、以下の行為を行ってはなりません：</p>
          <ul className="space-y-2 ml-4">
            <li>1. 法令または公序良俗に違反する行為</li>
            <li>2. 犯罪行為に関連する行為</li>
            <li>3. 本アプリの運営を妨害する行為</li>
            <li>4. 本アプリの不正利用、リバースエンジニアリング、逆コンパイル</li>
            <li>5. 他のユーザーまたは第三者の権利を侵害する行為</li>
            <li>6. 本アプリのセキュリティを脅かす行為</li>
            <li>7. その他、当社が不適切と判断する行為</li>
          </ul>
        </div>
      </CinematicSection>

      <CinematicSection title="第4条（アプリ内課金）">
        <div className="space-y-4 text-gray-400 leading-[2em]">
          <ul className="space-y-2 ml-4">
            <li>1. 本アプリ内でPASS（買い切り課金）を購入することができます</li>
            <li>2. 購入はApple App Storeを通じて行われ、Appleの利用規約が適用されます</li>
            <li>3. 購入後の返金は、Appleの返金ポリシーに従います</li>
            <li>4. 購入した機能は、同一Apple IDでサインインしている端末で復元可能です</li>
          </ul>
        </div>
      </CinematicSection>

      <CinematicSection title="第5条（知的財産権）">
        <p className="text-gray-400 leading-[2em]">
          本アプリに関する著作権、商標権、その他の知的財産権は、当社または正当な権利者に帰属します。ユーザーは、これらの権利を侵害する行為を行ってはなりません。
        </p>
      </CinematicSection>

      <CinematicSection title="第6条（免責事項）">
        <div className="space-y-4 text-gray-400 leading-[2em]">
          <ul className="space-y-2 ml-4">
            <li>1. 当社は、本アプリの動作について一切の保証を行いません</li>
            <li>2. 本アプリの利用により生じた損害について、当社は責任を負いません</li>
            <li>3. 本アプリは予告なく変更、終了する場合があります</li>
            <li>4. カメラ使用時の安全性については、ユーザー自身の責任で管理してください</li>
          </ul>
        </div>
      </CinematicSection>

      <CinematicSection title="第7条（サービスの変更・終了）">
        <p className="text-gray-400 leading-[2em]">
          当社は、ユーザーへの事前通知なく、本アプリの内容を変更、またはサービスを終了することができます。これにより生じた損害について、当社は責任を負いません。
        </p>
      </CinematicSection>

      <CinematicSection title="第8条（規約の変更）">
        <p className="text-gray-400 leading-[2em]">
          当社は、必要に応じて本規約を変更することができます。変更後の規約は、本アプリ内またはウェブサイトに掲載した時点で効力を生じます。
        </p>
      </CinematicSection>

      <CinematicSection title="第9条（準拠法・管轄裁判所）">
        <div className="space-y-4 text-gray-400 leading-[2em]">
          <p>本規約の解釈および適用については、日本法に準拠します。</p>
          <p>本規約に関する紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とします。</p>
        </div>
      </CinematicSection>

      <CinematicSection title="第10条（お問い合わせ）">
        <GlassCard>
          <p className="text-gray-400 mb-4 leading-[2em]">
            本規約に関するお問い合わせは、以下までご連絡ください。
          </p>
          <p className="text-gray-600 text-sm">
            メールアドレス: <span className="text-gray-500">[お問い合わせメールアドレス設定予定]</span>
          </p>
          <p className="text-gray-600 text-sm mt-2">
            サポートページ:{' '}
            <a href="/support" className="transition-colors duration-300" style={{ color: 'rgba(14, 165, 233, 0.6)' }} onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.9)'} onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(14, 165, 233, 0.6)'}>
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
            <p className="text-gray-600 text-xs tracking-wide">制定日: 2025年3月</p>
            <p className="mt-1 text-gray-600 text-xs tracking-wide">最終更新日: 2025年3月</p>
          </motion.div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
