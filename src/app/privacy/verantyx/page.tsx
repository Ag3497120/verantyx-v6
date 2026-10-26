'use client';

import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';
import { PageHero, CinematicSection, FeatureCheck } from '@/components/CinematicSection';

// Privacy policy for VeRantyx: the Mac app (VeRantyx for Mac) and the Apple Vision Pro app.
export default function VeRantyxPrivacyPage() {
  const { lang } = useLanguage();
  const ja = lang === 'ja';
  const list = (items: string[]) => (
    <div className="space-y-2.5">
      {items.map((item, i) => (
        <FeatureCheck key={i}>{item}</FeatureCheck>
      ))}
    </div>
  );

  const sections = [
    {
      id: 'intro',
      title: { ja: 'はじめに', en: 'Introduction' },
      content: (
        <p className="text-gray-400 leading-[2em]">
          {ja
            ? 'Verantyx が提供する VeRantyx（Mac アプリ「VeRantyx for Mac」と Apple Vision Pro のアプリ）のプライバシーポリシーです。VeRantyx はデータを集めません。'
            : 'This is the privacy policy for VeRantyx (the Mac app "VeRantyx for Mac" and the Apple Vision Pro app), provided by Verantyx. VeRantyx does not collect any data.'}
        </p>
      ),
    },
    {
      id: 'vision',
      title: { ja: 'Apple Vision Pro のアプリ', en: 'The Apple Vision Pro app' },
      content: list(
        ja
          ? [
              '頭と手の位置、コントローラーの入力を、ご自身の Mac の VeRantyx にだけ送ります（ローカルネットワークか、Wi-Fi の直接の通信）',
              'Mac からゲームの映像と音を受け取ります',
              '手の位置などは Verantyx やほかの誰にも送りません',
              '設定と診断用のログは端末の中にだけ保存され、送信されません',
            ]
          : [
              'Sends your head and hand positions and controller input only to VeRantyx on your own Mac (over your local network or a direct Wi-Fi connection)',
              "Receives the game's picture and sound from your Mac",
              'Hand positions and the like are never sent to Verantyx or anyone else',
              'Settings and a diagnostics log are stored only on the device and never uploaded',
            ],
      ),
    },
    {
      id: 'mac',
      title: { ja: 'Mac のアプリ（VeRantyx for Mac）', en: 'The Mac app (VeRantyx for Mac)' },
      content: list(
        ja
          ? [
              'ゲームと SteamVR は、ご自身の Steam アカウントで Valve から入手します。サインインはご自身で Steam（steamcmd や Windows 版 Steam）に入力し、VeRantyx は資格情報を保存・送信しません',
              'アプリ自身がインターネットから取りに行くのは、Valve の steamcmd（Steam のダウンロードの道具）だけです',
              '映像と音は、ご自身のヘッドセット（Vision Pro のアプリ、または ALVR のアプリ）にだけ送ります',
              'ログはご自身の Mac の中にだけ保存されます',
              '「問題を報告」は、内容を入れた報告のページをブラウザで開くだけです。送るかどうかはご自身が決め、内容も送る前に確認できます',
            ]
          : [
              'Games and SteamVR come from Valve through your own Steam account. You sign in to Steam yourself (steamcmd or the Windows Steam client); VeRantyx never stores or sends your credentials',
              "The only thing the app itself downloads from the internet is Valve's steamcmd (Steam's download tool)",
              'The picture and sound go only to your own headset (the Vision Pro app or an ALVR app)',
              'Logs stay on your own Mac',
              '"Report a problem" only opens a prefilled report page in your browser; you decide whether to send it and can review it first',
            ],
      ),
    },
    {
      id: 'none',
      title: { ja: 'ないもの', en: 'What there is none of' },
      content: list(
        ja
          ? ['アカウント', '利用状況の分析', '広告', 'トラッキング', '第三者へのデータの提供']
          : ['Accounts', 'Analytics', 'Advertising', 'Tracking', 'Sharing data with third parties'],
      ),
    },
    {
      id: 'site',
      title: { ja: 'ダウンロードの数', en: 'Download counts' },
      content: (
        <p className="text-gray-400 leading-[2em]">
          {ja
            ? 'verantyx.ai からアプリがダウンロードされた回数を、ファイルごと・日ごとに数えています。IP アドレス、ブラウザの情報、Cookie など、ダウンロードした人に関する情報は記録しません。'
            : 'verantyx.ai counts how many times each app file is downloaded, per day. Nothing about the person downloading is recorded: no IP address, no browser information, no cookies.'}
        </p>
      ),
    },
    {
      id: 'contact',
      title: { ja: 'お問い合わせ', en: 'Contact' },
      content: (
        <p className="text-gray-400">
          {ja ? 'ウェブサイト: ' : 'Website: '}
          <a href="/support/" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>
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
            {ja ? '内容を変えるときは、このページを更新します。' : 'If this policy changes, this page will be updated.'}
          </p>
          <p className="text-gray-600 text-xs tracking-wide">{ja ? '最終更新日: 2026年10月26日' : 'Last updated: October 26, 2026'}</p>
        </>
      ),
    },
  ];

  return (
    <main className="relative bg-black text-white overflow-x-hidden">
      <Navbar />
      <PageHero
        title={ja ? 'VeRantyx のプライバシーポリシー' : 'VeRantyx Privacy Policy'}
        subtitle={ja ? 'Mac と Apple Vision Pro のアプリ' : 'The Mac and Apple Vision Pro apps'}
        gradient="linear-gradient(135deg, #0EA5E9, #7C3AED)"
      />
      {sections.map((s) => (
        <CinematicSection key={`${s.id}-${lang}`} title={s.title[lang]}>
          {s.content}
        </CinematicSection>
      ))}
      <Footer />
    </main>
  );
}
