'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

type L = { en: string; ja: string };

/* Written for somebody who has never opened a terminal, because that is who
 * the tool is for. Three deliberate choices:
 *
 *   the browser demo comes first    it needs nothing installed at all, and a
 *                                   person who can see what the thing does in
 *                                   ten seconds will decide whether the rest
 *                                   is worth their afternoon
 *   every command is one line       a step that says "then edit this file" is
 *                                   a step where half of readers stop
 *   the failure cases are on the    "python: command not found" is the single
 *   page, not in a FAQ              most likely outcome, and sending someone
 *                                   to search for it is abandoning them
 *
 * GitHub is mentioned once, at the bottom, for people who want the source.
 * It is not the download path.
 */

const STEPS: { n: string; title: L; body: L; cmd?: string; note?: L }[] = [
  {
    n: '1',
    title: { en: 'Check Python is there', ja: 'Python があるか確かめる' },
    body: {
      en: 'Open Terminal (macOS: ⌘+Space, type "terminal") and paste this. A version number means you are ready.',
      ja: 'ターミナルを開いて（macOS は ⌘+スペースで「ターミナル」）、これを貼り付けます。バージョン番号が出れば準備完了です。',
    },
    cmd: 'python3 --version',
    note: {
      en: 'Nothing appears, or "command not found"? Install Python from python.org — the standard installer, no options to change — then come back to this step.',
      ja: '何も出ない・「command not found」と出た場合は、python.org から Python を入れてください（標準のインストーラのまま、設定変更は不要です）。入れたらこの手順に戻ります。',
    },
  },
  {
    n: '2',
    title: { en: 'Install Vera', ja: 'Vera を入れる' },
    body: {
      en: 'One line. It downloads about 1 MB and takes a few seconds.',
      ja: '1行です。1MB ほどをダウンロードして数秒で終わります。',
    },
    cmd: 'python3 -m pip install "verantyx-vera[docs]"',
    note: {
      en: 'The [docs] part adds PDF and Word reading. Leave it out and it still runs, but only on plain text and HTML.',
      ja: '[docs] は PDF と Word を読むための追加です。外しても動きますが、テキストと HTML だけになります。',
    },
  },
  {
    n: '3',
    title: { en: 'Open it', ja: '開く' },
    body: {
      en: 'This opens the app in your browser. It runs entirely on your own machine — nothing is sent anywhere.',
      ja: 'ブラウザでアプリが開きます。すべてあなたの機械の中で動き、どこにも送信されません。',
    },
    cmd: 'vera field',
    note: {
      en: 'To stop it, go back to Terminal and press Control+C. To start it again another day, run the same line.',
      ja: '止めるにはターミナルに戻って Control+C。別の日にまた使うときは同じ行を実行するだけです。',
    },
  },
];

const TROUBLE: { q: L; a: L }[] = [
  {
    q: { en: '“vera: command not found”', ja: '「vera: command not found」と出る' },
    a: {
      en: 'Use python3 -m verantyx.cli field instead. It does exactly the same thing; only the shortcut is missing.',
      ja: '代わりに python3 -m verantyx.cli field を実行してください。中身は同じで、短縮名だけが登録されていない状態です。',
    },
  },
  {
    q: { en: 'The browser did not open', ja: 'ブラウザが開かない' },
    a: {
      en: 'Type http://127.0.0.1:8900 into the address bar yourself. The app is already running.',
      ja: 'アドレス欄に http://127.0.0.1:8900 と自分で入力してください。アプリはもう動いています。',
    },
  },
  {
    q: { en: 'A PDF was refused', ja: 'PDF が読めなかった' },
    a: {
      en: 'A scanned PDF is a picture of text, and no software can pull letters out of it without OCR. Try a PDF where you can select the text with the mouse, or the Word/HTML original.',
      ja: 'スキャン画像の PDF は「文字の写真」なので、OCR なしでは文字を取り出せません。マウスで文字を選択できる PDF か、Word/HTML の元資料をお試しください。',
    },
  },
  {
    q: { en: 'Our network blocks installs', ja: '職場のネットワークで入れられない' },
    a: {
      en: 'Install it on a machine that can reach the internet, then copy the whole folder across. It never needs a network to run — only to be installed.',
      ja: 'インターネットに繋がる機械で入れてから、フォルダごとコピーしてください。動作にネットワークは一切不要で、必要なのは導入のときだけです。',
    },
  },
];

export default function DownloadPage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';

  return (
    <main lang={lang} className="relative text-white min-h-screen" style={{ overflowX: 'clip' }}>
      <Navbar />

      <section className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-8">
        <div className="mx-auto w-full max-w-3xl">
          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: ja ? 'clamp(1.9rem, 6.5vw, 3.4rem)' : 'clamp(2.1rem, 7.5vw, 4rem)',
              lineHeight: ja ? 1.25 : 1.06,
              marginBottom: '1rem',
            }}
          >
            {t({ en: 'Install it on your own machine', ja: '自分の機械に入れる' })}
          </motion.h1>
          <p style={{ color: 'var(--ink-2)', fontSize: '1.02rem', lineHeight: ja ? 1.95 : 1.7 }}>
            {t({
              en: 'Three steps, about two minutes. Once installed it never touches the network again — which is why documents you cannot share are safe with it.',
              ja: '3手順、2分ほどです。入れたあとは二度とネットワークに触れません。共有できない資料を扱えるのはそのためです。',
            })}
          </p>
        </div>
      </section>

      {/* Try before installing — the fastest way to decide */}
      <section className="relative px-5 sm:px-6 pb-10">
        <div className="mx-auto w-full max-w-3xl">
          <div
            className="rounded-2xl p-5 sm:p-6"
            style={{
              border: '1px solid rgba(var(--accent-rgb), 0.35)',
              background: 'rgba(var(--accent-rgb), 0.06)',
            }}
          >
            <p className="font-semibold mb-2">
              {t({ en: 'Not sure yet? Try it without installing.', ja: '迷っていますか。入れずに試せます。' })}
            </p>
            <p className="mb-4" style={{ color: 'var(--ink-2)', fontSize: '.93rem', lineHeight: ja ? 1.9 : 1.65 }}>
              {t({
                en: 'The browser demo runs the same engine on fictional data. Nothing to install, nothing to uninstall.',
                ja: 'ブラウザ版が同じエンジンを架空データで動かします。インストールも削除も不要です。',
              })}
            </p>
            <a href="/vera/demo/" className="btn-accent rounded-xl px-5 py-3 text-sm font-semibold" style={{ textDecoration: 'none' }}>
              {t({ en: 'Open the browser demo', ja: 'ブラウザ版を開く' })} →
            </a>
          </div>
        </div>
      </section>

      <section className="relative px-5 sm:px-6 pb-12">
        <div className="mx-auto w-full max-w-3xl space-y-5">
          {STEPS.map((s) => (
            <div
              key={s.n}
              className="rounded-2xl p-5 sm:p-6"
              style={{ border: '1px solid var(--line, rgba(255,255,255,0.1))' }}
            >
              <div className="flex items-baseline gap-3 mb-2">
                <span
                  className="rounded-lg px-2.5 py-1 font-bold"
                  style={{
                    background: 'rgba(var(--accent-rgb), 0.14)',
                    color: 'rgba(var(--accent-rgb), 0.95)',
                    fontSize: '.85rem',
                  }}
                >
                  {s.n}
                </span>
                <h2 className="font-semibold" style={{ fontSize: '1.08rem' }}>{t(s.title)}</h2>
              </div>
              <p style={{ color: 'var(--ink-2)', fontSize: '.93rem', lineHeight: ja ? 1.9 : 1.65 }}>
                {t(s.body)}
              </p>
              {s.cmd && (
                <pre
                  className="mt-3 rounded-xl px-4 py-3 overflow-x-auto"
                  style={{
                    background: 'var(--surface-2, rgba(255,255,255,0.05))',
                    border: '1px solid var(--line, rgba(255,255,255,0.09))',
                    fontSize: '.82rem',
                    lineHeight: 1.7,
                  }}
                >
                  <code>{s.cmd}</code>
                </pre>
              )}
              {s.note && (
                <p className="mt-3" style={{ color: 'var(--ink-3)', fontSize: '.86rem', lineHeight: ja ? 1.85 : 1.6 }}>
                  {t(s.note)}
                </p>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="relative px-5 sm:px-6 pb-20">
        <div className="mx-auto w-full max-w-3xl">
          <h2 className="font-display font-bold mb-5" style={{ fontSize: 'clamp(1.25rem, 3.4vw, 1.7rem)' }}>
            {t({ en: 'If something goes wrong', ja: 'うまくいかないとき' })}
          </h2>
          <div className="space-y-3">
            {TROUBLE.map((x) => (
              <div key={x.q.en} className="rounded-xl p-4" style={{ border: '1px solid var(--line, rgba(255,255,255,0.09))' }}>
                <p className="font-semibold mb-1" style={{ fontSize: '.95rem' }}>{t(x.q)}</p>
                <p style={{ color: 'var(--ink-3)', fontSize: '.88rem', lineHeight: ja ? 1.85 : 1.6 }}>{t(x.a)}</p>
              </div>
            ))}
          </div>
          <p className="mt-8" style={{ color: 'var(--ink-3)', fontSize: '.85rem' }}>
            {t({ en: 'Source code: ', ja: 'ソースコード: ' })}
            <a
              href="https://github.com/Ag3497120/Verantyx"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'rgba(var(--accent-rgb), 0.95)' }}
            >
              github.com/Ag3497120/Verantyx
            </a>
          </p>
        </div>
      </section>

      <Footer />
    </main>
  );
}
