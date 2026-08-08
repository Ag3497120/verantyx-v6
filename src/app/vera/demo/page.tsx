'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

const SPACE = 'https://kofdai-verantyx-vera.hf.space';
const SPACE_PAGE = 'https://huggingface.co/spaces/kofdai/verantyx-vera';
const ENGINE = 'https://github.com/Ag3497120/Verantyx';

type L = { en: string; ja: string };

/* The demo runs on a Hugging Face Space, not here. That is deliberate twice
 * over: this site stays a static export with no server to leak anything
 * through, and the boundary between "a page you read" and "a server you send
 * a document to" stays somewhere a visitor can see it. */

const TABS: { name: L; body: L }[] = [
  {
    name: { en: 'Two texts', ja: 'テキスト2件' },
    body: {
      en: 'Settled, updated, contested and unanswered, separated and never blended — each claim carrying the source that made it.',
      ja: '確定・更新・係争・未回答を分離し、決して混ぜません。すべての主張が出典を連れています。',
    },
  },
  {
    name: { en: 'Self-evolution', ja: '自己進化' },
    body: {
      en: 'The engine proves a defect in its own reading with no answer key, repairs what it can, and hands you what it cannot.',
      ja: 'エンジンが自分の読みの欠陥を答え合わせなしで証明し、直せるものを直し、直せないものをあなたに渡すまで。',
    },
  },
  {
    name: { en: 'Closed loop', ja: '閉ループ' },
    body: {
      en: 'Candidate repairs built from what was PROVEN, then measured — accepted or rejected, with the numbers that decided it.',
      ja: '証明された欠陥から修復候補を作り、測って受理か却下を決めます。決め手になった数字も一緒に出ます。',
    },
  },
  {
    name: { en: 'Lexicon', ja: '辞書' },
    body: {
      en: 'A jgen carrying only its embed table, so it cannot generate. It answers two questions that were measured usable, and refuses the one that was not.',
      ja: 'embed 表だけを持つ jgen なので生成できません。測って使えると分かった二つに答え、使えなかった一つは答えません。',
    },
  },
  {
    name: { en: 'Files', ja: 'ファイル' },
    body: {
      en: 'PDF, Word, HTML, CSV. Public documents only — the same warning as everywhere else on this page.',
      ja: 'PDF・Word・HTML・CSV。公開資料のみ。このページの他の場所と同じ警告です。',
    },
  },
];

const LINES: { closed: boolean; title: L; body: L }[] = [
  {
    closed: true,
    title: { en: 'Layout defects', ja: 'レイアウト欠陥' },
    body: {
      en: 'A space between two kanji was put there by a PDF extractor, not an author — Japanese does not space its words. So layout carries no information, and a claim that appears only in the spaced reading is provably spurious. Repaired unattended.',
      ja: '漢字の間の空白は著者ではなくPDF抽出器が入れたものです（日本語は語間を空けません）。レイアウトは情報を運ばないので、空白のある読みにだけ現れた主張は証明つきで偽です。無人で修復します。',
    },
  },
  {
    closed: true,
    title: { en: 'Guard conflicts', ja: 'ガード矛盾' },
    body: {
      en: "A guard's meaning is: match this after a polar term and the term asserts nothing. A placed pole whose tail a guard matches is an internal contradiction — and both live inside the same process, so no world knowledge enters. Repaired unattended.",
      ja: 'ガードの意味は「極性語の直後にこれが来たら、その語は何も主張していない」。極が置かれた文の語尾にガードが一致していたら内部矛盾です。両方とも同じプロセス内にあるので世界の知識は不要。無人で修復します。',
    },
  },
  {
    closed: false,
    title: { en: 'Vocabulary', ja: '語彙' },
    body: {
      en: 'No transformation of a document reveals what an unseen word means. Candidates are found, anchored, damage-tested and queued — and the one judgement left is yours. On the real corpora that queue held two true candidates and two false ones, and nothing inside the engine can tell them apart.',
      ja: 'どんな文書変換も、見たことのない語の意味を教えてくれません。候補は発見・錨付け・損傷テスト・待ち行列まで自動で進み、残る一つの判断があなたのものです。実コーパスではその待ち行列に本物2件と偽物2件が並び、エンジンには区別できません。',
    },
  },
];

export default function VeraDemoPage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';

  return (
    <main
      lang={lang}
      className="relative text-white min-h-screen"
      style={{ overflowX: 'clip' }}
    >
      <Navbar />

      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-10 sm:pb-14">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 70% 55% at 30% 25%, rgba(var(--accent-rgb), 0.10), transparent 62%)',
          }}
        />
        <div className="mx-auto w-full max-w-4xl relative z-10">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="uppercase mb-5 sm:mb-7"
            style={{
              color: 'rgba(var(--accent-rgb), 0.9)',
              fontSize: 'clamp(0.62rem, 1.6vw, 0.72rem)',
              letterSpacing: '0.3em',
            }}
          >
            Live demo
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.06 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: ja
                ? 'clamp(1.9rem, 7vw, 3.9rem)'
                : 'clamp(2.2rem, 8.5vw, 4.8rem)',
              lineHeight: ja ? 1.22 : 1.04,
              letterSpacing: ja ? '-0.01em' : '-0.03em',
              marginBottom: '1.1rem',
            }}
          >
            {t({ en: 'Try it in the browser', ja: 'ブラウザで試す' })}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="max-w-2xl"
            style={{
              color: 'var(--ink-2)',
              fontSize: 'clamp(1rem, 2.4vw, 1.2rem)',
              lineHeight: ja ? 1.95 : 1.7,
              fontWeight: 300,
            }}
          >
            {t({
              en: 'The full engine, running live — no install, no account. Deterministic and with no model anywhere in the answer path, so the same input always produces the same board.',
              ja: 'エンジンをそのまま動かしています。インストールもアカウントも不要。答えの経路にモデルが一切なく決定論的なので、同じ入力からは必ず同じ板が出ます。',
            })}
          </motion.p>
        </div>
      </section>

      {/* ── The warning, before the thing it is about ──────────── */}
      <section className="relative px-5 sm:px-6 pb-8">
        <div className="mx-auto w-full max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.5 }}
            className="rounded-2xl p-5 sm:p-6"
            style={{
              border: '1px solid rgba(var(--warn-rgb, 245 158 11), 0.42)',
              background: 'rgba(var(--warn-rgb, 245 158 11), 0.07)',
            }}
          >
            <p
              className="font-semibold mb-2"
              style={{ color: 'var(--warn, #f59e0b)', fontSize: '0.98rem' }}
            >
              {t({
                en: 'Never paste confidential documents here',
                ja: '機密文書は絶対に入れないでください',
              })}
            </p>
            <p
              style={{
                color: 'var(--ink-2)',
                fontSize: '0.93rem',
                lineHeight: ja ? 1.9 : 1.65,
              }}
            >
              {t({
                en: 'This is a public server. Anything you paste or upload is transmitted to it. Use fictional data, or documents that are already published. To run the same engine on documents you cannot share, install it locally — it makes no network calls at all.',
                ja: 'これは公開サーバです。貼り付け・アップロードした文書は送信されます。架空データか、すでに公開されている資料だけを使ってください。共有できない文書で同じエンジンを動かすには、ローカルに入れてください。外部接続を一切しません。',
              })}
            </p>
            <a
              href="/vera/download/"
              className="inline-block mt-4 rounded-xl px-5 py-3 text-sm font-semibold"
              style={{
                textDecoration: 'none',
                color: 'var(--ink)',
                border: '1px solid var(--line-strong, rgba(255,255,255,0.18))',
              }}
            >
              {t({ en: 'How to install it — two minutes', ja: '入れ方（2分）' })} →
            </a>
            <pre
              className="mt-4 rounded-xl px-4 py-3 overflow-x-auto"
              style={{
                background: 'var(--surface-2, rgba(255,255,255,0.045))',
                border: '1px solid var(--line, rgba(255,255,255,0.08))',
                fontSize: '0.82rem',
                lineHeight: 1.75,
                color: 'var(--ink-2)',
              }}
            >
              <code>{`vera field   # ${ja ? '外部接続なし・127.0.0.1' : 'no network, 127.0.0.1'}`}</code>
            </pre>
          </motion.div>
        </div>
      </section>

      {/* ── The embed ──────────────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-14 sm:pb-20">
        <div className="mx-auto w-full max-w-5xl">
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.55 }}
            className="rounded-2xl overflow-hidden"
            style={{
              border: '1px solid var(--line, rgba(255,255,255,0.1))',
              background: 'var(--surface, rgba(255,255,255,0.02))',
            }}
          >
            <iframe
              src={SPACE}
              title="Verantyx Vera demo"
              loading="lazy"
              /* No allow= list: the demo needs no camera, microphone,
               * geolocation or payment, and a permission not granted is a
               * permission that cannot be misused. */
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads"
              style={{
                width: '100%',
                height: 'min(80vh, 860px)',
                minHeight: '560px',
                border: 0,
                display: 'block',
              }}
            />
          </motion.div>

          <p
            className="mt-4 text-center"
            style={{ color: 'var(--ink-4, rgba(255,255,255,0.45))', fontSize: '0.82rem' }}
          >
            {t({ en: 'Hosted on Hugging Face Spaces · ', ja: 'Hugging Face Spaces 上で稼働 · ' })}
            <a
              href={SPACE_PAGE}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'rgba(var(--accent-rgb), 0.95)' }}
            >
              {t({ en: 'open in a new tab', ja: '別タブで開く' })} ↗
            </a>
            {t({
              en: ' · the Space sleeps when idle and takes a moment to wake',
              ja: ' · 無操作時はスリープするため初回は起動待ちがあります',
            })}
          </p>
        </div>
      </section>

      {/* ── What the three tabs show ───────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-14 sm:pb-20">
        <div className="mx-auto w-full max-w-4xl">
          <h2
            className="font-display font-bold mb-7"
            style={{ fontSize: 'clamp(1.35rem, 3.6vw, 2rem)' }}
          >
            {t({ en: 'Five tabs', ja: '五つのタブ' })}
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {TABS.map((tab, i) => (
              <motion.div
                key={tab.name.en}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.45, delay: i * 0.06 }}
                className="rounded-2xl p-5"
                style={{
                  border: '1px solid var(--line, rgba(255,255,255,0.09))',
                  background: 'var(--surface, rgba(255,255,255,0.02))',
                }}
              >
                <p className="font-semibold mb-2" style={{ fontSize: '0.95rem' }}>
                  {t(tab.name)}
                </p>
                <p
                  style={{
                    color: 'var(--ink-3, rgba(255,255,255,0.62))',
                    fontSize: '0.88rem',
                    lineHeight: ja ? 1.9 : 1.65,
                  }}
                >
                  {t(tab.body)}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Where the loop closes, and where it does not ────────── */}
      <section className="relative px-5 sm:px-6 pb-20 sm:pb-28">
        <div className="mx-auto w-full max-w-4xl">
          <h2
            className="font-display font-bold mb-3"
            style={{ fontSize: 'clamp(1.35rem, 3.6vw, 2rem)' }}
          >
            {t({
              en: 'What it repairs alone, and what it hands back',
              ja: '無人で直すもの、人に返すもの',
            })}
          </h2>
          <p
            className="mb-8 max-w-2xl"
            style={{
              color: 'var(--ink-2)',
              fontSize: '0.95rem',
              lineHeight: ja ? 1.95 : 1.7,
            }}
          >
            {t({
              en: 'The engine cannot decide whether its own reading is right — that needs the world. It can decide whether two readings of the same content agree, and if they do not, one of them is wrong. That is a proof, and it is the whole difference between the first two lines below and the third.',
              ja: 'エンジンは「自分の読みが正しいか」を決められません。それは世界を要ります。決められるのは「同じ内容の二つの読みが一致するか」で、一致しなければどちらかが誤りです。これは証明であり、下の最初の二つと三つ目を分けているのはその一点です。',
            })}
          </p>

          <div className="space-y-4">
            {LINES.map((line, i) => (
              <motion.div
                key={line.title.en}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.45, delay: i * 0.06 }}
                className="rounded-2xl p-5 sm:p-6"
                style={{
                  border: '1px solid var(--line, rgba(255,255,255,0.09))',
                  background: 'var(--surface, rgba(255,255,255,0.02))',
                  borderLeft: `3px solid ${
                    line.closed
                      ? 'rgba(var(--accent-rgb), 0.75)'
                      : 'var(--warn, #f59e0b)'
                  }`,
                }}
              >
                <div className="flex flex-wrap items-baseline gap-3 mb-2">
                  <p className="font-semibold" style={{ fontSize: '1rem' }}>
                    {t(line.title)}
                  </p>
                  <span
                    className="rounded-full px-2.5 py-0.5"
                    style={{
                      fontSize: '0.7rem',
                      letterSpacing: '0.04em',
                      color: line.closed
                        ? 'rgba(var(--accent-rgb), 0.95)'
                        : 'var(--warn, #f59e0b)',
                      border: `1px solid ${
                        line.closed
                          ? 'rgba(var(--accent-rgb), 0.4)'
                          : 'rgba(var(--warn-rgb, 245 158 11), 0.45)'
                      }`,
                    }}
                  >
                    {line.closed
                      ? t({ en: 'unattended', ja: '無人' })
                      : t({ en: 'your approval', ja: 'あなたの承認' })}
                  </span>
                </div>
                <p
                  style={{
                    color: 'var(--ink-3, rgba(255,255,255,0.62))',
                    fontSize: '0.9rem',
                    lineHeight: ja ? 1.9 : 1.65,
                  }}
                >
                  {t(line.body)}
                </p>
              </motion.div>
            ))}
          </div>

          <div className="mt-9 flex flex-wrap gap-3">
            <a
              href={ENGINE}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {t({ en: 'Engine on GitHub', ja: 'エンジン (GitHub)' })} →
            </a>
            <a
              href="/vera/"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{
                textDecoration: 'none',
                color: 'var(--ink-2)',
                border: '1px solid var(--line-strong, rgba(255,255,255,0.16))',
              }}
            >
              {t({ en: 'How it works', ja: '仕組みを読む' })}
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
