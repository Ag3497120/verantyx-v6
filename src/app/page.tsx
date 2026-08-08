'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

/* Temporary front page for the 2026 Kumamoto earthquake, aimed at someone
 * arriving from a link with no context. Restore the previous home with a
 * single git revert of the commit that added this file.
 *
 * Two honesty rules the whole page obeys:
 *
 *   every number is a measurement      each one names the corpus it came
 *                                      from, including the 0% row
 *   the "screenshot" is not a bitmap   the board below is the engine's real
 *                                      output on the real ministry PDFs,
 *                                      rendered as HTML and labelled as such.
 *                                      No stock disaster photos: we hold no
 *                                      rights to any, and a page about not
 *                                      guessing should not decorate itself
 *                                      with someone else's suffering.
 */

type L = { en: string; ja: string };

const ENGINE = 'https://github.com/Ag3497120/Verantyx';
const PYPI = 'https://pypi.org/project/verantyx-vera/';

/* The engine's actual findings on MLIT reports #3 → #33 (2026-07-29 →
 * 2026-08-07), reproduced verbatim — municipality, households, dates, file. */
const FINDINGS = [
  { core: '熊本市', n: '約20,970', span: '7/28〜8/3' },
  { core: '天草市', n: '約1,100', span: '7/28〜8/1' },
  { core: '御船町', n: '約2,600', span: '7/28〜8/1' },
  { core: '甲佐町', n: '約500', span: '7/28〜8/7' },
  { core: '南島原市', n: '約439', span: '7/28〜29' },
  { core: '太良町', n: '41', span: '7/28〜29' },
];

const USES: { title: L; body: L; fit: boolean }[] = [
  {
    fit: true,
    title: { en: 'Disaster information desks', ja: '災害対応の情報整理' },
    body: {
      en: 'Bulletin #3 says the water is out; bulletin #33 says restored. Which municipalities changed, which are still contested, and which report said what — without reading 250,000 characters by hand.',
      ja: '第3報は断水、第33報は復旧済。どの市町村が変わり、どこが未確定で、どの報が何を言ったか — 25万字を手で読まずに。',
    },
  },
  {
    fit: true,
    title: { en: 'Ledgers: contracts, permits, assets', ja: '台帳: 契約・権限・資産' },
    body: {
      en: 'Anything named that flips state — valid/expired, running/stopped, open/closed — across documents that disagree. The vocabulary is 12 oppositions and grows by approved proposal.',
      ja: '固有名を持つものが 有効/失効・稼働/停止・営業/休業 と状態を変え、資料どうしが食い違う形。語彙は12対で、承認制で育ちます。',
    },
  },
  {
    fit: true,
    title: { en: 'Auditing agent declarations', ja: 'エージェント申告の監査' },
    body: {
      en: 'An agent that says "sandbox on" while requesting "sandbox off" is two sources contradicting each other about a named thing — structurally the same detection, measured to work on typed declarations.',
      ja: '「サンドボックス有効」と言いながら無効化を要求するエージェントは、固有名についての出典2つの矛盾 — 構造的に同じ検出で、型つき宣言で動くことを実測済み。',
    },
  },
  {
    fit: false,
    title: { en: 'Not: wikis, meeting notes, prose', ja: '不向き: wiki・議事録・散文' },
    body: {
      en: 'Measured on 93 technical documents: 5 findings, 0 true. Abstract nouns recur across unrelated contexts, so comparing two mentions manufactures contradictions. We publish that number on purpose.',
      ja: '技術文書93本で実測: 検出5件、本物0件。抽象名詞は無関係な文脈に再来するので、比較すると存在しない矛盾ができます。この数字は意図して公開しています。',
    },
  },
];

const MECH: { title: L; body: L }[] = [
  {
    title: { en: '1 · Crosses, not embeddings', ja: '1 · ベクトルではなく十字' },
    body: {
      en: 'Each named thing gets one cross: a core and accumulating facets. A state word is stored as aspect:value — 復旧:断水 — so two poles on one aspect surface as a contradiction by structure, not by similarity score.',
      ja: '固有名ごとに十字が1本。状態語は 面:値（復旧:断水）で置かれるので、同じ面に両極が載った瞬間、類似度ではなく構造として矛盾が現れます。',
    },
  },
  {
    title: { en: '2 · A subject gate, measured in', ja: '2 · 主語ゲート(実測で入れた)' },
    body: {
      en: 'A pole lands only when the named thing is the subject of the sentence that carries it. Added after measuring 0-of-4 precision without it; with it, 14 of 14 findings on five disaster corpora were true.',
      ja: '極はその文の主語であるときだけ置かれます。無しでは適合率0/4だったのを実測してから導入し、以後、災害5コーパスで14/14が本物でした。',
    },
  },
  {
    title: { en: '3 · Typed refusal, never a guess', ja: '3 · 型つき拒否、推測はしない' },
    body: {
      en: 'No model anywhere in the answer path. When evidence is missing the answer is UNKNOWN_NO_EVIDENCE — a name for what is missing, not a fluent sentence about it. Same input, same output, offline.',
      ja: '答えの経路にモデルは一切ありません。証拠が無ければ UNKNOWN_NO_EVIDENCE — 欠けているものの名前が返り、流暢な作文は返りません。同じ入力は必ず同じ出力、オフラインで。',
    },
  },
  {
    title: { en: '4 · It repairs its own reader', ja: '4 · 自分の読みを自分で直す' },
    body: {
      en: 'It reads the same documents twice through transforms that cannot change meaning; a claim that appears in only one reading is provably spurious and gets repaired unattended. What a new word means still requires you.',
      ja: '意味を変えられない変換を通して同じ資料を二度読み、片方にしか現れない主張は証明つきで偽 — 無人で修復します。未知語の意味だけは、今もあなたの承認が要ります。',
    },
  },
];

export default function LandingPage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';

  return (
    <main lang={lang} className="relative text-white min-h-screen" style={{ overflowX: 'clip' }}>
      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-12">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              'radial-gradient(ellipse 70% 55% at 30% 22%, rgba(var(--accent-rgb), 0.11), transparent 62%)',
          }}
        />
        <div className="mx-auto w-full max-w-4xl relative z-10">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="uppercase mb-5"
            style={{
              color: 'rgba(var(--accent-rgb), 0.9)',
              fontSize: 'clamp(0.62rem, 1.6vw, 0.72rem)',
              letterSpacing: '0.28em',
            }}
          >
            {t({
              en: 'Kumamoto earthquake · August 2026 · a working example',
              ja: '2026年8月 熊本地震 — 実例で示します',
            })}
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.06 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: ja ? 'clamp(1.9rem, 6.8vw, 3.8rem)' : 'clamp(2.1rem, 7.6vw, 4.4rem)',
              lineHeight: ja ? 1.24 : 1.05,
              letterSpacing: ja ? '-0.01em' : '-0.03em',
              marginBottom: '1.1rem',
            }}
          >
            {t({
              en: '33 bulletins. Which towns have water back?',
              ja: '33本の速報。水が戻った町はどこか。',
            })}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="max-w-2xl"
            style={{
              color: 'var(--ink-2)',
              fontSize: 'clamp(1rem, 2.4vw, 1.18rem)',
              lineHeight: ja ? 1.95 : 1.72,
              fontWeight: 300,
            }}
          >
            {t({
              en: 'Vera reads the actual ministry PDFs and answers with sources — or refuses, with a typed reason. No LLM in the answer path. Deterministic. Offline. Below is its real output on the real documents.',
              ja: 'Vera は省庁の PDF をそのまま読み、出典つきで答えます — 答えられないときは理由の名前つきで拒否します。答えの経路に LLM なし、決定論、オフライン。下は実際の文書に対する実際の出力です。',
            })}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.26 }}
            className="mt-8 flex flex-wrap items-center gap-3"
          >
            <a href="/vera/demo/" className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold" style={{ textDecoration: 'none' }}>
              {t({ en: 'Try it in the browser', ja: 'ブラウザで試す' })} →
            </a>
            <a
              href="/vera/download/"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong, rgba(255,255,255,0.16))' }}
            >
              pip install verantyx-vera
            </a>
          </motion.div>
        </div>
      </section>

      {/* ── The real board ───────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-14">
        <div className="mx-auto w-full max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-80px' }}
            transition={{ duration: 0.55 }}
            className="rounded-2xl overflow-hidden"
            style={{ border: '1px solid var(--line, rgba(255,255,255,0.12))' }}
          >
            {/* window chrome, so it reads as an application */}
            <div
              className="flex items-center gap-2 px-4 py-2.5"
              style={{ background: 'var(--surface-2, rgba(255,255,255,0.05))', borderBottom: '1px solid var(--line, rgba(255,255,255,0.08))' }}
            >
              <span style={{ width: 10, height: 10, borderRadius: 5, background: '#f66' }} />
              <span style={{ width: 10, height: 10, borderRadius: 5, background: '#fb5' }} />
              <span style={{ width: 10, height: 10, borderRadius: 5, background: '#5c5' }} />
              <span className="ml-2" style={{ color: 'var(--ink-3)', fontSize: '0.78rem' }}>
                vera field — 127.0.0.1:8900
              </span>
            </div>
            <div className="p-4 sm:p-6" style={{ background: 'var(--surface, rgba(255,255,255,0.02))' }}>
              <p className="mb-1" style={{ fontSize: '0.86rem', color: 'var(--ink-2)' }}>
                {t({
                  en: 'Input: MLIT bulletins #3, #13, #23, #33 (Jul 29 – Aug 7) · 61,083 characters',
                  ja: '入力: 国交省 第3・13・23・33報（7/29〜8/7）・61,083字',
                })}
              </p>
              <p className="mb-4" style={{ fontSize: '0.86rem', color: 'var(--ink-2)' }}>
                {t({ en: 'Read: 2,319 / 2,701 sentences (85.9%) · 6 comparable pairs · ', ja: '読み取り: 2,319/2,701文（85.9%）・比較可能6組・' })}
                <b style={{ color: 'var(--ink)' }}>{t({ en: '6 findings, 6 true', ja: '検出6件、6件とも本物' })}</b>
              </p>
              <div className="space-y-2">
                {FINDINGS.map((f, i) => (
                  <motion.div
                    key={f.core}
                    initial={{ opacity: 0, x: -8 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true, margin: '-40px' }}
                    transition={{ duration: 0.35, delay: i * 0.06 }}
                    className="rounded-lg px-3 py-2.5 flex flex-wrap items-baseline gap-x-3 gap-y-1"
                    style={{ background: 'var(--surface-2, rgba(255,255,255,0.045))', fontSize: '0.88rem' }}
                  >
                    <b style={{ minWidth: '5.5em' }}>{f.core}</b>
                    <span style={{ color: '#e57373' }}>{t({ en: 'water out', ja: '断水あり' })}</span>
                    <span style={{ color: 'var(--ink-3)' }}>7/29 · mlit_03</span>
                    <span style={{ color: 'var(--ink-3)' }}>→</span>
                    <span style={{ color: '#81c784' }}>{t({ en: 'restored', ja: '復旧済' })}</span>
                    <span style={{ color: 'var(--ink-3)' }}>8/7 · mlit_33</span>
                    <span style={{ color: 'var(--ink-3)', marginLeft: 'auto' }}>
                      {f.n}{t({ en: ' households · ', ja: '戸・' })}{f.span}
                    </span>
                  </motion.div>
                ))}
              </div>
              <p className="mt-4" style={{ fontSize: '0.8rem', color: 'var(--ink-3)', lineHeight: 1.7 }}>
                {t({
                  en: 'This is the engine’s verbatim output rendered as HTML — not a screenshot and not an illustration. Every line names the file it came from, so you can disagree with the engine, which is the only way to find out it is wrong.',
                  ja: 'これはエンジンの出力をそのまま HTML にしたもので、スクリーンショットでも図解でもありません。全行が出典ファイルを名指すので、エンジンに反論できます — それが間違いを見つける唯一の方法です。',
                })}
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-14">
        <div className="mx-auto w-full max-w-4xl">
          <h2 className="font-display font-bold mb-2" style={{ fontSize: 'clamp(1.4rem, 3.8vw, 2.1rem)' }}>
            {t({ en: 'How it works', ja: '仕組み' })}
          </h2>
          <p className="mb-7 max-w-2xl" style={{ color: 'var(--ink-2)', fontSize: '0.95rem', lineHeight: ja ? 1.9 : 1.7 }}>
            {t({
              en: 'The name is the architecture: knowledge lives on stereo crosses, and disagreement is a geometric event.',
              ja: '名前が構造そのものです。知識は立体十字の上に置かれ、食い違いは幾何の事件として現れます。',
            })}
          </p>

          {/* stereo-cross diagram — inline SVG, loads nothing */}
          <div className="mb-8 flex justify-center">
            <svg viewBox="0 0 640 190" role="img" aria-label={t({ en: 'Two sources placing opposite poles on one cross', ja: '2つの出典が1本の十字に反対の極を置く図' })} style={{ width: '100%', maxWidth: 640 }}>
              <defs>
                <linearGradient id="lg" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="640" y2="190">
                  <stop offset="0%" stopColor="#3ec8c8" />
                  <stop offset="100%" stopColor="#7c5cf0" />
                </linearGradient>
              </defs>
              <g transform="translate(320,95)">
                <line x1="-70" y1="0" x2="70" y2="0" stroke="url(#lg)" strokeWidth="2" />
                <line x1="0" y1="-62" x2="0" y2="62" stroke="url(#lg)" strokeWidth="2" />
                <line x1="-48" y1="-38" x2="48" y2="38" stroke="url(#lg)" strokeWidth="1.4" opacity="0.6" />
                <circle r="5" fill="#3b82f6" />
                <text x="0" y="-72" textAnchor="middle" fill="var(--ink, #eee)" fontSize="14" fontWeight="700">熊本市</text>
                <text x="-84" y="4" textAnchor="end" fill="#e57373" fontSize="12.5">復旧:断水 ← mlit_03</text>
                <text x="84" y="4" textAnchor="start" fill="#81c784" fontSize="12.5">復旧:復旧 ← mlit_33</text>
                <text x="58" y="52" fill="var(--ink-3, #888)" fontSize="11">{t({ en: 'facets…', ja: '他の面…' })}</text>
              </g>
              <text x="320" y="182" textAnchor="middle" fill="var(--ink-3, #888)" fontSize="11.5">
                {t({
                  en: 'two poles on one aspect of one core = contested, with both sources attached',
                  ja: '1本のコアの1つの面に両極 = 係争。両方の出典が付いたまま',
                })}
              </text>
            </svg>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {MECH.map((m, i) => (
              <motion.div
                key={m.title.en}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.45, delay: i * 0.05 }}
                className="rounded-2xl p-5"
                style={{ border: '1px solid var(--line, rgba(255,255,255,0.09))', background: 'var(--surface, rgba(255,255,255,0.02))' }}
              >
                <p className="font-semibold mb-2" style={{ fontSize: '0.97rem' }}>{t(m.title)}</p>
                <p style={{ color: 'var(--ink-3)', fontSize: '0.88rem', lineHeight: ja ? 1.9 : 1.65 }}>{t(m.body)}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── What you could use it for ────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-14">
        <div className="mx-auto w-full max-w-4xl">
          <h2 className="font-display font-bold mb-2" style={{ fontSize: 'clamp(1.4rem, 3.8vw, 2.1rem)' }}>
            {t({ en: 'What would you point it at?', ja: '何に向けて使えるか' })}
          </h2>
          <p className="mb-7 max-w-2xl" style={{ color: 'var(--ink-2)', fontSize: '0.95rem', lineHeight: ja ? 1.9 : 1.7 }}>
            {t({
              en: 'The boundary is measured, not guessed: it works where named things change state across disagreeing sources.',
              ja: '境界は推測ではなく実測です。固有名を持つものが状態を変え、出典が食い違う場所で効きます。',
            })}
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            {USES.map((u, i) => (
              <motion.div
                key={u.title.en}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }}
                transition={{ duration: 0.45, delay: i * 0.05 }}
                className="rounded-2xl p-5"
                style={{
                  border: '1px solid var(--line, rgba(255,255,255,0.09))',
                  background: 'var(--surface, rgba(255,255,255,0.02))',
                  borderLeft: `3px solid ${u.fit ? 'rgba(var(--accent-rgb), 0.75)' : 'var(--warn, #f59e0b)'}`,
                }}
              >
                <p className="font-semibold mb-2" style={{ fontSize: '0.97rem' }}>{t(u.title)}</p>
                <p style={{ color: 'var(--ink-3)', fontSize: '0.88rem', lineHeight: ja ? 1.9 : 1.65 }}>{t(u.body)}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── The numbers ──────────────────────────────────────── */}
      <section className="relative px-5 sm:px-6 pb-16">
        <div className="mx-auto w-full max-w-4xl">
          <h2 className="font-display font-bold mb-5" style={{ fontSize: 'clamp(1.4rem, 3.8vw, 2.1rem)' }}>
            {t({ en: 'Measured, including the failure', ja: '実測 — 失敗も含めて' })}
          </h2>
          <div className="rounded-2xl overflow-x-auto" style={{ border: '1px solid var(--line, rgba(255,255,255,0.1))' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ background: 'var(--surface-2, rgba(255,255,255,0.05))' }}>
                  {[
                    t({ en: 'corpus', ja: 'コーパス' }),
                    t({ en: 'findings', ja: '検出' }),
                    t({ en: 'true', ja: '本物' }),
                    t({ en: 'precision', ja: '適合率' }),
                  ].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '0.7rem 1rem', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  [t({ en: 'Government disaster reports (5 corpora, 4 blind)', ja: '官庁災害文書（5コーパス、うち4はブラインド）' }), '14', '14', '100%'],
                  [t({ en: 'Naive keyword baseline, same documents', ja: '素朴なキーワード照合（同じ文書）' }), '38', '6', '16%'],
                  [t({ en: 'Technical prose, 93 documents', ja: '技術文書93本' }), '5', '0', '0%'],
                ].map((row, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--line, rgba(255,255,255,0.07))' }}>
                    {row.map((c, j) => (
                      <td key={j} style={{ padding: '0.7rem 1rem', color: j === 0 ? 'var(--ink-2)' : 'var(--ink)' }}>{c}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3" style={{ color: 'var(--ink-3)', fontSize: '0.82rem', lineHeight: 1.7 }}>
            {t({
              en: 'The 0% row is why the page above says “not for wikis”. Publishing where a tool fails is cheaper than an afternoon of your time finding out.',
              ja: '0%の行が「wiki には不向き」の根拠です。失敗する場所を公開するほうが、あなたの午後より安上がりです。',
            })}
          </p>

          <div className="mt-9 flex flex-wrap gap-3">
            <a href="/vera/demo/" className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold" style={{ textDecoration: 'none' }}>
              {t({ en: 'Try the live demo', ja: 'デモを試す' })} →
            </a>
            <a href={PYPI} target="_blank" rel="noopener noreferrer" className="rounded-xl px-6 py-3 text-sm font-semibold" style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong, rgba(255,255,255,0.16))' }}>
              PyPI
            </a>
            <a href={ENGINE} target="_blank" rel="noopener noreferrer" className="rounded-xl px-6 py-3 text-sm font-semibold" style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong, rgba(255,255,255,0.16))' }}>
              GitHub
            </a>
            <a href="/vera/" className="rounded-xl px-6 py-3 text-sm font-semibold" style={{ textDecoration: 'none', color: 'var(--ink-2)', border: '1px solid var(--line-strong, rgba(255,255,255,0.16))' }}>
              {t({ en: 'Design principles', ja: '設計思想' })}
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
