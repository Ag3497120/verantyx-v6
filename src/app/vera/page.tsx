'use client';

import { motion } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

const ENGINE = 'https://github.com/Ag3497120/Verantyx';
const IDE = 'https://github.com/Ag3497120/Verantyx';

type L = { en: string; ja: string };

/* Every number on this page is a measurement that can be re-run, and each is
 * stated with the corpus it was measured on. A figure without its corpus is
 * exactly the shape of claim this engine exists to refuse. */
const CORPORA: {
  source: L;
  detail: L;
  detections: number;
  recall: string;
  blind: boolean;
}[] = [
  {
    source: { en: 'Cabinet Office damage reports', ja: '内閣府 被害状況速報' },
    detail: { en: '4 revisions · 252,575 characters', ja: '4版 · 252,575字' },
    detections: 8,
    recall: '8 / 8',
    blind: false,
  },
  {
    source: { en: 'MLIT 第N報 series', ja: '国交省 第N報 系列' },
    detail: { en: '4 revisions · 61,083 characters', ja: '4版 · 61,083字' },
    detections: 6,
    recall: '6 / 6',
    blind: true,
  },
];

const FINDINGS: { topic: L; change: L; when: string }[] = [
  {
    topic: { en: 'Kumamoto City · water', ja: '熊本市 · 水道' },
    change: { en: 'out → restored', ja: '断水 → 復旧済' },
    when: '7/29 → 8/6',
  },
  {
    topic: { en: 'Kumamoto Prison · shelter', ja: '熊本刑務所 · 避難所' },
    change: { en: 'opened → closed', ja: '開設 → 閉鎖' },
    when: '8/3 → 8/6',
  },
  {
    topic: { en: 'National highways', ja: '直轄国道' },
    change: { en: 'closed → cleared', ja: '通行止 → 解除' },
    when: '7/29 → 7/31',
  },
  {
    topic: { en: 'Toll roads', ja: '有料道路' },
    change: { en: 'clear → new closure', ja: 'なし → 新規閉塞' },
    when: '8/3 → 8/6',
  },
];

const PRINCIPLES: { title: L; body: L }[] = [
  {
    title: { en: 'A refusal is a type', ja: '拒否は型である' },
    body: {
      en: 'UNKNOWN_NO_EVIDENCE. UNKNOWN_LOW_COVERAGE. UNKNOWN_DOMINANT_SOURCE. Each names what is missing, so the next step is a procedure rather than a matter of taste.',
      ja: 'UNKNOWN_NO_EVIDENCE、UNKNOWN_LOW_COVERAGE、UNKNOWN_DOMINANT_SOURCE。何が欠けているかを名指すので、次の一手が好みではなく手順になります。',
    },
  },
  {
    title: { en: 'An update is not a conflict', ja: '更新は係争ではない' },
    body: {
      en: 'A road closed at 09:00 and open at 15:00 is one story told twice. Showing it as a disagreement is how an information officer stops trusting the board. An unreadable date leaves the dispute standing rather than inventing an update.',
      ja: '9時に通行止、15時に通行可能は、一つの話が2度語られただけです。これを対立として見せることが、情報担当者がこの板を信じなくなる瞬間です。読めない日付は、偽の更新を作らず係争を残します。',
    },
  },
  {
    title: { en: 'Every finding names its line', ja: 'すべての所見が出典の行を名指す' },
    body: {
      en: 'Not a summary of the sources — the sentence itself, with the file it came from. A person can disagree with the engine, which is the only way to find out that it is wrong.',
      ja: '出典の要約ではなく、文そのものと、それが載っていたファイル名です。人がエンジンに反論できます。それが、エンジンが間違っていると分かる唯一の方法です。',
    },
  },
  {
    title: { en: 'No model in the answer path', ja: '答えの経路にモデルを置かない' },
    body: {
      en: 'The same documents always produce the same findings, on a laptop, offline. There is no matrix arithmetic anywhere in it, so there is nothing to be non-deterministic about.',
      ja: '同じ文書からは必ず同じ所見が出ます。ノートPCで、オフラインで。行列演算が一切ないので、非決定的になる余地がありません。',
    },
  },
];

const LIMITS: L[] = [
  {
    en: 'Recall is measured on two corpora, and both are Japanese government disaster reports. A third format may well expose a third layout defect — that is what the blind run is for, and it will be run again.',
    ja: '再現率は2コーパスでの測定で、どちらも日本の官庁災害報告です。3つ目の形式が3つ目のレイアウト欠陥を出す可能性は十分あります。それを見つけるためのブラインド検証であり、今後も繰り返します。',
  },
  {
    en: 'One false positive across 21.6 million characters of mixed English and Japanese prose. It is English, and it is honest: two documents using the word “channels” generically about different situations.',
    ja: '日英混在2,160万字に対して誤検出は1件。英語で、内容も正直なものです。2つの文書が「channels」を一般名詞として、別々の状況について使っています。',
  },
  {
    en: 'The vocabulary does not carry every state word a document might use. The most common one it omits is 障害 — deliberately, because 「障害のある方」 would otherwise read as a system failure, in the documents written for them.',
    ja: '語彙は、文書が使いうるすべての状態語を持っているわけではありません。最も多い未収録語は「障害」で、これは意図的です。入れると「障害のある方」がシステム障害と読まれ、しかもそれは障害のある方のために書かれた文書の中で起きます。',
  },
  {
    en: 'It does not write. No free-form prose, no summarisation, no translation, no open-domain chat. That is not a weakness being worked on — it is the trade that buys everything above.',
    ja: '文章を書きません。自由作文も要約も翻訳も雑談もしません。これは改善中の弱点ではなく、上のすべてを買うための取引です。',
  },
];

/* The jgen static dictionary. Documented here rather than in a feature list
 * because the interesting part is the third row: the question it REFUSES to
 * answer, and the measurement that put it out of reach. */
const LEXICON: { q: L; verdict: L; measured: L; ok: boolean }[] = [
  {
    ok: true,
    q: { en: 'Is this the kind of word that can carry a state?',
         ja: 'この語は状態を担える種類の語か' },
    verdict: { en: 'Usable', ja: '使える' },
    measured: {
      en: 'Separated the real proposal queue completely: true candidates at +0.164 / +0.128 / +0.082, false ones at −0.143 / −0.239. Unseen state words (滞留, 孤立, 冠水) landed on the right side too.',
      ja: '実際の候補列を完全に分離。本物が +0.164 / +0.128 / +0.082、偽物が −0.143 / −0.239。未知の状態語（滞留・孤立・冠水）も正しい側に落ちました。',
    },
  },
  {
    ok: true,
    q: { en: 'Which known words sit nearest to it?',
         ja: 'どの既知語がいちばん近いか' },
    verdict: { en: 'Usable as search', ja: '検索として使える' },
    measured: {
      en: '冠水 → 断水 (0.52), 停電 → 停止 (0.47). Shown to the operator as context beside a proposal, never as a decision.',
      ja: '冠水 → 断水 (0.52)、停電 → 停止 (0.47)。提案の横に文脈として示すだけで、判断はしません。',
    },
  },
  {
    ok: false,
    q: { en: 'Which pole is it — restored, or still out?',
         ja: 'どちらの極か — 解消側か、継続側か' },
    verdict: { en: 'Refused. Absent from the API.', ja: '拒否。API に存在しません' },
    measured: {
      en: '64.5% leave-one-out on the engine\u2019s own 31 terms — a coin flip. Opposite poles live in identical contexts: an outage and its restoration share a paragraph, so a frozen table holds no information that separates them. A 4B model scored 54.8% on the same test. Neither is usable, and no function returns a pole.',
      ja: 'エンジン自身の31語で leave-one-out 64.5% — ほぼコイン投げです。反対の極は同じ文脈に現れます（「断水が発生」と「復旧が完了」は同じ段落）ので、凍結された表には区別する情報がありません。4B のモデルは同じ試験で 54.8% でした。どちらも使い物にならず、極を返す関数は存在しません。',
    },
  },
];

export default function VeraPage() {
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
      <section className="relative px-5 sm:px-6 pt-28 sm:pt-36 pb-16 sm:pb-24">
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
            Vera-α
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.06 }}
            className="font-display font-extrabold gradient-brand"
            style={{
              fontSize: ja
                ? 'clamp(2rem, 7.5vw, 4.25rem)'
                : 'clamp(2.4rem, 9vw, 5.25rem)',
              lineHeight: ja ? 1.22 : 1.02,
              letterSpacing: ja ? '-0.01em' : '-0.03em',
              marginBottom: '1.1rem',
            }}
          >
            {t({ en: 'It refuses to guess', ja: '推測しないエンジン' })}
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
              en: 'Pour documents in. Vera separates what the sources agree on, what changed, what they disagree about, and what nobody answered — and never blends the four.',
              ja: '文書を投入すると、Vera は「一致していること」「変わったこと」「食い違っていること」「誰も答えていないこと」を分けます。決して混ぜません。',
            })}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.28 }}
            className="mt-8 sm:mt-11 flex flex-wrap items-center gap-3"
          >
            <a
              href="/vera/demo/"
              className="btn-accent rounded-xl px-6 py-3 text-sm font-semibold"
              style={{ textDecoration: 'none' }}
            >
              {t({ en: 'Try it in the browser', ja: 'ブラウザで試す' })} →
            </a>
            <a
              href={ENGINE}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-6 py-3 text-sm font-semibold"
              style={{
                textDecoration: 'none',
                color: 'var(--ink-2)',
                border: '1px solid var(--line-strong, rgba(255,255,255,0.16))',
              }}
            >
              {t({ en: 'Engine on GitHub', ja: 'エンジン (GitHub)' })}
            </a>
            <a
              href={IDE}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl px-6 py-3 text-sm font-semibold text-slate-300"
              style={{
                border: '1px solid var(--line-strong)',
                textDecoration: 'none',
              }}
            >
              {t({ en: 'macOS IDE', ja: 'macOS IDE' })}
            </a>
          </motion.div>
        </div>
      </section>

      {/* ── The difference ─────────────────────────────────────── */}
      <Section label={t({ en: 'The difference', ja: '違い' })}>
        <H2 ja={ja}>
          {t({
            en: 'Asked about something it has never read',
            ja: '読んだことのないものを聞かれたとき',
          })}
        </H2>
        <div
          className="rounded-xl border overflow-x-auto"
          style={{
            borderColor: 'rgba(var(--accent-rgb), 0.18)',
            background: 'var(--surface-2)',
          }}
        >
          <pre
            className="px-4 py-4 sm:px-6 sm:py-5 font-mono"
            style={{
              fontSize: 'clamp(0.75rem, 2.2vw, 0.875rem)',
              lineHeight: 1.75,
              color: 'var(--ink-2)',
              margin: 0,
            }}
          >
{`> 避難所
UNKNOWN_NO_EVIDENCE
no_candidate_cross`}
          </pre>
        </div>
        <Body ja={ja} className="mt-6">
          {t({
            en: 'A language model asked the same question writes a paragraph about shelters, because writing a paragraph is what it does. Vera has nothing stored, so it says which kind of nothing.',
            ja: '同じ問いを言語モデルにすれば、避難所についての段落が返ります。段落を作るのがその仕事だからです。Vera は何も保存していないので、どの種類の「無い」なのかを言います。',
          })}
        </Body>
      </Section>

      {/* ── Measured ───────────────────────────────────────────── */}
      <Section id="measured" label={t({ en: 'Measured', ja: '実測値' })}>
        <H2 ja={ja}>
          {t({ en: 'On real published documents', ja: '実在の公開文書で' })}
        </H2>
        <Body ja={ja} className="mb-8 sm:mb-10">
          {t({
            en: 'A planted test corpus is graded by whoever wrote it. These are two government report series about the same live disaster, read revision by revision, and every finding was checked against its source by a person.',
            ja: '仕込みのテストコーパスは、書いた本人が採点します。以下は同じ災害についての2つの官庁報告系列を版ごとに読ませたもので、全所見を人が原文と照合しています。',
          })}
        </Body>

        <div className="grid gap-4 sm:gap-5 sm:grid-cols-2 mb-10 sm:mb-12">
          {CORPORA.map((c) => (
            <div
              key={c.source.en}
              className="rounded-2xl border p-5 sm:p-6"
              style={{
                borderColor: c.blind
                  ? 'rgba(var(--accent-rgb), 0.32)'
                  : 'var(--line)',
                background: 'var(--surface)',
              }}
            >
              <div className="flex items-start justify-between gap-3 mb-1">
                <div
                  className="font-semibold"
                  style={{ fontSize: 'clamp(0.95rem, 2.4vw, 1.06rem)' }}
                >
                  {t(c.source)}
                </div>
                {c.blind && (
                  <span
                    className="shrink-0 uppercase rounded-full px-2.5 py-1"
                    style={{
                      fontSize: '0.6rem',
                      letterSpacing: '0.16em',
                      color: 'rgba(var(--accent-rgb), 0.95)',
                      border: '1px solid rgba(var(--accent-rgb), 0.32)',
                    }}
                  >
                    {t({ en: 'blind', ja: 'ブラインド' })}
                  </span>
                )}
              </div>
              <div
                className="text-slate-500 mb-6"
                style={{ fontSize: 'clamp(0.72rem, 2vw, 0.8rem)' }}
              >
                {t(c.detail)}
              </div>
              <div className="flex items-end gap-7 sm:gap-9">
                <Stat
                  value={c.recall}
                  label={t({ en: 'recall', ja: '再現率' })}
                  accent
                />
                <Stat
                  value={String(c.detections)}
                  label={t({ en: 'findings · 0 false', ja: '所見 · 誤り0' })}
                />
              </div>
            </div>
          ))}
        </div>

        <Body ja={ja} className="mb-6">
          {t({
            en: 'Recall has a denominator here, which is the part usually missing from such a claim: the water table names every affected municipality on both dates, so it is an answer key. It was read by hand, then compared.',
            ja: 'ここには再現率の分母があります。通常この種の主張でいちばん欠けている部分です。水道の表は影響自治体を両方の日付で名前ごとに挙げているので、それ自体が正解表です。手で読んでから突き合わせました。',
          })}
        </Body>

        <ul
          className="rounded-2xl border divide-y"
          style={{
            borderColor: 'var(--line)',
            // Tailwind's divide utility cannot see a CSS variable, so the
            // rule colour is set here instead of guessed from the palette.
            ['--tw-divide-opacity' as string]: '1',
          }}
        >
          {FINDINGS.map((f, i) => (
            <li
              key={i}
              className="px-4 sm:px-5 py-3.5 sm:py-4 flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4"
              style={{
                borderTop:
                  i === 0 ? 'none' : '1px solid rgba(148,163,184,0.11)',
              }}
            >
              <span
                className="font-semibold sm:min-w-[13rem]"
                style={{ fontSize: 'clamp(0.88rem, 2.3vw, 0.98rem)' }}
              >
                {t(f.topic)}
              </span>
              <span
                className="text-slate-300 flex-1"
                style={{ fontSize: 'clamp(0.85rem, 2.2vw, 0.95rem)' }}
              >
                {t(f.change)}
              </span>
              <span
                className="text-slate-500 font-mono shrink-0"
                style={{ fontSize: 'clamp(0.72rem, 2vw, 0.8rem)' }}
              >
                {f.when}
              </span>
            </li>
          ))}
        </ul>

        <Body ja={ja} className="mt-5" muted>
          {t({
            en: 'All four are reported as updates with dates, not as conflicts. The controls hold as well: municipalities still without water on the last revision are reported as still without water, never as restored.',
            ja: 'いずれも係争ではなく、日付つきの更新として報告されます。対照も成立しています。最終版でも断水が続く自治体は、そのとおり報告され、復旧とは一度も報告されません。',
          })}
        </Body>
      </Section>

      {/* ── Generalisation ─────────────────────────────────────── */}
      <Section label={t({ en: 'Generalisation', ja: '一般化' })}>
        <H2 ja={ja}>
          {t({
            en: 'The second corpus was the test',
            ja: '2つ目のコーパスが試験だった',
          })}
        </H2>
        <Body ja={ja} className="mb-5">
          {t({
            en: 'Every reading rule had been derived from one ministry’s format. Whether that generalises is not something you can reason about — so a second agency’s series, in its own format, was ingested with no code changes and read only afterwards.',
            ja: '読み取り規則はすべて1つの省庁の形式から導いたものでした。それが一般化するかは、考えて分かることではありません。そこで別機関の系列を、独自形式のまま、コードを1行も変えずに投入し、読んだのは後にしました。',
          })}
        </Body>
        <Body ja={ja}>
          {t({
            en: 'Five of six landed on the first try. The sixth exposed two layout defects — a full-width table row read as wrapped prose, and a thousands separator typed as a period. Both were fixed structurally, neither fix mentions an agency, and both made the first corpus read slightly better too. That is what a blind run is for.',
            ja: '6件中5件は初回で当たりました。残る1件がレイアウト欠陥を2つ露出させます。全幅の表行を折り返し本文と誤読したこと、千位区切りがピリオドで打たれていたこと。どちらも構造的に修正され、機関名は一切含まず、しかも1つ目のコーパスの読みも少し良くなりました。ブラインド検証はそのためにあります。',
          })}
        </Body>
      </Section>

      {/* ── Principles ─────────────────────────────────────────── */}
      <Section label={t({ en: 'How it behaves', ja: 'ふるまい' })}>
        <div className="grid gap-4 sm:gap-5 sm:grid-cols-2">
          {PRINCIPLES.map((p) => (
            <div
              key={p.title.en}
              className="rounded-2xl border p-5 sm:p-6"
              style={{
                borderColor: 'var(--line)',
                background: 'var(--surface)',
              }}
            >
              <h3
                className="font-semibold mb-3"
                style={{
                  fontSize: 'clamp(0.98rem, 2.5vw, 1.1rem)',
                  lineHeight: ja ? 1.6 : 1.35,
                }}
              >
                {t(p.title)}
              </h3>
              <p
                className="text-slate-400"
                style={{
                  fontSize: 'clamp(0.85rem, 2.2vw, 0.92rem)',
                  lineHeight: ja ? 1.95 : 1.68,
                }}
              >
                {t(p.body)}
              </p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── The static dictionary ──────────────────────────────── */}
      <Section label={t({ en: 'Static dictionary', ja: '静的辞書' })}>
        <H2 ja={ja}>
          {t({
            en: 'A model used as a dictionary, and only where it measured usable',
            ja: 'モデルを辞書として使う — 測って使えた範囲だけ',
          })}
        </H2>
        <Body ja={ja}>
          {t({
            en: 'Vera can grow its vocabulary from the documents themselves, and a person approves each word. To put the likely-real candidates in front of that person first, it can consult a jgen — a local model file converted with `--parts lexicon`, carrying its embedding table and nothing else. It has no layers that generate, so it physically cannot write. It is opened once and read a row at a time: pure standard library, no inference engine, no network.',
            ja: 'Vera は文書自身から語彙を育て、一語ずつ人が承認します。その人の前に「本物らしい候補」を先に置くために、jgen を引きます — `--parts lexicon` で変換したローカルのモデルファイルで、埋め込み表だけを持ち、生成する層を持ちません。物理的に文章を書けません。一度開いて必要な行だけを読む、標準ライブラリのみ・推論エンジンなし・通信なしの実装です。',
          })}
        </Body>

        <div className="mt-8 space-y-4 max-w-3xl">
          {LEXICON.map((row) => (
            <div
              key={row.q.en}
              className="rounded-2xl p-5"
              style={{
                border: '1px solid var(--line, rgba(255,255,255,0.09))',
                background: 'var(--surface, rgba(255,255,255,0.02))',
                borderLeft: `3px solid ${
                  row.ok ? 'rgba(var(--accent-rgb), 0.75)' : 'var(--warn, #f59e0b)'
                }`,
              }}
            >
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-2">
                <p className="font-semibold" style={{ fontSize: '0.97rem' }}>
                  {t(row.q)}
                </p>
                <span
                  className="rounded-full px-2.5 py-0.5"
                  style={{
                    fontSize: '0.7rem',
                    letterSpacing: '0.04em',
                    color: row.ok
                      ? 'rgba(var(--accent-rgb), 0.95)'
                      : 'var(--warn, #f59e0b)',
                    border: `1px solid ${
                      row.ok
                        ? 'rgba(var(--accent-rgb), 0.4)'
                        : 'rgba(var(--warn-rgb, 245 158 11), 0.5)'
                    }`,
                  }}
                >
                  {t(row.verdict)}
                </span>
              </div>
              <p
                className="text-slate-400"
                style={{
                  fontSize: 'clamp(0.85rem, 2.2vw, 0.9rem)',
                  lineHeight: ja ? 1.9 : 1.65,
                }}
              >
                {t(row.measured)}
              </p>
            </div>
          ))}
        </div>

        <div
          className="mt-6 rounded-2xl p-5 max-w-3xl"
          style={{
            border: '1px solid rgba(var(--accent-rgb), 0.3)',
            background: 'rgba(var(--accent-rgb), 0.05)',
          }}
        >
          <Body ja={ja}>
            {t({
              en: 'The dictionary orders the queue. It never accepts a word — that stays with the person, and it stays there because of the third row, not out of caution. It is optional too: without one configured, the queue simply arrives unsorted, and nothing else changes.',
              ja: '辞書は並べ替えるだけで、語を受理しません。受理は人のもので、その理由は用心ではなく3行目の実測です。設定は任意で、辞書が無ければ候補列が並べ替えられないだけ、他は何も変わりません。',
            })}
          </Body>
        </div>
      </Section>

      {/* ── Limits ─────────────────────────────────────────────── */}
      <Section label={t({ en: 'Limits', ja: '限界' })}>
        <H2 ja={ja}>
          {t({
            en: 'Stated here, not left to be discovered',
            ja: '見つけられる前に、ここに書く',
          })}
        </H2>
        <ul className="space-y-5 max-w-2xl">
          {LIMITS.map((l, i) => (
            <li key={i} className="flex gap-3.5">
              <span
                aria-hidden
                className="mt-2.5 shrink-0 rounded-full"
                style={{
                  width: 5,
                  height: 5,
                  background: 'rgba(var(--accent-rgb), 0.55)',
                }}
              />
              <Body ja={ja}>{t(l)}</Body>
            </li>
          ))}
        </ul>
      </Section>

      <Footer />
    </main>
  );
}

/* ── Small building blocks ──────────────────────────────────── */

function Stat({
  value,
  label,
  accent,
}: {
  value: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <div>
      <div
        className={`font-display font-bold ${accent ? 'gradient-brand' : ''}`}
        style={{ fontSize: 'clamp(1.6rem, 5.5vw, 2.1rem)', lineHeight: 1.1 }}
      >
        {value}
      </div>
      <div
        className="text-slate-500 mt-1.5"
        style={{ fontSize: 'clamp(0.66rem, 1.9vw, 0.74rem)' }}
      >
        {label}
      </div>
    </div>
  );
}

function H2({ ja, children }: { ja: boolean; children: React.ReactNode }) {
  return (
    <h2
      className="font-display font-bold"
      style={{
        fontSize: ja
          ? 'clamp(1.4rem, 4.4vw, 2.1rem)'
          : 'clamp(1.55rem, 5vw, 2.35rem)',
        lineHeight: ja ? 1.45 : 1.15,
        letterSpacing: ja ? '0' : '-0.02em',
        marginBottom: '1.25rem',
      }}
    >
      {children}
    </h2>
  );
}

function Body({
  ja,
  children,
  className = '',
  muted,
}: {
  ja: boolean;
  children: React.ReactNode;
  className?: string;
  muted?: boolean;
}) {
  return (
    <p
      className={`${muted ? 'text-slate-500' : 'text-slate-400'} max-w-2xl ${className}`}
      style={{
        fontSize: muted
          ? 'clamp(0.8rem, 2.1vw, 0.875rem)'
          : 'clamp(0.9rem, 2.3vw, 1rem)',
        lineHeight: ja ? 1.95 : 1.72,
      }}
    >
      {children}
    </p>
  );
}

function Section({
  id,
  label,
  children,
}: {
  id?: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="relative px-5 sm:px-6 py-12 sm:py-16 md:py-20">
      <div className="mx-auto w-full max-w-4xl">
        <div className="flex items-center gap-4 mb-6 sm:mb-8">
          <span
            className="h-px w-8 sm:w-12 shrink-0"
            style={{
              background:
                'linear-gradient(90deg, rgba(var(--accent-rgb), 0.5), transparent)',
            }}
          />
          <span
            className="uppercase text-slate-500 font-semibold"
            style={{ fontSize: '0.66rem', letterSpacing: '0.26em' }}
          >
            {label}
          </span>
        </div>
        {children}
      </div>
    </section>
  );
}
