'use client';

import { useEffect, useState, type FormEvent } from 'react';
import type { HomeLang, L } from './useHomeLang';

/* The /try/ surface (also embedded on the home page).
 *
 * Two modes, decided at build time:
 *   - NEXT_PUBLIC_VERA_API set   → POST the question to that URL (functions/api/vera/ask.js relays it)
 *   - unset (the published build) → replay public/vera-demo/recorded.json, which holds the real
 *     output of the commands, recorded once in the development tree
 * Every verdict and answer on screen is taken from the recorded (or returned) JSON. Nothing here
 * writes an answer of its own, and a question that was not recorded is not "run" at all. */

const API = process.env.NEXT_PUBLIC_VERA_API || '';
const RECORD_URL = '/vera-demo/recorded.json';

/* The fixed document. Must equal recorded.json document.sentences (checked on load). */
const SENTENCES = [
  '先生が生徒に地図を渡した。',
  '校長が職員室で書類を確認した。',
  '母が弟に話した人を兄が呼んだ。',
  '係が荷物を倉庫に運んだ。',
];
const QUESTIONS = [
  '誰が生徒に地図を渡した？',
  '兄は誰を呼んだ？',
  '係は荷物をどこに運んだ？',
  '校長はどこで書類を確認した？',
];
const DOCUMENT_TEXT = SENTENCES.map((s) => s + '\n').join('');

type Json = unknown;
type AskView = {
  verdict: string | null;
  answer: string | null;
  values: string[] | null;
  evidence: string[] | null;
  policy: string | null;
  basis: string | null;
  reason: string | null;
  how_to_resolve: string | null;
};
type Clause = {
  predicate: string;
  roles: Record<string, string>;
  polarity?: string | null;
  tense?: string | null;
  modality?: string | null;
  voice?: string | null;
};
type Unsupported = { predicate?: string; reasons?: string[] };
type ReadView = {
  readable: boolean | null;
  clauses: Clause[] | null;
  relations: Json[] | null;
  abstain: Json;
  unsupported: Unsupported[] | null;
  events_status: string | null;
};
type RecordedAsk = { question: string; rc: number; raw: Json; view: AskView };
type RecordedRead = { text: string; rc: number; raw: Json; view: ReadView };
type Recorded = {
  recorded_on: string;
  placement: string;
  document: { name: string; text: string; sentences: string[] };
  asks: RecordedAsk[];
  reads: RecordedRead[];
};

type Shown =
  | { kind: 'ask'; source: 'record'; item: RecordedAsk }
  | { kind: 'ask'; source: 'api'; question: string; raw: Json; view: AskView }
  | { kind: 'api_error'; question: string; why: string }
  | { kind: 'pending'; question: string }
  | { kind: 'not_recorded'; question: string }
  | { kind: 'read'; item: RecordedRead };

/* Same projection as the recorder's ask_view, applied to a live API answer. */
function viewOf(raw: Json): AskView {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
  const bp = (r.basis_policy && typeof r.basis_policy === 'object' ? r.basis_policy : {}) as Record<
    string,
    unknown
  >;
  const s = (v: unknown) => (typeof v === 'string' ? v : null);
  const arr = (v: unknown) => (Array.isArray(v) ? v.map((x) => String(x)) : null);
  return {
    verdict: s(r.verdict),
    answer: s(r.text),
    values: arr(r.values),
    evidence: arr(r.evidence),
    policy: s(bp.outcome),
    basis: s(bp.basis),
    reason: s(r.reason),
    how_to_resolve: s(r.how_to_resolve),
  };
}

const T = {
  badgeRecord: { ja: '記録済みの実行結果（2026-10-04、配置 r8）', en: 'Recorded run output (2026-10-04, placement r8)' },
  badgeApi: { ja: 'Vera API に問い合わせています', en: 'Asking the Vera API' },
  badgeNoteRecord: {
    ja: 'この画面はいま Vera を動かしていません。開発ツリーで実際に実行した出力を、そのまま再生しています。',
    en: 'Vera is not running behind this page. It replays, unchanged, the output of real runs in the development tree.',
  },
  badgeNoteApi: {
    ja: '問いはサイトの中継を通って Vera のサーバーに送られます。',
    en: 'Questions go through the site relay to a Vera server.',
  },
  doc: { ja: '文書（固定の 4 文）', en: 'Document (four fixed sentences)' },
  readBtn: { ja: '読解を見る', en: 'Show reading' },
  questions: { ja: '問い', en: 'Questions' },
  free: { ja: '自由入力', en: 'Your own question' },
  freePh: { ja: '例: 誰が生徒に地図を渡した？', en: 'e.g. 誰が生徒に地図を渡した？' },
  ask: { ja: '聞く', en: 'Ask' },
  loading: { ja: '記録を読み込んでいます…', en: 'Loading the record…' },
  loadFail: { ja: '記録を読み込めません', en: 'Cannot load the record' },
  mismatch: {
    ja: '記録の文書がこの画面の文書と一致しません。結果を表示しません。',
    en: 'The recorded document does not match this page. Nothing is shown.',
  },
  pick: { ja: '問いを選ぶか、文の「読解を見る」を押してください。', en: 'Pick a question, or press “Show reading” on a sentence.' },
  pending: { ja: '問い合わせ中…', en: 'Asking…' },
  apiError: { ja: 'API が使えません', en: 'API unavailable' },
  apiErrorNote: {
    ja: '記録の再生には切り替えていません。表示しているのは API の失敗の型だけです。',
    en: 'This has not fallen back to the recording. Only the failure type is shown.',
  },
  notRecorded: {
    ja: '記録に無い入力は手元の vera chat で試せます。この画面は記録に無い問いを実行しません。',
    en: 'Input that is not in the record can be tried with vera chat locally. This page does not run unrecorded questions.',
  },
  verdict: { ja: '判定', en: 'Verdict' },
  answered: { ja: '答えた', en: 'answered' },
  abstained: { ja: '型つきで黙った', en: 'abstained, with a type' },
  otherVerdict: { ja: 'その他の判定', en: 'other verdict' },
  answer: { ja: '答え', en: 'Answer' },
  evidence: { ja: '証拠の文', en: 'Evidence' },
  policy: { ja: '方針', en: 'Policy' },
  whyText: { ja: '黙った理由（Vera の文）', en: 'Why it abstained (Vera’s sentence)' },
  reason: { ja: '理由の型', en: 'Reason' },
  resolve: { ja: '解消の手がかり', en: 'How to resolve' },
  raw: { ja: '生の JSON（加工なし）', en: 'Raw JSON (unmodified)' },
  fromRecord: { ja: '記録から', en: 'from the record' },
  fromApi: { ja: 'API から', en: 'from the API' },
  readTitle: { ja: '読解', en: 'Reading' },
  readNoteApi: {
    ja: '読解は API を通さず、記録済みの実行結果を表示しています。',
    en: 'Readings do not go through the API; this shows the recorded run.',
  },
  readable: { ja: '読めた', en: 'readable' },
  unreadable: { ja: '読めない', en: 'not readable' },
  events: { ja: '事象の十字', en: 'Event cross' },
  predicate: { ja: '述語', en: 'Predicate' },
  roles: { ja: '役割', en: 'Roles' },
  form: { ja: '極性・時制・態', en: 'Polarity · tense · voice' },
  relations: { ja: '節の関係', en: 'Clause relations' },
  unsupported: { ja: '型で決めきれなかったところ', en: 'What the types could not settle' },
  question: { ja: '問い', en: 'Question' },
  fullscreen: { ja: '全画面で試す', en: 'Open full page' },
} satisfies Record<string, L>;

const ROLE_LABEL: Record<string, L> = {
  agent: { ja: '誰が', en: 'who' },
  patient: { ja: '何を・誰を', en: 'what / whom' },
  recipient: { ja: '誰に', en: 'to whom' },
  place: { ja: 'どこで', en: 'where' },
  goal: { ja: 'どこへ', en: 'to where' },
  source: { ja: 'どこから', en: 'from where' },
  time: { ja: 'いつ', en: 'when' },
};

const CMD_CHAT = 'python -m verantyx.cli chat --mode round5 --document <ファイル>';
const CMD_ASK = 'python -m verantyx.cli ask --mode round5 --document <ファイル> -- <問い>';

export default function TryVera({ view, showFullLink }: { view: HomeLang; showFullLink?: boolean }) {
  const t = (o: L) => o[view];
  const ja = view === 'ja';
  const [rec, setRec] = useState<Recorded | null>(null);
  const [recState, setRecState] = useState<'loading' | 'ok' | 'fail' | 'mismatch'>('loading');
  const [shown, setShown] = useState<Shown | null>(null);
  const [free, setFree] = useState('');

  useEffect(() => {
    let alive = true;
    fetch(RECORD_URL)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((j: Recorded) => {
        if (!alive) return;
        const same =
          Array.isArray(j?.document?.sentences) &&
          j.document.sentences.length === SENTENCES.length &&
          j.document.sentences.every((s, i) => s === SENTENCES[i]);
        setRec(j);
        setRecState(same ? 'ok' : 'mismatch');
      })
      .catch(() => {
        if (alive) setRecState('fail');
      });
    return () => {
      alive = false;
    };
  }, []);

  function fromRecord(q: string): Shown {
    const item = rec?.asks.find((a) => a.question === q);
    return item ? { kind: 'ask', source: 'record', item } : { kind: 'not_recorded', question: q };
  }

  async function askApi(q: string) {
    setShown({ kind: 'pending', question: q });
    try {
      const r = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, document: DOCUMENT_TEXT }),
      });
      let body: Json = null;
      try {
        body = await r.json();
      } catch {
        body = null;
      }
      if (r.status !== 200) {
        const why =
          body && typeof body === 'object' && typeof (body as { why?: unknown }).why === 'string'
            ? (body as { why: string }).why
            : `http_${r.status}`;
        setShown({ kind: 'api_error', question: q, why });
        return;
      }
      setShown({ kind: 'ask', source: 'api', question: q, raw: body, view: viewOf(body) });
    } catch {
      setShown({ kind: 'api_error', question: q, why: 'network_error' });
    }
  }

  function ask(q: string) {
    const question = q.trim();
    if (!question) return;
    if (API) {
      void askApi(question);
      return;
    }
    if (recState !== 'ok') return;
    setShown(fromRecord(question));
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    ask(free);
  }

  function showRead(i: number) {
    if (recState !== 'ok' || !rec) return;
    const item = rec.reads.find((r) => r.text === SENTENCES[i]);
    if (item) setShown({ kind: 'read', item });
  }

  const recordReady = recState === 'ok';
  const canAsk = API ? true : recordReady;

  return (
    <div className="w-full">
      {/* Always rendered statically, so the label is in the exported HTML. */}
      <div
        className="rounded-xl border px-4 py-3 sm:px-5 mb-5"
        style={{
          borderColor: API ? 'rgba(var(--accent-rgb), 0.4)' : 'rgba(var(--warn-rgb) / 0.45)',
          background: API ? 'rgba(var(--accent-rgb), 0.06)' : 'rgba(var(--warn-rgb) / 0.07)',
        }}
      >
        <div className="font-semibold" style={{ color: API ? 'var(--accent)' : 'var(--warn)', fontSize: '0.92rem' }}>
          {API ? t(T.badgeApi) : t(T.badgeRecord)}
        </div>
        <div className="mt-1" style={{ color: 'var(--ink-3)', fontSize: '0.82rem', lineHeight: ja ? 1.8 : 1.6 }}>
          {API ? t(T.badgeNoteApi) : t(T.badgeNoteRecord)}
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <div className="min-w-0">
          <Label>{t(T.doc)}</Label>
          <ol className="flex flex-col gap-2">
            {SENTENCES.map((s, i) => (
              <li
                key={s}
                className="rounded-lg border px-3 py-2.5 flex flex-wrap items-center justify-between gap-2"
                style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}
              >
                <span style={{ color: 'var(--ink)', fontSize: '0.92rem' }}>{s}</span>
                <button
                  type="button"
                  onClick={() => showRead(i)}
                  disabled={!recordReady}
                  className="rounded-md px-2.5 py-1 shrink-0 disabled:opacity-40"
                  style={{
                    fontSize: '0.74rem',
                    color: 'var(--ink-3)',
                    border: '1px solid var(--line-strong)',
                    cursor: recordReady ? 'pointer' : 'default',
                  }}
                >
                  {t(T.readBtn)}
                </button>
              </li>
            ))}
          </ol>
        </div>

        <div className="min-w-0">
          <Label>{t(T.questions)}</Label>
          <div className="flex flex-col gap-2">
            {QUESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => ask(q)}
                disabled={!canAsk}
                className="text-left rounded-lg border px-3 py-2.5 disabled:opacity-40"
                style={{
                  borderColor: 'rgba(var(--accent-rgb), 0.28)',
                  background: 'rgba(var(--accent-rgb), 0.05)',
                  color: 'var(--ink)',
                  fontSize: '0.92rem',
                  cursor: canAsk ? 'pointer' : 'default',
                }}
              >
                {q}
              </button>
            ))}
          </div>
          <form onSubmit={onSubmit} className="mt-3 flex gap-2">
            <label className="sr-only" htmlFor="vera-free">
              {t(T.free)}
            </label>
            <input
              id="vera-free"
              value={free}
              onChange={(e) => setFree(e.target.value)}
              placeholder={t(T.freePh)}
              maxLength={200}
              className="min-w-0 flex-1 rounded-lg border px-3 py-2"
              style={{
                borderColor: 'var(--line-strong)',
                background: 'var(--surface-2)',
                color: 'var(--ink)',
                fontSize: '0.9rem',
              }}
            />
            <button
              type="submit"
              disabled={!canAsk}
              className="btn-accent rounded-lg px-4 py-2 shrink-0 font-semibold disabled:opacity-40"
              style={{ fontSize: '0.85rem' }}
            >
              {t(T.ask)}
            </button>
          </form>
        </div>
      </div>

      <div
        className="mt-5 rounded-2xl border p-4 sm:p-6 min-w-0"
        style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}
        aria-live="polite"
      >
        <Result
          view={view}
          shown={shown}
          recState={recState}
        />
      </div>

      {showFullLink && (
        <div className="mt-4">
          <a
            href="/try/"
            className="inline-block rounded-xl px-5 py-2.5 text-sm font-semibold"
            style={{ color: 'var(--ink-2)', border: '1px solid var(--line-strong)', textDecoration: 'none' }}
          >
            {t(T.fullscreen)} →
          </a>
        </div>
      )}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="uppercase font-semibold mb-2.5"
      style={{ fontSize: '0.66rem', letterSpacing: '0.2em', color: 'var(--ink-4)' }}
    >
      {children}
    </div>
  );
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 py-2 border-t" style={{ borderColor: 'var(--line)' }}>
      <div className="shrink-0 sm:w-40" style={{ color: 'var(--ink-4)', fontSize: '0.78rem' }}>
        {k}
      </div>
      <div className="min-w-0 break-words" style={{ color: 'var(--ink)', fontSize: '0.9rem' }}>
        {children}
      </div>
    </div>
  );
}

function Raw({ view, raw }: { view: HomeLang; raw: unknown }) {
  return (
    <details className="mt-4">
      <summary className="cursor-pointer" style={{ color: 'var(--ink-3)', fontSize: '0.8rem' }}>
        {T.raw[view]}
      </summary>
      <div className="mt-2 rounded-lg border overflow-x-auto max-h-96 overflow-y-auto" style={{ borderColor: 'var(--line)', background: 'var(--surface-2)' }}>
        <pre className="p-3 font-mono" style={{ fontSize: '0.72rem', lineHeight: 1.55, color: 'var(--ink-2)', margin: 0 }}>
          {JSON.stringify(raw, null, 2)}
        </pre>
      </div>
    </details>
  );
}

function Result({
  view,
  shown,
  recState,
}: {
  view: HomeLang;
  shown: Shown | null;
  recState: 'loading' | 'ok' | 'fail' | 'mismatch';
}) {
  const t = (o: L) => o[view];
  const muted = { color: 'var(--ink-3)', fontSize: '0.88rem' } as const;

  if (!API && recState === 'loading') return <p style={muted}>{t(T.loading)}</p>;
  if (!API && recState === 'fail') return <p style={{ ...muted, color: 'var(--warn)' }}>{t(T.loadFail)}</p>;
  if (!API && recState === 'mismatch') return <p style={{ ...muted, color: 'var(--warn)' }}>{t(T.mismatch)}</p>;
  if (!shown) return <p style={muted}>{t(T.pick)}</p>;

  if (shown.kind === 'pending')
    return (
      <p style={muted}>
        {shown.question} — {t(T.pending)}
      </p>
    );

  if (shown.kind === 'api_error')
    return (
      <div>
        <Row k={t(T.question)}>{shown.question}</Row>
        <Row k={t(T.verdict)}>
          <span style={{ color: 'var(--warn)' }}>
            {t(T.apiError)}: <code className="font-mono">{shown.why}</code>
          </span>
        </Row>
        <p className="mt-3" style={muted}>
          {t(T.apiErrorNote)}
        </p>
      </div>
    );

  if (shown.kind === 'not_recorded')
    return (
      <div>
        <Row k={t(T.question)}>{shown.question}</Row>
        <p className="mt-3" style={muted}>
          {t(T.notRecorded)}
        </p>
        <div className="mt-3 rounded-lg border overflow-x-auto" style={{ borderColor: 'var(--line)', background: 'var(--surface-2)' }}>
          <pre className="p-3 font-mono" style={{ fontSize: '0.74rem', lineHeight: 1.7, color: 'var(--ink-2)', margin: 0 }}>
            {CMD_CHAT + '\n' + CMD_ASK}
          </pre>
        </div>
      </div>
    );

  if (shown.kind === 'read') {
    const v = shown.item.view;
    return (
      <div>
        <div className="mb-2" style={{ color: 'var(--ink-4)', fontSize: '0.74rem' }}>
          {t(T.readTitle)} · {API ? t(T.readNoteApi) : t(T.fromRecord)}
        </div>
        <Row k={t(T.readTitle)}>{shown.item.text}</Row>
        <Row k={t(T.readable)}>
          <code className="font-mono">{String(v.readable)}</code> ({v.readable ? t(T.readable) : t(T.unreadable)})
        </Row>
        <Row k={t(T.events)}>
          <code className="font-mono">{String(v.events_status)}</code>
        </Row>
        {(v.clauses || []).map((c, i) => (
          <div key={i} className="mt-3 rounded-lg border p-3 min-w-0" style={{ borderColor: 'var(--line)' }}>
            <div style={{ color: 'var(--ink)', fontSize: '0.92rem' }}>
              <span style={{ color: 'var(--ink-4)', fontSize: '0.74rem' }}>{t(T.predicate)} </span>
              <strong>{c.predicate}</strong>
            </div>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left" style={{ fontSize: '0.84rem' }}>
                <tbody>
                  {Object.entries(c.roles || {}).map(([role, val]) => (
                    <tr key={role} className="border-t" style={{ borderColor: 'var(--line)' }}>
                      <td className="py-1.5 pr-3 whitespace-nowrap" style={{ color: 'var(--ink-3)' }}>
                        {ROLE_LABEL[role] ? t(ROLE_LABEL[role]) : role}
                      </td>
                      <td className="py-1.5 pr-3 font-mono whitespace-nowrap" style={{ color: 'var(--ink-4)' }}>
                        {role}
                      </td>
                      <td className="py-1.5" style={{ color: 'var(--ink)' }}>
                        {val}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-2" style={{ color: 'var(--ink-4)', fontSize: '0.76rem' }}>
              {t(T.form)}: <code className="font-mono">{[c.polarity, c.tense, c.voice].map((x) => String(x)).join(' · ')}</code>
            </div>
          </div>
        ))}
        {(v.relations || []).length > 0 && (
          <Row k={t(T.relations)}>
            <code className="font-mono break-all" style={{ fontSize: '0.78rem' }}>
              {JSON.stringify(v.relations)}
            </code>
          </Row>
        )}
        {(v.unsupported || []).length > 0 && (
          <Row k={t(T.unsupported)}>
            <ul className="flex flex-col gap-1">
              {(v.unsupported || []).map((u, i) => (
                <li key={i} style={{ fontSize: '0.84rem' }}>
                  <strong>{u.predicate}</strong>: {(u.reasons || []).join(' / ')}
                </li>
              ))}
            </ul>
          </Row>
        )}
        <Raw view={view} raw={shown.item.raw} />
      </div>
    );
  }

  // kind === 'ask'
  const v = shown.source === 'record' ? shown.item.view : shown.view;
  const question = shown.source === 'record' ? shown.item.question : shown.question;
  const raw = shown.source === 'record' ? shown.item.raw : shown.raw;
  const verdict = v.verdict || '';
  const answered = verdict === 'ANSWER';
  const abstained = verdict.startsWith('UNKNOWN');
  const verdictNote = answered ? t(T.answered) : abstained ? t(T.abstained) : t(T.otherVerdict);
  const tone = answered ? 'var(--accent)' : 'var(--warn)';

  return (
    <div>
      <div className="mb-2" style={{ color: 'var(--ink-4)', fontSize: '0.74rem' }}>
        {shown.source === 'record' ? t(T.fromRecord) : t(T.fromApi)}
      </div>
      <Row k={t(T.question)}>{question}</Row>
      <Row k={t(T.verdict)}>
        <span className="font-semibold" style={{ color: tone }}>
          {verdictNote}
        </span>{' '}
        <code className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--ink-3)' }}>
          {verdict || String(v.verdict)}
        </code>
      </Row>
      {answered && (
        <>
          <Row k={t(T.answer)}>
            <strong>{v.answer}</strong>
          </Row>
          <Row k={t(T.evidence)}>
            <ul className="flex flex-col gap-1">
              {(v.evidence || []).map((e, i) => (
                <li key={i}>「{e}」</li>
              ))}
            </ul>
          </Row>
        </>
      )}
      {!answered && (
        <>
          <Row k={t(T.whyText)}>{v.answer}</Row>
          <Row k={t(T.reason)}>
            <code className="font-mono" style={{ fontSize: '0.8rem' }}>
              {String(v.reason)}
            </code>
          </Row>
          <Row k={t(T.resolve)}>{v.how_to_resolve}</Row>
        </>
      )}
      <Row k={t(T.policy)}>
        <code className="font-mono" style={{ fontSize: '0.8rem' }}>
          {String(v.policy)}
        </code>
      </Row>
      <Raw view={view} raw={raw} />
    </div>
  );
}
