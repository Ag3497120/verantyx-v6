'use client';

/* The resident bot, present on every page.
 *
 * It shows its verdict — ANSWER, UNKNOWN_NO_EVIDENCE, UNKNOWN_AMBIGUOUS —
 * beside every reply, and names the project each answer came from. Showing
 * the refusal rather than hiding it is the point: this is a site about an
 * engine that refuses to guess, so the bot demonstrates the behaviour
 * instead of describing it.
 */

import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { useLanguage } from '@/lib/i18n';
import { ask, CATALOGUE_SIZE, SUGGESTIONS, type Reply } from '@/lib/bot';
import Logo from '@/components/Logo';

type Turn = { role: 'user'; text: string } | { role: 'bot'; reply: Reply };

const VERDICT_STYLE: Record<
  Reply['verdict'],
  { color: string; border: string }
> = {
  ANSWER: {
    color: 'rgba(var(--accent-rgb), 0.95)',
    border: 'rgba(var(--accent-rgb), 0.35)',
  },
  UNKNOWN_NO_EVIDENCE: { color: '#f59e0b', border: 'rgba(245,158,11,0.35)' },
  UNKNOWN_AMBIGUOUS: { color: '#a78bfa', border: 'rgba(167,139,250,0.35)' },
};

export default function SiteBot() {
  const { lang } = useLanguage();
  const ja = lang === 'ja';
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [turns, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Escape closes, which is what a keyboard user expects from a panel that
  // sits over the page.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  function send(text: string) {
    const q = text.trim();
    if (!q) return;
    setTurns((prev) => [
      ...prev,
      { role: 'user', text: q },
      { role: 'bot', reply: ask(q, lang) },
    ]);
    setInput('');
  }

  return (
    <>
      {/* Launcher */}
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={ja ? 'Vera ボットを開く' : 'Open the Vera bot'}
        style={{
          position: 'fixed',
          right: 'clamp(14px, 3vw, 26px)',
          bottom: 'clamp(14px, 3vw, 26px)',
          zIndex: 1200,
          width: 52,
          height: 52,
          // The button held its contents with default inline flow, so the
          // block-level SVG sat against the top-left of the padding box and
          // the ✕ glyph sat on a text baseline — two different offsets in
          // the same button. Centring both explicitly is the fix.
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          borderRadius: 16,
          border: '1px solid rgba(var(--accent-rgb), 0.4)',
          background: 'var(--chrome-solid)',
          backdropFilter: 'blur(14px)',
          WebkitBackdropFilter: 'blur(14px)',
          color: 'var(--accent)',
          cursor: 'pointer',
          fontSize: '1.25rem',
          lineHeight: 1,
          boxShadow: '0 10px 30px rgba(0,0,0,0.28)',
          transition: 'transform 0.25s ease, border-color 0.25s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
        }}
      >
        {open ? '✕' : <Logo size={26} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            lang={lang}
            style={{
              position: 'fixed',
              zIndex: 1199,
              right: 'clamp(10px, 3vw, 26px)',
              left: 'auto',
              bottom: 'clamp(76px, 12vw, 90px)',
              width: 'min(calc(100vw - 20px), 390px)',
              maxHeight: 'min(72vh, 560px)',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 18,
              border: '1px solid var(--line-strong)',
              background: 'var(--chrome-solid)',
              backdropFilter: 'blur(18px)',
              WebkitBackdropFilter: 'blur(18px)',
              boxShadow: '0 24px 60px rgba(0,0,0,0.35)',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: '14px 16px',
                borderBottom: '1px solid var(--line)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  fontWeight: 700,
                  fontSize: '0.94rem',
                  color: 'var(--ink)',
                  fontFamily: 'var(--font-display)',
                }}
              >
                <Logo size={17} glow={false} />
                {ja ? 'Vera ボット' : 'Vera bot'}
              </div>
              <div
                style={{
                  fontSize: '0.7rem',
                  color: 'var(--ink-4)',
                  marginTop: 3,
                  lineHeight: 1.6,
                }}
              >
                {ja
                  ? `${CATALOGUE_SIZE} リポジトリの README から答えるか、型で拒否します。モデルは使いません。`
                  : `Answers from ${CATALOGUE_SIZE} repositories' READMEs, or refuses with a type. No model.`}
              </div>
            </div>

            {/* Log */}
            <div
              ref={logRef}
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '14px 16px',
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}
            >
              {turns.length === 0 && (
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 7,
                  }}
                >
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s.en}
                      onClick={() => send(s[lang])}
                      style={{
                        padding: '7px 11px',
                        borderRadius: 9,
                        border: '1px solid var(--line-strong)',
                        background: 'transparent',
                        color: 'var(--ink-3)',
                        fontSize: '0.76rem',
                        cursor: 'pointer',
                        lineHeight: 1.4,
                      }}
                    >
                      {s[lang]}
                    </button>
                  ))}
                </div>
              )}

              {turns.map((turn, i) =>
                turn.role === 'user' ? (
                  <div
                    key={i}
                    style={{
                      alignSelf: 'flex-end',
                      maxWidth: '88%',
                      padding: '8px 12px',
                      borderRadius: '12px 12px 3px 12px',
                      background: 'rgba(var(--accent-rgb), 0.14)',
                      border: '1px solid rgba(var(--accent-rgb), 0.24)',
                      color: 'var(--ink)',
                      fontSize: '0.83rem',
                      lineHeight: ja ? 1.85 : 1.55,
                      overflowWrap: 'anywhere',
                    }}
                  >
                    {turn.text}
                  </div>
                ) : (
                  <BotTurn key={i} reply={turn.reply} ja={ja} onPick={send} />
                )
              )}
            </div>

            {/* Composer */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send(input);
              }}
              style={{
                display: 'flex',
                gap: 8,
                padding: '12px 14px',
                borderTop: '1px solid var(--line)',
              }}
            >
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={ja ? 'プロジェクトについて聞く' : 'Ask about a project'}
                style={{
                  flex: 1,
                  minWidth: 0,
                  padding: '9px 12px',
                  borderRadius: 10,
                  border: '1px solid var(--line-strong)',
                  background: 'var(--surface-2)',
                  color: 'var(--ink)',
                  fontSize: '0.83rem',
                  outline: 'none',
                }}
              />
              <button
                type="submit"
                className="btn-accent"
                style={{
                  padding: '9px 14px',
                  borderRadius: 10,
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                {ja ? '送信' : 'Ask'}
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function BotTurn({
  reply,
  ja,
  onPick,
}: {
  reply: Reply;
  ja: boolean;
  onPick: (text: string) => void;
}) {
  const v = VERDICT_STYLE[reply.verdict];
  return (
    <div style={{ maxWidth: '95%' }}>
      <div
        style={{
          display: 'inline-block',
          padding: '2px 8px',
          borderRadius: 6,
          border: `1px solid ${v.border}`,
          color: v.color,
          fontSize: '0.6rem',
          letterSpacing: '0.1em',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          marginBottom: 7,
        }}
      >
        {reply.verdict}
        {reply.project ? `  ·  ${reply.project}` : ''}
      </div>
      <div
        style={{
          padding: '10px 13px',
          borderRadius: '12px 12px 12px 3px',
          border: '1px solid var(--line)',
          background: 'var(--surface)',
          color: 'var(--ink-2)',
          fontSize: '0.83rem',
          lineHeight: ja ? 1.9 : 1.62,
          overflowWrap: 'anywhere',
        }}
      >
        {reply.text.split('\n\n').map((para, i) => (
          <p key={i} style={{ marginTop: i === 0 ? 0 : 9 }}>
            {para}
          </p>
        ))}
      </div>

      {reply.quotes && reply.quotes.length > 0 && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {reply.quotes.map((q, i) => (
            <div
              key={i}
              style={{
                paddingLeft: 10,
                borderLeft: '2px solid rgba(var(--accent-rgb), 0.35)',
                color: 'var(--ink-3)',
                fontSize: '0.76rem',
                lineHeight: ja ? 1.85 : 1.6,
                overflowWrap: 'anywhere',
              }}
            >
              {q}
            </div>
          ))}
        </div>
      )}

      {reply.source && (
        <a
          href={reply.source.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: 'inline-block',
            marginTop: 8,
            marginRight: 12,
            fontSize: '0.72rem',
            color: 'var(--ink-4)',
            textDecoration: 'none',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          }}
        >
          {reply.source.name} ↗
        </a>
      )}

      {reply.options && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
          {reply.options.map((o) => (
            <button
              key={o}
              onClick={() => onPick(o)}
              style={{
                padding: '5px 10px',
                borderRadius: 8,
                border: '1px solid var(--line-strong)',
                background: 'transparent',
                color: 'var(--ink-3)',
                fontSize: '0.74rem',
                cursor: 'pointer',
              }}
            >
              {o}
            </button>
          ))}
        </div>
      )}

      {reply.href && (
        <a
          href={reply.href}
          style={{
            display: 'inline-block',
            marginTop: 8,
            fontSize: '0.76rem',
            color: 'var(--accent)',
            textDecoration: 'none',
            borderBottom: '1px solid rgba(var(--accent-rgb), 0.35)',
          }}
        >
          {reply.hrefLabel ?? reply.href} →
        </a>
      )}

      {reply.matched && reply.matched.length > 0 && (
        <div
          style={{
            marginTop: 7,
            fontSize: '0.64rem',
            color: 'var(--ink-4)',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          }}
        >
          {ja ? '一致したキー: ' : 'matched: '}
          {reply.matched.join(', ')}
        </div>
      )}
    </div>
  );
}
