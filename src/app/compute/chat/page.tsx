'use client';

import { useEffect, useRef, useState } from 'react';

type Message = { role: 'user' | 'assistant'; content: string };
type Graph = { role: string; version: number; depth?: number; axis?: string };
const graphNames: Record<string, string> = { borrower: '借り手 / このMac', cloud_ingress: 'クラウド / 入口', cloud_egress: 'クラウド / 出口', lender: '貸し手 / M1 Max' };

export default function ComputeChat() {
  const [token, setToken] = useState('');
  const [draftToken, setDraftToken] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('接続コードを入力して開始');
  const [error, setError] = useState('');
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const controller = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);

  async function connect(value: string) {
    if (!value.trim()) return;
    setError(''); setStatus('２台のMacとの接続を確認中');
    try {
      const response = await fetch('/api/compute/status', { headers: { Authorization: `Bearer ${value.trim()}` } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '接続できませんでした');
      setToken(value.trim()); sessionStorage.setItem('four-cross-pairing', value.trim());
      setStatus(result.ready ? '接続済み / Qwenの応答を待機中' : '接続済み / モデルの読み込み中');
    } catch (e) { setStatus('未接続'); setError(e instanceof Error ? e.message : '接続エラー'); }
  }

  useEffect(() => {
    const hash = new URLSearchParams(window.location.hash.slice(1));
    const saved = hash.get('pair') || sessionStorage.getItem('four-cross-pairing') || '';
    if (hash.has('pair')) window.history.replaceState(null, '', window.location.pathname);
    if (saved) { setDraftToken(saved); void connect(saved); }
  }, []);

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages]);

  async function send() {
    if (!input.trim() || busy || !token) return;
    const next: Message[] = [...messages, { role: 'user' as const, content: input.trim() }].slice(-14);
    setMessages([...next, { role: 'assistant', content: '' }]); setInput(''); setError(''); setBusy(true);
    setStatus('４基を通して貸し手へ計算を依頼中');
    const abort = new AbortController(); controller.current = abort;
    let answer = '', completed = false;
    try {
      const response = await fetch('/api/compute/chat', { method: 'POST', signal: abort.signal,
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, body: JSON.stringify({ messages: next }) });
      if (!response.ok || !response.body) { const result = await response.json(); throw new Error(result.error || '計算依頼に失敗しました'); }
      const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true }).replaceAll('\r\n', '\n');
        let boundary: number;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
          const event = frame.split('\n').find(line => line.startsWith('event:'))?.slice(6).trim();
          const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
          if (!data || data === '[DONE]') continue;
          const value = JSON.parse(data);
          if (event === 'error') throw new Error(value.error || '計算中に接続が切れました');
          if (event === 'topology' || event === 'complete') {
            setGraphs(value.graphs || []);
            if (event === 'complete') completed = true;
          } else {
            const text = value.choices?.[0]?.delta?.content || '';
            if (text) { answer += text; setMessages([...next, { role: 'assistant', content: answer }]); setStatus('貸し手のGPUで生成中'); }
          }
        }
      }
      if (!completed || !answer) throw new Error('応答が完了する前に接続が終了しました');
      setStatus('応答完了 / ４基の状態を更新');
    } catch (e) {
      if (abort.signal.aborted) setStatus('画面への受信を停止しました');
      else { setError(e instanceof Error ? e.message : '通信エラー'); setStatus('エラー / 接続状態を確認してください'); }
    } finally { setBusy(false); controller.current = null; }
  }

  return <main className="fc-chat">
    <style>{`
      .fc-chat{min-height:100dvh;color:var(--ink);background:radial-gradient(ellipse at 10% 0%,rgba(var(--accent-rgb),.1),transparent 48%),var(--bg);padding:36px clamp(18px,4vw,64px) 52px;font-family:var(--font-dm-sans),var(--font-noto-sans-jp),sans-serif}
      .fc-wrap{max-width:1260px;margin:auto}.fc-top{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:32px}.fc-brand{font-size:13px;letter-spacing:.2em;font-weight:700}.fc-tag{font-size:11px;border:1px solid var(--line);padding:7px 11px;border-radius:30px;color:var(--ink-2)}
      .fc-heading{font-family:var(--font-syne),sans-serif;font-size:clamp(30px,4vw,52px);line-height:1.05;letter-spacing:-.045em;margin:0 0 12px}.fc-sub{font-size:14px;color:var(--ink-2);line-height:1.9;max-width:760px;margin:0 0 28px}.fc-layout{display:grid;grid-template-columns:minmax(0,1fr) 272px;gap:24px;align-items:start}
      .fc-panel{border:1px solid var(--line);background:var(--surface);border-radius:22px;overflow:hidden}.fc-status{padding:14px 20px;border-bottom:1px solid var(--line);font-size:12px;display:flex;align-items:center;gap:9px;color:var(--ink-2)}.fc-dot{width:7px;height:7px;border-radius:50%;background:rgb(var(--accent-rgb));box-shadow:0 0 0 4px rgba(var(--accent-rgb),.1)}
      .fc-messages{height:clamp(330px,51vh,640px);overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:26px;scrollbar-width:thin}.fc-empty{margin:auto;max-width:430px;color:var(--ink-2);text-align:center;line-height:1.9}.fc-empty h2{font-size:22px;color:var(--ink);margin-bottom:12px}.fc-message{max-width:92%;white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.9}.fc-message.user{align-self:flex-end;border:1px solid var(--line);background:var(--surface-2);border-radius:16px 16px 4px 16px;padding:12px 17px}.fc-message.assistant{align-self:flex-start}.fc-who{font-size:10px;letter-spacing:.12em;color:var(--ink-3);margin-bottom:7px}.fc-composer{padding:16px;border-top:1px solid var(--line);display:flex;gap:12px;align-items:flex-end}.fc-composer textarea{resize:vertical;min-height:64px;max-height:180px;flex:1;background:transparent;color:var(--ink);border:0;outline:none;font:inherit;font-size:15px;line-height:1.7;padding:6px}.fc-button{border:0;border-radius:12px;background:rgb(var(--accent-rgb));color:#fff;cursor:pointer;padding:12px 18px;font-size:13px;font-weight:700;white-space:nowrap}.fc-button:disabled{opacity:.4;cursor:not-allowed}.fc-button.secondary{background:var(--surface-2);border:1px solid var(--line);color:var(--ink)}
      .fc-side-title{font-size:11px;letter-spacing:.18em;color:var(--ink-3);margin:0 0 16px}.fc-node{border-left:2px solid var(--line);padding:2px 0 23px 18px;position:relative}.fc-node:before{content:'';position:absolute;width:8px;height:8px;border:2px solid var(--ink-3);background:var(--bg);border-radius:50%;left:-5px;top:6px}.fc-node.active:before{border-color:rgb(var(--accent-rgb))}.fc-node b{font-size:13px}.fc-node p{font-size:12px;color:var(--ink-3);margin:6px 0 0;line-height:1.7}.fc-pair{padding:16px;border:1px solid var(--line);border-radius:16px;margin-top:12px}.fc-pair input{box-sizing:border-box;width:100%;padding:10px;background:var(--surface);border:1px solid var(--line);border-radius:8px;color:var(--ink);font-size:12px;margin:10px 0}.fc-note{font-size:11px;line-height:1.8;color:var(--ink-3);margin-top:18px}.fc-error{padding:12px 20px;color:#d35743;background:rgba(211,87,67,.08);font-size:13px}.fc-chat button:focus-visible,.fc-chat input:focus-visible,.fc-chat textarea:focus-visible{outline:2px solid rgb(var(--accent-rgb));outline-offset:3px}
      @media(max-width:820px){.fc-layout{grid-template-columns:1fr}.fc-side{display:flex;flex-wrap:wrap;gap:12px}.fc-side-title{width:100%}.fc-node{flex:1;min-width:120px;padding-bottom:8px}.fc-pair,.fc-note{width:100%}.fc-top{margin-bottom:26px}.fc-messages{height:48vh;padding:18px}.fc-message{max-width:100%}.fc-tag{font-size:10px}.fc-composer{padding:12px;gap:6px}.fc-button{padding:12px}}
      @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
    `}</style>
    <div className="fc-wrap">
      <header className="fc-top"><a className="fc-brand" href="/">VERANTYX / COMPUTE</a><span className="fc-tag">INTERNET · FOUR CROSS</span></header>
      <h1 className="fc-heading">ひとつの対話。<br />ふたつのMac。</h1>
      <p className="fc-sub">Qwen3.6 35B-A3B の演算を、接続したMacのGPUへ。<br />クラウドは小さな制御だけを担当し、４基の立体十字構造体が処理をつなぎます。</p>
      <div className="fc-layout">
        <section className="fc-panel" aria-label="Qwenとのチャット">
          <div className="fc-status" role="status"><span className="fc-dot" />{status}</div>
          {error && <div className="fc-error" role="alert">{error}</div>}
          <div className="fc-messages" aria-live="polite">
            {!messages.length && <div className="fc-empty"><div className="fc-who">QWEN3.6 : 35B-A3B</div><h2>接続した計算資源で、話す。</h2><p>短い質問から始めてください。返答は貸し手のGPUで実際に生成されます。会話はこの画面のメモリだけに保持します。</p></div>}
            {messages.map((message, index) => <article className={`fc-message ${message.role}`} key={index}><div className="fc-who">{message.role === 'user' ? 'YOU' : 'QWEN / REMOTE COMPUTE'}</div>{message.content || (busy ? '計算を準備しています…' : '応答なし')}</article>)}
            <div ref={end} />
          </div>
          <form className="fc-composer" onSubmit={event => { event.preventDefault(); void send(); }}>
            <textarea aria-label="メッセージ" placeholder={token ? 'Qwenにメッセージを送る' : '接続コードでペアリングしてください'} value={input} maxLength={6000} disabled={!token || busy} onChange={event => setInput(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(); } }} />
            {busy ? <button className="fc-button secondary" type="button" onClick={() => controller.current?.abort()}>受信停止</button> : <button className="fc-button" disabled={!token || !input.trim()} type="submit">送信</button>}
          </form>
        </section>
        <aside className="fc-side">
          <h2 className="fc-side-title">LIVE COMPUTE PATH</h2>
          {Object.entries(graphNames).map(([role, name], index) => { const graph = graphs.find(item => item.role === role); return <div className={`fc-node ${graph ? 'active' : ''}`} key={role}><b>{String.fromCharCode(65 + index)} / {name}</b><p>{graph ? `構造 v${graph.version}${graph.axis ? ` · ${graph.axis}` : ''}` : '処理前 / 状態未取得'}<br />{role === 'borrower' ? 'モデルの保持・推論制御' : role === 'lender' ? 'Metal / 汎用テンソル演算' : '認証・経路・版の管理'}</p></div>; })}
          <div className="fc-pair"><label htmlFor="pairing" style={{fontSize:12}}>このMacの接続コード</label><input id="pairing" type="password" autoComplete="off" value={draftToken} onChange={e => setDraftToken(e.target.value)} placeholder="ローカルのペアリングコード" /><button className="fc-button secondary" type="button" disabled={busy} onClick={() => void connect(draftToken)}>接続を確認</button></div>
          <p className="fc-note">実験用・所有者限定。２台のMacと接続プログラムの起動が必要です。クラウドと貸し手の管理者からのデータ秘匿は保証しません。受信停止はGPUカーネルの即時停止を保証しません。</p>
        </aside>
      </div>
    </div>
  </main>;
}
