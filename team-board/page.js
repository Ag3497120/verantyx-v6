// ブラウザで開く「意見の場」の画面（Worker が / で返す）
export const PAGE_HTML = `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>意見の場</title>
<style>
:root {
  --bg: #f6f5f2; --card: #ffffff; --fg: #1d1d1b; --muted: #6b6a66; --line: #e3e1db;
  --accent: #2f6f5e; --accent-fg: #ffffff; --accent-soft: #e6f0ec; --ai: #f3efe4; --err: #b42318;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --card: #201f1d; --fg: #ecebe7; --muted: #a09e98; --line: #34322e;
    --accent: #5fb39a; --accent-fg: #0f1a16; --accent-soft: #1f2f29; --ai: #2a2720; --err: #ff8a80; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.7 -apple-system, "Hiragino Sans", "Yu Gothic UI", "Segoe UI", sans-serif; }
main { max-width: 640px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 14px; margin: 0 0 20px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-bottom: 16px; }
.card h2 { font-size: 15px; margin: 0 0 8px; }
.hint { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
textarea, input[type=password] { width: 100%; font: inherit; color: var(--fg); background: var(--bg);
  border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
textarea { min-height: 96px; resize: vertical; }
.row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
button { font: inherit; font-size: 14px; border-radius: 999px; padding: 8px 14px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--fg); }
button.primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); font-weight: 600; }
button:disabled { opacity: .5; cursor: default; }
.ai { background: var(--ai); border-radius: 10px; padding: 10px 12px; margin-top: 12px; }
.ai .label { font-size: 12px; color: var(--muted); }
.status { font-size: 13px; color: var(--muted); margin-top: 8px; min-height: 1em; }
.error { color: var(--err); }
.msg { border-top: 1px solid var(--line); padding: 12px 0; }
.msg:first-of-type { border-top: 0; }
.meta { font-size: 12px; color: var(--muted); }
.anon { display: inline-block; font-size: 11px; padding: 0 6px; border-radius: 6px; background: var(--accent-soft); color: var(--accent); margin-right: 4px; }
.text { white-space: pre-wrap; word-break: break-word; margin: 2px 0 6px; }
.react button { font-size: 12px; padding: 3px 10px; }
.react button.on { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
.me { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.link { background: none; border: 0; padding: 0; color: var(--muted); text-decoration: underline; font-size: 12px; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <div class="me">
    <h1>意見の場</h1>
    <span id="who" class="meta"></span>
  </div>
  <p class="sub">まとまっていなくて大丈夫。ひとりで書いて、出すかどうかは自分で決められます。</p>

  <section id="login" class="card" hidden>
    <h2>合言葉を入れてください</h2>
    <p class="hint">リーダーから個別に受け取った合言葉（tb_ で始まる文字列）です。この端末にだけ保存されます。</p>
    <input id="token" type="password" autocomplete="off" placeholder="tb_...">
    <div class="row"><button id="enter" class="primary">はじめる</button></div>
    <div id="loginStatus" class="status"></div>
  </section>

  <div id="app" hidden>
    <section class="card">
      <h2>書く</h2>
      <p class="hint">ここに書いた下書きは、共有を押すまで誰にも見えません。</p>
      <textarea id="draft" placeholder="思っていることを、そのまま書いてください"></textarea>
      <div class="row">
        <button id="shape">Claude と整える</button>
        <button id="named" class="primary">名前つきで共有</button>
        <button id="anon">名前なしで共有</button>
      </div>
      <div id="ai" class="ai" hidden>
        <div class="label">【AI案】Claude が整えた文章</div>
        <div id="aiText" class="text"></div>
        <div class="row">
          <button id="useAi">この文章を下書きにする</button>
          <button id="dropAi">使わない</button>
        </div>
      </div>
      <div id="status" class="status"></div>
    </section>

    <section class="card">
      <h2>みんなの意見</h2>
      <p class="hint">「名前なし」の意見は、リーダーにも誰が書いたか分かりません。</p>
      <div id="list"></div>
    </section>

    <button id="logout" class="link">この端末から合言葉を消す</button>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const store = {
  get() { try { return localStorage.getItem('tb_token') || ''; } catch { return ''; } },
  set(v) { try { localStorage.setItem('tb_token', v); } catch {} },
  clear() { try { localStorage.removeItem('tb_token'); } catch {} },
};
let token = store.get();
let lastSeen = 0;
const mine = new Set();

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const time = at => { const d = new Date(at); const p = n => String(n).padStart(2, '0');
  return (d.getMonth() + 1) + '/' + d.getDate() + ' ' + p(d.getHours()) + ':' + p(d.getMinutes()); };

async function api(path, init = {}) {
  const res = await fetch(path, { ...init, headers: { authorization: 'Bearer ' + token, 'content-type': 'application/json' } });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { const e = new Error(data.error || ('エラー ' + res.status)); e.status = res.status; throw e; }
  return data;
}

function say(text, isError) { $('status').textContent = text || ''; $('status').classList.toggle('error', !!isError); }

function render(messages) {
  const list = [...messages].reverse();
  $('list').innerHTML = list.length ? list.map(m => \`
    <div class="msg">
      <div class="meta">\${m.author ? esc(m.author) : '<span class="anon">名前なし</span>'}・\${time(m.at)}</div>
      <div class="text">\${esc(m.text)}</div>
      <div class="react">
        <button data-id="\${m.id}" data-kind="like" class="\${m.mine.includes('like') ? 'on' : ''}">いいね \${m.reactions.like || 0}</button>
        <button data-id="\${m.id}" data-kind="more" class="\${m.mine.includes('more') ? 'on' : ''}">もっと聞きたい \${m.reactions.more || 0}</button>
      </div>
    </div>\`).join('') : '<p class="hint">まだ意見はありません。最初のひとことをどうぞ。</p>';
}

async function refresh() {
  try {
    const { messages } = await api('/team/api/v1/messages?after=0');
    const fresh = lastSeen ? messages.filter(m => m.id > lastSeen && !mine.has(m.id)) : [];
    lastSeen = messages.reduce((n, m) => Math.max(n, m.id), lastSeen);
    render(messages);
    if (fresh.length && document.hidden) document.title = '(' + fresh.length + ') 意見の場';
  } catch (e) {
    if (e.status === 401) return showLogin('合言葉が違うようです。もう一度入れてください。');
    say('更新できませんでした：' + e.message, true);
  }
}

function busy(on) { for (const id of ['shape', 'named', 'anon']) $(id).disabled = on; }

async function share(anonymous) {
  const text = $('draft').value.trim();
  if (!text) return say('下書きが空です。', true);
  busy(true); say(anonymous ? '名前なしで共有しています…' : '共有しています…');
  try {
    const { message } = await api('/team/api/v1/messages', { method: 'POST', body: JSON.stringify({ text, anonymous }) });
    mine.add(message.id);
    $('draft').value = ''; $('ai').hidden = true;
    say(anonymous ? '名前なしで共有しました。' : '共有しました。');
    await refresh();
  } catch (e) { say(e.message, true); }
  busy(false);
}

async function shape() {
  const text = $('draft').value.trim();
  if (!text) return say('先に下書きを書いてください。', true);
  busy(true); say('Claude と整えています…');
  try {
    const r = await api('/team/api/v1/shape', { method: 'POST', body: JSON.stringify({ text }) });
    $('aiText').textContent = r.text; $('ai').hidden = false; say('');
  } catch (e) { say(e.message, true); }
  busy(false);
}

function showLogin(msg) {
  $('app').hidden = true; $('login').hidden = false; $('who').textContent = '';
  $('loginStatus').textContent = msg || ''; $('loginStatus').classList.toggle('error', !!msg);
}

async function start() {
  if (!token) return showLogin();
  try {
    const me = await api('/team/api/v1/me');
    $('who').textContent = me.name + ' さん';
    $('login').hidden = true; $('app').hidden = false;
    await refresh();
  } catch (e) {
    showLogin(e.status === 401 ? '合言葉が違うようです。もう一度入れてください。' : 'つながりませんでした：' + e.message);
  }
}

$('enter').onclick = () => { token = $('token').value.trim(); if (!token) return; store.set(token); start(); };
$('token').onkeydown = e => { if (e.key === 'Enter') $('enter').click(); };
$('named').onclick = () => share(false);
$('anon').onclick = () => share(true);
$('shape').onclick = shape;
$('useAi').onclick = () => { $('draft').value = $('aiText').textContent; $('ai').hidden = true; };
$('dropAi').onclick = () => { $('ai').hidden = true; };
$('logout').onclick = () => { store.clear(); token = ''; showLogin(); };
$('list').onclick = async e => {
  const b = e.target.closest('button[data-id]'); if (!b) return;
  try { await api('/team/api/v1/messages/' + b.dataset.id + '/react', { method: 'POST', body: JSON.stringify({ kind: b.dataset.kind }) }); await refresh(); }
  catch (err) { say(err.message, true); }
};
document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  document.title = '意見の場';
  if (token && !$('app').hidden) refresh();
});
setInterval(() => { if (token && !$('app').hidden) refresh(); }, 15000);
start();
</script>
</body>
</html>`;
