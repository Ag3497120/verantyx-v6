// team-board: チームの「意見の場」（verantyx.ai/team）。Pages Function functions/team/[[path]].js から呼ばれる。
//
// 決めごと
// - メンバーは 1 人 1 つの合言葉（トークン）で入る。サーバーには合言葉のハッシュだけを保存する。
// - 「名前なし」で共有した意見は、書いた人を一切保存しない（管理者にも分からない）。
// - 回数の記録は 1 時間ごとの合計だけ。細かい時刻を残すと、名前なしの意見と照らし合わせて書いた人が分かるため。
// - リアクションは、二重に押せないように押した人のハッシュを保存する（誰が押したかは画面に出さない）。
// - 「Claude と整える」は、Pages の秘密の値 ANTHROPIC_API_KEY（チームプランの API クレジット）で動かす。
//   整える前の文章も整えた結果も保存しない。
//   Pages の公開はビルドなし（コミットした out/ をそのまま配る）なので、npm の SDK を使わず API を直接呼ぶ。
//
// D1: env.TEAM_DB（表は team-board/schema.sql）。未設定なら 503 で理由を返す。

import { PAGE_HTML } from "./page.js";

const MAX_TEXT = 1000;
const SENDS_PER_HOUR = 30;
const SHAPES_PER_HOUR = 20;
const PAGE = 100;
const SHAPE_MODEL = "claude-haiku-5-5"; // 安く速いモデル（整えるだけの短い仕事なので）

const SHAPE_SYSTEM = `あなたはチームの意見を出しやすくする手伝い役です。
相手が書いたまとまっていない考えを、チームに共有しやすい 1〜3 文に整えてください。
- 相手の言いたいことを変えない。足さない。結論を勝手に強めたり弱めたりしない。
- 責める言い方や強すぎる言い方は、やわらかく言い換える。
- 何を相談したいのか（意見・質問・提案・困りごと）が伝わるようにする。
- 整えた文章だけを返す。前置きや説明は書かない。`;

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

const page = () =>
  new Response(PAGE_HTML, {
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      "content-security-policy":
        "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
      "referrer-policy": "no-referrer",
      "x-content-type-options": "nosniff",
    },
  });

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function member(req, db) {
  const m = /^Bearer\s+(.+)$/.exec(req.headers.get("authorization") || "");
  if (!m) return null;
  const hash = await sha256(m[1].trim());
  const row = await db.prepare("SELECT name, role FROM members WHERE token_hash = ?").bind(hash).first();
  return row ? { ...row, hash } : null;
}

// 1 時間ごとの回数を数える。上限を超えていたら false
async function countHour(db, table, hash, limit) {
  const hour = Math.floor(Date.now() / 3600_000);
  const row = await db.prepare(`SELECT n FROM ${table} WHERE member_hash = ? AND hour = ?`).bind(hash, hour).first();
  if ((row?.n ?? 0) >= limit) return false;
  await db.prepare(`INSERT INTO ${table} (member_hash, hour, n) VALUES (?, ?, 1) ON CONFLICT (member_hash, hour) DO UPDATE SET n = n + 1`)
    .bind(hash, hour).run();
  await db.prepare(`DELETE FROM ${table} WHERE hour < ?`).bind(hour - 24).run();
  return true;
}

async function listMessages(db, me, after) {
  const { results } = await db.prepare(
    `SELECT id, author, text, created_at FROM messages WHERE id > ? ORDER BY id DESC LIMIT ${PAGE}`,
  ).bind(after).all();
  const ids = results.map(r => r.id);
  const counts = {}, mine = {};
  if (ids.length) {
    const marks = ids.map(() => "?").join(",");
    const r1 = await db.prepare(
      `SELECT message_id, kind, COUNT(*) AS n, SUM(CASE WHEN member_hash = ? THEN 1 ELSE 0 END) AS me
       FROM reactions WHERE message_id IN (${marks}) GROUP BY message_id, kind`,
    ).bind(me.hash, ...ids).all();
    for (const r of r1.results) {
      (counts[r.message_id] ||= {})[r.kind] = r.n;
      if (r.me) (mine[r.message_id] ||= []).push(r.kind);
    }
  }
  return results.reverse().map(r => ({
    id: r.id, author: r.author, text: r.text, at: r.created_at,
    reactions: counts[r.id] || {}, mine: mine[r.id] || [],
  }));
}

// 整えた文章を返す。断られたときは null。
async function shapeText(apiKey, text, fetchImpl = fetch) {
  const res = await fetchImpl("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({
      model: SHAPE_MODEL,
      max_tokens: 1000,
      output_config: { effort: "low" },
      system: SHAPE_SYSTEM,
      messages: [{ role: "user", content: text }],
    }),
  });
  if (res.status === 429 || res.status === 529) throw Object.assign(new Error("busy"), { busy: true });
  if (!res.ok) throw new Error(`api ${res.status}`);
  const data = await res.json();
  if (data.stop_reason === "refusal") return null;
  return (data.content || []).filter(b => b.type === "text").map(b => b.text).join("").trim() || null;
}

async function forwardToTeams(env, msg) {
  if (!env.TEAMS_WEBHOOK) return;
  try {
    await fetch(env.TEAMS_WEBHOOK, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: `${msg.author ?? "名前なし"}：${msg.text}` }),
    });
  } catch { /* 転送の失敗で共有自体は止めない */ }
}

// path は /team 以下（"/"、"/api/v1/me" など）
export async function handle(req, env, path, opts = {}) {
  if (req.method === "GET" && (path === "/" || path === "")) return page();
  if (path === "/api/health") return json({ ok: true, name: "team-board", db: !!env.TEAM_DB });
  if (!path.startsWith("/api/")) return json({ error: "ありません" }, 404);

  const db = env.TEAM_DB;
  if (!db) return json({ error: "保存先（D1）がまだ準備されていません" }, 503);

  const me = await member(req, db);
  if (!me) return json({ error: "合言葉が違うか、設定されていません" }, 401);
  const url = new URL(req.url);
  const p = path.slice(4); // "/v1/..."

  if (req.method === "GET" && p === "/v1/me") return json({ name: me.name, role: me.role });

  if (req.method === "GET" && p === "/v1/messages") {
    const after = Math.max(0, Number(url.searchParams.get("after") || 0) | 0);
    return json({ messages: await listMessages(db, me, after) });
  }

  if (req.method === "POST" && p === "/v1/messages") {
    let body;
    try { body = await req.json(); } catch { return json({ error: "送る内容の形が違います" }, 400); }
    const text = String(body.text ?? "").trim();
    if (!text) return json({ error: "空の意見は送れません" }, 400);
    if (text.length > MAX_TEXT) return json({ error: `${MAX_TEXT} 文字までです` }, 400);
    if (!(await countHour(db, "sends", me.hash, SENDS_PER_HOUR))) return json({ error: "少し時間をおいてから送ってください" }, 429);
    const now = Date.now();
    const author = body.anonymous ? null : me.name;
    const res = await db.prepare("INSERT INTO messages (author, text, created_at) VALUES (?, ?, ?)").bind(author, text, now).run();
    const msg = { id: res.meta.last_row_id, author, text, at: now, reactions: {}, mine: [] };
    await forwardToTeams(env, msg);
    return json({ message: msg }, 201);
  }

  if (req.method === "POST" && p === "/v1/shape") {
    if (!env.ANTHROPIC_API_KEY) return json({ error: "Claude と整える機能は、まだ準備されていません" }, 503);
    let body;
    try { body = await req.json(); } catch { return json({ error: "送る内容の形が違います" }, 400); }
    const text = String(body.text ?? "").trim();
    if (!text) return json({ error: "下書きが空です" }, 400);
    if (text.length > MAX_TEXT) return json({ error: `${MAX_TEXT} 文字までです` }, 400);
    if (!(await countHour(db, "shape_uses", me.hash, SHAPES_PER_HOUR))) return json({ error: "少し時間をおいてから使ってください" }, 429);
    try {
      const shaped = await shapeText(env.ANTHROPIC_API_KEY, text, opts.fetch);
      if (!shaped) return json({ error: "この文章は整えられませんでした。そのまま共有するか、書き直してみてください" }, 422);
      return json({ text: shaped });
    } catch (err) {
      return json({ error: err.busy ? "Claude が混み合っています。少し待ってください" : "Claude を呼び出せませんでした" }, err.busy ? 503 : 502);
    }
  }

  const react = /^\/v1\/messages\/(\d+)\/react$/.exec(p);
  if (req.method === "POST" && react) {
    const id = Number(react[1]);
    let body;
    try { body = await req.json(); } catch { body = {}; }
    const kind = body.kind;
    if (!["like", "more"].includes(kind)) return json({ error: "リアクションの種類が違います" }, 400);
    if (!(await db.prepare("SELECT 1 AS x FROM messages WHERE id = ?").bind(id).first())) return json({ error: "その意見はありません" }, 404);
    const had = await db.prepare("SELECT 1 AS x FROM reactions WHERE message_id = ? AND member_hash = ? AND kind = ?").bind(id, me.hash, kind).first();
    if (had) await db.prepare("DELETE FROM reactions WHERE message_id = ? AND member_hash = ? AND kind = ?").bind(id, me.hash, kind).run();
    else await db.prepare("INSERT INTO reactions (message_id, member_hash, kind) VALUES (?, ?, ?)").bind(id, me.hash, kind).run();
    return json({ on: !had });
  }

  const del = /^\/v1\/messages\/(\d+)$/.exec(p);
  if (req.method === "DELETE" && del && me.role === "leader") {
    const id = Number(del[1]);
    await db.prepare("DELETE FROM reactions WHERE message_id = ?").bind(id).run();
    await db.prepare("DELETE FROM messages WHERE id = ?").bind(id).run();
    return json({ deleted: id });
  }

  return json({ error: "ありません" }, 404);
}
