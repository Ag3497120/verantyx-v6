// /team をこの Mac で試す（D1 の代わりに Node の SQLite を使う）。
//   node team-board/local-dev.mjs [port]        → http://127.0.0.1:8788/team
//   FAKE_CLAUDE=1 node team-board/local-dev.mjs  →「Claude と整える」を固定の文章で返す（API キーなしで画面を確かめる用）
// 試し用の合言葉: 本西=tb_test_leader（leader）, 大島=tb_test_member
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { handle } from "./core.js";

const db = new DatabaseSync(":memory:");
db.exec(readFileSync(new URL("./schema.sql", import.meta.url), "utf8"));
const h = t => createHash("sha256").update(t).digest("hex");
db.prepare("INSERT INTO members VALUES (?, ?, ?)").run(h("tb_test_leader"), "本西", "leader");
db.prepare("INSERT INTO members VALUES (?, ?, ?)").run(h("tb_test_member"), "大島", "member");

// D1 と同じ呼び方（prepare().bind().first()/all()/run()）をする小さな代役
const D1 = {
  prepare(sql) {
    const st = db.prepare(sql);
    let args = [];
    const api = {
      bind(...a) { args = a; return api; },
      async first() { return st.get(...args) ?? null; },
      async all() { return { results: st.all(...args) }; },
      async run() { const r = st.run(...args); return { meta: { last_row_id: Number(r.lastInsertRowid), changes: r.changes } }; },
    };
    return api;
  },
};

const fake = process.env.FAKE_CLAUDE === "1";
const env = { TEAM_DB: D1, ANTHROPIC_API_KEY: fake ? "fake" : process.env.ANTHROPIC_API_KEY };
const opts = fake ? {
  fetch: async () => new Response(JSON.stringify({
    stop_reason: "end_turn",
    content: [{ type: "text", text: "（試し）定例の時間を少し短くできないか、相談させてください。" }],
  }), { status: 200 }),
} : {};

const port = Number(process.argv[2] || 8788);
createServer(async (req, res) => {
  let body = "";
  for await (const c of req) body += c;
  const url = `http://127.0.0.1:${port}${req.url}`;
  const pathname = new URL(url).pathname;
  if (pathname !== "/team" && !pathname.startsWith("/team/")) { res.writeHead(302, { location: "/team" }); return res.end(); }
  const request = new Request(url, {
    method: req.method, headers: req.headers, body: ["GET", "HEAD"].includes(req.method) ? undefined : body,
  });
  const out = await handle(request, env, pathname.replace(/^\/team/, "") || "/", opts);
  console.log(`${new Date().toLocaleTimeString("ja-JP")} ${req.method} ${req.url} → ${out.status}`);
  res.writeHead(out.status, Object.fromEntries(out.headers));
  res.end(await out.text());
}).listen(port, "127.0.0.1", () => console.log(`team-board (local): http://127.0.0.1:${port}/team${fake ? "（Claude は代役）" : ""}`));
