// The download counts, for the site's owner only: /api/downloads/stats?key=<secret>.
//
// The secret is the Pages secret DOWNLOAD_STATS_KEY (wrangler pages secret put); without
// it set, or with a wrong key, the answer is 404 as if nothing were here. JSON by default,
// a small table with ?format=html.

function same(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const nothing = new Response("Not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  if (!env.DOWNLOAD_STATS_KEY || !same(url.searchParams.get("key") || "", env.DOWNLOAD_STATS_KEY)) return nothing;
  if (!env.VERA_DEMAND) return new Response("KV not bound", { status: 503 });
  const files = {};
  let cursor;
  do {
    const page = await env.VERA_DEMAND.list({ prefix: "dl:", cursor });
    for (const k of page.keys) {
      const [, file, when] = k.name.split(":");
      const n = parseInt((await env.VERA_DEMAND.get(k.name)) || "0", 10) || 0;
      files[file] = files[file] || { total: 0, days: {} };
      if (when === "total") files[file].total = n; else files[file].days[when] = n;
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  const headers = { "Cache-Control": "no-store", "X-Robots-Tag": "noindex" };
  if (url.searchParams.get("format") !== "html")
    return new Response(JSON.stringify(files, null, 2), { headers: { ...headers, "Content-Type": "application/json; charset=utf-8" } });
  const rows = Object.keys(files).sort().reverse().map((f) => {
    const days = Object.keys(files[f].days).sort().reverse().slice(0, 14)
      .map((d) => `${d}: ${files[f].days[d]}`).join("<br>");
    return `<tr><td>${f}</td><td style="text-align:right">${files[f].total}</td><td>${days}</td></tr>`;
  }).join("");
  const html = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">` +
    `<title>Downloads</title><style>body{font:14px system-ui;margin:16px}td,th{border-bottom:1px solid #ccc;padding:6px 10px;vertical-align:top}</style>` +
    `<h1>VeRantyx のダウンロード数</h1><table><tr><th>ファイル</th><th>合計</th><th>日ごと（直近14日、UTC）</th></tr>${rows}</table>` +
    `<p>数えているのは回数だけです（IP などは保存しません）。同時のダウンロードが1回にまとまることがあるので、おおよその数です。</p>`;
  return new Response(html, { headers: { ...headers, "Content-Type": "text/html; charset=utf-8" } });
}
