// Web search and page reading for the Cleanroom page, without installing anything.
// Only a search query or a public address the person approved reaches here; no project
// files, no records, no keys. Nothing is stored. Answers are for verantyx.ai pages only.

export const ALLOWED = ["https://verantyx.ai", "https://www.verantyx.ai", "http://127.0.0.1:8765", "http://localhost:8765"];
const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15";
const MAX_BYTES = 2_000_000, MAX_TEXT = 20_000, WINDOW = 60_000, PER_WINDOW = 30;
const seen = new Map(); // per-isolate, per-address request counts

export function reply(status, body, origin) {
  const headers = { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", Vary: "Origin" };
  if (ALLOWED.includes(origin)) headers["Access-Control-Allow-Origin"] = origin;
  return new Response(JSON.stringify(body), { status, headers });
}
export function gate(request) {
  const origin = request.headers.get("Origin") || "";
  if (origin && !ALLOWED.includes(origin) && !origin.endsWith(".verantyx-site.pages.dev")) return reply(403, { error: { code: "ORIGIN_NOT_ALLOWED" } }, origin);
  const who = request.headers.get("CF-Connecting-IP") || "?", now = Date.now();
  const b = seen.get(who);
  if (!b || now - b.start > WINDOW) seen.set(who, { start: now, n: 1 });
  else if (++b.n > PER_WINDOW) return reply(429, { error: { code: "TOO_MANY_REQUESTS" } }, origin);
  return null;
}
export function preflight(request) {
  const origin = request.headers.get("Origin") || "";
  const h = { "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "600" };
  if (ALLOWED.includes(origin)) h["Access-Control-Allow-Origin"] = origin;
  return new Response(null, { status: 204, headers: h });
}
function publicUrl(raw) {
  let u; try { u = new URL(raw); } catch { throw new Error("BAD_URL"); }
  if (!/^https?:$/.test(u.protocol)) throw new Error("ONLY_HTTP_URLS");
  const h = u.hostname;
  if (h === "localhost" || h.endsWith(".local") || h.endsWith(".internal") || /^(127\.|10\.|192\.168\.|169\.254\.|0\.|172\.(1[6-9]|2\d|3[01])\.)/.test(h) || h.startsWith("[")) throw new Error("PRIVATE_ADDRESS_REFUSED");
  return u.toString();
}
async function get(url, init = {}) {
  const r = await fetch(publicUrl(url), { ...init, headers: { "User-Agent": UA, ...(init.headers || {}) }, redirect: "follow" });
  publicUrl(r.url || url);
  const buf = await r.arrayBuffer();
  return { url: r.url || url, text: new TextDecoder().decode(buf.slice(0, MAX_BYTES)) };
}
const clean = s => (s || "").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&nbsp;/g, " ").trim();

export async function search(query) {
  query = String(query || "").trim();
  if (!query || query.length > 400) throw new Error("BAD_QUERY");
  const { text } = await get("https://html.duckduckgo.com/html/", { method: "POST", body: new URLSearchParams({ q: query, kl: "jp-jp" }), headers: { "Content-Type": "application/x-www-form-urlencoded" } });
  const results = [];
  for (const chunk of text.split('class="result__a"').slice(1)) {
    const link = /href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/.exec(chunk); if (!link) continue;
    let href = link[1].replace(/&amp;/g, "&");
    const m = /uddg=([^&]+)/.exec(href); if (m) href = decodeURIComponent(m[1]);
    if (href.startsWith("//")) href = "https:" + href;
    if (!href.startsWith("http") || href.includes("duckduckgo.com/y.js")) continue;
    const snip = /class="result__snippet"[^>]*>([\s\S]*?)<\/(?:a|div|td)>/.exec(chunk);
    results.push({ title: clean(link[2]), url: href, snippet: clean(snip && snip[1]) });
    if (results.length >= 8) break;
  }
  if (!results.length) {
    // The full page sometimes answers with a bot check; the light page is the fallback.
    const lite = await get("https://lite.duckduckgo.com/lite/", { method: "POST", body: new URLSearchParams({ q: query, kl: "jp-jp" }), headers: { "Content-Type": "application/x-www-form-urlencoded" } });
    const links = [...lite.text.matchAll(/<a rel="nofollow" href="([^"]+)" class='result-link'>([\s\S]*?)<\/a>/g)];
    const snippets = [...lite.text.matchAll(/class='result-snippet'>([\s\S]*?)<\/td>/g)];
    links.slice(0, 8).forEach((m, i) => { if (m[1].startsWith("http")) results.push({ title: clean(m[2]), url: m[1], snippet: clean(snippets[i] && snippets[i][1]) }); });
  }
  if (!results.length) throw new Error("NO_RESULTS_OR_BLOCKED");
  return { query, engine: "duckduckgo", via: "verantyx.ai", results };
}
export async function read(url) {
  const page = await get(String(url || ""));
  const title = clean((/<title[^>]*>([\s\S]*?)<\/title>/i.exec(page.text) || [])[1]).slice(0, 300);
  const body = page.text.replace(/<(script|style|noscript|svg|nav|footer|header|form|iframe)[\s\S]*?<\/\1>/gi, " ")
    .replace(/<\/(p|div|li|h[1-6]|tr|section|article|pre|blockquote)>|<br\s*\/?>/gi, "\n");
  const text = clean(body).replace(/[ \t　]+/g, " ").replace(/\n\s*\n+/g, "\n\n").trim();
  return { url: page.url, title, text: text.slice(0, MAX_TEXT), truncated: text.length > MAX_TEXT, via: "verantyx.ai" };
}
