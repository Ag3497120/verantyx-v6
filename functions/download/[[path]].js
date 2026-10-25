// Download counter for the Mac app's DMGs (verantyx.ai/download/*).
//
// Serves the file exactly as before (context.next() hands the request to the static
// asset) and, for a full GET of a .dmg, adds one to two counters in KV: the file's
// total and the file's count for the day (UTC). Nothing about the visitor is kept:
// no IP address, no user agent, no cookie. Range requests that resume a download
// (Range not starting at 0) are not counted again. KV increments are read-modify-write,
// so two downloads in the same instant can count as one: the numbers are close, not exact.
//
// KV: the site's existing namespace bound as VERA_DEMAND, keys "dl:<file>:total" and
// "dl:<file>:<YYYY-MM-DD>" (the demand inlet uses "d:<subject>", so they never mix).
// Unbound, files are still served and nothing is counted.

async function bump(kv, key) {
  const n = parseInt((await kv.get(key)) || "0", 10) || 0;
  await kv.put(key, String(n + 1));
}

export async function onRequestGet(context) {
  const { request, env, waitUntil } = context;
  const res = await context.next();
  const file = decodeURIComponent(new URL(request.url).pathname.split("/").pop() || "");
  const range = request.headers.get("Range");
  const fresh = !range || /^bytes=0-/.test(range);
  if (env.VERA_DEMAND && res.ok && fresh && /^VeRantyx-[0-9.]+\.dmg$/.test(file)) {
    const day = new Date().toISOString().slice(0, 10);
    waitUntil(Promise.all([bump(env.VERA_DEMAND, `dl:${file}:total`), bump(env.VERA_DEMAND, `dl:${file}:${day}`)]));
  }
  return res;
}
