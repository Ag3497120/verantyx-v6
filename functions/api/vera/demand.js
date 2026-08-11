// Demand collection — the zero-friction inlet.
//
// A visitor's refused SUBJECT (one word, no free text) is the most
// valuable growth signal Vera has, and it was dying in the visitor's tab.
// POST increments a counter per subject in KV; GET returns the ranked
// list for `grow --demand`. No accounts, no cookies, no payloads beyond
// the subject itself. Nothing here touches the model: entries only become
// knowledge after the owner approves and runs a release — the same two
// human gates every other inlet passes.
//
// Requires a KV namespace bound as VERA_DEMAND (Pages → Settings →
// Functions → KV bindings). Unbound, both verbs answer 503 with a clear
// note instead of pretending.

const MAX_SUBJECT = 64;
const OK = { "Access-Control-Allow-Origin": "*",
             "Content-Type": "application/json; charset=utf-8" };

export async function onRequestPost({ request, env }) {
  if (!env.VERA_DEMAND)
    return new Response(JSON.stringify({ ok: false, why: "kv_unbound" }),
                        { status: 503, headers: OK });
  let subject;
  try { subject = String((await request.json()).subject || "").trim(); }
  catch (e) { subject = ""; }
  if (!subject || subject.length > MAX_SUBJECT ||
      /[\n\r\t<>]/.test(subject))
    return new Response(JSON.stringify({ ok: false, why: "bad_subject" }),
                        { status: 400, headers: OK });
  const key = "d:" + subject;
  const cur = parseInt((await env.VERA_DEMAND.get(key)) || "0", 10);
  await env.VERA_DEMAND.put(key, String(cur + 1));
  return new Response(JSON.stringify({ ok: true, subject, count: cur + 1 }),
                      { headers: OK });
}

export async function onRequestGet({ env }) {
  if (!env.VERA_DEMAND)
    return new Response(JSON.stringify({ ok: false, why: "kv_unbound" }),
                        { status: 503, headers: OK });
  const list = await env.VERA_DEMAND.list({ prefix: "d:", limit: 1000 });
  const out = [];
  for (const k of list.keys) {
    const n = parseInt((await env.VERA_DEMAND.get(k.name)) || "0", 10);
    out.push({ subject: k.name.slice(2), count: n });
  }
  out.sort((a, b) => b.count - a.count);
  return new Response(JSON.stringify({ ok: true, demand: out }),
                      { headers: OK });
}

export async function onRequestOptions() {
  return new Response(null, { headers: {
    ...OK, "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type" } });
}
