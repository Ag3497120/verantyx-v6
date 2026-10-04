// Relay for the /try/ page — forwards one question (and the fixed document) to a Vera server.
//
// The page calls this only when it was built with NEXT_PUBLIC_VERA_API set. Without that the page
// replays recorded output and never reaches here. The upstream is the Pages environment variable
// VERA_ORIGIN (Pages → Settings → Environment variables). Unset, this answers 503 with the reason
// instead of pretending. Nothing is added to the forwarded body: no cookies, no headers from the
// visitor, no secrets. Nothing is stored or logged.

const MAX_QUESTION = 200;
const MAX_DOCUMENT = 2000;
const OK = { "Access-Control-Allow-Origin": "*",
             "Content-Type": "application/json; charset=utf-8" };

function fail(status, why, extra) {
  return new Response(JSON.stringify({ ok: false, why, ...(extra || {}) }),
                      { status, headers: OK });
}

export async function onRequestPost({ request, env }) {
  const origin = String((env && env.VERA_ORIGIN) || "").trim();
  if (!origin)
    return fail(503, "origin_unset",
                { note: "VERA_ORIGIN が設定されていないので Vera に中継できません" });
  let body;
  try { body = await request.json(); }
  catch (e) { return fail(400, "bad_json"); }
  if (!body || typeof body !== "object") return fail(400, "bad_json");
  const question = typeof body.question === "string" ? body.question.trim() : "";
  if (!question || question.length > MAX_QUESTION ||
      /[\u0000-\u001f\u007f<>]/.test(question))
    return fail(400, "bad_question");
  let document;
  if (body.document !== undefined && body.document !== null) {
    if (typeof body.document !== "string" || body.document.length > MAX_DOCUMENT)
      return fail(400, "bad_document");
    document = body.document;
  }
  let upstream;
  try {
    upstream = await fetch(origin, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document }),
    });
  } catch (e) {
    return fail(502, "upstream_unreachable");
  }
  const text = await upstream.text();
  return new Response(text, { status: upstream.status, headers: OK });
}

export async function onRequestOptions() {
  return new Response(null, { headers: {
    ...OK, "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type" } });
}
