import { gate, preflight, reply, search } from "./_lib.js";
export const onRequestOptions = ({ request }) => preflight(request);
export async function onRequestPost({ request }) {
  const origin = request.headers.get("Origin") || "", refused = gate(request); if (refused) return refused;
  try { return reply(200, await search((await request.json()).query), origin); }
  catch (e) { return reply(/^[A-Z_]+$/.test(e.message) ? 400 : 502, { error: { code: /^[A-Z_]+$/.test(e.message) ? e.message : "WEB_UNAVAILABLE" } }, origin); }
}
