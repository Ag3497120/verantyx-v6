// Small control plane only. No model weights, tensors or LLM inference here.
// The public enrollment key is safe to publish. Private keys remain on the Macs.
const BORROWER_PUBLIC = "yQ7b3T1NF5ONsaipHx3et+idBZD0AUMDYcBGfu3PM6Q=";
const PREFIX = "four-cross:qwen-chat:v1:";
const MODEL = "qwen3.6:35b-a3b";
const AXES = ["+x", "-x", "+y", "-y", "+z", "-z"];
const encoder = new TextEncoder();
const wire = (value) => JSON.stringify(sort(value));
function sort(value) {
  if (Array.isArray(value)) return value.map(sort);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(k => [k, sort(value[k])]));
  return value;
}
const bytes = (value) => Uint8Array.from(atob(value), c => c.charCodeAt(0));
const base64 = (value) => btoa(String.fromCharCode(...new Uint8Array(value)));
const hash = async (value) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(value)))).map(x => x.toString(16).padStart(2, "0")).join("");
const json = (value, status = 200) => new Response(JSON.stringify(value), {status, headers: {"Content-Type": "application/json", "Cache-Control": "no-store"}});
const fail = (code, status = 400) => { const error = new Error(code); error.status = status; throw error; };

async function body(request) {
  const data = await request.text();
  if (encoder.encode(data).length > 32768) fail("REQUEST_TOO_LARGE", 413);
  try { return JSON.parse(data); } catch { fail("INVALID_JSON"); }
}

async function enrolled(data) {
  if (!data || !data.payload || typeof data.signature !== "string") fail("SIGNATURE_REQUIRED", 401);
  const p = data.payload;
  if (!Number.isSafeInteger(p.issued_ms) || Math.abs(Date.now() - p.issued_ms) > 60000 || !/^[a-f0-9]{32}$/.test(p.nonce || "")) fail("STALE_REQUEST", 401);
  const key = await crypto.subtle.importKey("raw", bytes(BORROWER_PUBLIC), {name: "Ed25519"}, false, ["verify"]);
  if (!await crypto.subtle.verify("Ed25519", key, bytes(data.signature), encoder.encode(wire(p)))) fail("BAD_SIGNATURE", 401);
  return p;
}

async function keys(env, create = false) {
  let stored = await env.VERA_DEMAND.get(PREFIX + "keys", "json");
  if (!stored && create) {
    stored = {};
    for (const role of ["ingress", "egress"]) {
      const pair = await crypto.subtle.generateKey("Ed25519", true, ["sign", "verify"]);
      stored[role] = {private: await crypto.subtle.exportKey("jwk", pair.privateKey), public: await crypto.subtle.exportKey("jwk", pair.publicKey)};
    }
    await env.VERA_DEMAND.put(PREFIX + "keys", JSON.stringify(stored));
  }
  if (!stored) fail("NOT_ENROLLED", 503);
  return stored;
}

async function stamp(env, role, privateKey, feature, job) {
  const key = PREFIX + "graph:" + role;
  const old = await env.VERA_DEMAND.get(key, "json") || {count: 0, version: 1, memory: {}, head: "0".repeat(64)};
  const count = old.count + 1, version = 1 + Math.floor(count / 4);
  const seed = await hash(privateKey.d + ":" + role + ":" + version);
  const featureId = await hash(privateKey.d + ":feature:" + feature);
  const memory = {...old.memory, [featureId]: (old.memory[featureId] || 0) + 1};
  const entries = Object.entries(memory).sort((a, b) => b[1] - a[1]).slice(0, 144);
  const rotation = parseInt(seed.slice(0, 4), 16) % 6;
  const axes = [...AXES.slice(rotation), ...AXES.slice(0, rotation)];
  const cells = entries.map(([token, mass], index) => ({token, mass, axis: axes[Math.floor(index % 24 / 4)], face: ["north", "south", "east", "west"][index % 4], path: index < 24 ? [] : [axes[Math.floor(index / 24) % 6]]}));
  const head = await hash(old.head + ":" + job + ":" + featureId + ":" + count);
  const state = {role, count, version, depth: entries.length > 24 ? 1 : 0, memory: Object.fromEntries(entries), cells, head};
  state.commitment = await hash(wire(state));
  await env.VERA_DEMAND.put(key, JSON.stringify(state));
  return {role, version, count, depth: state.depth, commitment: state.commitment, axis: axes[0]};
}

async function sign(jwk, value) {
  const key = await crypto.subtle.importKey("jwk", jwk, {name: "Ed25519"}, false, ["sign"]);
  return base64(await crypto.subtle.sign("Ed25519", key, encoder.encode(wire(value))));
}

async function lease(env, job, requestHash, phase = "chat") {
  if (!/^[a-f0-9]{32}$/.test(job) || !/^[a-f0-9]{64}$/.test(requestHash)) fail("INVALID_JOB");
  const pair = await keys(env);
  const ingress = await stamp(env, "cloud_ingress", pair.ingress.private, phase, job);
  const egress = await stamp(env, "cloud_egress", pair.egress.private, phase, job);
  const payload = {protocol: "four-cross-chat-v1", job_id: job, request_hash: requestHash, phase, model: MODEL,
    issued_ms: Date.now(), expires_ms: Date.now() + (phase === "bootstrap" ? 1800000 : 300000),
    nonce: crypto.randomUUID().replaceAll("-", ""), ingress, egress};
  const ingress_signature = await sign(pair.ingress.private, payload);
  return {payload, ingress_signature, signature: await sign(pair.egress.private, {payload, ingress_signature})};
}

async function registration(env, request) {
  const stored = await env.VERA_DEMAND.get(PREFIX + "device", "json");
  if (!stored) fail("BORROWER_OFFLINE", 503);
  const authorization = request.headers.get("Authorization") || "";
  if (!authorization.startsWith("Bearer ") || await hash(authorization.slice(7)) !== stored.token_hash) fail("PAIRING_REQUIRED", 401);
  return {stored, authorization};
}

export async function onRequest({request, env, params}) {
  try {
    if (!env.VERA_DEMAND) fail("CONTROL_STORAGE_UNAVAILABLE", 503);
    const path = Array.isArray(params.path) ? params.path.join("/") : params.path || "";
    if (request.method === "POST" && path === "register") {
      const p = await enrolled(await body(request));
      const endpoint = new URL(p.endpoint);
      if (endpoint.protocol !== "https:" || !endpoint.hostname.endsWith(".trycloudflare.com") || endpoint.port || endpoint.username || endpoint.password || endpoint.pathname !== "/" || endpoint.search || endpoint.hash || !/^[a-f0-9]{64}$/.test(p.token_hash || "")) fail("INVALID_REGISTRATION");
      const pair = await keys(env, true);
      await env.VERA_DEMAND.put(PREFIX + "device", JSON.stringify({endpoint: endpoint.origin, token_hash: p.token_hash, registered_ms: Date.now()}));
      return json({ok: true, public_keys: {ingress: pair.ingress.public, egress: pair.egress.public}});
    }
    if (request.method === "POST" && path === "control") {
      const p = await enrolled(await body(request));
      if (!['bootstrap', 'chat'].includes(p.phase)) fail("INVALID_PHASE");
      return json({ticket: await lease(env, p.job_id, p.request_hash, p.phase)});
    }
    if (request.method === "GET" && path === "info") {
      return json({model: MODEL, cloud: "lightweight_control_plane", cloud_graphs: 2, tensor_transport: "internet_wss_aes256gcm", public_compute: false});
    }
    if (request.method === "GET" && path === "status") {
      const {stored, authorization} = await registration(env, request);
      const result = await fetch(stored.endpoint + "/health", {headers: {Authorization: authorization}, redirect: "error", signal: AbortSignal.timeout(10000)});
      return new Response(result.body, {status: result.status, headers: {"Content-Type": "application/json", "Cache-Control": "no-store"}});
    }
    if (request.method === "POST" && path === "chat") {
      const {stored, authorization} = await registration(env, request);
      const data = await body(request);
      const messages = data.messages;
      if (!Array.isArray(messages) || !messages.length || messages.length > 16 || messages.some(m => !m || !["user", "assistant"].includes(m.role) || typeof m.content !== "string" || m.content.length > 6000)) fail("INVALID_MESSAGES");
      const job = crypto.randomUUID().replaceAll("-", "");
      const ticket = await lease(env, job, await hash(wire(messages)));
      const result = await fetch(stored.endpoint + "/v1/chat", {method: "POST", headers: {"Content-Type": "application/json", Authorization: authorization}, body: JSON.stringify({messages, ticket}), redirect: "error", signal: AbortSignal.timeout(180000)});
      return new Response(result.body, {status: result.status, headers: {"Content-Type": result.headers.get("Content-Type") || "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}});
    }
    return json({error: "NOT_FOUND"}, 404);
  } catch (error) {
    return json({error: error.status ? error.message : "CONTROL_OR_DEVICE_UNAVAILABLE"}, error.status || 502);
  }
}
