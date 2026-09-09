// Cloudflare transports encrypted records; it never evaluates the LLM/RPC graph.
// Only the paired borrower can register a lender or open a transport channel.
const PUBLIC_KEY = "YDG1RYCxIQozHVphLXu2qPgVokEZR70Eqh61FuC06AQ=";
const PROTOCOL = "four-cross-internet-transport-v1";
const KEY = "four-cross:qwen-chat:v1:internet-relay";

function wire(value) {
  if (Array.isArray(value)) return "[" + value.map(wire).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.keys(value).sort().map(k => JSON.stringify(k) + ":" + wire(value[k])).join(",") + "}";
  return JSON.stringify(value);
}
function bytes(value) { return Uint8Array.from(atob(value), c => c.charCodeAt(0)); }
function json(value, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: {
    "Content-Type": "application/json", "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  } });
}
function validEndpoint(value) {
  const url = new URL(value);
  if (url.protocol !== "https:" || !/^[a-z0-9-]+\.trycloudflare\.com$/.test(url.hostname)
      || url.username || url.password || url.port || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("Invalid lender endpoint");
  }
  return url.origin;
}
async function authenticate(request) {
  const encoded = request.headers.get("X-Four-Cross-Authorization") || "";
  const signature = request.headers.get("X-Four-Cross-Signature") || "";
  if (encoded.length > 4096 || signature.length > 128) throw new Error("Invalid authorization");
  const payload = JSON.parse(new TextDecoder().decode(bytes(encoded)));
  const key = await crypto.subtle.importKey("raw", bytes(PUBLIC_KEY), "Ed25519", false, ["verify"]);
  if (!(await crypto.subtle.verify("Ed25519", key, bytes(signature), new TextEncoder().encode(wire(payload))))
      || payload.protocol !== PROTOCOL || !Number.isSafeInteger(payload.issued_ms)
      || Math.abs(Date.now() - payload.issued_ms) > 90000 || !/^[a-f0-9]{32}$/.test(payload.nonce)) {
    throw new Error("Invalid authorization");
  }
  return payload;
}

export async function onRequest({ request, env, params }) {
  if (!env.VERA_DEMAND) return json({ error: "Relay storage unavailable" }, 503);
  const path = Array.isArray(params.path) ? params.path.join("/") : params.path;
  let authorization;
  try { authorization = await authenticate(request); }
  catch { return json({ error: "Transport authorization required" }, 401); }
  if (request.method === "POST" && path === "register" && authorization.purpose === "register") {
    try {
      const endpoint = validEndpoint(authorization.endpoint);
      const record = { endpoint, registered_ms: Date.now() };
      await env.VERA_DEMAND.put(KEY, JSON.stringify(record));
      return json({ registered: true, endpoint, transport: "internet_wss_aes256gcm", direct_fallback: false });
    } catch { return json({ error: "Registration failed" }, 400); }
  }
  if (request.method !== "GET" || !["rpc", "control"].includes(path)
      || authorization.purpose !== "connect" || authorization.channel !== path
      || request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
    return json({ error: "Invalid transport channel" }, 400);
  }
  const lender = await env.VERA_DEMAND.get(KEY, "json");
  if (!lender) return json({ error: "Lender not registered" }, 503);
  try {
    const headers = new Headers({ Upgrade: "websocket",
      "X-Four-Cross-Authorization": request.headers.get("X-Four-Cross-Authorization"),
      "X-Four-Cross-Signature": request.headers.get("X-Four-Cross-Signature"),
    });
    const response = await fetch(validEndpoint(lender.endpoint) + "/" + path, {
      method: "GET", headers, redirect: "manual",
    });
    // Returning the upgrade directly lets the platform stream bounded frames.
    // The lender verifies signatures and rejects nonce reuse independently.
    if (response.status === 101 && response.webSocket) return response;
    return json({ error: "Lender refused transport" }, 502);
  } catch { return json({ error: "Lender transport unavailable" }, 502); }
}
