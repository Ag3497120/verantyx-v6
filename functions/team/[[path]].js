// verantyx.ai/team — チームの「意見の場」。中身は team-board/core.js。
// /team は画面、/team/api/* は API。D1 は TEAM_DB、Claude は秘密の値 ANTHROPIC_API_KEY。
import { handle } from "../../team-board/core.js";

export async function onRequest({ request, env }) {
  const path = new URL(request.url).pathname.replace(/^\/team/, "") || "/";
  return handle(request, env, path);
}
