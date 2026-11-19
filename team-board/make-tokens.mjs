// メンバーごとの合言葉を作る。
//   node make-tokens.mjs 本西:leader 大島 川端 チンソウ 林淳暉 横畑
// 出力: tokens.txt（各メンバーに個別に渡す。共有チャットに貼らない）と members.sql（サーバーに登録する）
import { randomBytes, createHash } from "node:crypto";
import { writeFileSync } from "node:fs";

const args = process.argv.slice(2);
if (!args.length) { console.error("使い方: node make-tokens.mjs 名前[:leader] ..."); process.exit(1); }

const lines = [], sql = [];
for (const a of args) {
  const [name, role = "member"] = a.split(":");
  const token = "tb_" + randomBytes(18).toString("base64url");
  const hash = createHash("sha256").update(token).digest("hex");
  lines.push(`${name}\t${token}`);
  sql.push(`INSERT OR REPLACE INTO members (token_hash, name, role) VALUES ('${hash}', '${name.replace(/'/g, "''")}', '${role === "leader" ? "leader" : "member"}');`);
}
writeFileSync("tokens.txt", lines.join("\n") + "\n", { mode: 0o600 });
writeFileSync("members.sql", sql.join("\n") + "\n");
console.log(`${args.length} 人分を tokens.txt と members.sql に書きました。tokens.txt は各メンバーに個別に渡してください。`);
