# team-board — verantyx.ai/team（チームの意見の場）

学内コンテストのチーム用。意見を出しにくいメンバーが、ひとりで下書きして（必要なら Claude と整えて）、名前つきか名前なしかを選んで共有する場所。

| 場所 | 中身 |
|---|---|
| `functions/team/[[path]].js` | Pages Function。`/team` は画面、`/team/api/*` は API |
| `team-board/core.js` | 本体（合言葉の確認、意見・いいね、Claude と整える） |
| `team-board/page.js` | 画面 |
| `team-board/schema.sql` | D1 の表 |
| `team-board/make-tokens.mjs` | メンバーごとの合言葉を作る |
| `team-board/local-dev.mjs` | 手元で試す（`FAKE_CLAUDE=1` で Claude を代役に） |

`_routes.json`（public と out の両方）に `/team` と `/team/*` を足してある。

## 名前なし・記録について

- 名前なしの意見は書いた人を保存しない（管理者にも分からない）。
- 送信と「整える」の回数は 1 時間ごとの合計だけ残す（細かい時刻を残すと名前なしの意見と照らし合わせられるため）。
- 「整える」で送った文章と結果は保存しない（Anthropic の API には送られる）。
- 6 人と少人数なので、文体や内容から推測されることはありうる。

## 公開の準備（1 回だけ）

1. D1 を作り、表と合言葉を入れる

   ```bash
   npx wrangler@4 d1 create team-board
   npx wrangler@4 d1 execute team-board --remote --file team-board/schema.sql
   node team-board/make-tokens.mjs 本西:leader 大島 川端 チンソウ 林淳暉 横畑
   npx wrangler@4 d1 execute team-board --remote --file members.sql
   ```

   `tokens.txt` と `members.sql` はリポジトリのルートにできる。**コミットしない**（`.gitignore` 済み）。合言葉は各メンバーに個別に渡し、渡し終えたら `tokens.txt` を消す。

2. `wrangler.toml` の Team board の `[[d1_databases]]` の 4 行の `#` を外し、`database_id` を入れる。
3. API キー（チームプランと連携した Claude Console の組織で作った、このサイト専用のキー）を登録する

   ```bash
   npx wrangler@4 pages secret put ANTHROPIC_API_KEY --project-name verantyx-site
   ```

4. main に入れて push すると公開される。`https://verantyx.ai/team/api/health` が `{"ok":true,...,"db":true}` なら準備完了。

D1 がまだのうちは、画面は開くが API は「保存先がまだ準備されていません」（503）を返す。

## 手元で試す

```bash
FAKE_CLAUDE=1 node team-board/local-dev.mjs
```

http://127.0.0.1:8788/team を開き、合言葉 `tb_test_leader`（本西）か `tb_test_member`（大島）。
