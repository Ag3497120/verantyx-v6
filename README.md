
## サイトの構成（W9-site）

- `/` … Vera のホーム（`src/app/page.tsx` → `src/components/vera-home/VeraHome.tsx`）。日本語が既定で、保存された言語が `en` のときだけ英語で出す。
- `/try/` … Vera の体験（`src/components/vera-home/TryVera.tsx`。ホームにも埋め込む）。
  - ビルド時に `NEXT_PUBLIC_VERA_API` が空なら、`public/vera-demo/recorded.json`（記録済みの実行結果）を再生する。記録に無い自由入力は実行せず、手元の `vera chat` を案内する。
  - `NEXT_PUBLIC_VERA_API` を設定してビルドすると、問いをその URL に POST する。`/api/vera/ask` を指せば `functions/api/vera/ask.js`（Pages Function）が Pages の環境変数 `VERA_ORIGIN` の先へ中継する。`VERA_ORIGIN` が未設定なら `503 origin_unset` を返す。
- `/cleanroom/` … Cleanroom。公開される中身は `cleanroom-export/`（github.com/Ag3497120/cleanroom を書き出したもの）の `index.html`・`index.txt` で、`scripts/overlay-cleanroom.mjs` がこれを `out/cleanroom/` に置く。以前は `out/` 直下に置いていたが、それではホーム（Vera）を上書きするため出力先を変えた。`src/app/cleanroom/page.tsx` は `cleanroom-export` が無いときの予備。
- `/vera/` … 以前の Vera のページ（「Vera の測定」）。

### 記録済みの実行結果の作り直し

開発ツリー（verantyx）で、`document.txt` という名前のファイルに固定の文書（下の 4 文を 1 行ずつ）を置き、そのディレクトリで次を実行して、出力の JSON を `public/vera-demo/recorded.json` にまとめる（`document.txt` の名前は出力の `source` に入るので変えない）。

```
先生が生徒に地図を渡した。
校長が職員室で書類を確認した。
母が弟に話した人を兄が呼んだ。
係が荷物を倉庫に運んだ。
```

```
export VERA_PLACEMENT=<配置 r8 のディレクトリ> PYTHONPATH=<開発ツリー> PYTHONDONTWRITEBYTECODE=1
python -m verantyx.cli ask --mode round5 --document document.txt --request-kind factual --human-present -- 誰が生徒に地図を渡した？
python -m verantyx.cli ask --mode round5 --document document.txt --request-kind factual --human-present -- 兄は誰を呼んだ？
python -m verantyx.cli ask --mode round5 --document document.txt --request-kind factual --human-present -- 係は荷物をどこに運んだ？
python -m verantyx.cli ask --mode round5 --document document.txt --request-kind factual --human-present -- 校長はどこで書類を確認した？
python -m verantyx.semantic_read --text=先生が生徒に地図を渡した。 --events
python -m verantyx.semantic_read --text=校長が職員室で書類を確認した。 --events
python -m verantyx.semantic_read --text=母が弟に話した人を兄が呼んだ。 --events
python -m verantyx.semantic_read --text=係が荷物を倉庫に運んだ。 --events
```

`recorded.json` は各項目に argv・終了コード・生の JSON（`raw`）・人が読む形（`view`）を持つ。生成後に手で編集しない。記録を作り直したら、画面の札「記録済みの実行結果（日付、配置）」も `TryVera.tsx` で合わせる。
