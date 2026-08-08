'use client';

import Footer from '@/components/Footer';
import { useLanguage } from '@/lib/i18n';

/* An article page, deliberately not a product page.
 *
 * It does not use <Navbar/>. The site nav carries a "GitHub · CLI" call to
 * action, a theme picker and a language toggle, and a reader arriving from a
 * link aggregator reads that chrome as an advertisement before they have read
 * a sentence. What replaces it is one line back to the site root.
 *
 * No install command above the fold, no buttons, no demo embed. Code links
 * are three lines at the bottom, after the argument is finished. If the
 * writing does not earn the click, nothing above it should try to.
 */

type L = { en: string; ja: string };

export default function MetamorphicParserPage() {
  const { lang } = useLanguage();
  const t = (o: L) => o[lang];
  const ja = lang === 'ja';

  const body: React.CSSProperties = {
    color: 'var(--ink-2)',
    fontSize: '1.02rem',
    lineHeight: ja ? 1.95 : 1.72,
    margin: '0 0 1.15rem',
  };
  const h2: React.CSSProperties = {
    fontSize: 'clamp(1.2rem, 3vw, 1.55rem)',
    fontWeight: 700,
    margin: '2.6rem 0 0.9rem',
    letterSpacing: ja ? '0' : '-0.01em',
  };
  const pre: React.CSSProperties = {
    background: 'var(--surface-2, rgba(255,255,255,0.05))',
    border: '1px solid var(--line, rgba(255,255,255,0.09))',
    borderRadius: '0.6rem',
    padding: '0.9rem 1.1rem',
    overflowX: 'auto',
    fontSize: '0.83rem',
    lineHeight: 1.75,
    margin: '0 0 1.3rem',
  };
  const note: React.CSSProperties = {
    ...body,
    color: 'var(--ink-3)',
    fontSize: '0.92rem',
    borderLeft: '3px solid var(--line, rgba(255,255,255,0.14))',
    paddingLeft: '1rem',
  };

  return (
    <main lang={lang} className="relative text-white min-h-screen" style={{ overflowX: 'clip' }}>
      {/* One line of chrome. No CTA. */}
      <div
        className="px-5 sm:px-6 py-4"
        style={{ borderBottom: '1px solid var(--line, rgba(255,255,255,0.08))' }}
      >
        <div className="mx-auto w-full max-w-3xl">
          <a href="/" style={{ color: 'var(--ink-3)', fontSize: '0.85rem', textDecoration: 'none' }}>
            ← Verantyx
          </a>
        </div>
      </div>

      <article className="px-5 sm:px-6 pt-12 sm:pt-16 pb-16">
        <div className="mx-auto w-full max-w-3xl">
          <h1
            className="font-display font-extrabold"
            style={{
              fontSize: ja ? 'clamp(1.7rem, 5vw, 2.5rem)' : 'clamp(1.9rem, 5.5vw, 2.9rem)',
              lineHeight: ja ? 1.35 : 1.15,
              margin: '0 0 0.9rem',
            }}
          >
            {t({
              en: 'My parser finds bugs in its own reading, without an answer key',
              ja: '答え合わせなしで、パーサが自分の読みの誤りを見つける',
            })}
          </h1>
          <p style={{ color: 'var(--ink-3)', fontSize: '0.88rem', margin: '0 0 2.4rem' }}>
            {t({ en: '8 August 2026', ja: '2026年8月8日' })}
          </p>

          <p style={body}>
            {t({
              en: 'I have a parser that reads government documents and reports when two sources contradict each other about the same thing. It runs on Japanese disaster bulletins — one report says a town’s water is out, a later one says restored, and someone has to notice.',
              ja: '官庁の文書を読み、同じ対象について二つの出典が食い違ったときに報告するパーサを書いています。日本の災害速報に対して動きます — ある報は断水と書き、後の報は復旧済と書く。誰かがそれに気づかなければなりません。',
            })}
          </p>
          <p style={body}>
            {t({
              en: 'The parser has bugs. Every new document format produces a new one. What I could not work out for a long time was how to find them without me reading the output, because “is this reading correct?” needs somebody who knows what the document means.',
              ja: 'このパーサには欠陥があります。新しい形式の文書を入れるたびに新しいものが出ます。長いあいだ分からなかったのは、私が出力を読まずにそれを見つける方法でした。「この読みは正しいか」は、文書の意味を知っている人を要するからです。',
            })}
          </p>
          <p style={body}>
            {t({
              en: 'Then I noticed I had been asking the wrong question. There is a different one that does not need the world:',
              ja: 'そこで、問いを間違えていたことに気づきました。世界を必要としない別の問いがあります。',
            })}
          </p>
          <pre style={pre}>
            <code>{t({
              en: 'not:  is this reading correct?\nbut:  do two readings of the SAME CONTENT agree?',
              ja: '不可: この読みは正しいか？\n可能: 同じ内容の二つの読みは一致するか？',
            })}</code>
          </pre>
          <p style={body}>
            {t({
              en: 'If they disagree, one of them is wrong. That is a proof, not a heuristic, and it costs no human. This is metamorphic testing — the trick is finding a transform where you can argue the direction, not just the disagreement.',
              ja: '一致しなければ、どちらかが誤りです。これは発見的手法ではなく証明で、人手を一切要しません。メタモルフィックテストの一種ですが、要点は「食い違った」だけでなく「どちらが誤りか」まで言える変換を見つけることです。',
            })}
          </p>

          <h2 style={h2}>{t({ en: 'The transform', ja: '変換' })}</h2>
          <p style={body}>
            {t({
              en: 'Japanese does not put spaces between words. So a space between two kanji in running prose was put there by the PDF extractor, not by the author. That gives you something stronger than “these two readings differ”:',
              ja: '日本語は語と語の間に空白を置きません。したがって散文中の漢字と漢字の間の空白は、著者ではなく PDF 抽出器が入れたものです。ここから「二つの読みが違う」より強いことが言えます。',
            })}
          </p>
          <p style={{ ...body, fontWeight: 700, color: 'var(--ink)' }}>
            {t({ en: 'LAYOUT CANNOT ADD INFORMATION.', ja: 'レイアウトは情報を足せない。' })}
          </p>
          <p style={body}>
            {t({
              en: 'If closing up an extractor’s space makes a claim disappear, the claim was manufactured by the whitespace. Not “suspicious” — spurious. No arrangement of spaces is evidence that a town has water.',
              ja: '抽出器が入れた空白を詰めたときに主張が消えるなら、その主張は空白が作り出したものです。「疑わしい」ではなく偽です。空白の並びが「町に水が来ている」証拠になったことは一度もありません。',
            })}
          </p>
          <p style={body}>{t({ en: 'Concretely, from a real ministry PDF:', ja: '実際の省庁 PDF から、具体的には:' })}</p>
          <pre style={pre}>
            <code>{t({
              en: '「全 12 戸が断水しています」 →  parser reads: 全 ("all") is out of water\n「全12戸が断水しています」   →  parser reads: 全12戸 ("all 12 households")',
              ja: '「全 12 戸が断水しています」 →  パーサの読み: 「全」が断水している\n「全12戸が断水しています」   →  パーサの読み: 「全12戸」が断水している',
            })}</code>
          </pre>
          <p style={body}>
            {t({
              en: 'The first is a fragment. Nobody had to read either one to know that one of them is wrong.',
              ja: '前者は断片です。どちらかが誤りであると知るのに、誰も読む必要がありませんでした。',
            })}
          </p>

          <h2 style={h2}>{t({ en: 'What it actually found', ja: '実際に見つかったもの' })}</h2>
          <p style={body}>
            {t({
              en: 'Run across five corpora of real published documents:',
              ja: '実際に公開された文書、5コーパスに対して実行した結果:',
            })}
          </p>
          <pre style={pre}>
            <code>{t({
              en: '13  proven defects on two ministry PDF series\n 0  on statutes, municipal HTML, operator press releases',
              ja: '13 件  省庁 PDF 2系列で証明された欠陥\n 0 件  法令・自治体HTML・事業者リリース',
            })}</code>
          </pre>
          <p style={body}>
            {t({
              en: 'Not typos. Things like a claim about water restoration filed under 自治体 (“municipality”, the generic word) instead of the actual municipality’s name, and a service disruption filed under 路線 (“route”) instead of the line name. Anyone asking about their own town or their own train line got nothing back.',
              ja: '誤字ではありません。復旧に関する主張が、実際の市町村名ではなく「自治体」という一般語に載っていた。運休が路線名ではなく「路線」に載っていた。自分の町や自分の路線について尋ねた人には、何も返っていませんでした。',
            })}
          </p>

          <h2 style={h2}>
            {t({
              en: 'The repair is mechanical, because the answer key is internal',
              ja: '答えが内部にあるので、修復は機械的にできる',
            })}
          </h2>
          <p style={body}>
            {t({
              en: 'Once you can prove a defect, you can propose a fix and measure it. The gate:',
              ja: '欠陥を証明できるなら、修復を提案して測ることができます。関門は次の4つです。',
            })}
          </p>
          <pre style={pre}>
            <code>{t({
              en: 'the planted test suite still passes\nno confirmed finding across five corpora is lost\ncoverage does not fall\nthe count of proven defects strictly falls',
              ja: '植え込み検査に通ること\n5コーパスの確定した検出を1件も失わないこと\n被覆率が下がらないこと\n証明済みの欠陥の数が厳密に減ること',
            })}</code>
          </pre>
          <p style={body}>
            {t({
              en: 'Two candidates, and both outcomes happened — which is why both code paths exist:',
              ja: '候補は2つで、両方の結末が実際に起きました。だから両方の経路が存在します。',
            })}
          </p>
          <pre style={pre}>
            <code>{t({
              en:
'counter_split  ACCEPTED\n  A numeral and its counter are one word (12戸, 15炉).\n  Proven defects 13 → 12, coverage 73.39% → 73.39%,\n  the same 9 confirmed findings, the same 18,460 sentences placed.\n\nlayout_space   REJECTED\n  Close up ANY single space between two CJK characters.\n  Removes every proven defect — and costs 79 sentences their\n  subject, of which only 8 were the spurious claims.',
              ja:
'counter_split  受理\n  数詞とその助数詞は一語（12戸、15炉）。\n  証明済み欠陥 13 → 12、被覆率 73.39% → 73.39%、\n  確定検出は同じ9件、配置文も同じ18,460。\n\nlayout_space   却下\n  CJK 文字どうしの単一空白を「すべて」詰める。\n  証明済み欠陥は全て消えるが、79文がコアを失う。\n  そのうち偽の主張だったのは8件だけ。',
            })}</code>
          </pre>
          <p style={note}>
            {t({
              en: 'The rejected one is the more useful record. Without it, the same losing candidate gets proposed on every single run, forever. So the rejection goes into a ledger.',
              ja: '却下の方が有用な記録です。これが無いと、負ける候補が毎回、永久に提案され続けます。だから却下は台帳に書かれます。',
            })}
          </p>

          <h2 style={h2}>
            {t({
              en: 'A second oracle: the output versus the parser’s own rules',
              ja: '第二のオラクル — 出力と、パーサ自身の規則',
            })}
          </h2>
          <p style={body}>
            {t({
              en: 'The parser has guards that mean “if this pattern follows the term, the term asserts nothing”: 〜のため (“for the purpose of”), 〜による (“caused by”), 〜と認める (“deemed to be”).',
              ja: 'パーサには「この語尾が続いたら、その語は何も主張していない」というガードがあります。〜のため、〜による、〜と認める。',
            })}
          </p>
          <p style={body}>
            {t({
              en: 'So a placed claim whose tail one of those guards matches is an internal contradiction. Both the output and the rules live in the same process — no world knowledge enters. That found 7 more:',
              ja: 'したがって、置かれた主張の語尾にそのガードが一致していたら、内部矛盾です。出力も規則も同じプロセスの中にあるので、世界の知識は要りません。これで7件が出ました。',
            })}
          </p>
          <pre style={pre}>
            <code>{t({
              en: '「災害復旧のため派遣された職員」\n   →  filed "restored" on 災害派遣手当 ("disaster dispatch allowance")',
              ja: '「災害復旧のため派遣された職員」\n   →  「災害派遣手当」に 復旧 が置かれていた',
            })}</code>
          </pre>
          <p style={body}>
            {t({
              en: 'A dispatch allowance is not a restored water main.',
              ja: '派遣手当は、復旧した水道管ではありません。',
            })}
          </p>
          <p style={body}>
            {t({
              en: 'And this is the class I had hit four separate times by hand: a guard applied on the prose path and skipped on the table path. Enumeration, deeming, until, and now のため. Every time, a human found it. So I fixed the class instead of the instance — suppressions are now consulted at the one line every claim passes through, which means the hole cannot reopen as a path-skip.',
              ja: 'そしてこれは、私が手作業で4度踏んだクラスです。散文の経路にガードを適用し、表の経路で飛ばす。列挙、みなし、〜まで、そして今回の のため。毎回、人が見つけていました。だからインスタンスではなくクラスを直しました — 抑制は、すべての主張が通る唯一の一行で参照されるようになり、この穴は経路飛ばしとして再発できません。',
            })}
          </p>

          <h2 style={h2}>{t({ en: 'Where it stops', ja: '止まる場所' })}</h2>
          <p style={body}>
            {t({
              en: 'This does not make the parser self-improving in any general sense. It repairs what the parser’s own reader broke. It cannot tell you what a word it has never seen MEANS — no transformation of a document reveals that — so new vocabulary arrives as a queue with an approve button, and nothing in that path can write to the config without a person pressing it.',
              ja: 'これはパーサを一般的な意味で自己改善させるものではありません。直すのは、パーサ自身の読み取りが壊したものだけです。見たことのない語が何を意味するかは教えてくれません — どんな文書変換もそれを明かさないので — 新しい語彙は承認ボタン付きの待ち行列として届き、その経路のどこも、人が押さない限り設定を書き換えられません。',
            })}
          </p>
          <p style={body}>
            {t({
              en: 'The honest summary: metamorphic relations gave me an answer key for the class of bug where the input was misread, and nothing at all for the class where the parser is simply ignorant. That turned out to be a bigger fraction than I expected, and a smaller one than I wanted.',
              ja: '正直にまとめると、メタモルフィックな関係は「入力を読み違えた」種類の欠陥に対して答えをくれ、「単に知らない」種類には何もくれませんでした。前者は予想より大きく、望んだほどではありませんでした。',
            })}
          </p>

          <h2 style={h2}>
            {t({
              en: 'Numbers, because a post like this is worthless without them',
              ja: '数字 — これが無ければ、この種の文章に価値はない',
            })}
          </h2>
          <div style={{ overflowX: 'auto', margin: '0 0 1.2rem' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ background: 'var(--surface-2, rgba(255,255,255,0.05))' }}>
                  {[
                    t({ en: 'corpus', ja: 'コーパス' }),
                    t({ en: 'findings', ja: '検出' }),
                    t({ en: 'true', ja: '本物' }),
                  ].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '0.6rem 0.9rem', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  [t({ en: 'government disaster reports, 5 corpora, 4 read blind', ja: '官庁災害文書 5コーパス（うち4はブラインド）' }), '14', '14'],
                  [t({ en: 'naive keyword baseline, same documents', ja: '素朴なキーワード照合（同じ文書）' }), '38', '6'],
                  [t({ en: 'technical prose, 93 mixed EN/JA documents', ja: '技術文書93本（日英混在）' }), '5', '0'],
                ].map((row, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--line, rgba(255,255,255,0.07))' }}>
                    {row.map((c, j) => (
                      <td key={j} style={{ padding: '0.6rem 0.9rem', color: j === 0 ? 'var(--ink-2)' : 'var(--ink)' }}>{c}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={body}>
            {t({
              en: 'That last row is on the front page of the README. The engine works where documents make state claims about NAMED things — a municipality, a route, a contract. On prose it manufactures contradictions out of abstract nouns that recur in unrelated contexts, and a wiki is the wrong input.',
              ja: '最後の行は README の冒頭に書いてあります。このエンジンが効くのは、文書が固有名を持つものについて状態を主張している場合です — 市町村、路線、契約。散文では、無関係な文脈に再来する抽象名詞から存在しない矛盾を作り出します。wiki は入力として間違っています。',
            })}
          </p>

          <hr style={{ border: 0, borderTop: '1px solid var(--line, rgba(255,255,255,0.1))', margin: '2.6rem 0 1.4rem' }} />
          <p style={{ ...body, fontSize: '0.9rem', margin: 0 }}>
            {t({ en: 'Code: ', ja: 'コード: ' })}
            <a href="https://github.com/Ag3497120/Verantyx" target="_blank" rel="noopener noreferrer" style={{ color: 'rgba(var(--accent-rgb), 0.95)' }}>
              github.com/Ag3497120/Verantyx
            </a>
            {' · '}
            <code style={{ fontSize: '0.85rem' }}>pip install verantyx-vera</code>
          </p>
          <p style={{ ...body, fontSize: '0.9rem', color: 'var(--ink-3)', margin: '0.5rem 0 0' }}>
            {t({
              en: 'No LLM anywhere in the answer path, no GPU, runs offline. That last part is not a flex — the people this is for have shelter registers and hospital lists on their laptops, and the correct place for those is nowhere.',
              ja: '答えの経路に LLM はなく、GPU も不要で、オフラインで動きます。最後の点は自慢ではありません — これを使う人たちのノートPCには避難所名簿や病院リストが入っていて、それらを置くべき場所はどこにもないからです。',
            })}
          </p>
        </div>
      </article>

      <Footer />
    </main>
  );
}
