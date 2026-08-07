/* The site bot, built the way Vera is built.
 *
 * It answers from a stored table of facts or returns a TYPED refusal saying
 * why it cannot. There is no model behind it and no network call: the same
 * question always produces the same answer, and every answer names the
 * project it came from. A bot that improvised about an engine whose entire
 * claim is "it refuses to guess" would be the worst possible advertisement
 * for it — so this one refuses too, and shows the refusal rather than
 * dressing it up as an apology.
 *
 * Matching is deterministic and explainable: a query is reduced to terms,
 * each fact is scored by which of its keys those terms hit, and the score is
 * shown. Japanese has no word boundaries, so terms are matched by substring
 * on that side, which is the same compromise the engine itself makes.
 */

export type Verdict = 'ANSWER' | 'UNKNOWN_NO_EVIDENCE' | 'UNKNOWN_AMBIGUOUS';

export type Lang = 'en' | 'ja';

export type Fact = {
  /** Which project this fact belongs to — shown with every answer. */
  project: string;
  /** Match keys. Latin keys match on word, Japanese keys on substring. */
  keys: string[];
  answer: { en: string; ja: string };
  /** Optional in-site destination, offered rather than followed. */
  href?: string;
  hrefLabel?: { en: string; ja: string };
};

export const FACTS: Fact[] = [
  // ── Vera ────────────────────────────────────────────────────────────
  {
    project: 'Vera-α',
    keys: ['vera', 'engine', 'what is vera', 'ヴェラ', 'ベラ', 'エンジン'],
    answer: {
      en: 'Vera-α is a deterministic knowledge engine. Documents go in; it separates what the sources agree on, what changed, what they disagree about, and what nobody answered. No language model sits in the answer path, so the same documents always produce the same findings.',
      ja: 'Vera-α は決定論的な知識エンジンです。文書を投入すると、「一致していること」「変わったこと」「食い違っていること」「誰も答えていないこと」を分けます。答えの経路に言語モデルが無いので、同じ文書からは必ず同じ所見が出ます。',
    },
    href: '/vera/',
    hrefLabel: { en: 'Vera page', ja: 'Vera のページ' },
  },
  {
    project: 'Vera-α',
    keys: [
      'accuracy', 'accurate', 'precision', 'recall', 'measured', 'measure',
      'benchmark', 'evidence', 'reliable', 'reliability', 'trust', 'proof',
      'how good', 'how well', 'result', 'results', 'number', 'numbers',
      '精度', '正確', '再現率', '実測', '測定', 'ベンチマーク', '根拠',
      '信頼', '結果', '数字', 'どれくらい',
    ],
    answer: {
      en: 'Measured on real published documents: 8 findings and 8/8 recall on the Cabinet Office damage reports (4 revisions, 252,575 characters), and 6 findings and 6/6 on the MLIT 第N報 series read blind (4 revisions, 61,083 characters). Every finding was checked against its source by a person. Zero false positives on both.',
      ja: '実在の公開文書での測定です。内閣府の被害状況速報(4版・252,575字)で所見8件・再現率 8/8、国交省の第N報系列をブラインドで読んで(4版・61,083字)所見6件・再現率 6/6。全所見を人が原文と照合し、どちらも誤検出ゼロです。',
    },
    href: '/vera/#measured',
    hrefLabel: { en: 'See the numbers', ja: '数字を見る' },
  },
  {
    project: 'Vera-α',
    keys: [
      'limit', 'limits', 'limitation', 'limitations', 'weakness', 'cannot',
      "can't", 'drawback', 'downside', 'problem', 'risk', 'fail',
      '限界', '弱点', '欠点', 'できない', '苦手', 'リスク',
    ],
    answer: {
      en: 'Stated plainly: recall is measured on two corpora and both are Japanese government disaster reports, so a third format may expose a third layout defect. There is one false positive across 21.6 million characters, and it is English. And it does not write — no prose, no summarisation, no translation, no open-domain chat.',
      ja: '正直に言います。再現率は2コーパスでの測定で、どちらも日本の官庁災害報告なので、3つ目の形式が3つ目のレイアウト欠陥を出す可能性があります。2,160万字に対して誤検出が1件あり、それは英語です。そして文章を書きません — 作文も要約も翻訳も雑談もしません。',
    },
    href: '/vera/',
    hrefLabel: { en: 'Limits in full', ja: '限界の全文' },
  },
  {
    project: 'Vera-α',
    keys: [
      'refuse', 'refusal', 'unknown', 'hallucination', 'guess', 'typed',
      '拒否', '推測', '幻覚', 'ハルシネーション', '型',
    ],
    answer: {
      en: 'A refusal is a type, not an apology. UNKNOWN_NO_EVIDENCE means nothing stored matched. UNKNOWN_LOW_COVERAGE means too little of the corpus was read to stand behind an answer. UNKNOWN_DOMINANT_SOURCE means one document supplied nearly everything, so agreement would be an illusion. Each names what is missing, so the next step is a procedure.',
      ja: '拒否は謝罪ではなく型です。UNKNOWN_NO_EVIDENCE は該当する保存事実が無いこと。UNKNOWN_LOW_COVERAGE はコーパスを十分読めておらず答えを保証できないこと。UNKNOWN_DOMINANT_SOURCE は1文書がほぼ全部を供給しており、一致が見かけ倒しになること。何が欠けているかを名指すので、次の一手が手順になります。',
    },
  },
  {
    project: 'Vera-α',
    keys: [
      'document', 'pdf', 'format', 'ingest', 'import', 'file',
      '文書', '取り込み', '投入', '形式', 'ファイル',
    ],
    answer: {
      en: 'PDF, Word, HTML, CSV, TSV, JSON, Markdown and plain text, a folder at a time. A format it cannot read returns a typed refusal naming the file rather than an empty document — an empty document is indistinguishable from one that genuinely said nothing.',
      ja: 'PDF・Word・HTML・CSV・TSV・JSON・Markdown・テキストを、フォルダ単位で。読めない形式は空の文書ではなく、ファイル名を名指した型付きの拒否を返します。空の文書は「本当に何も書いていない文書」と区別がつかないからです。',
    },
  },
  {
    project: 'Vera-α',
    keys: [
      'disaster', 'emergency', 'earthquake', 'shelter', 'evacuation',
      '災害', '被災', '地震', '避難', '防災',
    ],
    answer: {
      en: 'This is the case it was measured on. Several agencies publish about one event and the question is what is actually going on. Vera keeps the disagreement instead of dissolving it into fluent summary, and separates an update (the same story told twice, ordered by time) from a conflict (two sources that genuinely disagree).',
      ja: 'これが実測に使った用途です。複数の機関が一つの事象について発表し、問いは「実際どうなっているのか」。Vera は食い違いを流暢な要約に溶かさず保ちます。そして「更新」(同じ話が2度語られ時刻で順序づく)と「係争」(本当に食い違っている)を分けます。',
    },
    href: '/vera/#measured',
    hrefLabel: { en: 'The disaster measurement', ja: '災害での実測' },
  },

  // ── Verantyx IDE ────────────────────────────────────────────────────
  {
    project: 'Verantyx IDE',
    keys: ['ide', 'macos', 'mac', 'app', 'desktop', 'アプリ', 'デスクトップ'],
    answer: {
      en: 'Verantyx is a macOS IDE that carries Vera-α. Two engines sit behind one switch in the chat header: jgen council (LLM and agent deliberation) for exploration, and Vera-a only (deterministic, typed verdicts, no LLM in the path) for anything where being wrong is expensive. A refusal is never rewritten by a model.',
      ja: 'Verantyx は Vera-α を積んだ macOS の IDE です。チャットヘッダの切替で2つのエンジンを使い分けます。探索には jgen 合議(LLM とエージェントの合議)、間違いが高くつく場面には単体 Vera-a(決定論・型付き判定・LLM を経由しない)。拒否をモデルに言い直させることはしません。',
    },
  },

  // ── Verantyx-CLI ────────────────────────────────────────────────────
  {
    project: 'Verantyx-CLI',
    keys: ['cli', 'router', 'council', 'memory', 'local', 'ルーター', '合議', '記憶'],
    answer: {
      en: 'Verantyx-CLI keeps a small local router resident and wakes larger local models only when the task needs them, carrying memory across restarts. It is a harness, not an accuracy booster for small models — the claim boundaries are in the repository.',
      ja: 'Verantyx-CLI は小さなローカルルーターを常駐させ、必要なときだけ大型ローカルモデルを起こし、再起動をまたいで記憶を運びます。ハーネスであって、小さなモデルを魔法で強くする道具ではありません。主張の境界はリポジトリに公開しています。',
    },
    href: '/verantyx-cli/',
    hrefLabel: { en: 'CLI page', ja: 'CLI のページ' },
  },

  // ── .jcross ─────────────────────────────────────────────────────────
  {
    project: '.jcross',
    keys: ['jcross', 'crossword', 'dsl', 'language', 'クロスワード', '言語'],
    answer: {
      en: '.jcross is a small DSL for crossword puzzles, with its own guide on this site.',
      ja: '.jcross はクロスワードパズルのための小さな DSL です。専用のガイドをこのサイトに置いています。',
    },
    href: '/jcross-language/',
    hrefLabel: { en: 'Language guide', ja: '言語ガイド' },
  },

  // ── Apps ────────────────────────────────────────────────────────────
  {
    project: 'Apps',
    keys: [
      'game', 'ios', 'iphone', 'pakupaku', 'fishing', 'mouth', 'talkiepress',
      'ゲーム', 'アプリ一覧', 'パクパク', '釣り',
    ],
    answer: {
      en: 'The iOS side: PakuPaku Fishing and MouthEat are played with mouth movement through face tracking, and TalkiePress is a separate project. All are listed in the Apps catalogue.',
      ja: 'iOS 側です。パクパク釣りと MouthEat はフェイストラッキングで口の動きを使って遊びます。TalkiePress は別プロジェクトです。すべて Apps の一覧にあります。',
    },
    href: '/apps/',
    hrefLabel: { en: 'Apps catalogue', ja: 'アプリ一覧' },
  },

  // ── The site itself ─────────────────────────────────────────────────
  {
    project: 'This site',
    keys: [
      'bot', 'you', 'yourself', 'how do you work', 'chatbot', 'ai',
      'ボット', 'あなた', 'この bot', 'チャットボット',
    ],
    answer: {
      en: 'I am built the way Vera is: a stored table of facts, deterministic matching, and a typed refusal when nothing matches. No model, no network call. That is deliberate — a bot that improvised about an engine whose whole claim is that it refuses to guess would be the worst possible advertisement for it.',
      ja: '私は Vera と同じ作りです。保存された事実の表、決定論的な照合、該当が無ければ型付きの拒否。モデルもネットワーク呼び出しもありません。これは意図的です。「推測しない」ことが全ての主張であるエンジンについて、当のボットが即興で喋ったら、それは最悪の宣伝になります。',
    },
  },
  {
    project: 'This site',
    keys: [
      'contact', 'contribute', 'help', 'support', 'github', 'source',
      '連絡', '貢献', 'サポート', 'ソース', '協力',
    ],
    answer: {
      en: 'The most useful contribution right now is a corpus in a format Vera has not read yet — recall is measured, but on two corpora that are both Japanese government disaster reports, and every defect the second one exposed was in layout reading. Everything is on GitHub under Ag3497120.',
      ja: 'いま最も価値があるのは、Vera がまだ読んだことのない形式のコーパスです。再現率は測定済みですが、2コーパスとも日本の官庁災害報告で、2つ目が露出させた欠陥はすべてレイアウト読取りにありました。すべて GitHub の Ag3497120 に公開しています。',
    },
    href: '/support/',
    hrefLabel: { en: 'Support', ja: 'サポート' },
  },
];

export const SUGGESTIONS: { en: string; ja: string }[] = [
  { en: 'What is Vera?', ja: 'Vera とは？' },
  { en: 'How accurate is it?', ja: '精度はどれくらい？' },
  { en: 'What are its limits?', ja: '限界は？' },
  { en: 'What documents can it read?', ja: 'どんな文書を読める？' },
];

export type Reply = {
  verdict: Verdict;
  project?: string;
  text: string;
  href?: string;
  hrefLabel?: string;
  /** Which stored keys the query hit — shown so the match is inspectable. */
  matched?: string[];
  /** Offered when the query was ambiguous between projects. */
  options?: string[];
};

const LATIN = /[a-z0-9]+/gi;
const HAS_CJK = /[぀-ヿ㐀-䶿一-鿿]/;

function sharesStem(a: string, b: string): boolean {
  const n = Math.min(a.length, b.length);
  if (n < 5) return false;
  return a.slice(0, 5) === b.slice(0, 5);
}

function terms(query: string): string[] {
  return (query.toLowerCase().match(LATIN) ?? []).filter((w) => w.length > 1);
}

/** Latin matches on whole words; Japanese has no word boundaries, so it
 *  matches on substring — the same compromise the engine itself makes. */
function score(fact: Fact, query: string): { score: number; hits: string[] } {
  const q = query.toLowerCase();
  const words = new Set(terms(query));
  const hits: string[] = [];
  let total = 0;
  for (const key of fact.keys) {
    const k = key.toLowerCase();
    if (HAS_CJK.test(key)) {
      if (q.includes(k)) {
        hits.push(key);
        total += k.length >= 3 ? 3 : 2;
      }
    } else if (k.includes(' ')) {
      if (q.includes(k)) {
        hits.push(key);
        total += 4;
      }
    } else if (words.has(k)) {
      hits.push(key);
      total += 2;
    } else if (k.length >= 5 && [...words].some((w) => sharesStem(w, k))) {
      // English inflects, and a key list cannot enumerate every form.
      // 「how accurate is it」 missed the key `accuracy` outright, which is a
      // silent recall loss rather than an honest refusal. A shared prefix of
      // five characters is enough for accurate/accuracy and
      // limit/limits/limitation, and short enough to stay explainable.
      hits.push(key);
      total += 1;
    }
  }
  return { score: total, hits };
}

export function ask(query: string, lang: Lang): Reply {
  const trimmed = query.trim();
  if (!trimmed) {
    return {
      verdict: 'UNKNOWN_NO_EVIDENCE',
      text:
        lang === 'ja'
          ? '質問を入力してください。'
          : 'Ask something and I will look it up.',
    };
  }

  const ranked = FACTS.map((fact) => ({ fact, ...score(fact, trimmed) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score);

  if (ranked.length === 0) {
    return {
      verdict: 'UNKNOWN_NO_EVIDENCE',
      text:
        lang === 'ja'
          ? 'この問いに答えられる事実を持っていません。推測はしません。扱えるのは Vera、Verantyx IDE、Verantyx-CLI、.jcross、iOS アプリ、そしてこのサイト自身についてです。'
          : 'I have no stored fact that answers that, and I will not guess. What I can speak to: Vera, the Verantyx IDE, Verantyx-CLI, .jcross, the iOS apps, and this site itself.',
    };
  }

  // A tie between two different projects is genuinely ambiguous. Saying so
  // and offering the choice is more useful than picking one and sounding
  // certain about it.
  const top = ranked[0];
  const tied = ranked.filter(
    (r) => r.score === top.score && r.fact.project !== top.fact.project
  );
  if (tied.length > 0) {
    const projects = Array.from(
      new Set([top.fact.project, ...tied.map((r) => r.fact.project)])
    );
    return {
      verdict: 'UNKNOWN_AMBIGUOUS',
      text:
        lang === 'ja'
          ? `その言葉は ${projects.join(' / ')} のどれにも当てはまります。どれについてですか。`
          : `That matches ${projects.join(' / ')} equally well. Which one do you mean?`,
      options: projects,
    };
  }

  return {
    verdict: 'ANSWER',
    project: top.fact.project,
    text: top.fact.answer[lang],
    href: top.fact.href,
    hrefLabel: top.fact.hrefLabel?.[lang],
    matched: top.hits,
  };
}
