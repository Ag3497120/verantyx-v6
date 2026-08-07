"""Build the site's catalogue from the repositories' own READMEs.

The extraction runs through Vera rather than beside it: the same loaders, the
same sentence splitter, the same Japanese segmentation. That matters for one
reason — the bot on the site claims to answer from stored facts, and this is
where those facts come from. If the pipeline that built them were a separate
pile of string handling, the claim would be about a different system than the
one the site is describing.

What is stored per project, and why each one:

    summary     the first real sentence of the README, verbatim. Not a
                paraphrase: a paraphrase is the tool marking its own homework.
    topics      the cores Vera extracted, ranked by document frequency inside
                that README. These become the bot's match keys, so what a
                visitor can ask about is exactly what the document is about.
    facts       sentences carrying a number or a named claim, verbatim, with
                nothing added.

Nothing here is generated text. Every string a visitor sees came out of a
README the author wrote.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, "/Users/motonishikoudai/Projects/Verantyx-Vera-alpha")

from verantyx.cross_store import CrossStore                       # noqa: E402
from verantyx.document_ingest import (Document, _rejoin_abbreviations,  # noqa: E402
                                      _SENT, ingest_documents)
from verantyx.document_loaders import load_path                   # noqa: E402
from verantyx.lang import ja_content_runs, detect                 # noqa: E402

ROOT = pathlib.Path(
    "/private/tmp/claude-501/-Users-motonishikoudai-Projects-Vera/"
    "a06a7c3f-b137-42fd-9f8c-f1cedeeae514/scratchpad/repo_readmes"
)
OUT = pathlib.Path(
    "/private/tmp/claude-501/-Users-motonishikoudai-Projects-Vera/"
    "a06a7c3f-b137-42fd-9f8c-f1cedeeae514/scratchpad/verantyx-v6/src/lib/catalogue.ts"
)

#: Words that are true of every repository and so distinguish none of them.
#: A key that matches everything is a key that answers nothing.
UNIVERSAL = {
    "github", "repository", "repo", "license", "mit", "install", "usage",
    "python", "run", "code", "file", "files", "project", "version", "release",
    "com", "https", "http", "www", "org", "src", "readme", "docs", "doc",
    "リポジトリ", "インストール", "使い方", "ライセンス", "実行", "ファイル",
    "プロジェクト", "バージョン", "以下", "場合", "こと", "もの", "ため",
}

#: A topic is meant to be a thing, and the English decomposer hands back
#: whatever word carried the sentence — which for a README is very often the
#: imperative verb of an instruction. `fix`, `download`, `converts` and
#: `achieve` were all ranking as topics of repositories they say nothing
#: about. Verbs are excluded by list because English gives no reliable
#: morphological tell, and the list only has to cover the register a README
#: is written in.
VERBS = {
    "fix", "fixes", "fixed", "achieve", "achieves", "convert", "converts",
    "load", "loads", "contain", "contains", "download", "downloads", "add",
    "adds", "use", "uses", "using", "make", "makes", "get", "gets", "set",
    "sets", "build", "builds", "start", "starts", "open", "opens", "create",
    "creates", "check", "checks", "see", "note", "notes", "want", "need",
    "needs", "give", "gives", "take", "takes", "put", "keep", "keeps",
    "return", "returns", "call", "calls", "send", "sends", "show", "shows",
    "launch", "symptom", "symptoms", "introduction", "overview", "example",
    "examples", "step", "steps", "result", "results", "output", "input",
    "user", "users", "text", "web", "roles", "base", "setup", "tools",
    "reference", "rationale", "event", "events", "application", "model",
    "models", "architecture", "structure", "problem", "solution", "feature",
    "features", "option", "options", "command", "commands", "config",
}

#: The English decomposer marks a proper noun by appending `#p`. That is
#: internal bookkeeping and has no business on a page.
PROPER_TAG = "#p"

#: A sentence worth quoting states something checkable — a measurement, a
#: capability, a boundary. These are the shapes those take.
CLAIM = re.compile(
    r"\d+\s*%|\d[\d,]{2,}|\bno\b|\bnever\b|\bwithout\b|\bzero\b|\bonly\b"
    r"|できません|しません|ありません|不要|のみ|だけ|ゼロ|なし"
)

NOISE = re.compile(r"^[\s#>*\-=|`\[\]!]+$|^\s*$|^!\[|^\[!\[|^<")


#: A README often opens with YAML front matter, and `title: "…"` is metadata
#: rather than the first thing the document says.
FRONT_MATTER = re.compile(r"^\s*(title|emoji|colorFrom|colorTo|sdk|app_file|"
                          r"pinned|license|tags|short_description)\s*:", re.I)
#: Markdown emphasis and quote marks survive the loader because they are part
#: of the sentence; they are stripped only where a sentence is QUOTED, so the
#: reader sees prose rather than syntax.
MD_NOISE = re.compile(r"\*{1,3}|^>+\s*|`+|_{2,}")


def clean_quote(s: str) -> str:
    return re.sub(r"\s+", " ", MD_NOISE.sub("", s)).strip(" -:—")


def sentences(text: str) -> list[str]:
    out = []
    for raw in _rejoin_abbreviations(_SENT.split(text or "")):
        s = raw.strip()
        if not s or NOISE.match(s):
            continue
        if FRONT_MATTER.match(s):
            continue
        s = clean_quote(s)
        if 24 <= len(s) <= 260:
            out.append(s)
    return out


def topics_for(text: str, name: str) -> list[str]:
    """Cores Vera extracts, ranked by how often they carry a sentence."""
    one = CrossStore()
    one.track_provenance = True
    ingest_documents(one, [Document(source=name, text=text)])
    counts: dict[str, int] = {}
    for core, facets in one.crosses.items():
        clean = core[: -len(PROPER_TAG)] if core.endswith(PROPER_TAG) else core
        clean = clean.strip("_-. ")
        if len(clean) < 3:
            continue
        low = clean.lower()
        if low in UNIVERSAL or clean in UNIVERSAL or low in VERBS:
            continue
        if re.fullmatch(r"[\d.,/_-]+", clean):
            continue
        counts[clean] = counts.get(clean, 0) + sum(facets.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [c for c, _ in ranked[:14]]


def main() -> None:
    repos = {
        r["name"]: r
        for r in json.loads((ROOT / "_repos.json").read_text(encoding="utf-8"))
    }
    entries = []
    for path in sorted(ROOT.glob("*.md")):
        name = path.stem
        meta = repos.get(name)
        if not meta:
            continue
        loaded = load_path(str(path))
        if loaded.get("verdict") != "ANSWER":
            print(f"  skipped {name}: {loaded.get('verdict')}")
            continue
        text = loaded["document"].text
        sents = sentences(text)
        if not sents:
            print(f"  skipped {name}: no prose")
            continue

        summary = next((s for s in sents if len(s) >= 40), sents[0])
        facts = [s for s in sents if CLAIM.search(s)][:4]
        entries.append(
            {
                "name": name,
                "url": meta["url"],
                "description": (meta.get("description") or "").strip(),
                "language": (meta.get("primaryLanguage") or {}).get("name") or "",
                "stars": meta.get("stargazerCount", 0),
                "updated": (meta.get("updatedAt") or "")[:10],
                "lang": detect(text[:4000]),
                "summary": summary,
                "topics": topics_for(text, name),
                "facts": facts,
                "chars": len(text),
                "sentences": len(sents),
            }
        )
        print(f"  {len(text):7,d}  {len(sents):4d} sent  {len(entries[-1]['topics']):2d} topics  {name}")

    total_chars = sum(e["chars"] for e in entries)
    header = f"""/* GENERATED — do not edit by hand.
 *
 * Built by scripts/build_catalogue.py, which reads the READMEs of every
 * non-fork repository and runs them through Vera itself: the same loaders,
 * the same sentence splitter, the same Japanese segmentation the engine uses
 * on any other corpus.
 *
 * {len(entries)} repositories · {total_chars:,} characters read.
 *
 * Every string below is verbatim from a README. Summaries are the first real
 * sentence, not a paraphrase — a paraphrase would be the tool marking its own
 * homework. Topics are the cores Vera extracted, ranked by mass within that
 * document, which is why they are also the bot's match keys: what a visitor
 * can ask about is exactly what the document is about.
 */

export type CatalogueEntry = {{
  name: string;
  url: string;
  description: string;
  language: string;
  stars: number;
  updated: string;
  lang: string;
  summary: string;
  topics: string[];
  facts: string[];
  chars: number;
  sentences: number;
}};

export const CATALOGUE_CHARS = {total_chars};

export const CATALOGUE: CatalogueEntry[] = """
    OUT.write_text(
        header + json.dumps(entries, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"\n  {len(entries)} entries, {total_chars:,} chars -> {OUT.name}")


if __name__ == "__main__":
    main()
