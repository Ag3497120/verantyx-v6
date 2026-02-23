"""
wiki_knowledge_fetcher_v2.py
Wikipedia 知識取得 v2 — より深い知識抽出

改善点:
  1. Summary API → Full sections API（定理・公式・数値を取得）
  2. 複数セクションから構造化 facts を抽出
  3. Infobox / 数値データの抽出
  4. 関連ページのリンク先も浅く取得（1-hop）
  5. 日本語 Wikipedia fallback
  6. より賢い概念名 → 検索語変換

設計原則（鉄の壁）: 概念名のみで検索。問題文は渡さない。
"""

from __future__ import annotations
import json
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WikiFactV2:
    """Wikipedia から抽出した事実（v2: セクション情報付き）"""
    fact_id: str
    concept: str
    summary: str
    section: str = ""           # セクション名 (e.g. "Definition", "Properties")
    properties: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    numeric_values: dict = field(default_factory=dict)  # key-value の数値データ
    source_url: str = ""
    confidence: float = 1.0     # fact の信頼度


@dataclass
class WikiKnowledgeResponseV2:
    concept: str
    found: bool
    facts: list[WikiFactV2] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    raw_text: str = ""
    error: str = ""
    problem_text_shared: bool = False  # 常に False


class WikiKnowledgeFetcherV2:
    """Wikipedia 知識取得 v2 — セクション単位の深い取得"""

    WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    WIKI_SECTIONS_API = "https://en.wikipedia.org/api/rest_v1/page/mobile-sections/"
    WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
    WIKI_PARSE_API = "https://en.wikipedia.org/w/api.php"

    # 日本語 Wikipedia
    WIKI_JP_SUMMARY_API = "https://ja.wikipedia.org/api/rest_v1/page/summary/"
    WIKI_JP_SEARCH_API = "https://ja.wikipedia.org/w/api.php"

    # 知識的に重要なセクション名
    IMPORTANT_SECTIONS = {
        'definition', 'definitions', 'statement', 'theorem', 'formula',
        'properties', 'characteristics', 'description', 'overview',
        'classification', 'types', 'examples', 'applications',
        'formal definition', 'mathematical formulation', 'formal statement',
        'proof', 'derivation', 'history', 'nomenclature', 'taxonomy',
        'mechanism', 'structure', 'composition', 'function',
        'diagnosis', 'symptoms', 'treatment', 'pathophysiology',
        'etymology', 'usage', 'significance',
    }

    def __init__(self, max_chars: int = 4000, use_sections: bool = True,
                 use_jp_fallback: bool = True, follow_links: bool = False,
                 source_lang: str = "auto"):
        self.max_chars = max_chars
        self.use_sections = use_sections
        self.use_jp_fallback = use_jp_fallback
        self.follow_links = follow_links
        self.source_lang = source_lang  # "auto" | "en" | "ja" etc.

        # SSL context: macOS Python のデフォルト証明書パスが壊れている問題に対応
        import ssl
        try:
            import certifi
            self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            self._ssl_ctx = ssl.create_default_context()
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def fetch(self, concept: str, domain: str = "", kind: str = "definition") -> WikiKnowledgeResponseV2:
        """概念名からWikipedia知識を深く取得する"""
        search_term = self._concept_to_search_term(concept, domain)

        # 1) タイトル解決（複合語はそのまま、単語はドメイン付き）
        title = self._resolve_title(search_term)
        # タイトルが見つかっても内容が無関係な場合がある → 検証
        if title and not self._title_relevant(title, search_term):
            title = None
        if not title and domain:
            title = self._resolve_title(f"{search_term} {domain.replace('_', ' ')}")
        if not title:
            # 日本語 fallback（概念名が日本語の場合のみ）
            if self.use_jp_fallback and self._is_cjk(concept):
                return self._fetch_japanese(concept, domain, kind)
            return WikiKnowledgeResponseV2(concept=concept, found=False,
                                           error=f"Not found: {search_term}")

        # 2) Summary取得
        summary = self._fetch_summary(title)

        # 3) セクション取得（v2の主な改善点）
        sections_text = {}
        related = []
        if self.use_sections:
            sections_text, related = self._fetch_sections(title)

        # 4) Facts構築
        facts = self._build_facts(concept, title, summary, sections_text, kind)

        # 5) 数値抽出
        self._extract_numeric_values(facts, summary, sections_text)

        return WikiKnowledgeResponseV2(
            concept=concept,
            found=True,
            facts=facts,
            related_concepts=related[:10],
            raw_text=(summary or "")[:self.max_chars],
        )

    def _concept_to_search_term(self, concept: str, domain: str) -> str:
        """概念名を検索語に変換（v2: より賢い変換）"""
        term = concept.replace("_", " ")

        # 接尾辞の除去
        for suffix in ["general", "basics", "definition", "properties", "formula"]:
            term = re.sub(rf'\b{suffix}\b', '', term, flags=re.IGNORECASE).strip()

        # 一般的すぎる場合はドメイン付加
        if len(term.split()) <= 1 and domain:
            generic = {"test", "check", "rule", "law", "type", "class", "method"}
            if term.lower() in generic:
                term = f"{domain.replace('_', ' ')} {term}"

        return term.strip()

    @staticmethod
    def _is_cjk(text: str) -> bool:
        """テキストにCJK文字が含まれるか"""
        return any('\u3000' <= c <= '\u9fff' or '\uf900' <= c <= '\ufaff' for c in text)

    def _title_relevant(self, title: str, search_term: str) -> bool:
        """タイトルが検索語と関連しているか簡易チェック"""
        t = title.lower()
        s = search_term.lower()
        # 検索語の主要語がタイトルに含まれているか
        main_words = [w for w in s.split() if len(w) > 3]
        if not main_words:
            return True
        return any(w in t for w in main_words)

    def _resolve_title(self, search_term: str) -> Optional[str]:
        """検索語からWikipediaタイトルを解決"""
        # まず直接アクセス
        encoded = urllib.parse.quote(search_term.replace(" ", "_"), safe="")
        try:
            url = f"{self.WIKI_SUMMARY_API}{encoded}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Verantyx/2.0 (knowledge extraction)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=8, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                if data.get("extract") and len(data["extract"]) > 30:
                    return data.get("titles", {}).get("canonical", search_term)
        except Exception:
            pass

        # Search API fallback
        try:
            params = urllib.parse.urlencode({
                "action": "query", "list": "search",
                "srsearch": search_term, "srlimit": 5, "format": "json",
            })
            url = f"{self.WIKI_SEARCH_API}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Verantyx/2.0"})
            with urllib.request.urlopen(req, timeout=8, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("query", {}).get("search", [])
                if results:
                    return results[0]["title"]
        except Exception:
            pass
        return None

    def _fetch_summary(self, title: str) -> Optional[str]:
        """Summary API から要約取得"""
        try:
            encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
            url = f"{self.WIKI_SUMMARY_API}{encoded}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Verantyx/2.0", "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=8, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                return data.get("extract", "")
        except Exception:
            return None

    def _fetch_sections(self, title: str) -> tuple[dict[str, str], list[str]]:
        """
        Parse API でセクションテキストを取得。
        Returns: ({section_name: text}, [related_page_titles])
        """
        sections = {}
        related = []
        try:
            params = urllib.parse.urlencode({
                "action": "parse",
                "page": title,
                "prop": "wikitext|links|sections",
                "format": "json",
            })
            url = f"{self.WIKI_PARSE_API}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Verantyx/2.0"})
            with urllib.request.urlopen(req, timeout=12, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                parse = data.get("parse", {})

                # セクション情報
                sec_list = parse.get("sections", [])
                wikitext = parse.get("wikitext", {}).get("*", "")

                # セクション抽出
                if wikitext and sec_list:
                    sections = self._parse_wikitext_sections(wikitext, sec_list)

                # 関連リンク
                links = parse.get("links", [])
                for link in links:
                    if link.get("ns") == 0 and link.get("exists") is not None:
                        related.append(link.get("*", ""))
        except Exception:
            pass
        return sections, related

    def _parse_wikitext_sections(self, wikitext: str, sec_list: list[dict]) -> dict[str, str]:
        """wikitextからセクションを抽出（重要セクションのみ）"""
        sections = {}
        lines = wikitext.split('\n')
        current_section = "lead"
        current_text = []

        for line in lines:
            # セクションヘッダ検出
            m = re.match(r'^(={2,})\s*(.+?)\s*\1', line)
            if m:
                # 前のセクションを保存
                if current_text and current_section.lower() in self.IMPORTANT_SECTIONS or current_section == "lead":
                    text = self._clean_wikitext('\n'.join(current_text))
                    if len(text) > 30:
                        sections[current_section] = text[:1500]
                current_section = m.group(2).strip()
                current_text = []
            else:
                current_text.append(line)

        # 最後のセクション
        if current_text and current_section.lower() in self.IMPORTANT_SECTIONS:
            text = self._clean_wikitext('\n'.join(current_text))
            if len(text) > 30:
                sections[current_section] = text[:1500]

        return sections

    def _clean_wikitext(self, text: str) -> str:
        """wikitextからマークアップを除去"""
        # リンク [[target|text]] → text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
        # テンプレート {{...}} 除去（ネストなし簡易版）
        text = re.sub(r'\{\{[^}]{0,200}\}\}', '', text)
        # HTML タグ除去
        text = re.sub(r'<[^>]+>', '', text)
        # ref タグ除去
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/]*/>', '', text)
        # 余分な空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _build_facts(self, concept: str, title: str,
                     summary: Optional[str], sections: dict[str, str],
                     kind: str) -> list[WikiFactV2]:
        """Summary + セクションから構造化 facts を構築"""
        facts = []
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

        # Fact 0: Summary (定義)
        if summary and len(summary) > 30:
            facts.append(WikiFactV2(
                fact_id=f"wiki2_{concept}_summary",
                concept=concept,
                summary=summary[:800],
                section="summary",
                source_url=url,
            ))

        # Fact 1+: 重要セクション
        priority_sections = ["Definition", "Formal definition", "Statement",
                           "Properties", "Theorem", "Formula",
                           "Mathematical formulation", "Classification",
                           "Mechanism", "Structure", "Taxonomy"]

        added = 0
        for sec_name in priority_sections:
            for actual_name, text in sections.items():
                if actual_name.lower() == sec_name.lower() and text:
                    # セクション内の文を抽出
                    props = self._extract_properties(text)
                    formulas = self._extract_formulas(text)
                    facts.append(WikiFactV2(
                        fact_id=f"wiki2_{concept}_{actual_name.lower().replace(' ', '_')}",
                        concept=concept,
                        summary=text[:600],
                        section=actual_name,
                        properties=props,
                        formulas=formulas,
                        source_url=url,
                    ))
                    added += 1
                    if added >= 4:
                        break
            if added >= 4:
                break

        # 残りの重要セクション
        for sec_name, text in sections.items():
            if added >= 6:
                break
            if sec_name.lower() in self.IMPORTANT_SECTIONS:
                already = any(f.section.lower() == sec_name.lower() for f in facts)
                if not already and text:
                    facts.append(WikiFactV2(
                        fact_id=f"wiki2_{concept}_{sec_name.lower().replace(' ', '_')}_{added}",
                        concept=concept,
                        summary=text[:400],
                        section=sec_name,
                        source_url=url,
                    ))
                    added += 1

        # Fallback: セクションなしの場合
        if not facts and summary:
            facts.append(WikiFactV2(
                fact_id=f"wiki2_{concept}_raw",
                concept=concept,
                summary=summary[:500],
                source_url=url,
            ))

        return facts

    def _extract_properties(self, text: str) -> list[str]:
        """テキストから性質を抽出"""
        props = []
        patterns = [
            r'(?:states?\s+that|asserts?\s+that|implies?\s+that)\s+(.{20,200}?)(?:\.|$)',
            r'(?:is\s+equivalent\s+to)\s+(.{10,150}?)(?:\.|$)',
            r'(?:if\s+and\s+only\s+if)\s+(.{10,150}?)(?:\.|$)',
            r'(?:is\s+defined\s+(?:as|by))\s+(.{10,200}?)(?:\.|$)',
            r'(?:is\s+(?:always|never|exactly))\s+(.{10,100}?)(?:\.|$)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                props.append(m.group(1).strip())
                if len(props) >= 5:
                    return props
        return props

    def _extract_formulas(self, text: str) -> list[str]:
        """テキストから数式を抽出（LaTeX + wikitext math）"""
        formulas = []
        # <math>...</math>
        for m in re.finditer(r'<math[^>]*>(.*?)</math>', text, re.DOTALL):
            f = m.group(1).strip()
            if len(f) > 3:
                formulas.append(f)
        # $...$ style
        for m in re.finditer(r'\$([^$]+)\$', text):
            f = m.group(1).strip()
            if len(f) > 3:
                formulas.append(f)
        # Explicit equations: X = ...
        for m in re.finditer(r'([A-Z][a-z]?\s*=\s*[^,.\n]{5,60})', text):
            formulas.append(m.group(1).strip())
        return formulas[:5]

    def _extract_numeric_values(self, facts: list[WikiFactV2],
                                summary: Optional[str], sections: dict) -> None:
        """数値データを抽出して facts に付与"""
        all_text = (summary or "") + " ".join(sections.values())
        # パターン: "X is Y" where Y is numeric
        num_patterns = [
            (r'(?:has|is|equals?|approximately|about)\s+(\d+[\d,.]*)\s+(\w+)', 'value'),
            (r'(\d+[\d,.]*)\s*(?:km|m|cm|mm|kg|g|mg|°C|°F|K|Hz|eV|MeV|GeV)\b', 'measurement'),
            (r'(?:density|mass|charge|radius|period|wavelength|frequency)\s+(?:of|is|=)\s*([\d.eE+-]+)', 'physical'),
        ]
        for fact in facts:
            for pat, cat in num_patterns:
                for m in re.finditer(pat, fact.summary, re.IGNORECASE):
                    fact.numeric_values[f"{cat}_{len(fact.numeric_values)}"] = m.group(0)
                    if len(fact.numeric_values) >= 5:
                        break

    def _fetch_japanese(self, concept: str, domain: str, kind: str) -> WikiKnowledgeResponseV2:
        """日本語Wikipedia fallback"""
        term = concept.replace("_", " ")
        try:
            params = urllib.parse.urlencode({
                "action": "query", "list": "search",
                "srsearch": term, "srlimit": 3, "format": "json",
            })
            url = f"{self.WIKI_JP_SEARCH_API}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Verantyx/2.0"})
            with urllib.request.urlopen(req, timeout=8, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("query", {}).get("search", [])
                if results:
                    title = results[0]["title"]
                    # Summary取得
                    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
                    sum_url = f"{self.WIKI_JP_SUMMARY_API}{encoded}"
                    req2 = urllib.request.Request(sum_url, headers={
                        "User-Agent": "Verantyx/2.0", "Accept": "application/json",
                    })
                    with urllib.request.urlopen(req2, timeout=8, context=self._ssl_ctx) as resp2:
                        sdata = json.loads(resp2.read().decode())
                        extract = sdata.get("extract", "")
                        if extract and len(extract) > 20:
                            return WikiKnowledgeResponseV2(
                                concept=concept,
                                found=True,
                                facts=[WikiFactV2(
                                    fact_id=f"wiki2_jp_{concept}",
                                    concept=concept,
                                    summary=extract[:800],
                                    section="summary_ja",
                                    source_url=f"https://ja.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                                )],
                            )
        except Exception:
            pass
        return WikiKnowledgeResponseV2(concept=concept, found=False, error="Not found (ja)")

    def fetch_batch(self, needs: list[dict]) -> list[WikiKnowledgeResponseV2]:
        """バッチ取得"""
        return [self.fetch(
            concept=n.get("concept", ""),
            domain=n.get("domain", ""),
            kind=n.get("kind", "definition"),
        ) for n in needs]
