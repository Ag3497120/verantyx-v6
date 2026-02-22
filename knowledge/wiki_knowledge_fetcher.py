"""
wiki_knowledge_fetcher.py
Wikipedia から知識を取得する。

設計原則（鉄の壁）:
  - 検索クエリは概念名のみ（問題文は含まない）
  - 返ってきた知識は facts 形式に構造化
  - Qwen は構造化のみに使う（問題文は渡さない）

フロー:
  KnowledgeNeed.concept → Wikipedia API → 要約テキスト → Qwen(構造化) → facts JSON
"""

from __future__ import annotations
import json
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WikiFact:
    """Wikipedia から抽出した事実"""
    fact_id: str
    concept: str
    summary: str           # Wikipedia要約（最初の数段落）
    properties: list[str] = field(default_factory=list)  # 抽出された性質
    formulas: list[str] = field(default_factory=list)    # 数式
    source_url: str = ""


@dataclass
class WikiKnowledgeResponse:
    """Wikipedia知識取得の結果"""
    concept: str
    found: bool
    facts: list[WikiFact] = field(default_factory=list)
    raw_text: str = ""
    error: str = ""
    problem_text_shared: bool = False  # 常にFalse（鉄の壁保証）


class WikiKnowledgeFetcher:
    """
    Wikipedia API を使って概念の知識を取得する。

    問題文は一切使わない。概念名だけで検索する。
    """

    WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, max_chars: int = 3000, use_llm_structuring: bool = True):
        self.max_chars = max_chars
        self.use_llm_structuring = use_llm_structuring

    def fetch(self, concept: str, domain: str = "", kind: str = "definition") -> WikiKnowledgeResponse:
        """
        概念名からWikipediaの知識を取得する。

        Args:
            concept: 概念名 (e.g. "Barcan_formula", "checkmate_patterns")
            domain: ドメインヒント (e.g. "modal_logic")
            kind: 知識の種類 (e.g. "definition", "theorem", "property")

        Returns:
            WikiKnowledgeResponse
        """
        # 概念名を検索クエリに変換
        search_term = self._concept_to_search_term(concept, domain)

        # Wikipedia Summary API を試す
        summary = self._fetch_summary(search_term)

        if not summary:
            # 検索APIでフォールバック
            search_result = self._search_wikipedia(search_term)
            if search_result:
                summary = self._fetch_summary(search_result)

        if not summary:
            return WikiKnowledgeResponse(
                concept=concept,
                found=False,
                error=f"Wikipedia article not found for: {search_term}",
            )

        # 要約テキストから facts を抽出（ルールベース）
        facts = self._extract_facts_rule_based(concept, summary, kind)

        return WikiKnowledgeResponse(
            concept=concept,
            found=True,
            facts=facts,
            raw_text=summary[:self.max_chars],
        )

    def _concept_to_search_term(self, concept: str, domain: str) -> str:
        """概念名をWikipedia検索用の文字列に変換"""
        # underscore → space
        term = concept.replace("_", " ")

        # 一般的すぎる概念にドメインを付加
        generic_concepts = [
            "general", "basics", "definition", "properties",
            "formula", "theorem", "criterion",
        ]
        if any(g in term.lower() for g in generic_concepts):
            if domain:
                term = f"{domain.replace('_', ' ')} {term}"

        # 特殊な接尾辞を除去
        for suffix in ["_general", "_basics"]:
            term = term.replace(suffix.replace("_", " "), "").strip()

        return term

    def _fetch_summary(self, title: str) -> Optional[str]:
        """Wikipedia Summary API から要約を取得"""
        try:
            encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
            url = f"{self.WIKI_API}{encoded}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Verantyx/1.0 (knowledge extraction; no problem text shared)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                extract = data.get("extract", "")
                if extract and len(extract) > 50:
                    return extract
        except Exception:
            pass
        return None

    def _search_wikipedia(self, query: str) -> Optional[str]:
        """Wikipedia Search API でタイトルを検索"""
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
            })
            url = f"{self.WIKI_SEARCH_API}?{params}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Verantyx/1.0",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("query", {}).get("search", [])
                if results:
                    return results[0]["title"]
        except Exception:
            pass
        return None

    def _extract_facts_rule_based(
        self, concept: str, text: str, kind: str
    ) -> list[WikiFact]:
        """ルールベースでテキストから facts を抽出"""
        facts = []

        # 文分割
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 定義文（最初の1-2文）
        if kind in ("definition", "fact"):
            definition = " ".join(sentences[:2])
            if definition:
                facts.append(WikiFact(
                    fact_id=f"wiki_{concept}_def",
                    concept=concept,
                    summary=definition[:500],
                    source_url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(concept.replace(' ', '_'))}",
                ))

        # 性質・定理（"is", "states that", "implies" を含む文）
        property_patterns = [
            r'(?:states?\s+that|asserts?\s+that|implies?\s+that)',
            r'(?:is\s+equivalent\s+to|if\s+and\s+only\s+if)',
            r'(?:is\s+(?:a|an|the)\s+\w+)',
            r'(?:can\s+be\s+(?:expressed|written|stated|defined))',
        ]
        for sent in sentences[2:]:
            for pat in property_patterns:
                if re.search(pat, sent, re.IGNORECASE):
                    facts.append(WikiFact(
                        fact_id=f"wiki_{concept}_prop_{len(facts)}",
                        concept=concept,
                        summary=sent[:300],
                        properties=[sent[:200]],
                    ))
                    break
            if len(facts) >= 5:
                break

        # 数式パターンの抽出
        formula_patterns = re.findall(
            r'(?:\$[^$]+\$|\\[a-z]+\{[^}]+\}|[A-Z]\s*=\s*[^.,]+)',
            text
        )
        for f in formula_patterns[:3]:
            if len(f) > 5:
                if facts:
                    facts[0].formulas.append(f.strip())

        # テキスト全体も保存（Qwen構造化用）
        if not facts:
            facts.append(WikiFact(
                fact_id=f"wiki_{concept}_raw",
                concept=concept,
                summary=text[:500],
            ))

        return facts

    def fetch_batch(
        self, needs: list[dict]
    ) -> list[WikiKnowledgeResponse]:
        """
        複数の KnowledgeNeed を一括取得。

        Args:
            needs: [{"concept": "...", "domain": "...", "kind": "..."}]

        Returns:
            list[WikiKnowledgeResponse]
        """
        responses = []
        for need in needs:
            resp = self.fetch(
                concept=need.get("concept", ""),
                domain=need.get("domain", ""),
                kind=need.get("kind", "definition"),
            )
            responses.append(resp)
        return responses


# ──────────────────────────────────────────────────────────────
# 便利関数
# ──────────────────────────────────────────────────────────────

def fetch_wiki_knowledge(concept: str, domain: str = "") -> Optional[str]:
    """
    概念名だけでWikipediaの要約を取得する簡易関数。

    Returns:
        要約テキスト or None
    """
    fetcher = WikiKnowledgeFetcher()
    resp = fetcher.fetch(concept, domain)
    if resp.found and resp.facts:
        return resp.facts[0].summary
    return None
