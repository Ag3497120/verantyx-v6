"""
knowledge_pipeline_v2.py
統合パイプライン v2: 深いWikipedia取得 + 概念自動抽出

変更点:
  - WikiKnowledgeFetcherV2（セクション取得、日本語fallback）
  - concept_extractor_v2 によるNER的概念抽出
  - Wikipedia facts の CrossPiece変換を構造化
  - LLM呼び出し前にWikipediaを最大活用
"""

from dataclasses import dataclass, field

from knowledge.knowledge_gap_detector import KnowledgeGapDetector
from knowledge.knowledge_query_ir import KnowledgeQueryBuilder
from knowledge.llm_knowledge_fetcher import LLMKnowledgeFetcher
from knowledge.wiki_knowledge_fetcher_v2 import WikiKnowledgeFetcherV2
from knowledge.knowledge_sanitizer import KnowledgeSanitizer
from knowledge.cross_knowledge_mapper import CrossKnowledgeMapper, CrossPiece
from knowledge.knowledge_audit_bundle import (
    KnowledgeAuditBundle,
    KnowledgeAuditBundleBuilder,
)


@dataclass
class KnowledgePipelineResult:
    new_pieces: list[CrossPiece] = field(default_factory=list)
    audit: KnowledgeAuditBundle = field(default_factory=lambda: KnowledgeAuditBundle(problem_text_shared=False))
    sufficient: bool = False
    accepted_count: int = 0
    rejected_count: int = 0
    wiki_hits: int = 0         # v2: Wikipedia ヒット数
    wiki_sections: int = 0     # v2: 取得セクション数


class KnowledgePipelineV2:
    """Knowledge パイプライン v2"""

    def __init__(self, piece_db=None):
        self.gap_detector = KnowledgeGapDetector(piece_db=piece_db)
        self.query_builder = KnowledgeQueryBuilder(batch_mode=True)
        self.fetcher = LLMKnowledgeFetcher(max_tokens=1024)
        self.wiki_fetcher = WikiKnowledgeFetcherV2(
            max_chars=4000,
            use_sections=False,   # Parse API重すぎ→Summary APIのみ
            use_jp_fallback=True,
            follow_links=False,
        )
        self.mapper = CrossKnowledgeMapper()
        self.piece_db = piece_db

    def run(self, ir: dict, extra_concepts: list = None) -> KnowledgePipelineResult:
        """
        Args:
            ir: problem_ir
            extra_concepts: concept_extractor_v2 で抽出された追加概念
        """
        audit_builder = KnowledgeAuditBundleBuilder()
        all_new_pieces = []
        total_accepted = 0
        total_rejected = 0
        wiki_hits = 0
        wiki_sections = 0

        # Phase 1: Gap検出
        gap_report = self.gap_detector.detect(ir)
        audit_builder.record_gap(gap_report)

        if gap_report.sufficient:
            return KnowledgePipelineResult(
                new_pieces=[], audit=audit_builder.build(),
                sufficient=True,
            )

        # Phase 1.5: 追加概念をgapに統合
        all_gaps = list(gap_report.gaps)
        seen_symbols = {g.symbol.lower() for g in all_gaps}

        if extra_concepts:
            from knowledge.knowledge_gap_detector import KnowledgeGap
            for i, ec in enumerate(extra_concepts):
                name = ec.name if hasattr(ec, 'name') else ec.get('name', '')
                sym = name.replace(' ', '_').lower()
                if sym not in seen_symbols:
                    all_gaps.append(KnowledgeGap(
                        gap_id=f"gap_extra_{i:03d}",
                        kind=ec.kind if hasattr(ec, 'kind') else ec.get('kind', 'definition'),
                        symbol=sym,
                        domain_hint=[ec.domain_hint if hasattr(ec, 'domain_hint') else ec.get('domain_hint', 'general')],
                        max_facts=3,
                    ))
                    seen_symbols.add(sym)

        # Phase 2: Wikipedia v2 取得（セクション単位）
        wiki_pieces = []
        for gap in all_gaps:
            try:
                wiki_resp = self.wiki_fetcher.fetch(
                    concept=gap.symbol,
                    domain=gap.domain_hint[0] if gap.domain_hint else "",
                    kind=gap.kind,
                )
                if wiki_resp.found and wiki_resp.facts:
                    wiki_hits += 1
                    for wf in wiki_resp.facts:
                        if wf.summary and len(wf.summary) > 30:
                            _domain = gap.domain_hint[0] if gap.domain_hint else "general"

                            # 構造化された transform
                            transform = {"plain": wf.summary[:600]}
                            if wf.properties:
                                transform["properties"] = wf.properties[:3]
                            if wf.formulas:
                                transform["formulas"] = wf.formulas[:3]
                            if wf.numeric_values:
                                transform["numeric"] = dict(list(wf.numeric_values.items())[:5])

                            cp = CrossPiece(
                                piece_id=wf.fact_id,
                                domain=_domain,
                                kind=gap.kind,
                                symbol=wf.concept,
                                pattern={"domain": _domain, "symbol": wf.concept,
                                         "section": wf.section},
                                transform=transform,
                                verify_spec={},
                                worldgen_spec={},
                                grammar_binding={},
                                metadata={"source": "wikipedia_v2",
                                          "url": wf.source_url,
                                          "section": wf.section},
                            )
                            wiki_pieces.append(cp)
                            wiki_sections += 1
                            total_accepted += 1
            except Exception:
                pass

        if wiki_pieces:
            all_new_pieces.extend(wiki_pieces)

        # Phase 3: LLM呼び出し — DISABLED（問題文をLLMに渡さない原則）
        # Wikipedia知識のみで回答を構成する（Phase 2の結果をそのまま使用）

        # Phase 4: Audit
        audit_builder.record_cross_pieces(all_new_pieces)
        final_audit = audit_builder.build()

        return KnowledgePipelineResult(
            new_pieces=all_new_pieces,
            audit=final_audit,
            sufficient=False,
            accepted_count=total_accepted,
            rejected_count=total_rejected,
            wiki_hits=wiki_hits,
            wiki_sections=wiki_sections,
        )
