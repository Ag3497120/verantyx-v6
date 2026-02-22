"""
knowledge_pipeline.py
統合パイプライン: GapDetector → QueryBuilder → Fetcher → Sanitizer → Mapper → AuditBuilder

問題IRを受け取り、不足知識を検出してLLMに問い合わせ、検疫してCrossPieceに変換する。
"""

from dataclasses import dataclass, field

from knowledge.knowledge_gap_detector import KnowledgeGapDetector
from knowledge.knowledge_query_ir import KnowledgeQueryBuilder
from knowledge.llm_knowledge_fetcher import LLMKnowledgeFetcher
from knowledge.wiki_knowledge_fetcher import WikiKnowledgeFetcher
from knowledge.knowledge_sanitizer import KnowledgeSanitizer
from knowledge.cross_knowledge_mapper import CrossKnowledgeMapper, CrossPiece
from knowledge.knowledge_audit_bundle import (
    KnowledgeAuditBundle,
    KnowledgeAuditBundleBuilder,
)


@dataclass
class KnowledgePipelineResult:
    """Knowledge パイプライン全体の結果。"""
    new_pieces: list[CrossPiece] = field(default_factory=list)
    audit: KnowledgeAuditBundle = field(default_factory=lambda: KnowledgeAuditBundle(problem_text_shared=False))
    sufficient: bool = False           # True = LLM不要（既存ピースで解けた）
    accepted_count: int = 0            # 採用された facts 件数
    rejected_count: int = 0            # 棄却された facts 件数


class KnowledgePipeline:
    """
    Knowledge パイプライン全体を統括するクラス。

    フロー:
        1. GapDetector: IR → KnowledgeGapReport
        2. QueryBuilder: KnowledgeGapReport → KnowledgeQueryIR[]
        3. Fetcher: KnowledgeQueryIR → LLMKnowledgeResponse
        4. Sanitizer: LLMKnowledgeResponse → SanitizationResult (Gate-K1〜K6)
        5. Mapper: SanitizedFact[] → CrossPiece[]
        6. AuditBuilder: 全ステップの結果を KnowledgeAuditBundle にまとめる
    """

    def __init__(self, piece_db=None):
        """
        Args:
            piece_db: PieceDB インターフェース（既存ピース検索用）。
                     None の場合は辞書ベースのみで動く。
        """
        self.gap_detector = KnowledgeGapDetector(piece_db=piece_db)
        self.query_builder = KnowledgeQueryBuilder(batch_mode=True)
        self.fetcher = LLMKnowledgeFetcher(max_tokens=1024)
        self.wiki_fetcher = WikiKnowledgeFetcher(max_chars=2000)
        self.mapper = CrossKnowledgeMapper()
        self.piece_db = piece_db

    def run(self, ir: dict) -> KnowledgePipelineResult:
        """
        問題IRを受け取り、Knowledge パイプライン全体を実行する。

        Args:
            ir: problem_ir (Decomposerが作ったもの)
                {
                    "domain_hint": ["number_theory"],
                    "entities": [...],
                    "query": {"ask": "is_prime", "of": "n"},
                    "candidate_programs": [...],
                    ...
                }

        Returns:
            KnowledgePipelineResult: 新規ピース、監査ログ、採用/棄却件数
        """
        audit_builder = KnowledgeAuditBundleBuilder()
        all_new_pieces = []
        total_accepted = 0
        total_rejected = 0

        # ================================================================
        # Phase 1: Gap検出
        # ================================================================
        gap_report = self.gap_detector.detect(ir)
        audit_builder.record_gap(gap_report)

        # LLM 不要なら即終了
        if gap_report.sufficient:
            return KnowledgePipelineResult(
                new_pieces=[],
                audit=audit_builder.build(),
                sufficient=True,
                accepted_count=0,
                rejected_count=0,
            )

        # ================================================================
        # Phase 1.5: Wikipedia 知識取得（鉄の壁: 概念名のみで検索）
        # LLM の前にまず Wikipedia を引く（速い + 高品質）
        # ================================================================
        wiki_pieces = []
        for gap in gap_report.gaps:
            try:
                wiki_resp = self.wiki_fetcher.fetch(
                    concept=gap.symbol,
                    domain=gap.domain_hint[0] if gap.domain_hint else "",
                    kind=gap.kind,
                )
                if wiki_resp.found and wiki_resp.facts:
                    for wf in wiki_resp.facts:
                        if wf.summary and len(wf.summary) > 30:
                            # Wikipedia fact → CrossPiece 直接変換
                            _domain = gap.domain_hint[0] if gap.domain_hint else "general"
                            cp = CrossPiece(
                                piece_id=wf.fact_id,
                                domain=_domain,
                                kind=gap.kind,
                                symbol=wf.concept,
                                pattern={"domain": _domain, "symbol": wf.concept},
                                transform={"plain": wf.summary[:500]},
                                verify_spec={},
                                worldgen_spec={},
                                grammar_binding={},
                                metadata={"source": "wikipedia", "url": wf.source_url},
                            )
                            wiki_pieces.append(cp)
                            total_accepted += 1
            except Exception:
                pass

        if wiki_pieces:
            all_new_pieces.extend(wiki_pieces)

        # ================================================================
        # Phase 2: Query 構築 → LLM 呼び出し（Wikipedia で不足の場合）
        # ================================================================
        queries = self.query_builder.build(gap_report, query_prefix="kq")

        for query_ir in queries:
            # LLM 呼び出し
            llm_response = self.fetcher.fetch(query_ir)

            # ================================================================
            # Phase 3: Gate-K 検疫
            # ================================================================
            existing_symbols = set()
            if self.piece_db:
                # piece_db から既存シンボル取得（K5: dedup 用）
                # piece_db.all_symbols() が無い場合は空 set
                existing_symbols = getattr(self.piece_db, "all_symbols", lambda: set())()

            sanitizer = KnowledgeSanitizer(
                requested_domains=ir.get("domain_hint", []),
                existing_symbols=existing_symbols,
            )
            san_result = sanitizer.sanitize(llm_response)
            audit_builder.record_llm_query(query_ir, llm_response, san_result)

            total_accepted += len(san_result.accepted)
            total_rejected += len(san_result.rejected)

            # ================================================================
            # Phase 4: Cross 変換
            # ================================================================
            if san_result.accepted:
                new_pieces = self.mapper.map_all(san_result.accepted)
                all_new_pieces.extend(new_pieces)

        # ================================================================
        # Phase 5: Audit 構築
        # ================================================================
        audit_builder.record_cross_pieces(all_new_pieces)
        final_audit = audit_builder.build()

        return KnowledgePipelineResult(
            new_pieces=all_new_pieces,
            audit=final_audit,
            sufficient=False,
            accepted_count=total_accepted,
            rejected_count=total_rejected,
        )
