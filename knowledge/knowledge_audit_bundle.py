"""
knowledge_audit_bundle.py
Phase 8: Knowledge パイプライン全体の採用/棄却追跡。

既存の AuditBundle に `knowledge_audit` フィールドを追加する形で使う。

監査で見えること:
  - problem_text_shared: false（設計保証の証明）
  - 何を LLM に聞いたか（gap_ids）
  - 何が採用/棄却されたか（facts）
  - どのピースに変換されたか（cross_pieces）
"""

from dataclasses import dataclass, field
from typing import Any

from knowledge.knowledge_gap_detector import KnowledgeGapReport
from knowledge.knowledge_query_ir import KnowledgeQueryIR
from knowledge.llm_knowledge_fetcher import LLMKnowledgeResponse
from knowledge.knowledge_sanitizer import SanitizationResult
from knowledge.cross_knowledge_mapper import CrossPiece


# ---------------------------------------------------------------------------
# 1. KnowledgeAuditBundle の型
# ---------------------------------------------------------------------------

@dataclass
class LLMQueryAuditEntry:
    """LLM 問い合わせ1回分の監査ログ。"""
    query_id: str
    problem_text_shared: bool = False    # 常に False（設計保証）
    domain_hint: list[str] = field(default_factory=list)
    need_count: int = 0                  # 何件の知識を要求したか
    facts_returned: int = 0             # LLMが返した件数
    facts_accepted: int = 0             # Gate-K通過件数
    facts_rejected: int = 0             # Gate-K reject件数
    rejection_reasons: list[str] = field(default_factory=list)
    llm_error: str = ""                 # LLM呼び出しエラー


@dataclass
class KnowledgeAuditBundle:
    """
    Knowledge パイプライン全体の監査ログ。
    最終 AuditBundle の `knowledge_audit` フィールドとして埋め込む。
    """
    # Gap検出
    gap_sufficient: bool = False         # True = LLM不要（既存ピースで解けた）
    gap_count: int = 0
    gap_ids: list[str] = field(default_factory=list)

    # LLM問い合わせ
    llm_queries: list[LLMQueryAuditEntry] = field(default_factory=list)
    total_facts_accepted: int = 0
    total_facts_rejected: int = 0

    # Cross変換
    cross_pieces_created: int = 0
    cross_piece_ids: list[str] = field(default_factory=list)

    # 設計保証
    problem_text_shared: bool = False    # 設計上の保証（常に False）

    def to_dict(self) -> dict:
        return {
            "gap_sufficient": self.gap_sufficient,
            "gap_count": self.gap_count,
            "gap_ids": self.gap_ids,
            "llm_queries": [
                {
                    "query_id": q.query_id,
                    "problem_text_shared": q.problem_text_shared,
                    "domain_hint": q.domain_hint,
                    "need_count": q.need_count,
                    "facts_returned": q.facts_returned,
                    "facts_accepted": q.facts_accepted,
                    "facts_rejected": q.facts_rejected,
                    "rejection_reasons": q.rejection_reasons,
                    "llm_error": q.llm_error,
                }
                for q in self.llm_queries
            ],
            "total_facts_accepted": self.total_facts_accepted,
            "total_facts_rejected": self.total_facts_rejected,
            "cross_pieces_created": self.cross_pieces_created,
            "cross_piece_ids": self.cross_piece_ids,
            "problem_text_shared": self.problem_text_shared,
        }


# ---------------------------------------------------------------------------
# 2. KnowledgeAuditBundleBuilder
# ---------------------------------------------------------------------------

class KnowledgeAuditBundleBuilder:
    """
    Knowledge パイプライン全体の各ステップ結果を受け取り、
    KnowledgeAuditBundle を構築する。
    """

    def __init__(self):
        self._bundle = KnowledgeAuditBundle(problem_text_shared=False)

    def record_gap(self, gap_report: KnowledgeGapReport) -> "KnowledgeAuditBundleBuilder":
        self._bundle.gap_sufficient = gap_report.sufficient
        self._bundle.gap_count = gap_report.gap_count
        self._bundle.gap_ids = [g.gap_id for g in gap_report.gaps]
        return self

    def record_llm_query(
        self,
        query_ir: KnowledgeQueryIR,
        llm_response: LLMKnowledgeResponse,
        sanitization: SanitizationResult,
    ) -> "KnowledgeAuditBundleBuilder":
        # 問題文非共有は型で保証されているが、監査としても記録
        assert query_ir.problem_text_shared is False
        assert llm_response.problem_text_shared is False

        facts_returned = len(llm_response.parsed.get("facts", [])) if llm_response.ok else 0

        entry = LLMQueryAuditEntry(
            query_id=query_ir.query_id,
            problem_text_shared=False,  # 常に False
            domain_hint=query_ir.domain_hint,
            need_count=len(query_ir.need),
            facts_returned=facts_returned,
            facts_accepted=len(sanitization.accepted),
            facts_rejected=len(sanitization.rejected),
            rejection_reasons=[r.get("reason", "") for r in sanitization.rejected],
            llm_error=llm_response.error,
        )
        self._bundle.llm_queries.append(entry)
        self._bundle.total_facts_accepted += len(sanitization.accepted)
        self._bundle.total_facts_rejected += len(sanitization.rejected)
        return self

    def record_cross_pieces(self, pieces: list[CrossPiece]) -> "KnowledgeAuditBundleBuilder":
        self._bundle.cross_pieces_created = len(pieces)
        self._bundle.cross_piece_ids = [p.piece_id for p in pieces]
        return self

    def build(self) -> KnowledgeAuditBundle:
        return self._bundle


# ---------------------------------------------------------------------------
# 3. 最終 AuditBundle への埋め込み用ヘルパー
# ---------------------------------------------------------------------------

def embed_knowledge_audit(
    base_audit: dict,
    knowledge_audit: KnowledgeAuditBundle,
) -> dict:
    """
    既存の AuditBundle dict に knowledge_audit を追加する。

    使い方:
        audit_dict = base_audit_bundle.to_dict()
        audit_dict = embed_knowledge_audit(audit_dict, knowledge_audit_bundle)
        return audit_dict
    """
    base_audit["knowledge_audit"] = knowledge_audit.to_dict()
    return base_audit
