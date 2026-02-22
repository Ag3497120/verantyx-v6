"""
knowledge_query_ir.py
Phase 2: KnowledgeGap → LLM に渡す Knowledge Query IR を生成する。

最重要制約:
  - 問題文を含めない（problem_text_shared: false）
  - 答えを求めない（forbidden フィールドで明示）
  - 知識の構造を固定（facts スキーマ）

LLMにはこのJSONだけ渡す。
"""

import json
from dataclasses import dataclass, field
from knowledge.knowledge_gap_detector import KnowledgeGap, KnowledgeGapReport


# ---------------------------------------------------------------------------
# 1. Knowledge Query IR の型定義
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeQueryIR:
    query_id: str
    query_type: str = "knowledge_fill"
    problem_text_shared: bool = False   # 常に False（設計上の保証）
    domain_hint: list[str] = field(default_factory=list)
    need: list[dict] = field(default_factory=list)
    output_schema: dict = field(default_factory=dict)
    forbidden: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "problem_text_shared": self.problem_text_shared,
            "domain_hint": self.domain_hint,
            "need": self.need,
            "output_schema": self.output_schema,
            "forbidden": self.forbidden,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2. 固定の出力スキーマ（LLMに守らせる）
# ---------------------------------------------------------------------------

FACT_OUTPUT_SCHEMA = {
    "facts": [
        {
            "id": "string",
            "type": "definition|theorem|lemma|criterion",
            "symbol": "string",
            "statement_plain": "string (concise)",
            "preconditions": ["string"],
            "consequences": ["string"],
        }
    ]
}

# LLMに禁止する内容（forbidden フィールドで明示）
FORBIDDEN_BEHAVIORS = [
    "solve_original_problem",
    "predict_final_answer",
    "infer_user_intent_beyond_schema",
    "include_answer_to_any_question",
    "include_fields_named: answer, final, solution, correct, result",
    "provide_opinion_or_recommendation",
    "reference_original_problem_context",
]


# ---------------------------------------------------------------------------
# 3. System Prompt（LLMに渡す）
# ---------------------------------------------------------------------------

KNOWLEDGE_SYSTEM_PROMPT = """Output valid JSON only. No markdown. Be concise.
Return factual knowledge as {"facts":[...]}.
Each fact: {"id":"string","type":"definition|theorem","symbol":"string","statement_plain":"concise text","preconditions":[],"consequences":[]}.
Never include answer/solution/correct fields. Never reference any problem."""


# ---------------------------------------------------------------------------
# 4. Query Builder
# ---------------------------------------------------------------------------

class KnowledgeQueryBuilder:
    """
    KnowledgeGapReport → KnowledgeQueryIR のリスト に変換する。

    1 Gap を 1 Query に対応させる（並列問い合わせ可能にするため）。
    または Gap をバッチにまとめて 1 Query にすることもできる。
    """

    def __init__(self, batch_mode: bool = True):
        """
        batch_mode=True: 全Gapをまとめて1つのQueryにする（LLM呼び出し1回）
        batch_mode=False: Gap毎に個別Query（並列化したい場合）
        """
        self.batch_mode = batch_mode

    def build(self, gap_report: KnowledgeGapReport, query_prefix: str = "kq") -> list[KnowledgeQueryIR]:
        if gap_report.sufficient or not gap_report.gaps:
            return []

        if self.batch_mode:
            return [self._build_batch(gap_report.gaps, query_prefix)]
        else:
            return [self._build_single(gap, f"{query_prefix}_{i:03d}")
                    for i, gap in enumerate(gap_report.gaps)]

    def _build_single(self, gap: KnowledgeGap, query_id: str) -> KnowledgeQueryIR:
        need_item: dict = {
            "kind": gap.kind,
            "symbol": gap.symbol,
            "scope": gap.scope,
            "max_items": gap.max_facts,
        }
        if gap.relation:
            need_item["relation"] = gap.relation
        if gap.preconditions_needed:
            need_item["preconditions_required"] = True
        if gap.examples_needed:
            need_item["checkable_examples_required"] = True

        return KnowledgeQueryIR(
            query_id=query_id,
            query_type="knowledge_fill",
            problem_text_shared=False,
            domain_hint=gap.domain_hint,
            need=[need_item],
            output_schema=FACT_OUTPUT_SCHEMA,
            forbidden=FORBIDDEN_BEHAVIORS,
        )

    def _build_batch(self, gaps: list[KnowledgeGap], query_prefix: str) -> KnowledgeQueryIR:
        all_domains: list[str] = []
        need_items: list[dict] = []

        for gap in gaps:
            all_domains.extend(gap.domain_hint)
            need_item: dict = {
                "gap_id": gap.gap_id,
                "kind": gap.kind,
                "symbol": gap.symbol,
                "scope": gap.scope,
                "max_items": gap.max_facts,
            }
            if gap.relation:
                need_item["relation"] = gap.relation
            if gap.preconditions_needed:
                need_item["preconditions_required"] = True
            if gap.examples_needed:
                need_item["checkable_examples_required"] = True
            need_items.append(need_item)

        # domain dedup
        unique_domains = list(dict.fromkeys(all_domains))

        return KnowledgeQueryIR(
            query_id=f"{query_prefix}_batch",
            query_type="knowledge_fill",
            problem_text_shared=False,
            domain_hint=unique_domains,
            need=need_items,
            output_schema=FACT_OUTPUT_SCHEMA,
            forbidden=FORBIDDEN_BEHAVIORS,
        )


# ---------------------------------------------------------------------------
# 5. User Prompt Builder（LLMに渡すユーザーメッセージ）
# ---------------------------------------------------------------------------

def build_knowledge_user_prompt(query_ir: KnowledgeQueryIR) -> str:
    """
    LLMへのユーザーメッセージ。
    問題文は含まない。Knowledge Query IR だけ渡す。
    """
    return (
        "KNOWLEDGE QUERY:\n"
        f"{query_ir.to_json()}\n\n"
        "Return JSON matching output_schema. No answers. No problem-solving. Facts only."
    )
