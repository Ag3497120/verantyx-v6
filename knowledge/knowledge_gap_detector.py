"""
knowledge_gap_detector.py
Phase 1-2: Verantyx分解後、「不足知識」を特定する。

入力: problem_ir（Decomposerが作ったもの）
出力: KnowledgeGapRequest のリスト

問題文はここでは見ない。ir の構造だけを見る。
"""

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 1. 「不足知識」の型定義
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeGap:
    """
    Verantyxが「足りない」と判定した知識片の要求仕様。
    LLMにはこれだけを渡す（問題文は渡さない）。
    """
    gap_id: str                          # e.g. "gap_001"
    kind: str                            # "definition" | "theorem" | "criterion" | "alias" | "counterexample_rule"
    symbol: str                          # 対象記号/概念 e.g. "fundamental_group"
    domain_hint: list[str] = field(default_factory=list)   # e.g. ["algebraic_topology"]
    scope: str = "undergraduate"         # 知識の深さ
    max_facts: int = 5                   # 最大何件返してほしいか
    relation: str = ""                   # theorem/lemma の場合: どんな関係か e.g. "covering_space -> fundamental_group"
    preconditions_needed: bool = True    # preconditions を必ず含める
    examples_needed: bool = True         # checkable_examples を必ず含める


@dataclass
class KnowledgeGapReport:
    gaps: list[KnowledgeGap] = field(default_factory=list)
    gap_count: int = 0
    sufficient: bool = False  # True = Verantyx単独で解ける（LLM不要）
    reason: str = ""


# ---------------------------------------------------------------------------
# 2. PieceDB へのアクセス（差し替えポイント）
# ---------------------------------------------------------------------------

class PieceDBInterface:
    """
    現行 PieceDB の検索インターフェース。
    piece_db.py の実装に合わせて差し替える。
    """
    def has_piece(self, piece_name: str) -> bool:
        raise NotImplementedError

    def search_by_domain_and_op(self, domain: str, operation: str) -> list[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. ドメイン × オペレーション → 必要知識の辞書
#    （Cross DBに無いときに何を補充すべきかのルール）
# ---------------------------------------------------------------------------

# domain -> operation -> 何を聞けばいいか
DOMAIN_KNOWLEDGE_NEEDS: dict[str, dict[str, list[dict]]] = {
    "number_theory": {
        "is_prime": [
            {"kind": "definition", "symbol": "prime_number"},
            {"kind": "criterion",  "symbol": "primality_test", "relation": "trial_division|miller_rabin"},
        ],
        "factorial": [
            {"kind": "definition", "symbol": "factorial"},
        ],
        "gcd": [
            {"kind": "definition", "symbol": "greatest_common_divisor"},
            {"kind": "theorem",    "symbol": "euclidean_algorithm"},
        ],
    },
    "graph_theory": {
        "shortest_path": [
            {"kind": "theorem", "symbol": "dijkstra", "relation": "weighted_graph -> shortest_path"},
        ],
        "spanning_tree": [
            {"kind": "theorem", "symbol": "kruskal", "relation": "undirected_graph -> MST"},
        ],
        "is_bipartite": [
            {"kind": "criterion", "symbol": "bipartite_check", "relation": "graph -> 2-colorability"},
        ],
    },
    "algebraic_topology": {
        "fundamental_group": [
            {"kind": "definition", "symbol": "fundamental_group"},
            {"kind": "theorem",    "symbol": "van_kampen", "relation": "pushout -> fundamental_group"},
        ],
        "covering_space": [
            {"kind": "definition", "symbol": "covering_space"},
            {"kind": "theorem",    "symbol": "lifting_theorem"},
        ],
    },
    "combinatorics": {
        "binomial_coefficient": [
            {"kind": "definition", "symbol": "binomial_coefficient"},
            {"kind": "theorem",    "symbol": "pascals_identity"},
        ],
        "permutation": [
            {"kind": "definition", "symbol": "permutation"},
        ],
    },
    "logic": {
        "satisfiability": [
            {"kind": "definition", "symbol": "cnf_form"},
            {"kind": "criterion",  "symbol": "dpll", "relation": "cnf -> satisfiability"},
        ],
        "validity": [
            {"kind": "definition", "symbol": "logical_validity"},
        ],
    },
    # 追加: biology, chemistry, physics 等は随時
}

# IR の operations が PieceDB に存在しない場合の fallback
FALLBACK_NEEDS: list[dict] = [
    {"kind": "definition", "symbol": "{symbol}", "scope": "undergraduate-to-PhD concise"},
    {"kind": "theorem_candidates", "relation": "{domain} -> {operation}", "max_items": 3},
]


# ---------------------------------------------------------------------------
# 4. メイン: GapDetector
# ---------------------------------------------------------------------------

class KnowledgeGapDetector:
    """
    problem_ir を受け取り、PieceDB との照合で不足知識を検出する。
    問題文は一切見ない。
    """

    def __init__(self, piece_db: PieceDBInterface | None = None):
        self.piece_db = piece_db  # None の場合は辞書ベースのみで動く

    def detect(self, problem_ir: dict) -> KnowledgeGapReport:
        """
        Args:
            problem_ir: {
                "domain_hint": ["number_theory"],
                "entities": [...],
                "query": {"ask": "is_prime", "of": "n"},
                "candidate_programs": [...],
                "missing": [...],  # NEW: Decomposer が抽出した不足知識ニーズ
                ...
            }
        """
        # ── NEW: Decomposer の missing フィールドを直接 Gap に変換 ──
        missing_needs = problem_ir.get("missing", [])
        if missing_needs:
            gaps = []
            for i, need in enumerate(missing_needs):
                gap = KnowledgeGap(
                    gap_id=f"gap_{i+1:03d}",
                    kind=need.get("kind", "definition"),
                    symbol=need.get("concept", "unknown"),
                    domain_hint=[need.get("domain", "general")],
                    scope=need.get("scope", "concise"),
                    max_facts=need.get("max_facts", 5),
                    relation=need.get("relation", ""),
                )
                gaps.append(gap)
            return KnowledgeGapReport(
                gaps=gaps,
                gap_count=len(gaps),
                sufficient=False,
                reason=f"{len(gaps)}_gaps_from_decomposer_missing",
            )

        # ── Legacy path: missing がない場合は従来ロジック ──
        domain_hints = problem_ir.get("domain_hint", [])
        query = problem_ir.get("query", {})
        operation = query.get("ask", "")
        candidate_programs = problem_ir.get("candidate_programs", [])

        gaps: list[KnowledgeGap] = []
        gap_counter = 1

        for domain in domain_hints:
            domain_lower = domain.lower()

            # ① PieceDB にあるか確認
            if self.piece_db and self.piece_db.has_piece(f"{domain_lower}_{operation}"):
                continue  # 持っている → Gap なし

            # ② 辞書から必要知識を取得
            domain_needs = DOMAIN_KNOWLEDGE_NEEDS.get(domain_lower, {})
            op_needs = domain_needs.get(operation, [])

            if not op_needs:
                # fallback: 汎用クエリ
                op_needs = [
                    {"kind": "definition", "symbol": operation or domain_lower},
                    {"kind": "theorem_candidates",
                     "relation": f"{domain_lower} -> {operation}",
                     "max_items": 3},
                ]

            for need in op_needs:
                gap = KnowledgeGap(
                    gap_id=f"gap_{gap_counter:03d}",
                    kind=need.get("kind", "definition"),
                    symbol=need.get("symbol", operation),
                    domain_hint=[domain_lower],
                    scope=need.get("scope", "undergraduate"),
                    max_facts=need.get("max_items", 5),
                    relation=need.get("relation", ""),
                    preconditions_needed=True,
                    examples_needed=True,
                )
                gaps.append(gap)
                gap_counter += 1

        # candidate_programs があっても piece が全部未知なら gap
        if candidate_programs and self.piece_db:
            for cp in candidate_programs:
                piece_name = cp.get("piece", "")
                if piece_name and not self.piece_db.has_piece(piece_name):
                    gap = KnowledgeGap(
                        gap_id=f"gap_{gap_counter:03d}",
                        kind="definition",
                        symbol=piece_name,
                        domain_hint=domain_hints,
                        scope="technical",
                        max_facts=3,
                    )
                    gaps.append(gap)
                    gap_counter += 1

        sufficient = len(gaps) == 0
        return KnowledgeGapReport(
            gaps=gaps,
            gap_count=len(gaps),
            sufficient=sufficient,
            reason="piece_db_sufficient" if sufficient else f"{len(gaps)}_gaps_detected",
        )
