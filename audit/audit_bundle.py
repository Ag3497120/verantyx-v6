"""
AuditBundle.v1 — Transparent verification trace for Verantyx V6

This schema captures the full decision path: what LLM proposed (if any),
what CEGIS verified, and what answer was finally accepted.

Design principle: "LLM is a proposal generator, CEGIS is the sole arbiter"
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import json
import hashlib
import datetime


@dataclass
class RoutingTrace:
    """立体十字ルーティングのトレース（concept_search.RoutingTrace と同型）"""
    mode: str = ""                            # "keyword_maxpool" | "fulltext_mean"
    anchor_kws: List[str] = field(default_factory=list)
    top_domains: List[Tuple[str, float]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class TriggerInfo:
    """When and why LLM proposal was triggered"""
    executor_candidates_found: int
    used_llm_proposal: bool
    reason: Optional[str] = None  # "no_candidates" | "low_confidence" | None


@dataclass
class GateResult:
    """Gate check result (A/B/C gates)"""
    status: str  # "pass" | "reject" | "skip"
    rejected_fields: List[str] = field(default_factory=list)
    note: Optional[str] = None


@dataclass
class LLMProposalInfo:
    """LLM proposal metadata (never stores raw LLM output)"""
    raw_ir_digest: str       # sha256 of raw LLM output (privacy)
    gate_a: GateResult
    gate_b: GateResult
    gate_c: GateResult
    candidates_after_gates: int


@dataclass
class RefutationRecord:
    """A single counterexample found by CEGIS"""
    world_id: str
    program_id: str
    reason: str  # "output_mismatch" | "constraint_violation" | etc.


@dataclass
class CEGISInfo:
    """CEGIS verification layer info"""
    ran: bool
    iters: int = 0
    initial_candidates: int = 0
    survivors: int = 0
    selected_program_id: Optional[str] = None
    refutations: List[RefutationRecord] = field(default_factory=list)
    proved: bool = False
    inconclusive_reason: Optional[str] = None
    # inconclusive_reason values:
    # "worldgen_failed" | "insufficient_worlds" | "missing_spec"
    # "all_candidates_refuted" | "timeout" | "domain_unverifiable"
    # "empty_oracle"  ← A3 ハードゲート

    # ── A1: IR entity / registry ログ（2026-02-21 追加） ─────────────
    ir_entities: List[dict] = field(default_factory=list)    # 抽出した IR entities
    registry_hit_piece: Optional[str] = None                 # WORLDGEN_REGISTRY にヒットした piece_id
    oracle_worlds_count: int = 0                             # 生成した oracle worlds 数
    oracle_world_kinds: List[str] = field(default_factory=list)  # ["value_check", ...]


@dataclass
class VerifyInfo:
    """Verification layer metadata"""
    tool: str  # "sympy" | "z3" | "executor" | "simulation"
    worlds_generated: int = 0
    checks_passed: int = 0
    trivial_pass_guard: bool = False  # True = safety guard fired (worlds < threshold)
    certificate_type: Optional[str] = None  # "cegis_proved" | "simulation_proved" | "executor_computed"
    certificate_digest: Optional[str] = None


@dataclass
class AnswerInfo:
    """Final answer metadata"""
    value: Optional[str] = None
    status: str = "inconclusive"  # "cegis_proved" | "simulation_proved" | "executor_computed" | "inconclusive" | "rejected_by_gate"


@dataclass
class IntegrityInfo:
    """Bundle integrity check"""
    bundle_digest: str  # sha256 of entire bundle (computed at finalize)
    pipeline_version: str = "v6.3"


@dataclass
class AuditBundle:
    """
    AuditBundle.v1 — Complete audit trail for one problem

    This bundle is the single source of truth for:
    - What the LLM proposed (digest only, no raw text)
    - What CEGIS verified
    - What answer was accepted and why
    """
    bundle_version: str = "v1"
    question_id: str = ""
    timestamp_iso: str = ""
    trigger: Optional[TriggerInfo] = None
    llm_proposal: Optional[LLMProposalInfo] = None
    cegis: Optional[CEGISInfo] = None
    verify: Optional[VerifyInfo] = None
    answer: Optional[AnswerInfo] = None
    routing: Optional[RoutingTrace] = None   # 立体十字ルーティングトレース
    integrity: Optional[IntegrityInfo] = None

    def finalize(self) -> "AuditBundle":
        """Compute bundle_digest and timestamp, return self."""
        self.timestamp_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        # Digest everything except integrity
        tmp = asdict(self)
        tmp.pop("integrity", None)
        digest = hashlib.sha256(json.dumps(tmp, sort_keys=True, default=str).encode()).hexdigest()
        self.integrity = IntegrityInfo(bundle_digest=f"sha256:{digest[:16]}")
        return self

    def to_json(self, indent=2) -> str:
        """Serialize to JSON"""
        return json.dumps(asdict(self), indent=indent, default=str)

    def print_summary(self):
        """Human-readable one-liner summary for demo output."""
        q = self.question_id
        ans = self.answer
        ceg = self.cegis
        status = ans.status if ans else "no_answer"
        value = ans.value if ans else "—"
        worlds = self.verify.worlds_generated if self.verify else 0
        checks = self.verify.checks_passed if self.verify else 0
        llm_used = "LLM✓" if (self.trigger and self.trigger.used_llm_proposal) else "LLM✗"
        print(f"[{q}] status={status} value={repr(value)} worlds={worlds} checks={checks} {llm_used}")
        if ceg and not ceg.proved and ceg.inconclusive_reason:
            print(f"  inconclusive_reason={ceg.inconclusive_reason}")
        if self.llm_proposal:
            ga = self.llm_proposal.gate_a.status
            print(f"  gate_a={ga} candidates={self.llm_proposal.candidates_after_gates}")
