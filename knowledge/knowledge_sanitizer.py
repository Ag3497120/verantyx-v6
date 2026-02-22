"""
knowledge_sanitizer.py
Phase 3: LLM 知識応答の検疫（Gate-K: K1〜K6）

Gate-K1: Schema validation（必須キー）
Gate-K2: Answer contamination（answer/final/correct 汚染の除去）
Gate-K3: Domain consistency（問い合わせドメインと無関係な知識を除外）
Gate-K4: Formalizability check（Cross構造に落とせるか）
Gate-K5: Duplicate/redundancy merge（既存ピースとの重複統合）
Gate-K6: Confidence tagging（LLM由来タグを付与）
"""

import re
from dataclasses import dataclass, field
from typing import Any

from knowledge.llm_knowledge_fetcher import LLMKnowledgeResponse


# ---------------------------------------------------------------------------
# 1. サニタイズ済み知識片の型
# ---------------------------------------------------------------------------

@dataclass
class SanitizedFact:
    fact_id: str
    kind: str                              # definition / theorem / lemma / criterion
    symbol: str
    domain: str
    statement_formal: str
    statement_plain: str
    preconditions: list[str] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)
    checkable_examples: list[dict] = field(default_factory=list)
    confidence: str = "medium"
    source_hint: str = ""
    llm_origin: bool = True                # 常に True（監査用）
    gate_tags: list[str] = field(default_factory=list)  # 通過したGateのログ


@dataclass
class SanitizationResult:
    query_id: str
    accepted: list[SanitizedFact] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)   # 元のfactと reject理由
    gate_log: list[str] = field(default_factory=list)    # 監査ログ


# ---------------------------------------------------------------------------
# 2. Gate-K2: Answer Contamination
# ---------------------------------------------------------------------------

ANSWER_CONTAMINATION_PATTERNS = [
    re.compile(r"\bthe\s+answer\s+is\b", re.I),
    re.compile(r"\bfinal\s+answer\b", re.I),
    re.compile(r"\bcorrect\s+option\b", re.I),
    re.compile(r"\bcorrect\s+answer\b", re.I),
    re.compile(r"\bsolution\s+is\b", re.I),
    re.compile(r"\btherefore[,\s]+[A-E]\b", re.I),
    re.compile(r"\bthe\s+result\s+is\b", re.I),
    re.compile(r"\boption\s+[A-E]\s+is\s+correct\b", re.I),
    re.compile(r"\bchoice\s+[A-E]\b", re.I),
]

def _has_answer_contamination(text: str) -> bool:
    for pat in ANSWER_CONTAMINATION_PATTERNS:
        if pat.search(text):
            return True
    return False

def _check_answer_contamination(fact: dict) -> str | None:
    """汚染フィールドを検査。汚染があれば reject理由を返す。"""
    check_fields = [
        fact.get("statement_formal", ""),
        fact.get("statement_plain", ""),
        str(fact.get("consequences", "")),
    ]
    for text in check_fields:
        if _has_answer_contamination(str(text)):
            return f"answer_contamination_in_fact"
    # 禁止キーの存在確認
    banned = {"answer", "final", "solution", "correct", "result"}
    for k in fact:
        if k.lower() in banned:
            return f"banned_key='{k}'"
    return None


# ---------------------------------------------------------------------------
# 3. Gate-K1: Schema Validation
# ---------------------------------------------------------------------------

FACT_REQUIRED_KEYS = {"id", "type", "symbol", "statement_plain"}

def _check_schema(fact: dict) -> str | None:
    missing = FACT_REQUIRED_KEYS - set(fact.keys())
    if missing:
        return f"missing_keys={missing}"
    # type が有効か
    valid_types = {"definition", "theorem", "lemma", "criterion",
                   "counterexample_rule", "identity", "algorithm"}
    fact_type = fact.get("type", "").lower()
    if fact_type not in valid_types:
        return f"invalid_type='{fact_type}'"
    # statement が空でないか
    if not str(fact.get("statement_plain", "")).strip():
        return "empty_statement_plain"
    return None


# ---------------------------------------------------------------------------
# 4. Gate-K3: Domain Consistency
# ---------------------------------------------------------------------------

def _check_domain_consistency(fact: dict, requested_domains: list[str]) -> str | None:
    """
    fact.domain が requested_domains と整合するか確認する。
    完全一致不要。broad check（含まれているかどうか）。
    """
    if not requested_domains:
        return None  # domain 指定なし → チェックしない
    fact_domain = str(fact.get("domain", "")).lower()
    if not fact_domain:
        return None  # domain 未記載 → 通過（寛大に）
    # requested のどれかに partial match
    for req in requested_domains:
        if req.lower() in fact_domain or fact_domain in req.lower():
            return None
    return f"domain_mismatch: got='{fact_domain}', expected_one_of={requested_domains}"


# ---------------------------------------------------------------------------
# 5. Gate-K4: Formalizability Check
# ---------------------------------------------------------------------------

# Cross構造に落とすには「条件→結論」構造か「定義」が必要
# statement_formal か preconditions + consequences のいずれかがあればOK
def _check_formalizability(fact: dict) -> str | None:
    has_formal = bool(str(fact.get("statement_formal", "")).strip())
    has_plain = bool(str(fact.get("statement_plain", "")).strip())
    has_precond = isinstance(fact.get("preconditions"), list) and len(fact["preconditions"]) > 0
    has_conseq = isinstance(fact.get("consequences"), list) and len(fact["consequences"]) > 0
    has_examples = isinstance(fact.get("checkable_examples"), list) and len(fact["checkable_examples"]) > 0

    # 定義: statement_plain か statement_formal があればOK
    if fact.get("type") == "definition" and (has_formal or has_plain):
        return None
    # 定理/補題: preconditions + consequences が必要。ただし statement があれば許可
    if fact.get("type") in ("theorem", "lemma", "criterion"):
        if has_precond and has_conseq:
            return None
        if has_formal or has_plain:
            return None  # statement があれば許可
        return "theorem_missing_precond_or_conseq"
    # examples があれば通過
    if has_examples:
        return None
    if has_formal or has_plain or has_precond:
        return None
    return "insufficient_structure_for_cross_mapping"


# ---------------------------------------------------------------------------
# 6. 主エントリポイント: KnowledgeSanitizer
# ---------------------------------------------------------------------------

class KnowledgeSanitizer:
    """
    LLMKnowledgeResponse → SanitizationResult
    Gate-K1〜K4 を適用し、accepted / rejected に分類する。
    K5 (dedup) と K6 (tagging) はここで処理。
    """

    def __init__(self, requested_domains: list[str] | None = None,
                 existing_symbols: set[str] | None = None):
        self.requested_domains = requested_domains or []
        self.existing_symbols = existing_symbols or set()  # K5: 重複チェック用

    def sanitize(self, llm_response: LLMKnowledgeResponse) -> SanitizationResult:
        result = SanitizationResult(query_id=llm_response.query_id)

        if not llm_response.ok:
            result.gate_log.append(f"llm_response_failed:{llm_response.error}")
            return result

        facts_raw = llm_response.parsed.get("facts", [])
        if not isinstance(facts_raw, list):
            result.gate_log.append("facts_not_list")
            return result

        seen_symbols: set[str] = set()

        for i, fact in enumerate(facts_raw):
            if not isinstance(fact, dict):
                result.rejected.append({"index": i, "reason": "not_dict", "fact": fact})
                continue

            # --- Gate K1: Schema ---
            r = _check_schema(fact)
            if r:
                result.rejected.append({"index": i, "reason": f"gate_k1:{r}", "fact": fact})
                result.gate_log.append(f"fact[{i}] rejected: gate_k1:{r}")
                continue

            # --- Gate K2: Answer contamination ---
            r = _check_answer_contamination(fact)
            if r:
                result.rejected.append({"index": i, "reason": f"gate_k2:{r}", "fact": fact})
                result.gate_log.append(f"fact[{i}] rejected: gate_k2:{r}")
                continue

            # --- Gate K3: Domain consistency ---
            r = _check_domain_consistency(fact, self.requested_domains)
            if r:
                result.rejected.append({"index": i, "reason": f"gate_k3:{r}", "fact": fact})
                result.gate_log.append(f"fact[{i}] rejected: gate_k3:{r}")
                continue

            # --- Gate K4: Formalizability ---
            r = _check_formalizability(fact)
            if r:
                result.rejected.append({"index": i, "reason": f"gate_k4:{r}", "fact": fact})
                result.gate_log.append(f"fact[{i}] rejected: gate_k4:{r}")
                continue

            # --- Gate K5: Duplicate (soft dedup) ---
            symbol = str(fact.get("symbol", "")).lower().strip()
            if symbol in self.existing_symbols or symbol in seen_symbols:
                result.gate_log.append(f"fact[{i}] duplicate merged: symbol='{symbol}'")
                # reject ではなく skip（重複は採用せず）
                result.rejected.append({"index": i, "reason": "gate_k5:duplicate", "fact": fact})
                continue
            seen_symbols.add(symbol)

            # --- Gate K6: Confidence tagging (LLM由来を明示) ---
            gate_tags = ["gate_k1_ok", "gate_k2_ok", "gate_k3_ok", "gate_k4_ok", "gate_k5_ok"]
            confidence = fact.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            sanitized = SanitizedFact(
                fact_id=str(fact.get("id", f"fact_{i:03d}")),
                kind=fact.get("type", "definition"),
                symbol=fact.get("symbol", ""),
                domain=fact.get("domain", self.requested_domains[0] if self.requested_domains else ""),
                statement_formal=str(fact.get("statement_formal", "")),
                statement_plain=str(fact.get("statement_plain", "")),
                preconditions=fact.get("preconditions", []),
                consequences=fact.get("consequences", []),
                checkable_examples=fact.get("checkable_examples", []),
                confidence=confidence,
                source_hint=str(fact.get("source_hint", "")),
                llm_origin=True,
                gate_tags=gate_tags,
            )
            result.accepted.append(sanitized)
            result.gate_log.append(f"fact[{i}] accepted: symbol='{symbol}' confidence={confidence}")

        return result
