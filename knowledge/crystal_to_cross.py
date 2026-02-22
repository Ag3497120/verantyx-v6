"""
crystal_to_cross.py
Knowledge Crystallizer → Cross構造 接続層

結晶化された知識（FactAtom, RelationPiece）を
Verantyx の Cross構造（Piece, CEGIS, CrossSimulator）で使える形に変換する。

設計思想:
  facts → Crystallizer → FactAtom/RelationPiece
    → crystal_to_cross → Piece(verify_spec+worldgen_spec) + Candidate
      → CEGIS → proved/disproved
      → CrossSimulator → verified answer

これにより LLM に頼らず、小世界シミュレーションで答えを検証できる。
"""

from __future__ import annotations
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CrossCandidate:
    """CEGIS に渡す候補"""
    value: Any
    construction: List[str] = field(default_factory=list)
    confidence: float = 0.5
    constraints: List[str] = field(default_factory=list)
    verify_spec: Dict[str, Any] = field(default_factory=dict)
    worldgen_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossVerificationResult:
    """Cross検証結果"""
    answer: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    status: str = "unknown"  # proved | disproved | verified | unknown
    trace: List[str] = field(default_factory=list)


def crystal_to_candidates(
    ir_dict: dict,
    fact_atoms: list,
    relations: list,
    choices: Optional[Dict[str, str]] = None,
) -> List[CrossCandidate]:
    """
    結晶化データ → CEGIS Candidate リスト

    MCQ: 各選択肢を Candidate として、relations で検証
    exactMatch: FactAtom を Candidate として、constraints で検証
    """
    candidates = []

    if choices and len(choices) >= 2:
        # MCQ: 各選択肢を Candidate に変換
        candidates = _mcq_to_candidates(ir_dict, choices, fact_atoms, relations)
    else:
        # exactMatch: FactAtom を Candidate に変換
        candidates = _exact_to_candidates(ir_dict, fact_atoms, relations)

    return candidates


def verify_with_cross(
    ir_dict: dict,
    fact_atoms: list,
    relations: list,
    choices: Optional[Dict[str, str]] = None,
) -> Optional[CrossVerificationResult]:
    """
    結晶化データを使って Cross 構造で検証する。

    1. relations から predicate logic を構築
    2. MCQ: 各選択肢の主張を relations と照合
    3. exactMatch: FactAtom を constraints と照合
    4. 矛盾なく支持される候補を返す

    LLM不使用。ルールベース論理推論。
    """
    trace = []

    if not relations and not fact_atoms:
        return None

    if choices and len(choices) >= 2:
        return _verify_mcq(ir_dict, choices, relations, fact_atoms, trace)
    else:
        return _verify_exact(ir_dict, fact_atoms, relations, trace)


# ══════════════════════════════════════════════════════════════
#  MCQ Cross Verification
# ══════════════════════════════════════════════════════════════

def _mcq_to_candidates(
    ir_dict: dict,
    choices: Dict[str, str],
    fact_atoms: list,
    relations: list,
) -> List[CrossCandidate]:
    """MCQ選択肢 → CrossCandidate"""
    candidates = []
    for label, text in choices.items():
        constraints = []
        # 選択肢テキストから主張を抽出
        claims = _extract_claims_from_text(text)
        for claim in claims:
            constraints.append(f"claim:{claim}")

        candidates.append(CrossCandidate(
            value=label,
            construction=[f"choice_{label}"],
            confidence=0.5,
            constraints=constraints,
            verify_spec={
                "kind": "relation_consistency",
                "claims": claims,
            },
            worldgen_spec={
                "kind": "relation_world",
                "relations": [(r.subject, r.predicate, r.object, r.negated)
                              for r in relations if hasattr(r, 'subject')],
            },
        ))
    return candidates


def _verify_mcq(
    ir_dict: dict,
    choices: Dict[str, str],
    relations: list,
    fact_atoms: list,
    trace: list,
) -> Optional[CrossVerificationResult]:
    """
    MCQ を relations ベースで検証。

    各選択肢の主張が relations と整合するかチェック。
    - 選択肢の主張が relation と一致 → support
    - 選択肢の主張が relation と矛盾 → contradict
    - 一致する relation がない → unknown

    supports > 0 かつ contradicts == 0 の選択肢が1つ → proved
    """
    trace.append(f"cross_mcq_verify: {len(choices)} choices, {len(relations)} relations")

    # relations を検索可能な形に整理
    rel_index = {}
    for r in relations:
        if not hasattr(r, 'subject'):
            continue
        key = r.subject.lower().strip()
        if key not in rel_index:
            rel_index[key] = []
        rel_index[key].append(r)

    if not rel_index:
        trace.append("cross_mcq_verify: no indexable relations")
        return None

    # 各選択肢を検証
    choice_scores = {}
    for label, text in choices.items():
        supports = 0
        contradicts = 0
        text_lower = text.lower()

        for rel_key, rels in rel_index.items():
            for r in rels:
                # 選択肢テキストに subject と object が含まれているか
                subj_in = r.subject.lower() in text_lower
                obj_in = r.object.lower() in text_lower

                if not (subj_in or obj_in):
                    continue

                # predicate の方向性チェック
                pred = r.predicate.lower()

                if "valid" in pred or "holds" in pred:
                    if subj_in and obj_in:
                        if r.negated:
                            # "X is NOT valid in Y" で、選択肢が "X is valid in Y" を主張
                            if _text_claims_positive(text_lower, r.subject.lower(), r.object.lower()):
                                contradicts += 1
                                trace.append(f"  {label}: CONTRADICT ({r.subject} NOT {pred} {r.object})")
                            else:
                                supports += 1
                                trace.append(f"  {label}: SUPPORT (negation matches)")
                        else:
                            if _text_claims_positive(text_lower, r.subject.lower(), r.object.lower()):
                                supports += 1
                                trace.append(f"  {label}: SUPPORT ({r.subject} {pred} {r.object})")
                            else:
                                contradicts += 1
                                trace.append(f"  {label}: CONTRADICT (positive rel but negative claim)")

                elif "fails" in pred or "invalid" in pred:
                    if subj_in and obj_in:
                        if _text_claims_positive(text_lower, r.subject.lower(), r.object.lower()):
                            contradicts += 1
                        else:
                            supports += 1

                elif "causes" in pred or "is_a" in pred or "equals" in pred:
                    if subj_in and obj_in:
                        supports += 1

        choice_scores[label] = {"supports": supports, "contradicts": contradicts}

    trace.append(f"cross_mcq_scores: {choice_scores}")

    # 判定: supports > 0 & contradicts == 0
    clean_supporters = [
        (label, s) for label, s in choice_scores.items()
        if s["supports"] > 0 and s["contradicts"] == 0
    ]

    # supports > 0 の中で contradicts == 0 が1つだけ → proved
    if len(clean_supporters) == 1:
        winner = clean_supporters[0]
        return CrossVerificationResult(
            answer=winner[0],
            confidence=min(0.75, 0.5 + 0.1 * winner[1]["supports"]),
            method=f"cross_verified_mcq(supports={winner[1]['supports']},relations={len(relations)})",
            status="proved",
            trace=trace,
        )

    # contradicts > 0 で全部消えて1つだけ残った場合
    survivors = [
        (label, s) for label, s in choice_scores.items()
        if s["contradicts"] == 0
    ]
    if len(survivors) == 1 and any(s["contradicts"] > 0 for s in choice_scores.values()):
        winner = survivors[0]
        return CrossVerificationResult(
            answer=winner[0],
            confidence=0.6,
            method=f"cross_elimination_mcq(eliminated={len(choices)-1},relations={len(relations)})",
            status="verified",
            trace=trace,
        )

    return None


# ══════════════════════════════════════════════════════════════
#  exactMatch Cross Verification
# ══════════════════════════════════════════════════════════════

def _exact_to_candidates(
    ir_dict: dict,
    fact_atoms: list,
    relations: list,
) -> List[CrossCandidate]:
    """FactAtom → CrossCandidate"""
    candidates = []
    for atom in fact_atoms:
        if not hasattr(atom, 'value'):
            continue
        candidates.append(CrossCandidate(
            value=atom.value,
            construction=[f"atom_{atom.atom_type}_{atom.source_concept}"],
            confidence=atom.confidence,
            constraints=[f"type:{atom.atom_type}"],
            verify_spec={
                "kind": "fact_support",
                "support_count": len(atom.support_fact_ids),
            },
        ))
    return candidates


def _verify_exact(
    ir_dict: dict,
    fact_atoms: list,
    relations: list,
    trace: list,
) -> Optional[CrossVerificationResult]:
    """
    exactMatch を relations + fact_atoms で検証。

    relations の中に "equals" predicate がある場合:
      subject の value を抽出して回答

    fact_atoms で support >= 3 のものがあれば:
      強い証拠として回答
    """
    trace.append(f"cross_exact_verify: atoms={len(fact_atoms)}, relations={len(relations)}")

    # 1. "equals" relation から直接回答
    for r in relations:
        if not hasattr(r, 'predicate'):
            continue
        if r.predicate == "equals" and not r.negated:
            trace.append(f"  equals: {r.subject} = {r.object}")
            return CrossVerificationResult(
                answer=r.object,
                confidence=0.65,
                method=f"cross_equals({r.subject}={r.object})",
                status="verified",
                trace=trace,
            )

    # 2. 高 support の FactAtom
    strong_atoms = [
        a for a in fact_atoms
        if hasattr(a, 'support_fact_ids') and len(a.support_fact_ids) >= 3
    ]
    if strong_atoms:
        best = max(strong_atoms, key=lambda a: len(a.support_fact_ids))
        return CrossVerificationResult(
            answer=best.value,
            confidence=min(0.7, 0.4 + 0.1 * len(best.support_fact_ids)),
            method=f"cross_strong_atom(type={best.atom_type},supports={len(best.support_fact_ids)})",
            status="verified",
            trace=trace,
        )

    return None


# ══════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════

def _extract_claims_from_text(text: str) -> List[str]:
    """テキストから主張（claims）を抽出"""
    claims = []

    # "X holds/is valid/is true" パターン
    for m in re.finditer(
        r'([A-Z][\w\s\-]{2,30})\s+(holds?|is\s+valid|is\s+true|is\s+correct)',
        text, re.I
    ):
        claims.append(f"{m.group(1).strip()} HOLDS")

    # "X does not hold/fails" パターン
    for m in re.finditer(
        r'([A-Z][\w\s\-]{2,30})\s+(?:does\s+not\s+hold|fails?|is\s+not\s+valid|is\s+invalid)',
        text, re.I
    ):
        claims.append(f"{m.group(1).strip()} FAILS")

    # "Both X and Y" パターン
    if re.search(r'\bboth\b', text, re.I):
        claims.append("BOTH_HOLD")

    # "Neither X nor Y" パターン
    if re.search(r'\bneither\b', text, re.I):
        claims.append("NEITHER_HOLDS")

    return claims


def _text_claims_positive(text_lower: str, subject: str, obj: str) -> bool:
    """テキストが subject-object 関係について肯定的な主張をしているか"""
    # "X holds in Y", "X is valid in Y" などの肯定パターン
    positive_patterns = [
        f"{subject}.*(?:holds|valid|true|correct).*{obj}",
        f"{subject}.*(?:holds|valid|true|correct)",
        f"both.*{subject}.*{obj}",
    ]
    for pat in positive_patterns:
        if re.search(pat, text_lower):
            return True

    # "X does not hold" の否定パターン
    negative_patterns = [
        f"{subject}.*(?:does not|doesn't|not).*(?:hold|valid)",
        f"neither.*{subject}",
    ]
    for pat in negative_patterns:
        if re.search(pat, text_lower):
            return False

    # デフォルトは肯定と仮定
    return True


# ══════════════════════════════════════════════════════════════
#  Candidate Answer Verification (for ExactAnswerAssembler)
# ══════════════════════════════════════════════════════════════

def verify_candidate_answer(
    ir_dict: dict,
    candidate_answer: str,
    fact_atoms: list,
    relations: list,
) -> Optional[CrossVerificationResult]:
    """
    ExactAnswerAssemblerが生成した候補回答をCross構造で検証。
    
    検証方法:
    1. 候補回答がfact_atomsの中に支持されるか（support count）
    2. 候補回答がrelationsと矛盾しないか（consistency）
    3. 支持が多く矛盾がなければ verified
    
    Returns:
        CrossVerificationResult with status = proved | verified | unknown | contradicted
    """
    trace = []
    candidate_lower = candidate_answer.lower().strip()
    
    if not candidate_lower:
        return None
    
    supports = 0
    contradicts = 0
    
    # Check fact_atoms for support
    for atom in fact_atoms:
        atom_val = getattr(atom, 'value', getattr(atom, 'object', ''))
        if not atom_val:
            continue
        atom_lower = str(atom_val).lower().strip()
        
        # Direct match
        if candidate_lower == atom_lower:
            supports += 1
            trace.append(f"support:exact_match({atom_val})")
        # Containment match
        elif candidate_lower in atom_lower or atom_lower in candidate_lower:
            supports += 0.5
            trace.append(f"support:contains({atom_val[:30]})")
    
    # Check relations for consistency
    for rel in relations:
        if not hasattr(rel, 'subject'):
            continue
        rel_obj = str(getattr(rel, 'object', '')).lower()
        rel_subj = str(getattr(rel, 'subject', '')).lower()
        negated = getattr(rel, 'negated', False)
        
        # If relation mentions our candidate
        if candidate_lower in rel_obj or candidate_lower in rel_subj:
            if negated:
                contradicts += 1
                trace.append(f"contradict:negated_relation({rel.subject}→{rel.object})")
            else:
                supports += 0.3
    
    # Determine status
    if contradicts > 0 and supports == 0:
        status = "contradicted"
    elif supports >= 2 and contradicts == 0:
        status = "proved"
    elif supports >= 1 and contradicts == 0:
        status = "verified"
    else:
        status = "unknown"
    
    conf = min(supports * 0.3, 0.9)
    
    return CrossVerificationResult(
        answer=candidate_answer,
        confidence=conf,
        method=f"cross_verify_candidate(sup={supports:.1f},con={contradicts})",
        status=status,
        trace=trace,
    )
