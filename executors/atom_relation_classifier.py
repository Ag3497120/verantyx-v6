"""
atom_relation_classifier.py — Atom-based supports/contradicts/unknown classifier

LLM不使用。fact_atomizer で facts と choices を Atom 化し、
subject/predicate/object の Cross-match で関係を判定する。

`_llm_classify_relations` のドロップイン置換。
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ── Stopwords for tokenization ──
_STOPS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'must',
    'and', 'or', 'but', 'for', 'with', 'from', 'to', 'in', 'on',
    'at', 'by', 'of', 'it', 'its', 'this', 'that', 'which', 'who',
    'than', 'into', 'about', 'as', 'if', 'also', 'such', 'more',
    'most', 'other', 'some', 'any', 'each', 'every', 'all', 'both',
    'not', 'no',  # keep negation awareness separate
}

# ── Antonym pairs for contradiction detection ──
_ANTONYMS: List[Tuple[str, str]] = [
    ('increase', 'decrease'), ('rise', 'fall'), ('higher', 'lower'),
    ('more', 'less'), ('greater', 'fewer'), ('up', 'down'),
    ('positive', 'negative'), ('true', 'false'), ('correct', 'incorrect'),
    ('present', 'absent'), ('active', 'inactive'), ('legal', 'illegal'),
    ('possible', 'impossible'), ('stable', 'unstable'), ('soluble', 'insoluble'),
    ('organic', 'inorganic'), ('aerobic', 'anaerobic'),
    ('dominant', 'recessive'), ('proximal', 'distal'),
    ('anterior', 'posterior'), ('dorsal', 'ventral'),
    ('endothermic', 'exothermic'), ('oxidation', 'reduction'),
    ('anion', 'cation'), ('acid', 'base'), ('acidic', 'basic'),
    ('symmetric', 'asymmetric'), ('linear', 'nonlinear'),
    ('convergent', 'divergent'), ('finite', 'infinite'),
    ('continuous', 'discrete'), ('maximum', 'minimum'),
    ('internal', 'external'), ('intrinsic', 'extrinsic'),
    ('afferent', 'efferent'), ('inhibit', 'activate'),
    ('suppress', 'promote'), ('hydrophobic', 'hydrophilic'),
    ('prokaryote', 'eukaryote'), ('benign', 'malignant'),
]

# Build fast antonym lookup
_ANTONYM_MAP: Dict[str, Set[str]] = {}
for a, b in _ANTONYMS:
    _ANTONYM_MAP.setdefault(a, set()).add(b)
    _ANTONYM_MAP.setdefault(b, set()).add(a)

# ── Negation words ──
_NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'without',
    "n't", "cannot", "can't", "doesn't", "don't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
    'lack', 'absence', 'except', 'exclude', 'unlike', 'false',
    'incorrect', 'wrong', 'invalid',
}

# ── Synonym groups for support detection ──
_SYNONYM_GROUPS: List[Set[str]] = [
    {'increase', 'rise', 'grow', 'elevate', 'higher', 'greater', 'upregulate'},
    {'decrease', 'fall', 'drop', 'reduce', 'lower', 'decline', 'downregulate'},
    {'cause', 'lead', 'result', 'produce', 'induce', 'trigger', 'elicit'},
    {'prevent', 'inhibit', 'block', 'stop', 'suppress', 'hinder'},
    {'contain', 'include', 'comprise', 'consist', 'encompass'},
    {'require', 'need', 'necessitate', 'depend'},
    {'catalyze', 'accelerate', 'facilitate', 'promote', 'enhance'},
    {'bind', 'attach', 'associate', 'interact', 'connect'},
    {'encode', 'express', 'transcribe', 'translate'},
    {'secrete', 'release', 'emit', 'produce', 'generate'},
]
_SYNONYM_MAP: Dict[str, Set[str]] = {}
for group in _SYNONYM_GROUPS:
    for word in group:
        _SYNONYM_MAP[word] = group - {word}


@dataclass
class AtomMatch:
    """Single atom-level match between a choice and a fact."""
    choice_fragment: str
    fact_fragment: str
    match_type: str  # 'subject', 'object', 'predicate', 'full', 'keyword'
    score: float = 0.0


@dataclass
class ChoiceRelation:
    """Atom-based relation for one choice."""
    label: str
    relation: str  # supports | contradicts | unknown
    fact_ids: List[str] = field(default_factory=list)
    support_score: float = 0.0
    contradict_score: float = 0.0
    matched_atoms: List[AtomMatch] = field(default_factory=list)
    coverage: float = 0.0


def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenization, stop-word removal."""
    words = re.findall(r'[a-z][a-z0-9\-]{1,}', text.lower())
    return {w for w in words if w not in _STOPS}


def _extract_numbers(text: str) -> Set[str]:
    """Extract numeric values from text for exact matching."""
    return set(re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:×|x)\s*10\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?\b', text))


def _has_negation(text: str) -> bool:
    """Check if text contains negation."""
    lower = text.lower()
    # Check for n't contractions
    if "n't" in lower:
        return True
    words = set(re.findall(r'[a-z\']+', lower))
    return bool(words & _NEGATION_WORDS)


def _normalize_for_match(text: str) -> str:
    """Normalize text for flexible matching."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s\-]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def _subject_overlap(s1: str, s2: str) -> float:
    """Calculate normalized overlap between two subjects."""
    t1 = _tokenize(s1)
    t2 = _tokenize(s2)
    if not t1 or not t2:
        return 0.0
    common = t1 & t2
    # Also check synonyms
    for w in t1:
        syns = _SYNONYM_MAP.get(w, set())
        common |= (syns & t2)
    return len(common) / min(len(t1), len(t2))


def _check_antonym_contradiction(tokens1: Set[str], tokens2: Set[str]) -> bool:
    """Check if two token sets contain antonym pairs."""
    for w in tokens1:
        antonyms = _ANTONYM_MAP.get(w, set())
        if antonyms & tokens2:
            return True
    return False


def _numeric_match(text1: str, text2: str) -> bool:
    """Check if two texts share the same specific number."""
    nums1 = _extract_numbers(text1)
    nums2 = _extract_numbers(text2)
    if not nums1 or not nums2:
        return False
    return bool(nums1 & nums2)


def classify_relations_by_atoms(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
) -> Optional[dict]:
    """
    Atom-based relation classification — drop-in replacement for _llm_classify_relations.

    Returns dict with same schema:
        {
            "A": {"relation": "supports", "fact_ids": ["fact_0"]},
            "B": {"relation": "contradicts", "fact_ids": []},
            ...
            "decision": "",
            "survivors": [],
        }
    """
    from knowledge.fact_atomizer import FactAtomizer

    atomizer = FactAtomizer()

    # ── 1. Atomize facts ──
    fact_texts = []
    fact_id_map: Dict[int, str] = {}  # index → fact_id
    for idx, f in enumerate(facts):
        if isinstance(f, dict):
            s = (f.get("summary", "") or f.get("plain", ""))
            props = f.get("properties", [])
            formulas = f.get("formulas", [])
            parts = []
            if s:
                parts.append(s)
            for p in props:
                parts.append(str(p))
            for fl in formulas:
                parts.append(str(fl))
            text = " ".join(parts)
        elif hasattr(f, 'summary'):
            text = f.summary or ""
        else:
            continue
        if text.strip():
            fact_texts.append(text.strip())
            fact_id_map[len(fact_texts) - 1] = f"fact_{idx}"

    if not fact_texts:
        return None

    fact_atoms = atomizer.atomize_many(fact_texts)

    # Also keep raw fact texts for keyword fallback
    fact_tokens_list = [_tokenize(ft) for ft in fact_texts]

    # ── 2. Analyze each choice ──
    result = {}

    for label, choice_text in choices.items():
        # Atomize choice text
        choice_atoms = atomizer.atomize(choice_text)
        choice_tokens = _tokenize(choice_text)
        choice_neg = _has_negation(choice_text)
        choice_nums = _extract_numbers(choice_text)

        support_score = 0.0
        contradict_score = 0.0
        matched_fact_ids: List[str] = []
        matches: List[AtomMatch] = []

        # ── 2a. Atom × Atom cross-match ──
        for ca in choice_atoms:
            ca_subj_norm = _normalize_for_match(ca.subject)
            ca_obj_norm = _normalize_for_match(ca.object)
            ca_pred_norm = _normalize_for_match(ca.predicate)

            for fi, fa in enumerate(fact_atoms):
                fa_subj_norm = _normalize_for_match(fa.subject)
                fa_obj_norm = _normalize_for_match(fa.object)
                fa_pred_norm = _normalize_for_match(fa.predicate)

                # Subject overlap check
                subj_sim = _subject_overlap(ca.subject, fa.subject)
                if subj_sim < 0.3:
                    # Also try choice_subject == fact_object (reverse match)
                    subj_sim_rev = _subject_overlap(ca.subject, fa.object)
                    if subj_sim_rev < 0.3:
                        continue

                # Subject matches — now check predicate/object
                # Case 1: Same predicate, same object → SUPPORTS
                pred_sim = _subject_overlap(ca.predicate, fa.predicate)
                obj_sim = _subject_overlap(ca.object, fa.object)

                fid = f"fact_{fi}"
                # Find original fact_id
                for orig_idx, orig_id in fact_id_map.items():
                    if fi < len(fact_atoms) and fact_atoms[fi].raw_sentence in fact_texts[orig_idx] if orig_idx < len(fact_texts) else False:
                        fid = orig_id
                        break

                if pred_sim >= 0.3 and obj_sim >= 0.5:
                    # Strong match: same subject, similar predicate, similar object
                    # Check for negation flip
                    fa_neg = _has_negation(fa.raw_sentence)
                    if choice_neg != fa_neg:
                        contradict_score += 0.8
                        matches.append(AtomMatch(
                            choice_fragment=f"{ca.subject}|{ca.predicate}|{ca.object}",
                            fact_fragment=f"{fa.subject}|{fa.predicate}|{fa.object}",
                            match_type='full_negation_flip',
                            score=-0.8,
                        ))
                    else:
                        support_score += 0.8
                        matches.append(AtomMatch(
                            choice_fragment=f"{ca.subject}|{ca.predicate}|{ca.object}",
                            fact_fragment=f"{fa.subject}|{fa.predicate}|{fa.object}",
                            match_type='full',
                            score=0.8,
                        ))
                    if fid not in matched_fact_ids:
                        matched_fact_ids.append(fid)

                elif pred_sim >= 0.3 and obj_sim < 0.3:
                    # Same subject + predicate but different object
                    # Check antonyms
                    ca_obj_tokens = _tokenize(ca.object)
                    fa_obj_tokens = _tokenize(fa.object)
                    if _check_antonym_contradiction(ca_obj_tokens, fa_obj_tokens):
                        contradict_score += 0.6
                        matches.append(AtomMatch(
                            choice_fragment=ca.object,
                            fact_fragment=fa.object,
                            match_type='antonym_contradiction',
                            score=-0.6,
                        ))
                        if fid not in matched_fact_ids:
                            matched_fact_ids.append(fid)
                    # Check numeric mismatch
                    elif _numeric_match(ca.object, fa.object):
                        support_score += 0.5
                        matches.append(AtomMatch(
                            choice_fragment=ca.object,
                            fact_fragment=fa.object,
                            match_type='numeric_match',
                            score=0.5,
                        ))
                        if fid not in matched_fact_ids:
                            matched_fact_ids.append(fid)
                    elif ca_obj_tokens and fa_obj_tokens and not (ca_obj_tokens & fa_obj_tokens):
                        # Completely different objects for same predicate → weak contradiction
                        # Only if predicate is identity-like (is, was, equals, etc.)
                        identity_preds = {'is', 'was', 'equals', 'is_a', 'named', 'called', 'known_as'}
                        if _tokenize(ca.predicate) & identity_preds or _tokenize(fa.predicate) & identity_preds:
                            contradict_score += 0.3
                            matches.append(AtomMatch(
                                choice_fragment=ca.object,
                                fact_fragment=fa.object,
                                match_type='identity_mismatch',
                                score=-0.3,
                            ))
                            if fid not in matched_fact_ids:
                                matched_fact_ids.append(fid)

                elif obj_sim >= 0.5:
                    # Same subject, same object but different predicate
                    # Weak support
                    support_score += 0.3
                    matches.append(AtomMatch(
                        choice_fragment=f"{ca.subject}→{ca.object}",
                        fact_fragment=f"{fa.subject}→{fa.object}",
                        match_type='subject_object',
                        score=0.3,
                    ))
                    if fid not in matched_fact_ids:
                        matched_fact_ids.append(fid)

        # ── 2b. Keyword fallback (no atoms matched) ──
        if not matches:
            for fi, ft_tokens in enumerate(fact_tokens_list):
                if not ft_tokens:
                    continue
                common = choice_tokens & ft_tokens
                if not common:
                    # Try synonyms
                    for w in choice_tokens:
                        syns = _SYNONYM_MAP.get(w, set())
                        common |= (syns & ft_tokens)
                if len(common) < 2:
                    continue

                overlap = len(common) / max(len(choice_tokens), 1)
                if overlap < 0.25:
                    continue

                fid = fact_id_map.get(fi, f"fact_{fi}")

                # Check negation flip at keyword level
                fact_neg = _has_negation(fact_texts[fi])
                if choice_neg != fact_neg and _check_antonym_contradiction(choice_tokens, ft_tokens):
                    contradict_score += 0.4
                    matches.append(AtomMatch(
                        choice_fragment=choice_text[:60],
                        fact_fragment=fact_texts[fi][:60],
                        match_type='keyword_contradiction',
                        score=-0.4,
                    ))
                else:
                    kw_score = min(overlap * 0.6, 0.5)
                    support_score += kw_score
                    matches.append(AtomMatch(
                        choice_fragment=choice_text[:60],
                        fact_fragment=fact_texts[fi][:60],
                        match_type='keyword',
                        score=kw_score,
                    ))
                if fid not in matched_fact_ids:
                    matched_fact_ids.append(fid)

        # ── 2c. Numeric cross-check ──
        if choice_nums and not any(m.match_type == 'numeric_match' for m in matches):
            for fi, ft in enumerate(fact_texts):
                if _numeric_match(choice_text, ft):
                    fid = fact_id_map.get(fi, f"fact_{fi}")
                    support_score += 0.4
                    matches.append(AtomMatch(
                        choice_fragment=choice_text[:40],
                        fact_fragment=ft[:40],
                        match_type='numeric_cross',
                        score=0.4,
                    ))
                    if fid not in matched_fact_ids:
                        matched_fact_ids.append(fid)
                    break  # one numeric match is enough

        # ── 2d. Determine relation ──
        net_score = support_score - contradict_score
        if contradict_score >= 0.5 and contradict_score > support_score:
            relation = "contradicts"
        elif support_score >= 0.4 and support_score > contradict_score:
            relation = "supports"
        else:
            relation = "unknown"

        result[label] = {
            "relation": relation,
            "fact_ids": matched_fact_ids,
            "support_score": round(support_score, 3),
            "contradict_score": round(contradict_score, 3),
            "matches": len(matches),
        }

    result["decision"] = ""
    result["survivors"] = []
    result["_method"] = "atom_cross"

    # Log summary
    summary = {k: v["relation"] for k, v in result.items()
               if isinstance(v, dict) and "relation" in v}
    log.info(f"atom_classify: {summary}")

    return result
