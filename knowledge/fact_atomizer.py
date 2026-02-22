"""
fact_atomizer.py — Rule-based English sentence → structured FactAtom converter

LLM不使用。正規表現パターンで英文からsubject-predicate-objectトリプルを抽出。
Verantyx V6の知識層の心臓部。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re


@dataclass
class FactAtom:
    subject: str
    predicate: str
    object: str
    source: str = "wiki"          # "wiki" | "cross_piece"
    confidence: float = 0.7
    raw_sentence: str = ""
    when: Optional[str] = None    # temporal info
    where: Optional[str] = None   # location info
    numeric_value: Optional[float] = None
    unit: Optional[str] = None


# ── Helper patterns ──
_YEAR = r'(?:in\s+)?(\d{4})'
_NUM = r'([\d,]+(?:\.\d+)?)'
_UNIT = r'((?:meters?|km|miles?|kg|pounds?|degrees?|percent|%|years?|days?|hours?|seconds?|m/s|km/h|mph|cm|mm|liters?|gallons?|tons?|celsius|fahrenheit|kelvin|Hz|kHz|MHz|GHz|watts?|volts?|amps?|ohms?|joules?|calories?|eV|nm|μm|mol|moles?|ppm|ppb|atm|Pa|bar|daltons?)(?:\s+per\s+(?:second|minute|hour|day|year|meter|km|mole?))?)'
_NAME = r'([A-Z][a-zA-Z\u00C0-\u024F\'\-]+(?:\s+[A-Z][a-zA-Z\u00C0-\u024F\'\-]+)*)'
_NP = r'(.+?)'  # non-greedy noun phrase


def _clean(s: str) -> str:
    """Strip articles and extra whitespace"""
    s = s.strip()
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.I)
    return s.strip()


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(',', ''))
    except (ValueError, TypeError):
        return None


class FactAtomizer:
    """
    Rule-based English fact atomization.
    
    ~80 patterns covering: identity, possession, location, temporal,
    causation, authorship, measurement, classification, comparison, definition.
    """

    def __init__(self):
        self._patterns = self._build_patterns()

    def atomize(self, sentence: str, source: str = "wiki") -> List[FactAtom]:
        """Extract FactAtoms from a single English sentence."""
        atoms = []
        sentence = sentence.strip()
        if not sentence or len(sentence) < 5:
            return atoms

        for pat_name, regex, extractor, conf in self._patterns:
            m = regex.search(sentence)
            if m:
                try:
                    result = extractor(m)
                    if result:
                        for subj, pred, obj, extra in (result if isinstance(result[0], tuple) else [result]):
                            subj = _clean(subj)
                            obj = _clean(obj)
                            if not subj or not obj:
                                continue
                            if len(subj) < 2 and not subj.isdigit():
                                continue
                            if len(obj) < 1:
                                continue
                            atom = FactAtom(
                                subject=subj,
                                predicate=pred,
                                object=obj,
                                source=source,
                                confidence=conf,
                                raw_sentence=sentence,
                                when=extra.get('when') if extra else None,
                                where=extra.get('where') if extra else None,
                                numeric_value=extra.get('numeric') if extra else None,
                                unit=extra.get('unit') if extra else None,
                            )
                            atoms.append(atom)
                except Exception:
                    continue

        # Dedup
        seen = set()
        unique = []
        for a in atoms:
            key = (a.subject.lower(), a.predicate, a.object.lower())
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique

    def atomize_many(self, sentences: List[str], source: str = "wiki") -> List[FactAtom]:
        """Atomize multiple sentences."""
        all_atoms = []
        for s in sentences:
            # Split on period if multiple sentences
            for sub in re.split(r'(?<=[.!?])\s+', s):
                all_atoms.extend(self.atomize(sub, source))
        return all_atoms

    def _build_patterns(self):
        """Build all extraction patterns. Returns list of (name, regex, extractor_fn, confidence)."""
        P = []

        # ════════════════════════════════════════
        # IDENTITY / DEFINITION patterns
        # ════════════════════════════════════════

        # "{X} is the {ROLE} of {Y}"
        P.append(("is_role_of",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+the\s+(.+?)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_') + "_of", m.group(3), None),
            0.85))

        # "{X} is a/an {TYPE}"
        P.append(("is_a",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:a|an)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.7))

        # "{X}, also known as {Y}"
        P.append(("also_known_as",
            re.compile(r'^(.+?),?\s+(?:also\s+known\s+as|a\.k\.a\.?|or)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "alias", m.group(2), None),
            0.8))

        # "{X} is called {Y}"
        P.append(("is_called",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:called|named|known\s+as|referred\s+to\s+as)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "alias", m.group(2), None),
            0.8))

        # "{X} is {Y}" (generic identity, lower confidence)
        P.append(("is_generic",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is", m.group(2), None) if len(m.group(2)) > 2 else None,
            0.5))

        # ════════════════════════════════════════
        # LOCATION patterns
        # ════════════════════════════════════════

        # "{X} is located in/at {Y}"
        P.append(("located_in",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:located|situated|found)\s+(?:in|at|on|near)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "located_in", m.group(2), None),
            0.85))

        # "{X} is in {Y}" (when Y looks like a place)
        P.append(("in_place",
            re.compile(r'^(.+?)\s+(?:is|was)\s+in\s+([A-Z].+?)\.?$', re.I),
            lambda m: (m.group(1), "located_in", m.group(2), None),
            0.6))

        # ════════════════════════════════════════
        # AUTHORSHIP / CREATION patterns
        # ════════════════════════════════════════

        # "{X} was {VERB}ed by {Y} [in YEAR]" (passive)
        P.append(("passive_by_year",
            re.compile(r'^(.+?)\s+(?:was|were|is|are)\s+(\w+(?:ed|en|t))\s+by\s+(.+?)\s+in\s+(\d{4})\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_by", m.group(3), {"when": m.group(4)}),
            0.9))

        # "{X} was {VERB}ed by {Y}" (passive, no year)
        P.append(("passive_by",
            re.compile(r'^(.+?)\s+(?:was|were|is|are)\s+(\w+(?:ed|en|t))\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_by", m.group(3), None),
            0.85))

        # "{Y} {VERB}ed {X}" — active voice creation
        for verb, pred in [
            ("painted", "painted_by"), ("wrote", "written_by"), ("composed", "composed_by"),
            ("directed", "directed_by"), ("invented", "invented_by"), ("discovered", "discovered_by"),
            ("founded", "founded_by"), ("designed", "designed_by"), ("created", "created_by"),
            ("developed", "developed_by"), ("built", "built_by"), ("published", "published_by"),
        ]:
            P.append((f"active_{verb}",
                re.compile(rf'^(.+?)\s+{verb}\s+(.+?)\.?$', re.I),
                lambda m, p=pred: (m.group(2), p, m.group(1), None),
                0.8))

        # ════════════════════════════════════════
        # TEMPORAL patterns
        # ════════════════════════════════════════

        # "{X} was born in {YEAR}"
        P.append(("born_in",
            re.compile(r'^(.+?)\s+was\s+born\s+(?:in|on)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "born", m.group(2), {"when": m.group(2)}),
            0.9))

        # "{X} died in {YEAR}"
        P.append(("died_in",
            re.compile(r'^(.+?)\s+died\s+(?:in|on)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "died", m.group(2), {"when": m.group(2)}),
            0.9))

        # "{event} occurred/happened in {YEAR}"
        P.append(("occurred_in",
            re.compile(r'^(.+?)\s+(?:occurred|happened|took\s+place|began|started|ended)\s+(?:in|on)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "occurred_in", m.group(2), {"when": m.group(2)}),
            0.8))

        # ════════════════════════════════════════
        # POSSESSION / HAS patterns
        # ════════════════════════════════════════

        # "{X} has/had {N} {THING}"
        P.append(("has_count",
            re.compile(r'^(.+?)\s+(?:has|had|have|contains?|included?)\s+' + _NUM + r'\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "has_count", m.group(3),
                       {"numeric": _try_float(m.group(2))}),
            0.75))

        # "{X} has a population of {N}"
        P.append(("population",
            re.compile(r'^(.+?)\s+has\s+a\s+population\s+of\s+(?:about\s+|approximately\s+|around\s+)?' + _NUM + r'\.?$', re.I),
            lambda m: (m.group(1), "population", m.group(2),
                       {"numeric": _try_float(m.group(2))}),
            0.85))

        # ════════════════════════════════════════
        # MEASUREMENT / NUMERIC patterns
        # ════════════════════════════════════════

        # "{X} is/was {N} {UNIT}" (measurement)
        P.append(("measurement",
            re.compile(r'^(.+?)\s+(?:is|was|are|were|measures?|weighs?|equals?)\s+(?:about\s+|approximately\s+|around\s+|roughly\s+)?' + _NUM + r'\s+' + _UNIT + r'\.?$', re.I),
            lambda m: (m.group(1), "measures", m.group(2) + " " + m.group(3),
                       {"numeric": _try_float(m.group(2)), "unit": m.group(3)}),
            0.85))

        # "{X} is approximately/about {N}"
        P.append(("approx_value",
            re.compile(r'^(.+?)\s+(?:is|equals?|was)\s+(?:approximately|about|around|roughly|nearly)\s+' + _NUM + r'\.?$', re.I),
            lambda m: (m.group(1), "value", m.group(2),
                       {"numeric": _try_float(m.group(2))}),
            0.7))

        # ════════════════════════════════════════
        # CLASSIFICATION / BELONGS TO
        # ════════════════════════════════════════

        # "{X} belongs to {Y}"
        P.append(("belongs_to",
            re.compile(r'^(.+?)\s+(?:belongs?\s+to|is\s+part\s+of|is\s+a\s+member\s+of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "belongs_to", m.group(2), None),
            0.8))

        # "{X} is one of {Y}"
        P.append(("is_one_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+one\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "member_of", m.group(2), None),
            0.7))

        # ════════════════════════════════════════
        # COMPARISON patterns
        # ════════════════════════════════════════

        # "{X} is {COMP} than {Y}"
        for comp in ["larger", "smaller", "bigger", "greater", "higher", "lower",
                      "older", "younger", "faster", "slower", "longer", "shorter",
                      "more", "less", "heavier", "lighter", "earlier", "later"]:
            P.append((f"comp_{comp}",
                re.compile(rf'^(.+?)\s+(?:is|was|are|were)\s+{comp}\s+than\s+(.+?)\.?$', re.I),
                lambda m, c=comp: (m.group(1), c + "_than", m.group(2), None),
                0.75))

        # ════════════════════════════════════════
        # CAUSATION patterns
        # ════════════════════════════════════════

        # "{X} is caused by {Y}"
        P.append(("caused_by",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:caused|produced|triggered|induced)\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "caused_by", m.group(2), None),
            0.8))

        # "{X} causes {Y}"
        P.append(("causes",
            re.compile(r'^(.+?)\s+(?:causes?|produces?|leads?\s+to|results?\s+in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "causes", m.group(2), None),
            0.75))

        # ════════════════════════════════════════
        # PRIZE / AWARD patterns
        # ════════════════════════════════════════

        # "{X} won/received the {PRIZE} in {YEAR}"
        P.append(("won_prize",
            re.compile(r'^(.+?)\s+(?:won|received|was\s+awarded)\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})\.?$', re.I),
            lambda m: (m.group(1), "won", m.group(2), {"when": m.group(3)}),
            0.85))

        # "{X} won/received the {PRIZE}" (no year)
        P.append(("won_prize_noyear",
            re.compile(r'^(.+?)\s+(?:won|received|was\s+awarded)\s+(?:the\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "won", m.group(2), None),
            0.7))

        # ════════════════════════════════════════
        # CONTAINS / CONSISTS OF
        # ════════════════════════════════════════

        # "{X} contains/includes {Y}"
        P.append(("contains",
            re.compile(r'^(.+?)\s+(?:contains?|includes?|consists?\s+of|comprises?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "contains", m.group(2), None),
            0.7))

        # ════════════════════════════════════════
        # NAMED AFTER / DERIVED FROM
        # ════════════════════════════════════════

        P.append(("named_after",
            re.compile(r'^(.+?)\s+(?:is|was)\s+named\s+(?:after|for)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "named_after", m.group(2), None),
            0.85))

        P.append(("derived_from",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:derived|obtained|extracted)\s+from\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "derived_from", m.group(2), None),
            0.8))

        # ════════════════════════════════════════
        # FUNCTION / PURPOSE
        # ════════════════════════════════════════

        # "{X} is used to/for {Y}"
        P.append(("used_for",
            re.compile(r'^(.+?)\s+(?:is|are|was|were)\s+used\s+(?:to|for|in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "used_for", m.group(2), None),
            0.7))

        # "{X} serves as {Y}"
        P.append(("serves_as",
            re.compile(r'^(.+?)\s+(?:serves?|functions?|acts?|works?)\s+as\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "serves_as", m.group(2), None),
            0.7))

        # ════════════════════════════════════════
        # LANGUAGE patterns (for HLE)
        # ════════════════════════════════════════

        # "{X} is spoken in {Y}"
        P.append(("spoken_in",
            re.compile(r'^(.+?)\s+(?:is|are)\s+(?:spoken|used|official)\s+in\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "spoken_in", m.group(2), None),
            0.8))

        # "{X} means {Y}" / "{X} refers to {Y}"
        P.append(("means",
            re.compile(r'^(.+?)\s+(?:means?|refers?\s+to|denotes?|signifies?|represents?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "means", m.group(2), None),
            0.7))

        # ════════════════════════════════════════
        # SCIENTIFIC / FORMULA patterns
        # ════════════════════════════════════════

        # "{X} has a {PROPERTY} of {VALUE}"
        P.append(("has_property",
            re.compile(r'^(.+?)\s+has\s+(?:a|an)\s+(.+?)\s+of\s+(?:about\s+|approximately\s+)?' + _NUM + r'(?:\s+' + _UNIT + r')?\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_'),
                       m.group(3) + (" " + m.group(4) if m.group(4) else ""),
                       {"numeric": _try_float(m.group(3)), "unit": m.group(4)}),
            0.8))

        # "{X}'s {PROPERTY} is {VALUE}"
        P.append(("possessive_property",
            re.compile(r'^(.+?)\'s?\s+(.+?)\s+(?:is|was|equals?)\s+(?:about\s+|approximately\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_'), m.group(3), None),
            0.7))

        # "The {PROPERTY} of {X} is {VALUE}"
        P.append(("property_of_is",
            re.compile(r'^[Tt]he\s+(.+?)\s+of\s+(.+?)\s+(?:is|was|equals?)\s+(?:about\s+|approximately\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(2), m.group(1).lower().replace(' ', '_'), m.group(3), None),
            0.8))

        return P


# ── Module-level convenience ──
_default_atomizer = None

def get_atomizer() -> FactAtomizer:
    global _default_atomizer
    if _default_atomizer is None:
        _default_atomizer = FactAtomizer()
    return _default_atomizer


def atomize_facts(sentences: List[str], source: str = "wiki") -> List[FactAtom]:
    """Convenience: atomize a list of fact sentences."""
    return get_atomizer().atomize_many(sentences, source)
