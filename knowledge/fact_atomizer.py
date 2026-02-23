"""
fact_atomizer.py — Rule-based English sentence → structured FactAtom converter

LLM不使用。正規表現パターンで英文からsubject-predicate-objectトリプルを抽出。
Verantyx V6の知識層の心臓部。

Pattern categories:
  A. Identity / Definition (20+)
  B. Location / Geography (15+)
  C. Authorship / Creation (15+)
  D. Temporal / Historical (15+)
  E. Possession / Quantity (10+)
  F. Measurement / Physical (15+)
  G. Classification / Taxonomy (10+)
  H. Comparison / Ordering (10+)
  I. Causation / Effect (10+)
  J. Awards / Achievement (10+)
  K. Composition / Structure (10+)
  L. Language / Naming (10+)
  M. Science / Formula (15+)
  N. Function / Purpose (10+)
  O. Relationship / Association (10+)
  P. Process / Method (10+)
  Q. Legal / Political (10+)
  R. Medical / Biological (10+)
  S. Music / Art / Culture (10+)
  T. Mathematical / Logical (10+)
  
Total: ~200+ patterns
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import re


@dataclass
class FactAtom:
    subject: str
    predicate: str
    object: str
    source: str = "wiki"
    confidence: float = 0.7
    raw_sentence: str = ""
    when: Optional[str] = None
    where: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None


# ── Helpers ──
_NUM = r'([\d,]+(?:\.\d+)?)'
_UNIT = (r'((?:meters?|kilometres?|kilometers?|km|miles?|kg|kilograms?|grams?|g|'
         r'pounds?|lbs?|degrees?|percent|%|years?|days?|hours?|minutes?|seconds?|'
         r'm/s|km/h|mph|cm|mm|nm|μm|liters?|litres?|gallons?|tons?|tonnes?|'
         r'celsius|fahrenheit|kelvin|K|Hz|kHz|MHz|GHz|THz|watts?|W|volts?|V|'
         r'amps?|amperes?|A|ohms?|Ω|joules?|J|calories?|cal|eV|keV|MeV|GeV|'
         r'mol|moles?|ppm|ppb|atm|Pa|hPa|kPa|MPa|bar|mbar|daltons?|Da|'
         r'AU|light[- ]years?|ly|parsecs?|pc|'
         r'bits?|bytes?|KB|MB|GB|TB|'
         r'm²|m³|km²|cm²|ha|hectares?|acres?|'
         r'rpm|bpm|dB|decibels?|lumens?|lux|candela|'
         r'newtons?|N|pascals?|teslas?|T|webers?|henr(?:y|ies)|farads?|F|'
         r'siemens|S|becquerels?|Bq|grays?|Gy|sieverts?|Sv|'
         r'angstroms?|Å)(?:\s+per\s+(?:second|minute|hour|day|year|'
         r'meter|metre|km|mole?|liter|litre|kilogram|gram|cubic\s+\w+|square\s+\w+))?)')


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(',', ''))
    except (ValueError, TypeError):
        return None


def _ext(m, i):
    """Safe group extraction"""
    try:
        return m.group(i) or ""
    except (IndexError, AttributeError):
        return ""


class FactAtomizer:
    """
    Rule-based English fact atomization with ~200+ patterns.
    """

    def __init__(self):
        self._patterns = self._build_patterns()

    def atomize(self, sentence: str, source: str = "wiki") -> List[FactAtom]:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 5:
            return []

        atoms = []
        for pat_name, regex, extractor, conf in self._patterns:
            m = regex.search(sentence)
            if m:
                try:
                    result = extractor(m)
                    if not result:
                        continue
                    items = result if isinstance(result, list) else [result]
                    for item in items:
                        if isinstance(item, tuple) and len(item) >= 3:
                            subj, pred, obj = item[0], item[1], item[2]
                            extra = item[3] if len(item) > 3 else None
                        else:
                            continue
                        subj = _clean(subj)
                        obj = _clean(obj)
                        if not subj or not obj or (len(subj) < 2 and not subj.isdigit()):
                            continue
                        atom = FactAtom(
                            subject=subj, predicate=pred, object=obj,
                            source=source, confidence=conf, raw_sentence=sentence,
                            when=(extra or {}).get('when') if isinstance(extra, dict) else None,
                            where=(extra or {}).get('where') if isinstance(extra, dict) else None,
                            numeric_value=(extra or {}).get('numeric') if isinstance(extra, dict) else None,
                            unit=(extra or {}).get('unit') if isinstance(extra, dict) else None,
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
        all_atoms = []
        try:
            from knowledge.sentence_splitter import split_to_clauses
            use_splitter = True
        except ImportError:
            use_splitter = False

        for s in sentences:
            for sub in re.split(r'(?<=[.!?])\s+', s):
                if use_splitter:
                    clauses = split_to_clauses(sub)
                    for clause in clauses:
                        all_atoms.extend(self.atomize(clause, source))
                else:
                    all_atoms.extend(self.atomize(sub, source))
        return all_atoms

    def _build_patterns(self):
        P = []

        # ════════════════════════════════════════════════════════
        # A. IDENTITY / DEFINITION
        # ════════════════════════════════════════════════════════

        # "{X} is the {ROLE} of {Y}"
        P.append(("is_role_of",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+the\s+(.+?)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_') + "_of", m.group(3), None),
            0.85))

        # "{X}, also known as {Y}," — bounded alias (stop at comma/period/which/was)
        P.append(("also_known_as",
            re.compile(r'^(.+?),?\s+(?:also\s+known\s+as|a\.?k\.?a\.?|also\s+called|otherwise\s+known\s+as)\s+([^,]+?)(?:,|\.|$)', re.I),
            lambda m: (m.group(1), "alias", m.group(2).strip(), None),
            0.85))

        # "{X} is called/named/known as {Y}"
        P.append(("is_called",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:called|named|known\s+as|referred\s+to\s+as|termed|dubbed|designated|titled|labeled|labelled)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "alias", m.group(2), None),
            0.85))

        # "{X} is defined as {Y}"
        P.append(("defined_as",
            re.compile(r'^(.+?)\s+(?:is|are|was|were)\s+defined\s+as\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "defined_as", m.group(2), None),
            0.8))

        # "{X} refers to {Y}" / "{X} denotes {Y}"
        P.append(("refers_to",
            re.compile(r'^(.+?)\s+(?:refers?\s+to|denotes?|signifies?|represents?|stands?\s+for|symbolizes?|means?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "means", m.group(2), None),
            0.8))

        # "{X} is a type/form/kind/variety of {Y}"
        P.append(("type_of",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:a\s+)?(?:type|form|kind|variety|species|class|category|subtype|subclass|subset|variant|version|branch|division)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "type_of", m.group(2), None),
            0.8))

        # "{X} is a/an {TYPE}" — low confidence to avoid noise
        P.append(("is_a",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:a|an)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.35))  # LOW confidence — only used as last resort

        # "{X} is the {ROLE/TYPE} ..." — broader is_a with "the"
        P.append(("is_the",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+the\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.35))

        # "{X} is one of the {TYPE}" 
        P.append(("is_one_of",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+one\s+of\s+(?:the\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.35))

        # "{X} is considered/regarded/known to be {Y}"
        P.append(("is_considered",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:widely\s+)?(?:considered|regarded|known|recognized|thought|believed|understood)\s+(?:to\s+be\s+|as\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.4))

        # "{X} refers to {Y}" / "{X} denotes {Y}"
        P.append(("refers_to",
            re.compile(r'^(.+?)\s+(?:refers?\s+to|denotes?|means?|signifies?|represents?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_a", m.group(2), None),
            0.4))

        # "{X} is {ADJ}" (property)
        P.append(("is_adj",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+((?:not\s+)?(?:\w+(?:ly|ed|ous|ive|ble|al|ic|ful|less|ent|ant|ary|ory)))\s*\.?$', re.I),
            lambda m: (m.group(1), "property", m.group(2), None),
            0.4))

        # ════════════════════════════════════════════════════════
        # B. LOCATION / GEOGRAPHY
        # ════════════════════════════════════════════════════════

        # "{X} is located/situated in/at/on/near {Y}"
        P.append(("located_in",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:located|situated|found|based|headquartered|centered|centred)\s+(?:in|at|on|near|within|along|beside|between|across)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "located_in", m.group(2), None),
            0.85))

        # "{X} is in/at/on {Y}" (Y starts with uppercase = place)
        P.append(("in_place",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:in|at|on)\s+([A-Z].+?)\.?$', re.I),
            lambda m: (m.group(1), "located_in", m.group(2), None),
            0.55))

        # "{X} borders/adjoins {Y}"
        P.append(("borders",
            re.compile(r'^(.+?)\s+(?:borders?|adjoins?|is\s+adjacent\s+to|is\s+next\s+to|neighbors?|neighbours?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "borders", m.group(2), None),
            0.8))

        # "{X} flows through/into {Y}"
        P.append(("flows",
            re.compile(r'^(.+?)\s+(?:flows?\s+(?:through|into|across|past)|empties?\s+into|drains?\s+into)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "flows_through", m.group(2), None),
            0.8))

        # "{X} is the capital/largest city of {Y}"
        P.append(("capital_city",
            re.compile(r'^(.+?)\s+(?:is|was)\s+the\s+(?:capital(?:\s+city)?|largest\s+city|second[- ]largest\s+city|administrative\s+center|administrative\s+centre|chief\s+city|main\s+city)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "capital_of", m.group(2), None),
            0.9))

        # "The capital of {X} is {Y}"
        P.append(("capital_of_is",
            re.compile(r'^[Tt]he\s+capital\s+(?:city\s+)?of\s+(.+?)\s+(?:is|was)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(2), "capital_of", m.group(1), None),
            0.9))

        # "{X} lies/sits at coordinates/latitude/longitude"
        P.append(("coordinates",
            re.compile(r'^(.+?)\s+(?:lies|sits|is\s+located)\s+at\s+(?:coordinates?|latitude|a\s+latitude)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "coordinates", m.group(2), None),
            0.75))

        # "{X} is a city/town/village/country/state/province/region in {Y}"
        P.append(("geo_entity_in",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:a|an|the)\s+(?:city|town|village|hamlet|municipality|county|district|province|state|region|territory|department|prefecture|country|nation|island|peninsula|continent|mountain|river|lake|ocean|sea|desert|forest|park|valley)\s+(?:in|of|on|within)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "located_in", m.group(2), None),
            0.8))

        # "{X} has an area/elevation of {N}"
        P.append(("geo_measure",
            re.compile(r'^(.+?)\s+has\s+(?:an?\s+)?(?:area|elevation|altitude|depth|length|width|coastline)\s+of\s+(?:about\s+|approximately\s+)?' + _NUM + r'(?:\s+' + _UNIT + r')?\.?$', re.I),
            lambda m: (m.group(1), "area", m.group(2) + (" " + m.group(3) if m.group(3) else ""),
                       {"numeric": _try_float(m.group(2)), "unit": m.group(3)}),
            0.85))

        # ════════════════════════════════════════════════════════
        # C. AUTHORSHIP / CREATION
        # ════════════════════════════════════════════════════════

        # "{X} was {VERB}ed by {Y} in {YEAR}" (passive with year)
        P.append(("passive_by_year",
            re.compile(r'^(.+?)\s+(?:was|were|is|are)\s+(\w+(?:ed|en|t))\s+by\s+(.+?)\s+in\s+(\d{4})\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_by", m.group(3), {"when": m.group(4)}),
            0.9))

        # "{X} was {VERB}ed by {Y}" (passive, no year)
        P.append(("passive_by",
            re.compile(r'^(.+?)\s+(?:was|were|is|are)\s+(\w+(?:ed|en|t))\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_by", m.group(3), None),
            0.85))

        # Active voice creation verbs
        _creation_verbs = [
            ("painted", "painted_by"), ("wrote", "written_by"), ("composed", "composed_by"),
            ("directed", "directed_by"), ("invented", "invented_by"), ("discovered", "discovered_by"),
            ("founded", "founded_by"), ("designed", "designed_by"), ("created", "created_by"),
            ("developed", "developed_by"), ("built", "built_by"), ("published", "published_by"),
            ("produced", "produced_by"), ("recorded", "recorded_by"), ("performed", "performed_by"),
            ("sang", "sung_by"), ("sculpted", "sculpted_by"), ("photographed", "photographed_by"),
            ("choreographed", "choreographed_by"), ("formulated", "formulated_by"),
            ("proposed", "proposed_by"), ("theorized", "theorized_by"), ("coined", "coined_by"),
            ("introduced", "introduced_by"), ("established", "established_by"),
            ("patented", "patented_by"), ("synthesized", "synthesized_by"),
            ("isolated", "isolated_by"), ("derived", "derived_by"),
            ("proved", "proved_by"), ("solved", "solved_by"),
            ("constructed", "constructed_by"), ("engineered", "engineered_by"),
        ]
        for verb, pred in _creation_verbs:
            P.append((f"active_{verb}",
                re.compile(rf'^(.+?)\s+{verb}\s+(.+?)\.?$', re.I),
                lambda m, p=pred: (m.group(2), p, m.group(1), None),
                0.8))

        # "{X} is the author/creator/inventor of {Y}"
        P.append(("author_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+the\s+(?:author|creator|inventor|discoverer|founder|designer|architect|composer|writer|painter|sculptor|director|developer|pioneer|originator)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(2), "created_by", m.group(1), None),
            0.9))

        # ════════════════════════════════════════════════════════
        # D. TEMPORAL / HISTORICAL
        # ════════════════════════════════════════════════════════

        # "{X} was born on/in {DATE}"
        P.append(("born",
            re.compile(r'^(.+?)\s+was\s+born\s+(?:on|in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "born", m.group(2), {"when": m.group(2)}),
            0.9))

        # "{X} died on/in {DATE}"
        P.append(("died",
            re.compile(r'^(.+?)\s+died\s+(?:on|in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "died", m.group(2), {"when": m.group(2)}),
            0.9))

        # "{X} ({YEAR}–{YEAR})" birth-death
        P.append(("life_dates",
            re.compile(r'^(.+?)\s*\((\d{4})\s*[–\-]\s*(\d{4})\)', re.I),
            lambda m: [(m.group(1), "born", m.group(2), {"when": m.group(2)}),
                       (m.group(1), "died", m.group(3), {"when": m.group(3)})],
            0.85))

        # "{X} occurred/happened/took place in/on {DATE}"
        P.append(("occurred",
            re.compile(r'^(.+?)\s+(?:occurred|happened|took\s+place|began|started|ended|commenced|concluded|was\s+held|was\s+signed|was\s+enacted|was\s+ratified|was\s+established|was\s+formed|was\s+created)\s+(?:in|on|during|around|circa|c\.)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "occurred_in", m.group(2), {"when": m.group(2)}),
            0.85))

        # "In {YEAR}, {X} {VERBed} {Y}"
        P.append(("in_year_event",
            re.compile(r'^[Ii]n\s+(\d{4}),?\s+(.+?)\s+(was\s+\w+ed|\w+ed)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(2), "event", m.group(3) + " " + m.group(4), {"when": m.group(1)}),
            0.7))

        # "{X} was {VERBed} in {YEAR}"
        P.append(("verbed_in_year",
            re.compile(r'^(.+?)\s+was\s+(\w+ed)\s+in\s+(\d{4})\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower(), m.group(3), {"when": m.group(3)}),
            0.75))

        # "{X} dates from/back to {DATE}"
        P.append(("dates_from",
            re.compile(r'^(.+?)\s+dates?\s+(?:from|back\s+to|to)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "dates_from", m.group(2), {"when": m.group(2)}),
            0.75))

        # "{X} reigned/ruled from {Y} to {Z}"
        P.append(("reigned",
            re.compile(r'^(.+?)\s+(?:reigned|ruled|governed|served|held\s+office)\s+(?:from\s+)?(.+?)\s+(?:to|until|till)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "reigned", m.group(2) + " to " + m.group(3), {"when": m.group(2)}),
            0.8))

        # ════════════════════════════════════════════════════════
        # E. POSSESSION / QUANTITY
        # ════════════════════════════════════════════════════════

        # "{X} has/had/contains {N} {THING}"
        P.append(("has_count",
            re.compile(r'^(.+?)\s+(?:has|had|have|contains?|included?|comprises?|consists?\s+of|features?)\s+' + _NUM + r'\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "has_count", m.group(3), {"numeric": _try_float(m.group(2))}),
            0.75))

        # "{X} has a population of {N}"
        P.append(("population",
            re.compile(r'^(.+?)\s+has\s+(?:a\s+)?population\s+of\s+(?:about\s+|approximately\s+|around\s+|over\s+|nearly\s+|more\s+than\s+|less\s+than\s+)?' + _NUM + r'\.?$', re.I),
            lambda m: (m.group(1), "population", m.group(2), {"numeric": _try_float(m.group(2))}),
            0.85))

        # "There are {N} {X} in {Y}"
        P.append(("there_are",
            re.compile(r'^[Tt]here\s+(?:are|were|is|was)\s+' + _NUM + r'\s+(.+?)\s+in\s+(.+?)\.?$', re.I),
            lambda m: (m.group(3), "count_of_" + m.group(2).lower().rstrip('s'), m.group(1), {"numeric": _try_float(m.group(1))}),
            0.7))

        # ════════════════════════════════════════════════════════
        # F. MEASUREMENT / PHYSICAL
        # ════════════════════════════════════════════════════════

        # "{X} is/was/measures/weighs {N} {UNIT}"
        P.append(("measurement",
            re.compile(r'^(.+?)\s+(?:is|was|are|were|measures?|weighs?|equals?|has\s+a\s+(?:mass|weight|height|length|width|depth|diameter|radius|circumference|volume|area|speed|velocity|frequency|wavelength|temperature|pressure|density|charge|energy|power|force|capacity|distance|size)\s+of)\s+(?:about\s+|approximately\s+|around\s+|roughly\s+|nearly\s+|exactly\s+|precisely\s+)?' + _NUM + r'\s+' + _UNIT + r'\.?$', re.I),
            lambda m: (m.group(1), "measures", m.group(2) + " " + m.group(3),
                       {"numeric": _try_float(m.group(2)), "unit": m.group(3)}),
            0.85))

        # "{X} has a {PROPERTY} of {N} {UNIT}"
        P.append(("has_property_unit",
            re.compile(r'^(.+?)\s+has\s+(?:a|an)\s+(.+?)\s+of\s+(?:about\s+|approximately\s+)?' + _NUM + r'(?:\s+' + _UNIT + r')?\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_'),
                       m.group(3) + (" " + m.group(4) if m.group(4) else ""),
                       {"numeric": _try_float(m.group(3)), "unit": m.group(4)}),
            0.8))

        # "The {PROPERTY} of {X} is {VALUE}"
        P.append(("property_of_is",
            re.compile(r'^[Tt]he\s+(.+?)\s+of\s+(.+?)\s+(?:is|was|equals?)\s+(?:about\s+|approximately\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(2), m.group(1).lower().replace(' ', '_'), m.group(3), None),
            0.8))

        # "{X}'s {PROPERTY} is {VALUE}"
        P.append(("possessive_property",
            re.compile(r"^(.+?)'s?\s+(.+?)\s+(?:is|was|equals?)\s+(?:about\s+|approximately\s+)?(.+?)\.?$", re.I),
            lambda m: (m.group(1), m.group(2).lower().replace(' ', '_'), m.group(3), None),
            0.7))

        # "{X} is approximately/about {N}"
        P.append(("approx_value",
            re.compile(r'^(.+?)\s+(?:is|equals?|was)\s+(?:approximately|about|around|roughly|nearly|exactly|precisely)\s+' + _NUM + r'\.?$', re.I),
            lambda m: (m.group(1), "value", m.group(2), {"numeric": _try_float(m.group(2))}),
            0.7))

        # ════════════════════════════════════════════════════════
        # G. CLASSIFICATION / TAXONOMY
        # ════════════════════════════════════════════════════════

        # "{X} belongs to {Y}"
        P.append(("belongs_to",
            re.compile(r'^(.+?)\s+(?:belongs?\s+to|is\s+part\s+of|is\s+a\s+member\s+of|is\s+classified\s+(?:as|under|in)|falls?\s+under|is\s+included\s+in|is\s+categorized\s+(?:as|under))\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "belongs_to", m.group(2), None),
            0.8))

        # "{X} is one of the {Y}"
        P.append(("is_one_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+one\s+of\s+(?:the\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "member_of", m.group(2), None),
            0.7))

        # "{X} includes/comprises {Y}"
        P.append(("includes",
            re.compile(r'^(.+?)\s+(?:includes?|comprises?|encompasses?|incorporates?|embraces?|subsumes?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "includes", m.group(2), None),
            0.7))

        # "{X} consists of {Y}"
        P.append(("consists_of",
            re.compile(r'^(.+?)\s+(?:consists?\s+of|is\s+(?:composed|made\s+up|comprised)\s+of|is\s+formed\s+from|is\s+built\s+from)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "consists_of", m.group(2), None),
            0.75))

        # "{X} is divided into {Y}"
        P.append(("divided_into",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:divided|split|separated|partitioned|organized|organised|broken)\s+into\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "divided_into", m.group(2), None),
            0.75))

        # ════════════════════════════════════════════════════════
        # H. COMPARISON / ORDERING
        # ════════════════════════════════════════════════════════

        _comparatives = [
            "larger", "smaller", "bigger", "greater", "higher", "lower",
            "older", "younger", "faster", "slower", "longer", "shorter",
            "heavier", "lighter", "earlier", "later", "deeper", "shallower",
            "wider", "narrower", "thicker", "thinner", "warmer", "cooler",
            "hotter", "colder", "denser", "richer", "poorer", "stronger",
            "weaker", "harder", "softer", "brighter", "dimmer", "louder",
            "quieter", "more common", "less common", "more abundant", "less abundant",
            "more expensive", "less expensive", "more efficient", "less efficient",
        ]
        for comp in _comparatives:
            esc = re.escape(comp)
            P.append((f"comp_{comp.replace(' ','_')}",
                re.compile(rf'^(.+?)\s+(?:is|was|are|were)\s+{esc}\s+than\s+(.+?)\.?$', re.I),
                lambda m, c=comp: (m.group(1), c.replace(' ', '_') + "_than", m.group(2), None),
                0.75))

        # "{X} is the {SUPERLATIVE} {THING}"
        P.append(("superlative",
            re.compile(r'^(.+?)\s+(?:is|was)\s+the\s+((?:most|least|largest|smallest|oldest|youngest|longest|shortest|highest|lowest|fastest|slowest|first|last|best|worst|richest|poorest|biggest|tallest|deepest|widest|hottest|coldest|brightest|heaviest|lightest|densest|rarest|commonest)\s*\w*)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "is_" + m.group(2).lower().replace(' ', '_'), m.group(3), None),
            0.8))

        # ════════════════════════════════════════════════════════
        # I. CAUSATION / EFFECT
        # ════════════════════════════════════════════════════════

        # "{X} is caused by {Y}"
        P.append(("caused_by",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:caused|produced|triggered|induced|provoked|elicited|generated|brought\s+about)\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "caused_by", m.group(2), None),
            0.8))

        # "{X} causes/leads to {Y}"
        P.append(("causes",
            re.compile(r'^(.+?)\s+(?:causes?|produces?|leads?\s+to|results?\s+in|triggers?|induces?|gives?\s+rise\s+to|contributes?\s+to|is\s+responsible\s+for)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "causes", m.group(2), None),
            0.75))

        # "{X} is the result of {Y}"
        P.append(("result_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:the\s+)?(?:result|consequence|effect|outcome|product|byproduct)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "result_of", m.group(2), None),
            0.8))

        # "{X} prevents/inhibits {Y}"
        P.append(("prevents",
            re.compile(r'^(.+?)\s+(?:prevents?|inhibits?|blocks?|suppresses?|reduces?|decreases?|diminishes?|mitigates?|alleviates?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "prevents", m.group(2), None),
            0.7))

        # ════════════════════════════════════════════════════════
        # J. AWARDS / ACHIEVEMENT
        # ════════════════════════════════════════════════════════

        # "{X} won/received the {PRIZE} in {YEAR}"
        P.append(("won_prize_year",
            re.compile(r'^(.+?)\s+(?:won|received|was\s+awarded|earned|obtained|claimed|took\s+home)\s+(?:the\s+)?(.+?)\s+in\s+(\d{4})\.?$', re.I),
            lambda m: (m.group(1), "won", m.group(2), {"when": m.group(3)}),
            0.85))

        # "{X} won/received the {PRIZE}"
        P.append(("won_prize",
            re.compile(r'^(.+?)\s+(?:won|received|was\s+awarded|earned)\s+(?:the\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "won", m.group(2), None),
            0.65))

        # "The {PRIZE} was awarded to {X}"
        P.append(("prize_to",
            re.compile(r'^[Tt]he\s+(.+?)\s+was\s+(?:awarded|given|presented)\s+to\s+(.+?)\.?$', re.I),
            lambda m: (m.group(2), "won", m.group(1), None),
            0.8))

        # "{X} holds the record for {Y}"
        P.append(("holds_record",
            re.compile(r'^(.+?)\s+(?:holds?|set|broke|established)\s+(?:the\s+)?(?:record|world\s+record)\s+(?:for|in|of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "holds_record", m.group(2), None),
            0.8))

        # ════════════════════════════════════════════════════════
        # K. COMPOSITION / STRUCTURE
        # ════════════════════════════════════════════════════════

        # "{X} contains/includes {Y}"
        P.append(("contains",
            re.compile(r'^(.+?)\s+(?:contains?|is\s+composed\s+of|is\s+made\s+(?:of|from|up\s+of)|is\s+constructed\s+(?:of|from))\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "contains", m.group(2), None),
            0.75))

        # "{X} is an element/component/ingredient of {Y}"
        P.append(("component_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:an?\s+)?(?:element|component|ingredient|constituent|part|member|segment|portion|piece|factor)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "component_of", m.group(2), None),
            0.8))

        # ════════════════════════════════════════════════════════
        # L. LANGUAGE / NAMING / ETYMOLOGY
        # ════════════════════════════════════════════════════════

        # "{X} is spoken in {Y}"
        P.append(("spoken_in",
            re.compile(r'^(.+?)\s+(?:is|are)\s+(?:spoken|used|official|the\s+official\s+language)\s+(?:in|of|throughout)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "spoken_in", m.group(2), None),
            0.8))

        # "{X} comes from/derives from {Y}" (etymology)
        P.append(("etymology",
            re.compile(r'^(?:The\s+(?:word|term|name)\s+)?(.+?)\s+(?:comes?\s+from|derives?\s+from|originates?\s+from|is\s+derived\s+from|has\s+its\s+(?:origin|roots?)\s+in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "derived_from", m.group(2), None),
            0.8))

        # "{X} is named after/for {Y}"
        P.append(("named_after",
            re.compile(r'^(.+?)\s+(?:is|was)\s+named\s+(?:after|for|in\s+honor\s+of|in\s+honour\s+of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "named_after", m.group(2), None),
            0.85))

        # "{X} translates to/as {Y}"
        P.append(("translates_to",
            re.compile(r'^(.+?)\s+(?:translates?\s+(?:to|as|into)|means?\s+(?:literally\s+)?)\s+["\']?(.+?)["\']?\.?$', re.I),
            lambda m: (m.group(1), "translates_to", m.group(2), None),
            0.75))

        # "{X} is the {LANGUAGE} word for {Y}"
        P.append(("word_for",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:the\s+)?(\w+)\s+(?:word|term|name)\s+for\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "word_for_" + m.group(2).lower(), m.group(3), None),
            0.8))

        # ════════════════════════════════════════════════════════
        # M. SCIENCE / FORMULA
        # ════════════════════════════════════════════════════════

        # "{X} is expressed/given/described by {FORMULA}"
        P.append(("formula",
            re.compile(r'^(.+?)\s+(?:is|can\s+be)\s+(?:expressed|given|described|represented|calculated|computed|defined|stated|written)\s+(?:by|as)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "formula", m.group(2), None),
            0.75))

        # "{X} = {Y}" (equation)
        P.append(("equation",
            re.compile(r'^(.+?)\s*=\s*(.+?)\.?$', re.I),
            lambda m: (m.group(1).strip(), "equals", m.group(2).strip(), None),
            0.7))

        # "The formula for {X} is {Y}"
        P.append(("formula_for",
            re.compile(r'^[Tt]he\s+(?:formula|equation|expression)\s+for\s+(.+?)\s+is\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "formula", m.group(2), None),
            0.8))

        # "{X} has a molecular formula/weight of {Y}"
        P.append(("molecular",
            re.compile(r'^(.+?)\s+has\s+(?:a\s+)?(?:molecular\s+(?:formula|weight|mass)|chemical\s+formula|atomic\s+(?:number|mass|weight)|molar\s+mass)\s+(?:of\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "molecular_formula", m.group(2), None),
            0.85))

        # "{X} has a boiling/melting/freezing point of {N}"
        P.append(("phase_point",
            re.compile(r'^(.+?)\s+has\s+(?:a\s+)?(?:boiling|melting|freezing|flash|ignition|decomposition)\s+point\s+of\s+(?:about\s+)?' + _NUM + r'(?:\s+' + _UNIT + r')?\.?$', re.I),
            lambda m: (m.group(1), "boiling_point", m.group(2) + (" " + m.group(3) if m.group(3) else ""),
                       {"numeric": _try_float(m.group(2)), "unit": m.group(3)}),
            0.85))

        # "{X} has a half-life of {Y}"
        P.append(("half_life",
            re.compile(r'^(.+?)\s+has\s+(?:a\s+)?half[- ]life\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "half_life", m.group(2), None),
            0.85))

        # "{X} reacts with {Y} to form/produce {Z}"
        P.append(("reacts_with",
            re.compile(r'^(.+?)\s+reacts?\s+with\s+(.+?)\s+to\s+(?:form|produce|yield|give|create)\s+(.+?)\.?$', re.I),
            lambda m: [(m.group(1), "reacts_with", m.group(2), None),
                       (m.group(1) + " + " + m.group(2), "produces", m.group(3), None)],
            0.8))

        # ════════════════════════════════════════════════════════
        # N. FUNCTION / PURPOSE
        # ════════════════════════════════════════════════════════

        # "{X} is used to/for/in {Y}"
        P.append(("used_for",
            re.compile(r'^(.+?)\s+(?:is|are|was|were)\s+(?:used|utilized|employed|applied)\s+(?:to|for|in|as)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "used_for", m.group(2), None),
            0.7))

        # "{X} serves as/functions as {Y}"
        P.append(("serves_as",
            re.compile(r'^(.+?)\s+(?:serves?|functions?|acts?|works?|operates?)\s+as\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "serves_as", m.group(2), None),
            0.7))

        # "{X} is responsible for {Y}"
        P.append(("responsible_for",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:responsible|accountable|in\s+charge)\s+(?:for|of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "responsible_for", m.group(2), None),
            0.7))

        # "{X} plays a role in {Y}"
        P.append(("plays_role",
            re.compile(r'^(.+?)\s+(?:plays?|has)\s+(?:a|an)\s+(?:key\s+|important\s+|crucial\s+|major\s+|significant\s+|central\s+|vital\s+)?role\s+in\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "role_in", m.group(2), None),
            0.65))

        # ════════════════════════════════════════════════════════
        # O. RELATIONSHIP / ASSOCIATION
        # ════════════════════════════════════════════════════════

        # "{X} is related to {Y}"
        P.append(("related_to",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:related|connected|linked|tied|associated|affiliated|allied)\s+(?:to|with)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "related_to", m.group(2), None),
            0.6))

        # "{X} is similar to {Y}"
        P.append(("similar_to",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:similar|analogous|comparable|equivalent|identical|akin)\s+to\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "similar_to", m.group(2), None),
            0.65))

        # "{X} is the opposite of {Y}"
        P.append(("opposite_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+the\s+(?:opposite|reverse|inverse|antonym|converse|antithesis)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "opposite_of", m.group(2), None),
            0.8))

        # "{X} succeeded/replaced/preceded {Y}"
        P.append(("succession",
            re.compile(r'^(.+?)\s+(?:succeeded|replaced|followed|preceded|came\s+(?:before|after))\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "succeeded", m.group(2), None),
            0.75))

        # "{X} married/divorced {Y}"
        P.append(("married",
            re.compile(r'^(.+?)\s+(?:married|wed|divorced|was\s+married\s+to)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "married", m.group(2), None),
            0.8))

        # "{X} is the father/mother/son/daughter/sibling/spouse of {Y}"
        P.append(("family",
            re.compile(r'^(.+?)\s+(?:is|was)\s+the\s+(father|mother|son|daughter|brother|sister|sibling|spouse|wife|husband|parent|child|grandparent|grandfather|grandmother|grandson|granddaughter|uncle|aunt|nephew|niece|cousin)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_of", m.group(3), None),
            0.9))

        # ════════════════════════════════════════════════════════
        # P. PROCESS / METHOD
        # ════════════════════════════════════════════════════════

        # "{X} is produced/manufactured/synthesized by/through/via {Y}"
        P.append(("produced_by",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:produced|manufactured|synthesized|synthesised|fabricated|generated|obtained|extracted|harvested|grown|cultivated)\s+(?:by|through|via|using|from)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "produced_by", m.group(2), None),
            0.75))

        # "{X} involves {Y}"
        P.append(("involves",
            re.compile(r'^(.+?)\s+(?:involves?|requires?|entails?|necessitates?|demands?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "involves", m.group(2), None),
            0.6))

        # "{X} converts/transforms {Y} into {Z}"
        P.append(("converts",
            re.compile(r'^(.+?)\s+(?:converts?|transforms?|changes?|turns?)\s+(.+?)\s+(?:into|to|in)\s+(.+?)\.?$', re.I),
            lambda m: [(m.group(1), "converts", m.group(2) + " → " + m.group(3), None)],
            0.7))

        # ════════════════════════════════════════════════════════
        # Q. LEGAL / POLITICAL
        # ════════════════════════════════════════════════════════

        # "{X} is governed/ruled by {Y}"
        P.append(("governed_by",
            re.compile(r'^(.+?)\s+(?:is|was|are|were)\s+(?:governed|ruled|led|headed|administered|managed|controlled|run|overseen)\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "governed_by", m.group(2), None),
            0.8))

        # "{X} signed/ratified {Y}"
        P.append(("signed",
            re.compile(r'^(.+?)\s+(?:signed|ratified|enacted|passed|adopted|approved|vetoed|repealed|amended|abolished)\s+(?:the\s+)?(.+?)\.?$', re.I),
            lambda m: (m.group(1), "signed", m.group(2), None),
            0.7))

        # "{X} declared independence from {Y}"
        P.append(("independence",
            re.compile(r'^(.+?)\s+(?:declared|gained|achieved|won|obtained)\s+(?:its?\s+)?independence\s+(?:from|in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "independence_from", m.group(2), None),
            0.85))

        # ════════════════════════════════════════════════════════
        # R. MEDICAL / BIOLOGICAL
        # ════════════════════════════════════════════════════════

        # "{X} is a symptom/sign/marker of {Y}"
        P.append(("symptom_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:a|an)\s+(?:common\s+)?(?:symptom|sign|marker|indicator|hallmark|feature|manifestation|characteristic|biomarker)\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "symptom_of", m.group(2), None),
            0.8))

        # "{X} is treated with/by {Y}"
        P.append(("treated_with",
            re.compile(r'^(.+?)\s+(?:is|are|was|were|can\s+be)\s+(?:treated|managed|controlled|cured|alleviated|relieved)\s+(?:with|by|using|through)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "treated_with", m.group(2), None),
            0.75))

        # "{X} is caused by / is associated with {Y}" (medical)
        P.append(("med_associated",
            re.compile(r'^(.+?)\s+(?:is|are)\s+(?:commonly\s+)?(?:associated|linked|correlated|connected)\s+with\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "associated_with", m.group(2), None),
            0.65))

        # "{X} secretes/produces/synthesizes {Y}"
        P.append(("secretes",
            re.compile(r'^(.+?)\s+(?:secretes?|produces?|synthesizes?|synthesises?|releases?|expresses?|encodes?|transcribes?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "produces", m.group(2), None),
            0.7))

        # "{X} is found in / lives in {Y}" (biology)
        P.append(("found_in",
            re.compile(r'^(.+?)\s+(?:is|are|can\s+be)\s+(?:found|present|abundant|common|native|endemic|indigenous)\s+(?:in|to|throughout|across)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "found_in", m.group(2), None),
            0.7))

        # "{X} feeds on / eats {Y}"
        P.append(("feeds_on",
            re.compile(r'^(.+?)\s+(?:feeds?\s+on|eats?|preys?\s+(?:on|upon)|consumes?|grazes?\s+on)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "feeds_on", m.group(2), None),
            0.75))

        # ════════════════════════════════════════════════════════
        # S. MUSIC / ART / CULTURE
        # ════════════════════════════════════════════════════════

        # "{X} is a {GENRE} {TYPE} by {Y}"
        P.append(("art_by",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:a|an)\s+(?:\w+\s+)?(?:song|album|book|novel|film|movie|play|poem|painting|sculpture|symphony|concerto|opera|sonata|composition|work|piece|series|show|musical)\s+(?:by|from|of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "created_by", m.group(2), None),
            0.8))

        # "{X} was released/published in {YEAR}"
        P.append(("released_in",
            re.compile(r'^(.+?)\s+was\s+(?:released|published|premiered|aired|broadcast|exhibited|performed|debuted|launched|issued)\s+(?:in|on)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "released_in", m.group(2), {"when": m.group(2)}),
            0.8))

        # "{X} features/stars {Y}"
        P.append(("features",
            re.compile(r'^(.+?)\s+(?:features?|stars?|showcases?|highlights?|spotlights?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "features", m.group(2), None),
            0.6))

        # "{X} is set in {Y}"
        P.append(("set_in",
            re.compile(r'^(.+?)\s+(?:is|was)\s+set\s+in\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "set_in", m.group(2), None),
            0.7))

        # "{X} is in the key of {Y}"
        P.append(("key_of",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:in\s+the\s+key\s+of|written\s+in)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "key_of", m.group(2), None),
            0.8))

        # ════════════════════════════════════════════════════════
        # T. MATHEMATICAL / LOGICAL
        # ════════════════════════════════════════════════════════

        # "{X} is equal/equivalent to {Y}"
        P.append(("equal_to",
            re.compile(r'^(.+?)\s+(?:is|are|was|were)\s+(?:equal|equivalent|identical|congruent|isomorphic|homeomorphic|diffeomorphic)\s+to\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "equals", m.group(2), None),
            0.8))

        # "{X} is a special case of {Y}"
        P.append(("special_case",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:a\s+)?(?:special|particular|specific|limiting|degenerate|trivial)\s+case\s+of\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "special_case_of", m.group(2), None),
            0.8))

        # "{X} generalizes {Y}" / "{X} is a generalization of {Y}"
        P.append(("generalizes",
            re.compile(r'^(.+?)\s+(?:generalizes?|is\s+a\s+generalization\s+of|extends?|is\s+an\s+extension\s+of)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "generalizes", m.group(2), None),
            0.75))

        # "{X} implies {Y}"
        P.append(("implies",
            re.compile(r'^(.+?)\s+(?:implies?|entails?|necessitates?|guarantees?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "implies", m.group(2), None),
            0.65))

        # "{X} is a theorem/lemma/conjecture/axiom in {Y}"
        P.append(("theorem_in",
            re.compile(r'^(.+?)\s+(?:is|was)\s+(?:a|an)\s+(?:fundamental\s+|important\s+|key\s+|well-known\s+|famous\s+)?(?:theorem|lemma|corollary|conjecture|axiom|postulate|principle|law|rule|identity|inequality|formula|equation|result)\s+(?:in|of|from)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "theorem_in", m.group(2), None),
            0.75))

        # "{X} states that {Y}"
        P.append(("states_that",
            re.compile(r'^(.+?)\s+(?:states?|asserts?|claims?|postulates?|proves?|shows?|establishes?|demonstrates?)\s+that\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "states", m.group(2), None),
            0.7))

        # "{X} satisfies/obeys {Y}"
        P.append(("satisfies",
            re.compile(r'^(.+?)\s+(?:satisfies?|obeys?|follows?|adheres?\s+to|conforms?\s+to|fulfills?|meets?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), "satisfies", m.group(2), None),
            0.65))

        # ════════════════════════════════════════════════════════
        # U. ADDITIONAL CATCH-ALL PATTERNS
        # ════════════════════════════════════════════════════════

        # "{X} {VERB}s {Y}" — broad transitive verbs
        _transitive_verbs = (
            'produce|generate|emit|absorb|reflect|transmit|conduct|'
            'regulate|control|modulate|activate|deactivate|bind|inhibit|'
            'catalyze|catalyse|metabolize|metabolise|oxidize|oxidise|reduce|'
            'hydrolyze|hydrolyse|polymerize|polymerise|dissolve|precipitate|'
            'crystallize|crystallise|evaporate|condense|sublimate|ionize|ionise|'
            'decompose|corrode|erode|govern|manage|administer|oversee|'
            'protect|defend|support|maintain|sustain|preserve|conserve|'
            'cover|span|extend|reach|surround|enclose|separate|connect|'
            'supply|provide|deliver|distribute|transport|carry|convey|'
            'store|accumulate|collect|gather|concentrate|disperse|scatter|'
            'transform|modify|alter|adjust|adapt|convert|process|'
            'encode|decode|translate|interpret|analyze|analyse|evaluate|'
            'demonstrate|illustrate|exemplify|embody|represent|depict|portray|'
            'influence|affect|impact|shape|determine|define|characterize|'
            'surpass|exceed|outperform|dominate|rival|complement|supplement|'
            'require|need|demand|depend|rely|utilize|employ|exploit|'
            'enable|allow|permit|facilitate|promote|enhance|improve|'
            'limit|restrict|constrain|confine|prevent|prohibit|forbid|'
            'replace|substitute|supersede|displace|succeed|follow|precede|'
            'stimulate|trigger|initiate|launch|commence|terminate|complete|'
            'teach|train|educate|mentor|instruct|guide|direct|lead|'
            'measure|quantify|assess|test|verify|validate|confirm|'
            'observe|detect|identify|recognize|distinguish|classify|'
            'treat|diagnose|cure|heal|remedy|alleviate|relieve|'
            'attack|invade|conquer|defeat|capture|occupy|liberate|'
            'celebrate|commemorate|honor|honour|recognize|acknowledge'
        )
        P.append(("generic_svo",
            re.compile(rf'^([A-Z].+?)\s+((?:{_transitive_verbs})(?:s|es|ed|d)?)\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower().rstrip('sed').rstrip('e') or m.group(2).lower(), m.group(3), None),
            0.6))

        # "{X} is {VERB}ed by {Y}" — broad passive for all transitive verbs above
        P.append(("generic_passive",
            re.compile(rf'^(.+?)\s+(?:is|are|was|were)\s+((?:{_transitive_verbs})(?:ed|d|t))\s+by\s+(.+?)\.?$', re.I),
            lambda m: (m.group(1), m.group(2).lower() + "_by", m.group(3), None),
            0.7))

        return P


# ── Module-level convenience ──
_default_atomizer = None

def get_atomizer() -> FactAtomizer:
    global _default_atomizer
    if _default_atomizer is None:
        _default_atomizer = FactAtomizer()
    return _default_atomizer


def atomize_facts(sentences: List[str], source: str = "wiki") -> List[FactAtom]:
    return get_atomizer().atomize_many(sentences, source)
