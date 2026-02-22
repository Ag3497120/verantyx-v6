"""
exact_answer_assembler.py — Assemble exact answers from atomized facts + IR query

Cross構造的アプローチ: 問題のquery type × Atomのpredicate × 回答形式 の三軸交差で答えを構成。
LLM不使用。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import re

from knowledge.fact_atomizer import FactAtom, FactAtomizer, get_atomizer


# ════════════════════════════════════════
# Query Type Detection
# ════════════════════════════════════════

@dataclass
class QuerySpec:
    """Parsed query specification from IR"""
    query_type: str       # who, what, where, when, how_many, yes_no, which_value, identify, name
    target_entity: str    # what we're asking about
    target_slot: str      # what attribute we want (capital, author, year, etc.)
    answer_shape: str     # entity_name, integer, decimal, boolean, phrase
    keywords: List[str] = field(default_factory=list)


class QueryDetector:
    """Detect query type from IR dict and problem text."""

    # question word → query type
    Q_MAP = {
        "who": "who",
        "whom": "who",
        "whose": "who",
        "what": "what",
        "which": "which",
        "where": "where",
        "when": "when",
        "how many": "how_many",
        "how much": "how_many",
        "how old": "how_many",
        "how long": "how_many",
        "how far": "how_many",
        "how tall": "how_many",
        "how fast": "how_many",
        "is it true": "yes_no",
        "is it": "yes_no",
        "does": "yes_no",
        "do": "yes_no",
        "can": "yes_no",
        "are": "yes_no",
        "was": "yes_no",
        "will": "yes_no",
    }

    # target_slot hints
    SLOT_HINTS = {
        "capital": ["capital"],
        "population": ["population", "inhabitants", "people"],
        "area": ["area", "size", "square"],
        "height": ["height", "tall", "elevation", "altitude"],
        "author": ["author", "wrote", "written by", "writer"],
        "painter": ["painted", "painter", "artist"],
        "director": ["directed", "director"],
        "inventor": ["invented", "inventor"],
        "discoverer": ["discovered", "discoverer"],
        "founder": ["founded", "founder"],
        "year": ["year", "when", "date"],
        "born": ["born", "birth"],
        "died": ["died", "death"],
        "location": ["located", "where", "country", "city", "state", "continent"],
        "language": ["language", "spoken"],
        "name": ["name", "called", "known as"],
        "cause": ["cause", "caused by", "reason", "why"],
        "formula": ["formula", "equation", "expression"],
        "symbol": ["symbol", "denoted", "represented"],
        "count": ["how many", "number of", "count"],
        "value": ["value", "amount", "measure"],
    }

    def detect(self, ir_dict: dict, problem_text: str = "") -> QuerySpec:
        """Detect query type from IR and problem text."""
        text = problem_text.lower() if problem_text else ""
        
        # Detect query type
        query_type = "what"  # default
        for prefix, qtype in sorted(self.Q_MAP.items(), key=lambda x: -len(x[0])):
            if text.startswith(prefix) or f" {prefix} " in text:
                query_type = qtype
                break

        # Detect answer shape from IR
        schema = str(ir_dict.get("answer_schema", "")).lower()
        task = str(ir_dict.get("task", "")).lower()
        
        if schema in ("integer", "numeric") or "count" in task or "how many" in text:
            answer_shape = "integer"
        elif schema == "decimal" or "how much" in text:
            answer_shape = "decimal"
        elif schema == "boolean" or query_type == "yes_no":
            answer_shape = "boolean"
        elif query_type in ("who", "where"):
            answer_shape = "entity_name"
        else:
            answer_shape = "phrase"

        # Detect target slot
        target_slot = "unknown"
        for slot, hints in self.SLOT_HINTS.items():
            if any(h in text for h in hints):
                target_slot = slot
                break

        # Detect target entity (the thing being asked about)
        target_entity = self._extract_target(ir_dict, text)

        # Keywords from IR
        metadata = ir_dict.get("metadata", {})
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]

        return QuerySpec(
            query_type=query_type,
            target_entity=target_entity,
            target_slot=target_slot,
            answer_shape=answer_shape,
            keywords=keywords,
        )

    def _extract_target(self, ir_dict: dict, text: str) -> str:
        """Extract the main entity being asked about."""
        # From IR entities
        entities = ir_dict.get("entities", [])
        if entities:
            if isinstance(entities[0], dict):
                return entities[0].get("name", entities[0].get("value", ""))
            return str(entities[0])

        # From problem text: extract noun phrase after question word
        m = re.search(r'(?:what|who|where|which)\s+(?:is|was|are|were)\s+(?:the\s+)?(.+?)(?:\?|$)', text)
        if m:
            return m.group(1).strip().rstrip('?')

        m = re.search(r'(?:what|who|where|which)\s+(.+?)(?:\?|$)', text)
        if m:
            return m.group(1).strip().rstrip('?')

        return ""


# ════════════════════════════════════════
# Answer Matching & Ranking
# ════════════════════════════════════════

@dataclass
class AnswerCandidate:
    answer: str
    confidence: float
    method: str
    support_atoms: List[FactAtom]
    relevance_score: float = 0.0


class AtomMatcher:
    """Match FactAtoms against QuerySpec to produce answer candidates."""

    # Blocked generic answers
    BLOCKED = {
        'the', 'a', 'an', 'this', 'that', 'it', 'they', 'them',
        'he', 'she', 'his', 'her', 'its',
        'charge', 'charge is e', 'and opposite to the charge of',
        'algebra', 'none', 'unknown', 'n/a', 'various', 'many', 'some',
        'thing', 'something', 'everything', 'nothing', 'anything',
    }

    # Max answer length — block verbose definitions from being returned as answers
    MAX_ANSWER_LEN = 60

    # Predicate → query_type affinity
    PRED_QUERY_AFFINITY = {
        "who": ["painted_by", "written_by", "composed_by", "directed_by",
                "invented_by", "discovered_by", "founded_by", "created_by",
                "developed_by", "built_by", "designed_by", "published_by",
                "born", "died", "won"],
        "where": ["located_in", "capital_of", "spoken_in", "in_place"],
        "when": ["born", "died", "occurred_in", "won"],
        "how_many": ["has_count", "population", "measures", "value"],
        "what": ["is", "is_a", "alias", "means", "serves_as", "used_for",
                 "contains", "named_after", "derived_from"],
    }

    def match(self, query: QuerySpec, atoms: List[FactAtom]) -> List[AnswerCandidate]:
        """Match atoms against query and produce ranked candidates."""
        candidates = []

        for atom in atoms:
            # Check if atom's answer is blocked
            if atom.object.lower().strip() in self.BLOCKED:
                continue
            if len(atom.object.strip()) < 1:
                continue
            # Block verbose definitions (is_a returns full definitions)
            if len(atom.object) > self.MAX_ANSWER_LEN:
                continue

            # Calculate relevance
            relevance = self._relevance(query, atom)
            if relevance < 0.1:
                continue

            # Determine which field is the "answer"
            answer = self._extract_answer(query, atom)
            if not answer or answer.lower().strip() in self.BLOCKED:
                continue

            candidates.append(AnswerCandidate(
                answer=answer,
                confidence=atom.confidence * relevance,
                method=f"atom:{atom.predicate}",
                support_atoms=[atom],
                relevance_score=relevance,
            ))

        # Merge candidates with same answer
        merged = self._merge(candidates)

        # Sort by combined score: relevance * confidence * sqrt(supports)
        import math
        merged.sort(
            key=lambda c: c.relevance_score * c.confidence * math.sqrt(len(c.support_atoms)),
            reverse=True,
        )

        return merged

    def _relevance(self, query: QuerySpec, atom: FactAtom) -> float:
        """Score how relevant this atom is to the query."""
        score = 0.0

        # 1. Predicate-query type affinity
        affinities = self.PRED_QUERY_AFFINITY.get(query.query_type, [])
        if atom.predicate in affinities:
            score += 0.4
        
        # 2. Target slot match
        if query.target_slot != "unknown":
            slot_lower = query.target_slot.lower()
            if slot_lower in atom.predicate.lower():
                score += 0.3
            # Check for related predicates
            pred_lower = atom.predicate.lower()
            if (slot_lower == "author" and any(w in pred_lower for w in ["written", "wrote", "author"])) or \
               (slot_lower == "capital" and "capital" in pred_lower) or \
               (slot_lower == "location" and any(w in pred_lower for w in ["located", "in_place"])) or \
               (slot_lower == "year" and any(w in pred_lower for w in ["born", "died", "occurred"])):
                score += 0.3

        # 3. Target entity overlap
        if query.target_entity:
            target_words = set(query.target_entity.lower().split())
            subj_words = set(atom.subject.lower().split())
            obj_words = set(atom.object.lower().split())
            
            subj_overlap = len(target_words & subj_words) / max(len(target_words), 1)
            obj_overlap = len(target_words & obj_words) / max(len(target_words), 1)
            
            score += max(subj_overlap, obj_overlap) * 0.3

        # 4. Keyword overlap
        if query.keywords:
            kw_set = set(k.lower() for k in query.keywords)
            sentence_words = set(atom.raw_sentence.lower().split())
            kw_overlap = len(kw_set & sentence_words) / max(len(kw_set), 1)
            score += kw_overlap * 0.2

        # 5. Answer shape match
        if query.answer_shape == "integer" and atom.numeric_value is not None:
            score += 0.2
        elif query.answer_shape == "entity_name" and re.match(r'^[A-Z]', atom.object):
            score += 0.1
        elif query.answer_shape == "boolean":
            if atom.predicate in ("is", "is_a", "belongs_to"):
                score += 0.1

        return min(score, 1.0)

    def _extract_answer(self, query: QuerySpec, atom: FactAtom) -> Optional[str]:
        """Extract the answer field from atom based on query type."""
        
        if query.query_type == "who":
            # "Who painted X?" → atom.object (the person) if predicate is passive
            if "_by" in atom.predicate:
                return atom.object
            # "Who is X?" → atom.object
            if atom.predicate in ("is", "is_a"):
                return atom.object
            return atom.object

        elif query.query_type == "where":
            if atom.predicate in ("located_in", "in_place", "spoken_in"):
                return atom.object
            if "capital" in atom.predicate:
                return atom.subject  # "Paris is the capital of France" → Paris
            return atom.object

        elif query.query_type == "when":
            if atom.when:
                return atom.when
            if atom.predicate in ("born", "died", "occurred_in"):
                return atom.object
            return atom.object

        elif query.query_type == "how_many":
            if atom.numeric_value is not None:
                v = atom.numeric_value
                if v == int(v):
                    return str(int(v))
                return str(v)
            return atom.object

        elif query.query_type == "yes_no":
            # If we found a supporting atom, the answer is likely "Yes"
            return "Yes"

        elif query.query_type == "what":
            # "What is the capital of France?" → check if predicate matches slot
            if query.target_slot != "unknown":
                return atom.object
            return atom.object

        return atom.object

    def _merge(self, candidates: List[AnswerCandidate]) -> List[AnswerCandidate]:
        """Merge candidates with the same answer."""
        by_answer: Dict[str, AnswerCandidate] = {}
        for c in candidates:
            key = c.answer.lower().strip()
            if key in by_answer:
                existing = by_answer[key]
                existing.support_atoms.extend(c.support_atoms)
                existing.confidence = max(existing.confidence, c.confidence)
                existing.relevance_score = max(existing.relevance_score, c.relevance_score)
            else:
                by_answer[key] = c
        return list(by_answer.values())


# ════════════════════════════════════════
# Main Assembler
# ════════════════════════════════════════

class ExactAnswerAssembler:
    """
    Assemble exact answers from atomized facts + IR query.
    
    Cross構造: query_type × predicate × answer_shape の三軸交差で答えを決定。
    """

    def __init__(self):
        self.atomizer = get_atomizer()
        self.query_detector = QueryDetector()
        self.matcher = AtomMatcher()

    def assemble(
        self,
        ir_dict: dict,
        wiki_facts: List[str],
        problem_text: str = "",
        cross_pieces: Optional[List[dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Assemble exact answer from facts.
        
        Returns:
            {"answer": str, "confidence": float, "method": str} or None
        """
        # 1. Atomize all facts
        atoms = self.atomizer.atomize_many(wiki_facts, source="wiki")
        
        if cross_pieces:
            for cp in cross_pieces:
                text = cp.get("text", cp.get("description", ""))
                if text:
                    atoms.extend(self.atomizer.atomize_many([text], source="cross_piece"))

        if not atoms:
            return None

        # 2. Detect query type
        query = self.query_detector.detect(ir_dict, problem_text)

        # 3. Match atoms against query
        candidates = self.matcher.match(query, atoms)

        # 4. Also try keyword-based matching (for HLE where questions are complex)
        if problem_text:
            kw_candidates = self._keyword_match(atoms, problem_text, ir_dict)
            candidates.extend(kw_candidates)

        if not candidates:
            return None

        # 5. Re-sort all candidates
        import math
        candidates.sort(
            key=lambda c: c.relevance_score * c.confidence * math.sqrt(len(c.support_atoms)),
            reverse=True,
        )

        # 6. Select best candidate
        best = candidates[0]

        # 7. Confidence threshold — lower for HLE (PhD-level questions are hard to match)
        combined = best.relevance_score * best.confidence * math.sqrt(len(best.support_atoms))
        if combined < 0.15:
            return None

        # 8. Format answer
        answer = self._format_answer(best.answer, query.answer_shape)

        return {
            "answer": answer,
            "confidence": best.confidence,
            "method": f"exact_assembler:{best.method}(supports={len(best.support_atoms)},rel={best.relevance_score:.2f},atoms={len(atoms)})",
            "query_type": query.query_type,
            "target_slot": query.target_slot,
        }

    def _keyword_match(self, atoms: List[FactAtom], problem_text: str, ir_dict: dict) -> List[AnswerCandidate]:
        """
        Keyword-based matching for complex HLE questions.
        Instead of relying on query type detection, directly match
        problem text keywords against atom subjects/objects.
        """
        candidates = []
        
        # Extract significant keywords from problem text (lowercase, >3 chars, not stop words)
        stop = {'what','which','where','when','does','that','this','with','from',
                'have','been','will','would','could','should','about','their',
                'they','them','there','these','those','each','other','some',
                'than','then','into','also','most','only','such','very',
                'more','over','after','before','between','under','above',
                'through','during','following','answer','question','give',
                'find','determine','identify','name','list','describe','explain'}
        
        words = set(re.findall(r'[a-zA-Z]{4,}', problem_text.lower())) - stop
        
        # Find atoms where subject or predicate overlaps with problem keywords
        for atom in atoms:
            if atom.object.lower().strip() in AtomMatcher.BLOCKED:
                continue
            if len(atom.object) > AtomMatcher.MAX_ANSWER_LEN:
                continue
                
            atom_words = set(re.findall(r'[a-zA-Z]{4,}', 
                (atom.subject + " " + atom.predicate + " " + atom.raw_sentence).lower()))
            
            overlap = len(words & atom_words)
            if overlap < 2:
                continue
            
            relevance = min(overlap / max(len(words), 1) * 1.5, 1.0)
            
            # Prefer named entities and specific values over generic text
            answer = atom.object
            if atom.numeric_value is not None:
                answer = str(int(atom.numeric_value)) if atom.numeric_value == int(atom.numeric_value) else str(atom.numeric_value)
            
            # Skip if answer is just a common word
            if answer.lower() in stop or len(answer.strip()) < 2:
                continue
                
            candidates.append(AnswerCandidate(
                answer=answer,
                confidence=atom.confidence * 0.8,  # slight penalty for keyword-only match
                method=f"atom_kw:{atom.predicate}(overlap={overlap})",
                support_atoms=[atom],
                relevance_score=relevance,
            ))
        
        return candidates

    def _format_answer(self, raw: str, shape: str) -> str:
        """Format answer according to expected shape."""
        raw = raw.strip()
        
        if shape == "integer":
            # Try to extract integer
            m = re.search(r'-?\d[\d,]*', raw)
            if m:
                return m.group(0).replace(',', '')
            return raw
            
        elif shape == "decimal":
            m = re.search(r'-?\d[\d,]*\.?\d*', raw)
            if m:
                return m.group(0).replace(',', '')
            return raw
            
        elif shape == "boolean":
            lower = raw.lower()
            if lower in ("yes", "true", "correct"):
                return "Yes"
            elif lower in ("no", "false", "incorrect"):
                return "No"
            return raw
            
        elif shape == "entity_name":
            # Clean up
            raw = re.sub(r'\s*\(.*?\)\s*', '', raw)  # Remove parentheticals
            return raw.strip()
            
        return raw
