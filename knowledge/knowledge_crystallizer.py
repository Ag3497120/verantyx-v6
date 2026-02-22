"""
knowledge_crystallizer.py
Knowledge Crystallizer — Wikipedia/Qwen facts → Cross Piece + Fact Atom 変換

設計思想（kofdai 2026-02-22）:
  知識取得（Wikipedia hit）はできているが、
  知識を「使える形」に変換する層がない。
  
  facts → Cross Piece化 → 小世界検証 → answer生成
  
  「知識注入」はできているが「結晶化（crystallization）」ができていない。

出力:
  1. FactAtom — 答え候補の原子（数値、固有名詞、記号、短句）
  2. CrossPiece — CEGIS/CrossSimulator が扱える構造化知識
  3. RelationPiece — 概念間の関係（A causes B, X is_a Y）

鉄の壁: 問題文は一切参照しない。facts + IR のみ。
"""

from __future__ import annotations
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════

@dataclass
class FactAtom:
    """答え候補の原子 — facts から抽出した具体的な値"""
    value: str                      # "Paris", "48", "Euler's theorem"
    atom_type: str                  # "entity" | "numeric" | "formula" | "phrase"
    support_fact_ids: List[int] = field(default_factory=list)
    confidence: float = 0.5
    normalization: Dict[str, str] = field(default_factory=dict)  # casefold, stripped etc.
    source_concept: str = ""        # どの概念から抽出されたか


@dataclass
class RelationPiece:
    """概念間の関係"""
    subject: str                    # "Barcan formula"
    predicate: str                  # "valid_in", "causes", "is_a", "equals"
    object: str                     # "constant domain semantics"
    negated: bool = False           # "is NOT valid in ..."
    source_fact_id: int = -1
    confidence: float = 0.5


@dataclass
class CrossPiece:
    """CEGIS/CrossSimulator 用の構造化知識ピース"""
    piece_id: str
    piece_type: str                 # "definition" | "rule" | "property" | "criterion"
    domain: str
    symbols: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)


@dataclass
class CrystallizedKnowledge:
    """結晶化の出力"""
    fact_atoms: List[FactAtom] = field(default_factory=list)
    relations: List[RelationPiece] = field(default_factory=list)
    cross_pieces: List[CrossPiece] = field(default_factory=list)
    source_facts_count: int = 0


# ══════════════════════════════════════════════════════════════
#  Main Crystallizer
# ══════════════════════════════════════════════════════════════

class KnowledgeCrystallizer:
    """
    Wikipedia/Qwen facts → 結晶化された知識構造
    
    LLM不使用。ルールベース抽出。
    """

    # ── 数値パターン ──
    NUM_PATTERNS = [
        # "is 42", "equals 3.14", "= 7"
        re.compile(r'(?:is|equals?|=|was|are)\s+(-?[\d,]+(?:\.\d+)?)\b', re.I),
        # "approximately 2.718"
        re.compile(r'(?:approximately|about|roughly|around|circa)\s+(-?[\d,]+(?:\.\d+)?)\b', re.I),
        # "discovered in 1905"
        re.compile(r'(?:in|year|dated?)\s+(1[0-9]{3}|20[0-9]{2})\b', re.I),
        # standalone numbers with units: "48 states", "206 bones"
        re.compile(r'\b(\d{1,6})\s+(?:states?|bones?|elements?|species|countries|members?|layers?|dimensions?|vertices|edges|faces)\b', re.I),
    ]

    # ── 固有名詞パターン ──
    ENTITY_PATTERNS = [
        # "is called X", "known as X", "named X"
        re.compile(r'(?:is\s+called|known\s+as|named|termed|dubbed)\s+(?:the\s+)?([A-Z][a-zA-Z\s\-]{2,30})', re.I),
        # "X is a/an Y" (extract X)
        re.compile(r'^([A-Z][a-zA-Z\s\-]{2,40})\s+(?:is|was|are)\s+(?:a|an|the)\s+', re.M),
        # Capitalized multi-word terms
        re.compile(r'(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'),
    ]

    # ── 関係パターン ──
    RELATION_PATTERNS = [
        # "X is valid in Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+is\s+(valid|invalid|true|false)\s+in\s+([\w\s\-]{3,30})', re.I),
         lambda m: ("valid_in" if m.group(2).lower() in ("valid", "true") else "invalid_in", m.group(1).strip(), m.group(3).strip())),
        # "X causes Y" / "X leads to Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+(?:causes?|leads?\s+to|results?\s+in|produces?)\s+([\w\s\-]{3,30})', re.I),
         lambda m: ("causes", m.group(1).strip(), m.group(2).strip())),
        # "X is a type of Y" / "X is a Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+is\s+(?:a\s+)?(?:type|kind|form|variant|special\s+case)\s+of\s+([\w\s\-]{3,30})', re.I),
         lambda m: ("is_a", m.group(1).strip(), m.group(2).strip())),
        # "X holds/fails in Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+(holds?|fails?|breaks?)\s+in\s+([\w\s\-]{3,30})', re.I),
         lambda m: ("holds_in" if "hold" in m.group(2).lower() else "fails_in", m.group(1).strip(), m.group(3).strip())),
        # "X does not hold in Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+does\s+not\s+hold\s+in\s+([\w\s\-]{3,30})', re.I),
         lambda m: ("fails_in", m.group(1).strip(), m.group(2).strip())),
        # "X equals Y" / "X = Y"
        (re.compile(r'([A-Z][\w\s\-]{2,30})\s+(?:equals?|=)\s+([\w\s\-\d\.]{1,30})', re.I),
         lambda m: ("equals", m.group(1).strip(), m.group(2).strip())),
    ]

    # ── 定義パターン ──
    DEFINITION_PATTERN = re.compile(
        r'([A-Z][\w\s\-]{2,40})\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*'
        r'([\w\s\-,]{5,80}?)(?:\.|,\s)', re.M
    )

    # ── 否定パターン ──
    NEGATION_PATTERN = re.compile(
        r'(?:not|never|no|cannot|can\'t|doesn\'t|don\'t|isn\'t|aren\'t|neither|nor)\s',
        re.I
    )

    def crystallize(
        self,
        ir_dict: dict,
        raw_facts: List[dict],
    ) -> CrystallizedKnowledge:
        """
        Raw facts → CrystallizedKnowledge
        
        Args:
            ir_dict: IR.to_dict() 出力
            raw_facts: pipeline から取得した facts リスト
        """
        result = CrystallizedKnowledge(source_facts_count=len(raw_facts))

        for idx, fact in enumerate(raw_facts):
            text = self._get_fact_text(fact)
            if not text:
                continue

            concept = ""
            if isinstance(fact, dict):
                concept = fact.get("symbol", "") or fact.get("domain", "")

            # ── Fact Atoms 抽出 ──
            atoms = self._extract_atoms(text, idx, concept)
            result.fact_atoms.extend(atoms)

            # ── Relations 抽出 ──
            relations = self._extract_relations(text, idx)
            result.relations.extend(relations)

            # ── Properties/Formulas → CrossPiece ──
            if isinstance(fact, dict):
                props = fact.get("properties", [])
                formulas = fact.get("formulas", [])
                numeric = fact.get("numeric", {})
                if props or formulas or numeric:
                    piece = self._make_cross_piece(
                        fact_idx=idx,
                        concept=concept,
                        domain=fact.get("domain", ""),
                        properties=props,
                        formulas=formulas,
                        numeric=numeric,
                        summary=text[:200],
                    )
                    result.cross_pieces.append(piece)

                # numeric values → FactAtoms
                for key, val in numeric.items():
                    result.fact_atoms.append(FactAtom(
                        value=str(val),
                        atom_type="numeric",
                        support_fact_ids=[idx],
                        confidence=0.8,
                        normalization={"key": key},
                        source_concept=concept,
                    ))

        # 重複除去
        result.fact_atoms = self._dedupe_atoms(result.fact_atoms)

        log.info(
            f"Crystallized: atoms={len(result.fact_atoms)} "
            f"relations={len(result.relations)} "
            f"pieces={len(result.cross_pieces)} "
            f"from {result.source_facts_count} facts"
        )

        return result

    def _get_fact_text(self, fact) -> str:
        """fact からテキストを取得"""
        if isinstance(fact, dict):
            return (fact.get("summary", "") or fact.get("plain", ""))[:500]
        elif hasattr(fact, "summary"):
            return (fact.summary or "")[:500]
        return ""

    def _extract_atoms(self, text: str, fact_idx: int, concept: str) -> List[FactAtom]:
        """テキストから FactAtom を抽出"""
        atoms = []

        # 数値
        for pat in self.NUM_PATTERNS:
            for m in pat.finditer(text):
                val = m.group(1).replace(",", "")
                atoms.append(FactAtom(
                    value=val,
                    atom_type="numeric",
                    support_fact_ids=[fact_idx],
                    confidence=0.6,
                    normalization={"raw": m.group(0)[:50]},
                    source_concept=concept,
                ))

        # 固有名詞
        for pat in self.ENTITY_PATTERNS:
            for m in pat.finditer(text):
                entity = m.group(1).strip()
                if len(entity) > 2 and entity.lower() not in (
                    'the', 'this', 'that', 'these', 'those', 'they', 'there',
                    'which', 'where', 'when', 'what', 'however', 'also',
                ):
                    atoms.append(FactAtom(
                        value=entity,
                        atom_type="entity",
                        support_fact_ids=[fact_idx],
                        confidence=0.5,
                        normalization={"casefold": entity.lower()},
                        source_concept=concept,
                    ))

        # 数式（LaTeX）
        for m in re.finditer(r'\$([^$]+)\$|\\\((.+?)\\\)', text):
            formula = (m.group(1) or m.group(2) or "").strip()
            if formula and len(formula) > 1:
                atoms.append(FactAtom(
                    value=formula,
                    atom_type="formula",
                    support_fact_ids=[fact_idx],
                    confidence=0.4,
                    source_concept=concept,
                ))

        return atoms

    def _extract_relations(self, text: str, fact_idx: int) -> List[RelationPiece]:
        """テキストから RelationPiece を抽出"""
        relations = []
        for pattern, extractor in self.RELATION_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    predicate, subject, obj = extractor(m)
                    negated = bool(self.NEGATION_PATTERN.search(m.group(0)))
                    relations.append(RelationPiece(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        negated=negated,
                        source_fact_id=fact_idx,
                        confidence=0.6,
                    ))
                except Exception:
                    continue
        return relations

    def _make_cross_piece(
        self,
        fact_idx: int,
        concept: str,
        domain: str,
        properties: list,
        formulas: list,
        numeric: dict,
        summary: str,
    ) -> CrossPiece:
        """properties/formulas/numeric → CrossPiece"""
        symbols = [concept] if concept else []
        predicates = [str(p) for p in properties[:5]]
        constraints = [str(f) for f in formulas[:3]]
        if numeric:
            for k, v in list(numeric.items())[:3]:
                constraints.append(f"{k} = {v}")

        return CrossPiece(
            piece_id=f"crystal_{fact_idx}_{concept or 'unknown'}",
            piece_type="definition" if not formulas else "property",
            domain=domain or "unknown",
            symbols=symbols,
            predicates=predicates,
            constraints=constraints,
            provenance=[f"fact_{fact_idx}", summary[:60]],
        )

    def _dedupe_atoms(self, atoms: List[FactAtom]) -> List[FactAtom]:
        """重複するFactAtomを統合"""
        seen = {}
        for atom in atoms:
            key = (atom.value.lower().strip(), atom.atom_type)
            if key in seen:
                existing = seen[key]
                existing.support_fact_ids.extend(atom.support_fact_ids)
                existing.confidence = max(existing.confidence, atom.confidence)
            else:
                seen[key] = atom
        return list(seen.values())


# ══════════════════════════════════════════════════════════════
#  ExactAnswerSynthesizer — exactMatch問題用
# ══════════════════════════════════════════════════════════════

class ExactAnswerSynthesizer:
    """
    FactAtoms + IR → exactMatch 回答候補生成
    
    answer_contract に基づいて候補をフィルタ・ランキング
    """

    def synthesize(
        self,
        ir_dict: dict,
        crystal: CrystallizedKnowledge,
    ) -> Optional[Dict[str, Any]]:
        """
        exactMatch の回答候補を生成
        
        Returns:
            {"answer": str, "confidence": float, "method": str, "supports": list} or None
        """
        schema = ir_dict.get("answer_schema", "free_form")
        atoms = crystal.fact_atoms

        if not atoms:
            return None

        # answer_contract を推定
        contract = self._infer_contract(ir_dict)

        # contract に基づいてフィルタ
        candidates = []
        for atom in atoms:
            if self._matches_contract(atom, contract):
                candidates.append(atom)

        if not candidates:
            return None

        # support fact 数が多い順にソート
        candidates.sort(key=lambda a: (len(a.support_fact_ids), a.confidence), reverse=True)

        best = candidates[0]

        # 最低条件: support が2つ以上、または confidence が高い
        if len(best.support_fact_ids) >= 2 or best.confidence >= 0.7:
            return {
                "answer": best.value,
                "confidence": best.confidence,
                "method": f"exact_from_atoms(type={best.atom_type},supports={len(best.support_fact_ids)})",
                "supports": best.support_fact_ids,
            }

        return None

    def _infer_contract(self, ir_dict: dict) -> str:
        """IR から answer_contract を推定"""
        schema = ir_dict.get("answer_schema", "")
        task = ir_dict.get("task", "")

        if "numeric" in str(schema).lower() or "integer" in str(schema).lower():
            return "numeric"
        if "number" in str(task).lower() or "count" in str(task).lower():
            return "numeric"

        metadata = ir_dict.get("metadata", {})
        keywords = metadata.get("keywords", [])
        if any(k in keywords for k in ["number", "count", "how many", "calculate"]):
            return "numeric"

        return "any"

    def _matches_contract(self, atom: FactAtom, contract: str) -> bool:
        """atom が contract に合致するか"""
        if contract == "any":
            return True
        if contract == "numeric":
            return atom.atom_type == "numeric"
        if contract == "entity":
            return atom.atom_type == "entity"
        return True
