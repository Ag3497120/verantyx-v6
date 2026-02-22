"""
concept_extractor_v2.py
問題文から専門概念名を自動抽出する（v2: パターンマッチ + ヒューリスティクスNER）

鉄の壁: ここで抽出した概念名だけがLLM/Wikipediaに渡される。問題文は渡さない。

改善点:
  1. 大文字始まりの複合語を概念候補として抽出（Named Entity風）
  2. 学術用語パターン（X's theorem, X equation, X hypothesis 等）
  3. 括弧内の略語 (e.g., "polymerase chain reaction (PCR)")
  4. ドメインキーワード辞書の大幅拡充
  5. ノイズフィルタ（一般的すぎる語の除外）
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List


@dataclass
class ExtractedConcept:
    """抽出された概念"""
    name: str           # e.g. "Barcan formula"
    domain_hint: str    # e.g. "modal_logic"
    kind: str           # "definition" | "theorem" | "property" | "formula" | "entity"
    confidence: float   # 0.0-1.0
    source: str         # "pattern" | "ner" | "acronym" | "keyword"


# ── 一般的すぎて概念としては無価値な語 ──
STOP_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "which", "what", "where", "when", "who", "how", "why",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "not", "no", "yes", "true", "false",
    "if", "then", "else", "and", "or", "but", "nor",
    "for", "with", "from", "to", "in", "on", "at", "by", "of",
    "all", "each", "every", "some", "any", "most", "more",
    "than", "also", "only", "just", "very", "much",
    "one", "two", "three", "four", "five",
    "first", "second", "third", "last", "next",
    "new", "old", "other", "same", "different",
    "many", "few", "several", "both",
    "given", "find", "determine", "calculate", "compute",
    "consider", "suppose", "assume", "let",
    "answer", "question", "problem", "solution",
    "following", "below", "above",
    "show", "prove", "verify", "explain",
    "value", "number", "set", "point", "line",  # too generic in math
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Figure", "Table", "Example", "Note", "See", "Hint",
    "Part", "Section", "Chapter",
}

# ── 学術用語のサフィックスパターン ──
ACADEMIC_SUFFIXES = [
    "theorem", "lemma", "conjecture", "hypothesis", "principle",
    "law", "rule", "formula", "equation", "inequality",
    "axiom", "postulate", "corollary", "proposition",
    "effect", "paradox", "phenomenon", "mechanism",
    "process", "reaction", "syndrome", "disease", "disorder",
    "method", "algorithm", "transform", "decomposition",
    "space", "group", "ring", "field", "algebra", "manifold",
    "distribution", "function", "operator", "matrix",
    "number", "constant", "coefficient", "index",
    "model", "theory", "criterion", "test",
    "diagram", "spectrum", "series", "sequence",
    "cycle", "path", "graph", "tree", "network",
    "bond", "orbital", "configuration", "structure",
    "species", "genus", "family", "order", "class", "phylum",
]

# ── ドメイン推定キーワード（拡充版） ──
DOMAIN_KEYWORDS = {
    # 数学
    "topology": "topology", "manifold": "topology", "homotopy": "algebraic_topology",
    "homology": "algebraic_topology", "cohomology": "algebraic_topology",
    "polynomial": "algebra", "eigenvalue": "linear_algebra", "determinant": "linear_algebra",
    "integral": "calculus", "derivative": "calculus", "differential": "calculus",
    "probability": "probability", "stochastic": "probability",
    "prime": "number_theory", "modular": "number_theory", "congruence": "number_theory",
    "graph": "graph_theory", "vertex": "graph_theory", "edge": "graph_theory",
    "chromatic": "graph_theory", "hamiltonian": "graph_theory",
    "combinatorial": "combinatorics", "permutation": "combinatorics",
    "metric": "analysis", "convergence": "analysis", "compact": "analysis",
    "morphism": "category_theory", "functor": "category_theory",
    "sheaf": "algebraic_geometry", "scheme": "algebraic_geometry",

    # 物理
    "quantum": "quantum_mechanics", "photon": "quantum_mechanics",
    "quark": "particle_physics", "boson": "particle_physics", "fermion": "particle_physics",
    "entropy": "thermodynamics", "enthalpy": "thermodynamics",
    "magnetic": "electromagnetism", "electric": "electromagnetism",
    "gravitational": "gravity", "relativity": "relativity",
    "wavelength": "optics", "refraction": "optics", "diffraction": "optics",
    "superconductor": "condensed_matter", "semiconductor": "condensed_matter",

    # 化学
    "molecule": "chemistry", "atom": "chemistry", "ion": "chemistry",
    "covalent": "chemistry", "ionic": "chemistry", "metallic": "chemistry",
    "organic": "organic_chemistry", "aromatic": "organic_chemistry",
    "catalyst": "chemistry", "enzyme": "biochemistry",
    "protein": "biochemistry", "amino": "biochemistry",
    "nucleotide": "molecular_biology", "DNA": "molecular_biology", "RNA": "molecular_biology",
    "spectroscopy": "analytical_chemistry", "chromatography": "analytical_chemistry",

    # 生物
    "gene": "genetics", "allele": "genetics", "chromosome": "genetics",
    "evolution": "evolutionary_biology", "phylogenetic": "evolutionary_biology",
    "neuron": "neuroscience", "synapse": "neuroscience",
    "cell": "cell_biology", "mitosis": "cell_biology", "meiosis": "cell_biology",
    "ecosystem": "ecology", "population": "ecology",
    "virus": "virology", "bacteria": "microbiology",
    "antibody": "immunology", "antigen": "immunology",

    # CS
    "algorithm": "computer_science", "complexity": "computational_complexity",
    "automaton": "automata_theory", "Turing": "computability_theory",
    "compiler": "compiler_design", "parser": "compiler_design",
    "database": "databases", "SQL": "databases",
    "neural": "machine_learning", "gradient": "machine_learning",
    "cryptographic": "cryptography", "encryption": "cryptography",

    # その他
    "jurisprudence": "law", "statute": "law", "tort": "law",
    "GDP": "economics", "inflation": "economics",
    "syllogism": "logic", "predicate": "logic",
}


def extract_concepts_v2(problem_text: str, ir_dict: dict = None) -> list[ExtractedConcept]:
    """
    問題文から専門概念を自動抽出する。

    鉄の壁: 問題文はここで処理され、概念名だけが出力される。
    """
    concepts: list[ExtractedConcept] = []
    seen: set[str] = set()

    # ──────────────────────────────────────
    # 1. Named Entity 風: 大文字始まりの複合語
    # ──────────────────────────────────────
    # "Barcan formula", "Hilbert space", "Nash equilibrium"
    cap_pattern = r'\b([A-Z][a-zà-ÿ]+(?:\s+[A-Z]?[a-zà-ÿ]+){0,3})\s+(' + '|'.join(ACADEMIC_SUFFIXES) + r')\b'
    for m in re.finditer(cap_pattern, problem_text):  # NO re.IGNORECASE — require leading capital
        full = f"{m.group(1)} {m.group(2)}".strip()
        # All words must not be stop words
        words = full.lower().split()
        if any(w in STOP_WORDS for w in words[:len(words)-1]):  # allow suffix in stop
            continue
        if words[0] in STOP_WORDS:
            name_key = full.lower()
            if name_key not in seen:
                domain = _infer_domain(full)
                kind = _suffix_to_kind(m.group(2).lower())
                concepts.append(ExtractedConcept(
                    name=full, domain_hint=domain, kind=kind,
                    confidence=0.9, source="pattern"
                ))
                seen.add(name_key)

    # ──────────────────────────────────────
    # 2. 所有格パターン: "X's theorem", "X's law"
    # ──────────────────────────────────────
    poss_pattern = r"\b([A-Z][a-zà-ÿ]+(?:[-'][a-zà-ÿ]+)?)'s\s+(" + '|'.join(ACADEMIC_SUFFIXES) + r')\b'
    for m in re.finditer(poss_pattern, problem_text):
        full = f"{m.group(1)}'s {m.group(2)}"
        name_key = full.lower()
        if name_key not in seen:
            concepts.append(ExtractedConcept(
                name=full, domain_hint=_infer_domain(full),
                kind=_suffix_to_kind(m.group(2).lower()),
                confidence=0.9, source="pattern"
            ))
            seen.add(name_key)

    # ──────────────────────────────────────
    # 3. ハイフン付き複合名: "Born-Oppenheimer", "Cayley-Hamilton"
    # ──────────────────────────────────────
    for m in re.finditer(r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)+)\b', problem_text):
        name = m.group(1)
        name_key = name.lower()
        if name_key not in seen and len(name) > 5:
            concepts.append(ExtractedConcept(
                name=name, domain_hint=_infer_domain(name),
                kind="definition", confidence=0.8, source="ner"
            ))
            seen.add(name_key)

    # ──────────────────────────────────────
    # 4. 括弧内の略語: "polymerase chain reaction (PCR)"
    # ──────────────────────────────────────
    for m in re.finditer(r'([A-Za-z][a-z]+(?:\s+[a-z]+){1,4})\s+\(([A-Z]{2,6})\)', problem_text):
        full_name = m.group(1)
        abbrev = m.group(2)
        name_key = full_name.lower()
        if name_key not in seen:
            concepts.append(ExtractedConcept(
                name=full_name, domain_hint=_infer_domain(full_name + " " + abbrev),
                kind="definition", confidence=0.85, source="acronym"
            ))
            seen.add(name_key)

    # ──────────────────────────────────────
    # 5. 単独の大文字固有名（人名→定理名の可能性）
    # ──────────────────────────────────────
    for m in re.finditer(r'\b([A-Z][a-zà-ÿ]{3,})\b', problem_text):
        name = m.group(1)
        name_key = name.lower()
        if name_key not in seen and name_key not in STOP_WORDS:
            # 次の語がacademic suffixかチェック
            after = problem_text[m.end():m.end()+30]
            for suf in ACADEMIC_SUFFIXES[:20]:  # 主要なもののみ
                if re.match(rf'\s+{suf}\b', after, re.IGNORECASE):
                    full = f"{name} {suf}"
                    full_key = full.lower()
                    if full_key not in seen:
                        concepts.append(ExtractedConcept(
                            name=full, domain_hint=_infer_domain(full),
                            kind=_suffix_to_kind(suf),
                            confidence=0.85, source="ner"
                        ))
                        seen.add(full_key)
                    break

    # ──────────────────────────────────────
    # 6. ドメインキーワードマッチ
    # ──────────────────────────────────────
    text_lower = problem_text.lower()
    for kw, domain in DOMAIN_KEYWORDS.items():
        if kw.lower() in text_lower and kw.lower() not in seen:
            # キーワード周辺のコンテキストを概念名に
            concepts.append(ExtractedConcept(
                name=kw, domain_hint=domain,
                kind="definition", confidence=0.5, source="keyword"
            ))
            seen.add(kw.lower())

    # ──────────────────────────────────────
    # 7. 信頼度でソート、上位を返す
    # ──────────────────────────────────────
    concepts.sort(key=lambda c: c.confidence, reverse=True)
    return concepts[:15]  # 多すぎるとAPI呼び出しが爆発するので上限


def _infer_domain(text: str) -> str:
    """テキストからドメインを推定"""
    t = text.lower()
    for kw, domain in DOMAIN_KEYWORDS.items():
        if kw.lower() in t:
            return domain
    return "general"


def _suffix_to_kind(suffix: str) -> str:
    """学術サフィックスからkindを推定"""
    theorem_like = {"theorem", "lemma", "conjecture", "corollary", "proposition", "principle"}
    formula_like = {"formula", "equation", "inequality"}
    property_like = {"law", "rule", "axiom", "postulate", "criterion"}
    if suffix in theorem_like:
        return "theorem"
    if suffix in formula_like:
        return "formula"
    if suffix in property_like:
        return "property"
    return "definition"
