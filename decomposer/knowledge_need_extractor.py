"""
knowledge_need_extractor.py
Decomposer の出力 (IR) から「不足知識」を構造化して抽出する。

設計原則（kofdai指示 2026-02-22）:
  - 問題文は LLM に渡さない（鉄の壁）
  - Verantyx が分解した IR だけを見て missing を埋める
  - missing = LLM に聞くべき知識クエリのリスト
  - 各 missing エントリは「概念名 + 関係 + 必要な知識の種類」のみ

出力:
  List[KnowledgeNeed] — GapDetector が直接消費できる形式
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KnowledgeNeed:
    """LLM に聞くべき知識の1単位。問題文は含まない。"""
    concept: str               # e.g. "Barcan_formula", "checkmate", "Wasserstein_space"
    kind: str                  # "definition" | "theorem" | "property" | "criterion" | "formula" | "fact"
    domain: str                # e.g. "modal_logic", "chess", "measure_theory"
    relation: str = ""         # e.g. "Barcan_formula -> validity_conditions"
    scope: str = "concise"     # "concise" | "detailed" | "exhaustive"
    max_facts: int = 5
    context_hint: str = ""     # 非問題文の追加ヒント e.g. "in context of first-order modal logic"


# ──────────────────────────────────────────────────────────────────
# ドメイン × キーワード → 知識ニーズのルール辞書
# ──────────────────────────────────────────────────────────────────

# 各ドメインで「このキーワードが IR に含まれているなら、この知識が必要」
DOMAIN_NEED_RULES: dict[str, list[dict]] = {
    # ── 数学系 ──
    "number_theory": [
        {"keywords": ["prime", "primality"], "concept": "primality_test", "kind": "criterion"},
        {"keywords": ["gcd", "greatest_common_divisor"], "concept": "euclidean_algorithm", "kind": "theorem"},
        {"keywords": ["lcm"], "concept": "least_common_multiple", "kind": "definition"},
        {"keywords": ["modulo", "mod", "congruent"], "concept": "modular_arithmetic", "kind": "definition"},
        {"keywords": ["euler", "totient", "phi"], "concept": "euler_totient_function", "kind": "theorem"},
        {"keywords": ["fermat"], "concept": "fermat_little_theorem", "kind": "theorem"},
        {"keywords": ["divisor", "factor"], "concept": "divisor_function", "kind": "definition"},
    ],
    "combinatorics": [
        {"keywords": ["permutation"], "concept": "permutation_formula", "kind": "formula"},
        {"keywords": ["combination", "binomial"], "concept": "binomial_coefficient", "kind": "formula"},
        {"keywords": ["catalan"], "concept": "catalan_number", "kind": "formula"},
        {"keywords": ["stirling"], "concept": "stirling_number", "kind": "definition"},
        {"keywords": ["pigeonhole"], "concept": "pigeonhole_principle", "kind": "theorem"},
        {"keywords": ["inclusion", "exclusion"], "concept": "inclusion_exclusion", "kind": "theorem"},
        {"keywords": ["derangement"], "concept": "derangement", "kind": "formula"},
    ],
    "algebra": [
        {"keywords": ["polynomial", "roots"], "concept": "polynomial_roots", "kind": "theorem"},
        {"keywords": ["quadratic"], "concept": "quadratic_formula", "kind": "formula"},
        {"keywords": ["group", "subgroup"], "concept": "group_theory_basics", "kind": "definition"},
        {"keywords": ["ring", "ideal"], "concept": "ring_theory_basics", "kind": "definition"},
        {"keywords": ["field", "extension"], "concept": "field_extension", "kind": "definition"},
        {"keywords": ["galois"], "concept": "galois_theory", "kind": "theorem"},
    ],
    "linear_algebra": [
        {"keywords": ["eigenvalue", "eigenvector"], "concept": "eigenvalue_decomposition", "kind": "theorem"},
        {"keywords": ["determinant"], "concept": "determinant_properties", "kind": "property"},
        {"keywords": ["rank", "null"], "concept": "rank_nullity_theorem", "kind": "theorem"},
        {"keywords": ["orthogonal", "projection"], "concept": "orthogonal_projection", "kind": "definition"},
        {"keywords": ["singular", "svd"], "concept": "singular_value_decomposition", "kind": "theorem"},
    ],
    "calculus": [
        {"keywords": ["derivative", "differentiate"], "concept": "differentiation_rules", "kind": "formula"},
        {"keywords": ["integral", "integrate"], "concept": "integration_techniques", "kind": "formula"},
        {"keywords": ["taylor", "series"], "concept": "taylor_series", "kind": "formula"},
        {"keywords": ["limit", "converge"], "concept": "limit_evaluation", "kind": "criterion"},
        {"keywords": ["fourier"], "concept": "fourier_transform", "kind": "definition"},
    ],
    "geometry": [
        {"keywords": ["triangle", "area"], "concept": "triangle_area_formulas", "kind": "formula"},
        {"keywords": ["circle", "radius"], "concept": "circle_properties", "kind": "property"},
        {"keywords": ["polygon"], "concept": "polygon_properties", "kind": "property"},
        {"keywords": ["volume", "surface"], "concept": "solid_geometry_formulas", "kind": "formula"},
    ],
    "graph_theory": [
        {"keywords": ["chromatic", "coloring"], "concept": "graph_coloring", "kind": "theorem"},
        {"keywords": ["planar"], "concept": "planarity_criteria", "kind": "criterion"},
        {"keywords": ["hamiltonian"], "concept": "hamiltonian_path", "kind": "criterion"},
        {"keywords": ["eulerian"], "concept": "euler_circuit", "kind": "theorem"},
        {"keywords": ["spanning", "tree"], "concept": "spanning_tree_algorithms", "kind": "theorem"},
        {"keywords": ["bipartite"], "concept": "bipartite_graph", "kind": "criterion"},
    ],
    "probability": [
        {"keywords": ["bayes"], "concept": "bayes_theorem", "kind": "theorem"},
        {"keywords": ["binomial", "distribution"], "concept": "binomial_distribution", "kind": "formula"},
        {"keywords": ["poisson"], "concept": "poisson_distribution", "kind": "formula"},
        {"keywords": ["markov"], "concept": "markov_chain", "kind": "definition"},
        {"keywords": ["expected", "value"], "concept": "expected_value", "kind": "definition"},
    ],
    "modular_arithmetic": [
        {"keywords": ["chinese", "remainder"], "concept": "chinese_remainder_theorem", "kind": "theorem"},
        {"keywords": ["euler", "fermat"], "concept": "euler_fermat_theorem", "kind": "theorem"},
        {"keywords": ["primitive", "root"], "concept": "primitive_root", "kind": "definition"},
    ],

    # ── 科学系 ──
    "physics": [
        {"keywords": ["quantum", "wave_function"], "concept": "quantum_mechanics_postulates", "kind": "definition"},
        {"keywords": ["hamiltonian", "lagrangian"], "concept": "analytical_mechanics", "kind": "definition"},
        {"keywords": ["maxwell", "electric", "magnetic"], "concept": "electromagnetism", "kind": "theorem"},
        {"keywords": ["thermodynamic", "entropy"], "concept": "thermodynamics_laws", "kind": "theorem"},
        {"keywords": ["relativity", "lorentz"], "concept": "special_relativity", "kind": "theorem"},
        {"keywords": ["schrodinger"], "concept": "schrodinger_equation", "kind": "formula"},
        {"keywords": ["nuclear", "decay", "half-life"], "concept": "nuclear_physics", "kind": "formula"},
        {"keywords": ["optics", "refraction", "diffraction"], "concept": "wave_optics", "kind": "property"},
    ],
    "chemistry": [
        {"keywords": ["electron", "configuration"], "concept": "electron_configuration", "kind": "definition"},
        {"keywords": ["periodic", "table", "element"], "concept": "periodic_table_properties", "kind": "fact"},
        {"keywords": ["bond", "ionic", "covalent"], "concept": "chemical_bonding", "kind": "definition"},
        {"keywords": ["reaction", "equation"], "concept": "chemical_reaction_types", "kind": "definition"},
        {"keywords": ["acid", "base", "ph"], "concept": "acid_base_chemistry", "kind": "definition"},
        {"keywords": ["oxidation", "reduction", "redox"], "concept": "redox_reactions", "kind": "definition"},
        {"keywords": ["gibbs", "enthalpy", "entropy"], "concept": "thermochemistry", "kind": "formula"},
        {"keywords": ["organic", "functional_group"], "concept": "organic_chemistry_groups", "kind": "definition"},
        {"keywords": ["stoichiometry", "molar"], "concept": "stoichiometry", "kind": "formula"},
        {"keywords": ["equilibrium", "constant"], "concept": "chemical_equilibrium", "kind": "formula"},
        {"keywords": ["catalyst"], "concept": "catalysis", "kind": "definition"},
        {"keywords": ["nmr", "spectroscopy", "spectrum"], "concept": "spectroscopy_interpretation", "kind": "criterion"},
        {"keywords": ["synthesis", "retrosynthesis"], "concept": "organic_synthesis", "kind": "property"},
    ],
    "computer_science": [
        {"keywords": ["complexity", "big_o"], "concept": "computational_complexity", "kind": "definition"},
        {"keywords": ["turing", "automaton"], "concept": "automata_theory", "kind": "definition"},
        {"keywords": ["np", "reduction"], "concept": "complexity_classes", "kind": "criterion"},
        {"keywords": ["dynamic_programming"], "concept": "dynamic_programming", "kind": "definition"},
        {"keywords": ["sorting", "algorithm"], "concept": "sorting_algorithms", "kind": "property"},
        {"keywords": ["binary_search"], "concept": "binary_search", "kind": "property"},
        {"keywords": ["hash", "data_structure"], "concept": "data_structures", "kind": "property"},
    ],

    # ── 人文系 ──
    "philosophy": [
        {"keywords": ["syllogism", "modus"], "concept": "logical_argument_forms", "kind": "definition"},
        {"keywords": ["epistemology"], "concept": "epistemology_concepts", "kind": "definition"},
        {"keywords": ["ethics", "moral"], "concept": "ethical_theories", "kind": "definition"},
        {"keywords": ["metaphysics", "ontology"], "concept": "metaphysics_concepts", "kind": "definition"},
    ],

    # ── 特殊ドメイン ──
    "chess": [
        {"keywords": ["mate", "checkmate"], "concept": "checkmate_patterns", "kind": "criterion"},
        {"keywords": ["opening"], "concept": "chess_openings", "kind": "fact"},
        {"keywords": ["endgame"], "concept": "chess_endgame_theory", "kind": "property"},
    ],
}


# ──────────────────────────────────────────────────────────────────
# テキストから高次概念を抽出するパターン
# ──────────────────────────────────────────────────────────────────

# 問題文中の専門用語を concept として拾うパターン
# (regex, domain_hint, kind)
CONCEPT_EXTRACTION_PATTERNS: list[tuple[str, str, str]] = [
    # 数学概念
    (r'\b(moduli\s+space)\b', 'algebraic_geometry', 'definition'),
    (r'\b(fundamental\s+group)\b', 'algebraic_topology', 'definition'),
    (r'\b(homology|cohomology)\b', 'algebraic_topology', 'definition'),
    (r'\b(homotopy)\b', 'algebraic_topology', 'definition'),
    (r'\b(Wasserstein\s+(?:space|distance|metric))\b', 'measure_theory', 'definition'),
    (r'\b(Barcan\s+formula)\b', 'modal_logic', 'definition'),
    (r'\b(f-vector)\b', 'combinatorial_geometry', 'definition'),
    (r'\b(polytope)\b', 'combinatorial_geometry', 'definition'),
    (r'\b(Lie\s+(?:group|algebra))\b', 'lie_theory', 'definition'),
    (r'\b(Hilbert\s+space)\b', 'functional_analysis', 'definition'),
    (r'\b(Banach\s+space)\b', 'functional_analysis', 'definition'),
    (r'\b(Riemann\s+(?:surface|hypothesis|zeta))\b', 'complex_analysis', 'definition'),
    (r'\b(Galois\s+(?:group|extension|theory))\b', 'galois_theory', 'definition'),
    (r'\b(Sylow\s+theorem)\b', 'group_theory', 'theorem'),
    (r'\b(Jordan\s+(?:normal\s+)?form)\b', 'linear_algebra', 'definition'),
    (r'\b(characteristic\s+polynomial)\b', 'linear_algebra', 'definition'),
    (r'\b(Cayley-Hamilton)\b', 'linear_algebra', 'theorem'),
    (r'\b(Laplacian)\b', 'graph_theory', 'definition'),
    (r'\b(chromatic\s+(?:number|polynomial))\b', 'graph_theory', 'property'),

    # 物理概念
    (r'\b(Feynman\s+diagram)\b', 'quantum_field_theory', 'definition'),
    (r'\b(Lagrangian)\b', 'classical_mechanics', 'definition'),
    (r'\b(Hamiltonian)\b', 'quantum_mechanics', 'definition'),
    (r'\b(Noether.s\s+theorem)\b', 'theoretical_physics', 'theorem'),
    (r'\b(gauge\s+(?:theory|invariance|symmetry))\b', 'quantum_field_theory', 'definition'),
    (r'\b(renormalization)\b', 'quantum_field_theory', 'definition'),
    (r'\b(Born-Oppenheimer)\b', 'quantum_chemistry', 'definition'),

    # 化学概念
    (r'\b(IUPAC\s+name)\b', 'organic_chemistry', 'definition'),
    (r'\b(stereochemistry|chirality|enantiomer)\b', 'organic_chemistry', 'definition'),
    (r'\b(NMR\s+(?:spectrum|spectroscopy|shift))\b', 'analytical_chemistry', 'criterion'),
    (r'\b(mass\s+spectrometry)\b', 'analytical_chemistry', 'definition'),

    # 生物概念
    (r'\b(CRISPR)\b', 'molecular_biology', 'definition'),
    (r'\b(PCR)\b', 'molecular_biology', 'definition'),
    (r'\b(transcription|translation)\b', 'molecular_biology', 'definition'),
    (r'\b(phylogeny|phylogenetic)\b', 'evolutionary_biology', 'definition'),
    (r'\b(enzyme\s+kinetics|Michaelis-Menten)\b', 'biochemistry', 'formula'),

    # CS概念
    (r'\b(NP-(?:hard|complete))\b', 'computational_complexity', 'definition'),
    (r'\b(Turing\s+machine)\b', 'computability_theory', 'definition'),
    (r'\b(lambda\s+calculus)\b', 'theory_of_computation', 'definition'),
    (r'\b(Byzantine\s+(?:fault|generals))\b', 'distributed_systems', 'definition'),
    (r'\b(CAP\s+theorem)\b', 'distributed_systems', 'theorem'),

    # 法学・社会科学
    (r'\b(statute|legislation)\b', 'law', 'fact'),
    (r'\b(jurisprudence)\b', 'law', 'definition'),
    (r'\b(Nash\s+equilibrium)\b', 'game_theory', 'definition'),
    (r'\b(Pareto\s+(?:optimal|efficiency))\b', 'economics', 'definition'),
]


def extract_knowledge_needs(
    ir_dict: dict,
    problem_text: str,  # 概念抽出用（LLMには渡さない — ここで概念名だけ抜く）
) -> list[KnowledgeNeed]:
    """
    IR + 問題文のキーワードから、LLM に聞くべき知識ニーズを抽出する。

    重要: problem_text はこの関数内で概念名の抽出にだけ使う。
          抽出した概念名(concept)だけが KnowledgeNeed に入り、
          問題文自体は KnowledgeNeed に含まれない。
    """
    needs: list[KnowledgeNeed] = []
    seen_concepts: set[str] = set()

    domain = ir_dict.get("domain", ir_dict.get("domain_hint", ["unknown"]))
    if isinstance(domain, list):
        domain = domain[0] if domain else "unknown"
    domain_str = str(domain).lower()

    keywords = ir_dict.get("metadata", {}).get("keywords", [])
    text_lower = problem_text.lower()

    # ────────────────────────────────────────────────────
    # Phase 1: ドメインルールベースの知識ニーズ
    # ────────────────────────────────────────────────────
    # ドメインの正規化（advanced_* → base domain も試す）
    domains_to_check = [domain_str]
    if domain_str.startswith("advanced_"):
        domains_to_check.append(domain_str.replace("advanced_", ""))
    if domain_str == "multiple_choice":
        # MCQ はフォーマット。実ドメインをキーワードから推定
        domains_to_check = _infer_real_domains_from_keywords(keywords, text_lower)

    for dom in domains_to_check:
        rules = DOMAIN_NEED_RULES.get(dom, [])
        for rule in rules:
            rule_kws = rule["keywords"]
            # キーワードマッチ（IR keywords or テキスト中）
            matched = any(
                kw in keywords or kw.replace("_", " ") in text_lower or kw in text_lower
                for kw in rule_kws
            )
            if matched and rule["concept"] not in seen_concepts:
                needs.append(KnowledgeNeed(
                    concept=rule["concept"],
                    kind=rule["kind"],
                    domain=dom,
                    relation=rule.get("relation", ""),
                ))
                seen_concepts.add(rule["concept"])

    # ────────────────────────────────────────────────────
    # Phase 2: テキストから高次概念を直接抽出
    # ────────────────────────────────────────────────────
    for pattern, concept_domain, kind in CONCEPT_EXTRACTION_PATTERNS:
        m = re.search(pattern, problem_text, re.IGNORECASE)
        if m:
            concept_name = m.group(1).strip().replace(" ", "_").lower()
            if concept_name not in seen_concepts:
                needs.append(KnowledgeNeed(
                    concept=concept_name,
                    kind=kind,
                    domain=concept_domain,
                ))
                seen_concepts.add(concept_name)

    # ────────────────────────────────────────────────────
    # Phase 3: エンティティからの知識ニーズ
    # ────────────────────────────────────────────────────
    entities = ir_dict.get("entities", [])
    for ent in entities:
        ent_type = ent.get("type", "")
        ent_name = ent.get("name", "")
        if ent_type == "formula" and ent_name:
            concept_name = ent_name.replace(" ", "_").lower()
            if concept_name not in seen_concepts:
                needs.append(KnowledgeNeed(
                    concept=concept_name,
                    kind="definition",
                    domain=domain_str,
                ))
                seen_concepts.add(concept_name)

    # ────────────────────────────────────────────────────
    # Phase 4: fallback — ドメインが分かるが具体的な知識ニーズが
    #          見つからない場合、汎用クエリを生成
    # ────────────────────────────────────────────────────
    if not needs and domain_str not in ("unknown", "multiple_choice", "arithmetic"):
        task = ir_dict.get("task", "compute")
        needs.append(KnowledgeNeed(
            concept=f"{domain_str}_general",
            kind="definition",
            domain=domain_str,
            relation=f"{domain_str} -> {task}",
            scope="concise",
            context_hint=f"key concepts for {task} tasks in {domain_str}",
        ))

    return needs


def _infer_real_domains_from_keywords(keywords: list[str], text_lower: str) -> list[str]:
    """MCQ問題のキーワード/テキストから実ドメインを推定"""
    domain_scores: dict[str, int] = {}
    for dom, rules in DOMAIN_NEED_RULES.items():
        score = 0
        for rule in rules:
            for kw in rule["keywords"]:
                if kw in keywords or kw.replace("_", " ") in text_lower or kw in text_lower:
                    score += 1
        if score > 0:
            domain_scores[dom] = score

    if not domain_scores:
        # テキストから主要ドメインキーワードで推定
        domain_keyword_map = {
            "physics": ["energy", "force", "velocity", "quantum", "wave", "electron", "photon", "field"],
            "chemistry": ["molecule", "atom", "reaction", "bond", "acid", "base", "compound", "element"],
            "computer_science": ["algorithm", "program", "code", "complexity", "data", "compute"],
            "philosophy": ["argument", "premise", "logic", "ethics", "moral"],
            "number_theory": ["prime", "divisor", "modulo", "integer"],
            "algebra": ["equation", "polynomial", "solve", "variable"],
            "geometry": ["triangle", "circle", "angle", "area", "polygon"],
            "probability": ["probability", "random", "expected", "distribution"],
        }
        for dom, kws in domain_keyword_map.items():
            for kw in kws:
                if kw in text_lower:
                    domain_scores[dom] = domain_scores.get(dom, 0) + 1

    if domain_scores:
        sorted_doms = sorted(domain_scores, key=domain_scores.get, reverse=True)
        return sorted_doms[:2]  # 上位2ドメイン
    return ["unknown"]
