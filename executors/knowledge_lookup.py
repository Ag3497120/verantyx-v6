"""
Knowledge Lookup Executor

ルールベースの知識検索エグゼキュータ。
DeepSeek V3 671B の Cross座標から抽出されたピースに対応。
LLM不使用、完全にルールベース。
"""
import re
from typing import Any, Dict, Optional, List, Tuple


# =============================================================================
# 物理定数
# =============================================================================
PHYSICS_CONSTANTS = {
    "speed_of_light": {"value": 2.998e8, "unit": "m/s", "symbol": "c",
                       "keywords": ["speed of light", "c = 3", "velocity of light"]},
    "planck_constant": {"value": 6.626e-34, "unit": "J·s", "symbol": "h",
                        "keywords": ["planck constant", "planck's constant", "h = 6.626"]},
    "boltzmann": {"value": 1.381e-23, "unit": "J/K", "symbol": "k_B",
                  "keywords": ["boltzmann constant", "k_b", "boltzmann's constant"]},
    "avogadro": {"value": 6.022e23, "unit": "mol^-1", "symbol": "N_A",
                 "keywords": ["avogadro", "avogadro's number", "avogadro constant"]},
    "electron_charge": {"value": 1.602e-19, "unit": "C", "symbol": "e",
                        "keywords": ["elementary charge", "charge of electron", "electron charge"]},
    "electron_mass": {"value": 9.109e-31, "unit": "kg", "symbol": "m_e",
                      "keywords": ["mass of electron", "electron mass"]},
    "proton_mass": {"value": 1.673e-27, "unit": "kg", "symbol": "m_p",
                    "keywords": ["mass of proton", "proton mass"]},
    "gravitational": {"value": 6.674e-11, "unit": "m^3/(kg·s^2)", "symbol": "G",
                      "keywords": ["gravitational constant", "newton's gravitational"]},
    "gas_constant": {"value": 8.314, "unit": "J/(mol·K)", "symbol": "R",
                     "keywords": ["gas constant", "ideal gas constant", "universal gas constant"]},
    "stefan_boltzmann": {"value": 5.671e-8, "unit": "W/(m^2·K^4)", "symbol": "σ",
                         "keywords": ["stefan-boltzmann", "stefan boltzmann constant"]},
}

# =============================================================================
# 化学知識
# =============================================================================
CHEMISTRY_KNOWLEDGE = {
    "water_molecular_weight": {"value": 18.015, "unit": "g/mol",
                               "keywords": ["molecular weight of water", "molar mass of water", "H2O mass"]},
    "carbon_atomic_weight": {"value": 12.011, "unit": "g/mol",
                             "keywords": ["atomic weight of carbon", "molar mass of carbon"]},
    "oxygen_atomic_weight": {"value": 15.999, "unit": "g/mol",
                             "keywords": ["atomic weight of oxygen", "molar mass of oxygen"]},
    "nitrogen_atomic_weight": {"value": 14.007, "unit": "g/mol",
                               "keywords": ["atomic weight of nitrogen", "molar mass of nitrogen"]},
    "hydrogen_atomic_weight": {"value": 1.008, "unit": "g/mol",
                               "keywords": ["atomic weight of hydrogen", "molar mass of hydrogen"]},
    "ph_neutral": {"value": 7.0, "unit": "pH",
                   "keywords": ["neutral pH", "pH of pure water", "neutral solution pH"]},
    "avogadro_number": {"value": 6.022e23, "unit": "mol^-1",
                        "keywords": ["avogadro's number"]},
    "ideal_gas_molar_volume": {"value": 22.414, "unit": "L/mol",
                               "keywords": ["molar volume at STP", "22.4 L", "22.4 liters per mole"]},
}

# =============================================================================
# 数学定数
# =============================================================================
MATH_CONSTANTS = {
    "pi": {"value": 3.14159265358979, "symbol": "π",
           "keywords": ["value of pi", "π ≈", "pi ="]},
    "euler_e": {"value": 2.71828182845905, "symbol": "e",
                "keywords": ["euler's number", "base of natural log", "value of e"]},
    "golden_ratio": {"value": 1.61803398874989, "symbol": "φ",
                     "keywords": ["golden ratio", "phi", "1.618"]},
    "sqrt2": {"value": 1.41421356237310, "symbol": "√2",
              "keywords": ["square root of 2", "√2"]},
    "ln2": {"value": 0.693147180559945, "symbol": "ln 2",
            "keywords": ["natural log of 2", "ln 2", "ln(2)"]},
}

# =============================================================================
# 生物学知識
# =============================================================================
BIOLOGY_KNOWLEDGE = {
    "dna_bases": {"value": "ATCG", "keywords": ["dna bases", "nucleotides in DNA", "four bases"]},
    "rna_bases": {"value": "AUCG", "keywords": ["rna bases", "nucleotides in RNA"]},
    "codon_length": {"value": 3, "keywords": ["codon length", "triplet code", "three nucleotides per codon"]},
    "human_chromosomes": {"value": 46, "keywords": ["human chromosomes", "human chromosome number", "46 chromosomes"]},
    "human_genome_size": {"value": "3 billion base pairs", "keywords": ["human genome size", "human genome length"]},
    "atp_energy": {"value": "~30.5 kJ/mol", "keywords": ["ATP hydrolysis energy", "energy from ATP"]},
    "cell_membrane_components": {"value": "phospholipid bilayer", "keywords": ["cell membrane structure", "plasma membrane"]},
}

# =============================================================================
# CS知識
# =============================================================================
CS_KNOWLEDGE = {
    # Algorithm complexity
    "big_o_linear": {"value": "O(n)", "keywords": ["linear time", "linear search worst", "traversal complexity"]},
    "big_o_quadratic": {"value": "O(n^2)", "keywords": ["quadratic time", "bubble sort worst", "selection sort worst", "insertion sort worst", "naive string matching"]},
    "big_o_log": {"value": "O(log n)", "keywords": ["logarithmic time", "binary search", "sorted array search", "balanced BST search"]},
    "big_o_nlogn": {"value": "O(n log n)", "keywords": ["merge sort", "heap sort", "comparison sort lower bound", "sorting lower bound", "efficient sort"]},
    "big_o_constant": {"value": "O(1)", "keywords": ["constant time", "hash table lookup", "array access", "stack push pop"]},
    "big_o_exponential": {"value": "O(2^n)", "keywords": ["exponential time", "brute force subset", "naive fibonacci", "traveling salesman brute"]},
    "big_o_factorial": {"value": "O(n!)", "keywords": ["factorial time", "brute force permutation", "naive TSP"]},
    "quicksort_average": {"value": "O(n log n)", "keywords": ["quicksort average", "quick sort average case", "expected quicksort"]},
    "quicksort_worst": {"value": "O(n^2)", "keywords": ["quicksort worst case", "quick sort worst", "pivot worst"]},
    "mergesort_complexity": {"value": "O(n log n)", "keywords": ["merge sort complexity", "merge sort time", "mergesort worst"]},
    "heapsort_complexity": {"value": "O(n log n)", "keywords": ["heap sort complexity", "heapsort time"]},
    "insertion_sort_best": {"value": "O(n)", "keywords": ["insertion sort best case", "insertion sort nearly sorted"]},
    "binary_search_complexity": {"value": "O(log n)", "keywords": ["binary search complexity", "binary search time"]},
    "dfs_bfs_complexity": {"value": "O(V+E)", "keywords": ["DFS complexity", "BFS complexity", "depth first search time", "breadth first search time"]},
    "dijkstra_complexity": {"value": "O((V+E)log V)", "keywords": ["dijkstra complexity", "dijkstra algorithm time", "shortest path time"]},
    "bellman_ford": {"value": "O(VE)", "keywords": ["bellman ford complexity", "bellman-ford time"]},
    "floyd_warshall": {"value": "O(V^3)", "keywords": ["floyd warshall complexity", "all pairs shortest path time"]},
    "p_vs_np": {"value": "open problem", "keywords": ["P vs NP", "P = NP", "P equals NP", "millennium prize"]},
    # Data structures
    "stack_lifo": {"value": "LIFO", "keywords": ["stack order", "stack discipline", "last in first out"]},
    "queue_fifo": {"value": "FIFO", "keywords": ["queue order", "queue discipline", "first in first out"]},
    "hash_table_avg": {"value": "O(1)", "keywords": ["hash table average", "hash map lookup average", "dictionary lookup"]},
    "hash_table_worst": {"value": "O(n)", "keywords": ["hash table worst case", "hash collision worst"]},
    "bst_search_avg": {"value": "O(log n)", "keywords": ["BST search average", "binary search tree lookup", "balanced tree search"]},
    "heap_insert": {"value": "O(log n)", "keywords": ["heap insert", "priority queue insert", "heap push"]},
    "heap_extract": {"value": "O(log n)", "keywords": ["heap extract", "priority queue pop", "heapify", "heap delete max"]},
    # Graph theory
    "eulers_formula_graph": {"value": "V - E + F = 2", "keywords": ["euler formula planar", "planar graph euler", "vertices edges faces"]},
    "four_color_theorem": {"value": "4", "keywords": ["four color theorem", "planar graph coloring", "chromatic number planar", "minimum colors planar"]},
    "complete_graph_edges": {"value": "n(n-1)/2", "keywords": ["complete graph edges", "K_n edges", "fully connected graph edges"]},
    # Machine learning
    "gradient_descent": {"value": "gradient descent", "keywords": ["optimization algorithm neural", "weight update rule", "backpropagation update"]},
    "overfitting": {"value": "overfitting", "keywords": ["model too complex", "high variance low bias", "memorizes training"]},
    "underfitting": {"value": "underfitting", "keywords": ["model too simple", "high bias low variance", "poor training performance"]},
    # Boolean logic
    "de_morgan_and": {"value": "NOT(A AND B) = NOT A OR NOT B", "keywords": ["de morgan and", "nand equivalence"]},
    "de_morgan_or": {"value": "NOT(A OR B) = NOT A AND NOT B", "keywords": ["de morgan or", "nor equivalence"]},
    # Number representation
    "two_complement": {"value": "two's complement", "keywords": ["negative number representation", "two's complement", "signed integer representation"]},
    "ieee754": {"value": "IEEE 754", "keywords": ["floating point standard", "float representation", "double precision standard"]},
}

# =============================================================================
# 数学公式・定理
# =============================================================================
MATH_FORMULAS = {
    "pythagorean_theorem": {
        "formula": "a² + b² = c²",
        "keywords": ["pythagorean theorem", "right triangle", "hypotenuse"]
    },
    "quadratic_formula": {
        "formula": "x = (-b ± √(b²-4ac)) / (2a)",
        "keywords": ["quadratic formula", "quadratic equation solution"]
    },
    "euler_formula": {
        "formula": "e^(iπ) + 1 = 0",
        "keywords": ["euler's formula", "euler's identity", "e^(i*pi)"]
    },
    "sum_of_n_integers": {
        "formula": "n(n+1)/2",
        "keywords": ["sum of first n", "1+2+3+...+n", "sum of natural numbers"]
    },
    "geometric_series": {
        "formula": "a(1-r^n)/(1-r)",
        "keywords": ["geometric series", "geometric sum"]
    },
    "taylor_series_exp": {
        "formula": "e^x = Σ x^n/n!",
        "keywords": ["taylor series of e^x", "exponential taylor"]
    },
    "binomial_theorem": {
        "formula": "(a+b)^n = Σ C(n,k) a^(n-k) b^k",
        "keywords": ["binomial theorem", "binomial expansion"]
    },
    "fundamental_theorem_calculus": {
        "formula": "∫_a^b f'(x)dx = f(b) - f(a)",
        "keywords": ["fundamental theorem of calculus", "FTC"]
    },
}

# =============================================================================
# MCQ domain-specific heuristics
# =============================================================================
# キーワードベースでMCQの選択肢をブーストするための知識
DOMAIN_MCQ_HINTS = {
    "physics": {
        "positive": ["conservation", "energy", "momentum", "wave", "quantum", "field"],
        "negative": ["never", "impossible", "always"],
    },
    "chemistry": {
        "positive": ["bond", "reaction", "equilibrium", "oxidation", "reduction"],
        "negative": ["never"],
    },
    "biology": {
        "positive": ["protein", "DNA", "cell", "membrane", "enzyme", "receptor"],
        "negative": ["never"],
    },
    "math": {
        "positive": ["theorem", "proof", "converge", "bounded", "continuous"],
        "negative": [],
    },
    "cs": {
        "positive": ["algorithm", "complexity", "recursive", "dynamic", "greedy"],
        "negative": [],
    },
}


def lookup(
    question: str = "",
    domain: str = None,
    layer: int = 0,
    cross_xyz: list = None,
    knowledge_type: str = None,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    知識ベース検索エグゼキュータ

    Args:
        question: 問題文
        domain: ドメイン (math, physics, chemistry, biology, cs)
        layer: レイヤー番号 (0-60, Cross座標から)
        cross_xyz: Cross座標 [x, y, z]
        knowledge_type: 知識タイプ (formula, constant, fact, mcq_hint)

    Returns:
        知識結果または None
    """
    if not question:
        return None

    q_lower = question.lower()

    # 1. 具体的な定数/事実の検索
    result = _lookup_constants(q_lower, domain)
    if result:
        return result

    # 2. MCQ質問に対するドメインヒント
    result = _get_mcq_hints(q_lower, domain, question)
    if result:
        return result

    # 3. 数学公式の検索
    result = _lookup_math_formula(q_lower)
    if result:
        return result

    # 4. foundation_law_kb.jsonl から検索 (Phase 2 KB expansion)
    try:
        from puzzle.kb_loader import search_kb
        kb_results = search_kb(question, domain=domain, top_k=3)
        if kb_results:
            top = kb_results[0]
            # Return the most relevant known_values or formula
            kv = top.get("known_values", {})
            formula = top.get("formula")
            stmt = top.get("statement", "")
            return {
                "type": "kb_theorem",
                "kb_id": top.get("id"),
                "domain": top.get("domain"),
                "statement": stmt,
                "formula": formula,
                "known_values": kv,
                "schema": "text",
                "confidence": 0.65,
            }
    except Exception:
        pass

    return None


def _lookup_constants(q_lower: str, domain: str) -> Optional[Dict[str, Any]]:
    """定数・事実を検索"""
    all_knowledge = {}

    if domain in (None, "physics", "chemistry"):
        all_knowledge.update(PHYSICS_CONSTANTS)
    if domain in (None, "chemistry"):
        all_knowledge.update(CHEMISTRY_KNOWLEDGE)
    if domain in (None, "math"):
        all_knowledge.update(MATH_CONSTANTS)
    if domain in (None, "biology"):
        all_knowledge.update(BIOLOGY_KNOWLEDGE)
    if domain in (None, "cs", "computer_science"):
        all_knowledge.update(CS_KNOWLEDGE)

    for key, info in all_knowledge.items():
        keywords = info.get("keywords", [])
        for kw in keywords:
            if kw.lower() in q_lower:
                val = info.get("value")
                if val is not None:
                    return {
                        "type": "constant",
                        "key": key,
                        "value": val,
                        "unit": info.get("unit", ""),
                        "schema": "decimal" if isinstance(val, float) else "text",
                        "confidence": 0.85
                    }
    return None


def _get_mcq_hints(q_lower: str, domain: str, question: str) -> Optional[Dict[str, Any]]:
    """MCQ問題に対するドメインヒントを返す"""
    # MCQ形式でなければスキップ
    if not re.search(r'\n[A-Z][.\)]\s+', question):
        return None

    # ドメイン検出
    detected_domain = domain
    if not detected_domain:
        if any(kw in q_lower for kw in ["velocity", "momentum", "force", "energy", "wave", "quantum", "electric", "magnetic"]):
            detected_domain = "physics"
        elif any(kw in q_lower for kw in ["molecule", "reaction", "bond", "acid", "base", "oxidation"]):
            detected_domain = "chemistry"
        elif any(kw in q_lower for kw in ["protein", "dna", "rna", "cell", "enzyme", "organism", "species"]):
            detected_domain = "biology"
        elif any(kw in q_lower for kw in ["algorithm", "complexity", "program", "computer", "network", "data structure"]):
            detected_domain = "cs"

    if detected_domain and detected_domain in DOMAIN_MCQ_HINTS:
        hints = DOMAIN_MCQ_HINTS[detected_domain]
        return {
            "type": "mcq_hint",
            "domain": detected_domain,
            "positive_keywords": hints.get("positive", []),
            "negative_keywords": hints.get("negative", []),
            "schema": "knowledge",
            "confidence": 0.3
        }
    return None


def _lookup_math_formula(q_lower: str) -> Optional[Dict[str, Any]]:
    """数学公式を検索"""
    for key, info in MATH_FORMULAS.items():
        for kw in info.get("keywords", []):
            if kw.lower() in q_lower:
                return {
                    "type": "formula",
                    "key": key,
                    "formula": info["formula"],
                    "schema": "text",
                    "confidence": 0.7
                }
    return None


def lookup_by_type(knowledge_type: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    タイプ別知識検索

    Args:
        knowledge_type: constant, formula, mcq_hint など

    Returns:
        知識結果
    """
    return lookup(knowledge_type=knowledge_type, **kwargs)


# テスト用
if __name__ == "__main__":
    test_cases = [
        "What is the speed of light?",
        "Calculate the molecular weight of water.",
        "What is the value of pi?",
        "Which of the following algorithms has O(n log n) complexity?\nA. Bubble sort\nB. Merge sort\nC. Insertion sort",
    ]

    for q in test_cases:
        result = lookup(question=q)
        print(f"Q: {q[:60]}")
        print(f"  Result: {result}")
        print()
