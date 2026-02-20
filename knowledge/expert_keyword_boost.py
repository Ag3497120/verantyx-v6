"""
Expert Keyword Boost — 高信頼度Expertトークンに基づくドメインキーワードマップ

DeepSeek V3の高信頼度Expert（conf > 0.15）から抽出した
ドメイン特化キーワードを使ってMCQのドメイン推定を強化する

設計：
- token-levelの明示的キーワードマッチング
- concept_boost（ベクトル類似度）への追加レイヤーとして機能
- 特に entity_score ≈ 0 の問題で有効
"""

import re
import json
import os
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────
# 高信頼度Expertから抽出したドメイン特化キーワード
# (expert_vocab_domains.jsonのconf>0.15エントリから手動整理)
# ──────────────────────────────────────────────────────────

EXPERT_DOMAIN_KEYWORDS: Dict[str, Dict[str, float]] = {

    "algebra": {
        # L11E199 conf=0.50: Hamiltonian, symmetry, symmetric, Lagrangian
        "hamiltonian": 2.0, "symmetry": 1.8, "symmetric": 1.5,
        "lagrangian": 1.8, "subgroup": 2.0, "polynomial": 1.5,
        # L7E153: algebra, algebras, otimes, subgroup
        "algebra": 1.5, "algebras": 1.5, "otimes": 2.0,
        "mathfrak": 1.2, "isomorphism": 2.0, "homomorphism": 2.0,
        # L9E200: polynomials, polynomial, density
        "polynomials": 1.5, "eigenvalue": 1.8, "eigenvalues": 1.8,
        # L5E28: equation, Eq, density
        "equation": 0.8, "equations": 0.8, "linear system": 1.5,
        # L16E15, L19E125, L21E237: polynomial, polynomials, algebra
        "ring": 1.5, "field": 1.0, "group theory": 2.0,
        "quotient": 1.5, "kernel": 1.2, "ideal": 1.5,
        "galois": 2.0, "abelian": 2.0, "permutation group": 2.0,
        "coset": 2.0, "normal subgroup": 2.0,
    },

    "calculus": {
        # L4E89 conf=0.30: probability, exponential, derivative
        "derivative": 2.0, "derivatives": 1.8, "integral": 1.8,
        "integrals": 1.8, "differentiate": 2.0, "integrate": 1.8,
        "partial": 1.5, "partial derivative": 2.0,
        # L6E142: partial, frac
        "gradient": 1.5, "divergence": 1.5, "curl": 1.5,
        # L10E79: flux
        "flux": 1.8, "surface integral": 2.0, "line integral": 2.0,
        "limit": 1.0, "continuity": 1.0, "differentiable": 1.5,
        "taylor series": 2.0, "maclaurin": 2.0, "fourier": 1.8,
        "laplace": 1.5, "differential equation": 2.0, "ode": 1.8, "pde": 1.8,
    },

    "physics": {
        # L9E226 conf=0.47: mass, Gaussian, Hamiltonian
        "mass": 1.0, "energy": 1.0, "momentum": 1.5, "velocity": 1.0,
        # L5E147: magnetic, electron, Theorem
        "magnetic": 1.5, "electron": 1.5, "electric": 1.0,
        # L8E241: mass, density, energy
        "density": 0.8, "pressure": 1.0, "temperature": 0.8,
        # L10E187: Hamiltonian, matrix, energy, spectrum, matrices
        "spectrum": 1.5, "wave": 1.0, "photon": 1.5,
        "force": 0.8, "acceleration": 1.0, "gravity": 1.0,
        "quantum": 2.0, "bohr": 2.0, "schrodinger": 2.0,
        "planck": 2.0, "heisenberg": 2.0, "fermion": 2.0, "boson": 2.0,
        "relativity": 2.0, "lorentz": 2.0, "maxwell": 2.0,
        "entropy": 1.5, "thermodynamic": 1.5, "boltzmann": 2.0,
        "orbit": 1.2, "radiation": 1.2, "wavelength": 1.5,
    },

    "geometry": {
        # L9E111: Draw, angles, Layout
        "angle": 1.5, "angles": 1.5, "triangle": 1.5,
        # L9E106: angle, Sect
        "circle": 1.2, "polygon": 1.5, "parallelogram": 1.8,
        "perpendicular": 1.5, "parallel": 1.0,
        "area": 0.8, "volume": 0.8, "perimeter": 1.2,
        "vertex": 1.2, "edge": 0.8, "face": 0.8,
        "euclidean": 1.5, "coordinate": 0.8, "distance": 0.8,
        "tangent": 1.5, "chord": 1.5, "arc": 1.2,
        "pythagorean": 2.0, "congruent": 1.8, "similar": 1.0,
        "vector": 0.8, "cross product": 1.5, "dot product": 1.5,
    },

    "linear_algebra": {
        # L13E19: eigenvalues, matrix, mathcal
        "matrix": 1.0, "matrices": 1.0, "vector space": 1.5,
        "eigenvalue": 1.8, "eigenvector": 1.8, "eigenvalues": 1.8,
        "determinant": 1.5, "trace": 1.2, "rank": 1.0,
        "basis": 1.2, "span": 1.0, "linear independence": 1.8,
        "orthogonal": 1.5, "orthonormal": 1.8, "projection": 1.2,
        "svd": 2.0, "singular value": 2.0, "null space": 1.8,
        "row space": 1.8, "column space": 1.8,
        "diagonalizable": 2.0, "transpose": 1.2,
    },

    "number_theory": {
        # L12E229: sum, ≤, ollary (corollary)
        "prime": 1.5, "prime number": 2.0, "primes": 1.5,
        # L13E74: sum, algebra, polynomial, integer, quadratic
        "integer": 1.2, "integers": 1.2, "modulo": 1.8,
        "divisible": 1.5, "divisor": 1.5, "gcd": 2.0, "lcm": 2.0,
        "congruence": 1.8, "residue": 1.5, "remainder": 1.2,
        "fermat": 2.0, "euler": 1.5, "diophantine": 2.0,
        "fibonacci": 1.8, "arithmetic sequence": 1.5,
        "quadratic residue": 2.0, "legendre symbol": 2.0,
        "totient": 2.0, "coprime": 1.8,
    },

    "combinatorics": {
        # L15E27, L18E149, L18E154
        "combination": 1.5, "permutation": 1.5, "factorial": 1.5,
        "binomial": 1.5, "coefficient": 1.0, "choose": 1.2,
        "counting": 1.2, "pigeonhole": 2.0, "inclusion exclusion": 2.0,
        "graph": 1.0, "tree": 0.8, "cycle": 1.0, "path": 0.8,
        "coloring": 1.5, "chromatic": 2.0, "hamiltonian path": 2.0,
        "eulerian": 2.0, "planar": 1.5, "clique": 1.5,
        "generating function": 2.0, "recurrence": 1.5, "stirling": 2.0,
        "catalan": 2.0, "bell number": 2.0, "derangement": 2.0,
    },

    "probability": {
        # L19E86: polynomial, integrals, sum, vectors, matrix
        # L15E144: numerically, Bayesian, topological
        "probability": 1.5, "expected value": 1.8, "expectation": 1.5,
        "variance": 1.5, "standard deviation": 1.5,
        "distribution": 1.2, "normal distribution": 1.8,
        "bayesian": 2.0, "bayes": 2.0, "conditional": 1.5,
        "random variable": 1.5, "sample space": 1.5,
        "markov": 2.0, "stochastic": 1.8, "poisson": 1.8,
        "bernoulli": 1.8, "binomial distribution": 2.0,
        "central limit theorem": 2.0, "law of large numbers": 2.0,
        "independence": 1.2, "bayes theorem": 2.0,
    },

    "logic": {
        # L4E15: begingroup, Lemma, Theorem
        # L4E45: Theorem, Proposition
        "lemma": 1.8, "theorem": 1.5, "proposition": 1.5,
        "proof": 1.2, "corollary": 1.8,
        "if and only if": 1.5, "iff": 1.5, "implies": 1.2,
        "contradiction": 1.5, "contrapositive": 1.8,
        "induction": 1.5, "mathematical induction": 2.0,
        "axiom": 1.5, "definition": 0.8, "hypothesis": 1.0,
        "conjunction": 1.5, "disjunction": 1.5, "negation": 1.5,
        "predicate": 1.5, "quantifier": 1.5, "satisfiable": 1.5,
        "tautology": 2.0, "modal": 1.8, "kripke": 2.0,
    },

}


# ──────────────────────────────────────────────────────────
# ドメイン推定関数
# ──────────────────────────────────────────────────────────

def score_domain_keywords(text: str) -> Dict[str, float]:
    """
    問題テキストに含まれるキーワードからドメインスコアを計算
    
    Returns: {domain: score} - 高いほどそのドメインに近い
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}

    for domain, kw_weights in EXPERT_DOMAIN_KEYWORDS.items():
        score = 0.0
        matched = []
        for kw, weight in kw_weights.items():
            # フレーズマッチ (word boundary考慮)
            pattern = r'\b' + re.escape(kw) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                score += weight * min(count, 3)  # cap at 3 occurrences
                matched.append(kw)
        if score > 0:
            scores[domain] = score

    return scores


def get_top_domain(text: str, threshold: float = 0.5) -> Optional[str]:
    """最上位ドメインを返す（threshold未満なら None）"""
    scores = score_domain_keywords(text)
    if not scores:
        return None
    top = max(scores, key=scores.get)
    return top if scores[top] >= threshold else None


def get_domain_boost_vector(text: str) -> Dict[str, float]:
    """
    MCQソルバー用: ドメイン→ブーストスコアの辞書
    normalize して 0-1 range に
    """
    scores = score_domain_keywords(text)
    if not scores:
        return {}
    max_score = max(scores.values())
    return {d: s / max_score for d, s in scores.items()}


# ──────────────────────────────────────────────────────────
# Verantyx domain → expert keyword boost マッピング
# ──────────────────────────────────────────────────────────

# Verantyxの内部ドメイン名との対応
VERANTYX_DOMAIN_MAP = {
    "algebra":       ["math", "algebra"],
    "calculus":      ["math", "calculus"],
    "geometry":      ["math", "geometry"],
    "linear_algebra":["math", "linear_algebra"],
    "number_theory": ["math", "number_theory"],
    "combinatorics": ["math", "combinatorics"],
    "probability":   ["math", "probability", "statistics"],
    "physics":       ["physics", "engineering"],
    "logic":         ["logic", "math"],
}


def get_piece_domain_boost(text: str, piece_domain: str) -> float:
    """
    特定のpiece domain に対するテキストのキーワードブースト値
    piece_db のスコアリングに追加する
    """
    scores = score_domain_keywords(text)
    total = 0.0
    for expert_domain, verantyx_domains in VERANTYX_DOMAIN_MAP.items():
        if piece_domain in verantyx_domains and expert_domain in scores:
            total += scores[expert_domain]
    return total


# ──────────────────────────────────────────────────────────
# テスト
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_cases = [
        "Find all eigenvalues of the matrix A = [[1,2],[3,4]] and determine if it is diagonalizable.",
        "Compute the derivative of sin(x)*e^x and find its critical points.",
        "What is the probability that a Poisson distribution with parameter λ=3 gives a value greater than 5?",
        "Prove that every subgroup of a cyclic group is cyclic using mathematical induction.",
        "Find the Hamiltonian of a system with kinetic energy T and potential V.",
        "If p is prime and a is not divisible by p, what does Fermat's little theorem say about a^(p-1)?",
        "How many ways can 5 people be arranged in a circle considering rotational symmetry?",
        "In triangle ABC, angle A = 60°, side a = 5, side b = 8. Find angle B.",
    ]

    print("=== Expert Keyword Domain Scorer ===\n")
    for text in test_cases:
        scores = score_domain_keywords(text)
        top = get_top_domain(text)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        print(f"Q: {text[:70]}...")
        print(f"   Top domain: {top}")
        print(f"   Scores: {sorted_scores[:3]}")
        print()
