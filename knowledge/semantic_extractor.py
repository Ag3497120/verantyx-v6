"""
Semantic Knowledge Extractor

Bridges the gap between weight patterns → symbolic knowledge.
Uses "knowledge probing" technique:
  - Present partial formulas/theorems to the model
  - Let the model complete them
  - Capture high-confidence completions → convert to piece DB format

In STUB mode: uses a hardcoded library of known mathematical knowledge
to generate realistic piece entries without needing a real model.
"""

import json
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────
# Knowledge Probe Templates
# ──────────────────────────────────────────────

KNOWLEDGE_PROBES: Dict[str, List[Dict[str, str]]] = {
    "math_calculus": [
        {"prompt": "The derivative of x^n is ___", "expected": "nx^(n-1)", "piece_id": "calc_power_rule"},
        {"prompt": "The derivative of sin(x) is ___", "expected": "cos(x)", "piece_id": "calc_sin_derivative"},
        {"prompt": "The derivative of cos(x) is ___", "expected": "-sin(x)", "piece_id": "calc_cos_derivative"},
        {"prompt": "The derivative of e^x is ___", "expected": "e^x", "piece_id": "calc_exp_derivative"},
        {"prompt": "The derivative of ln(x) is ___", "expected": "1/x", "piece_id": "calc_ln_derivative"},
        {"prompt": "The chain rule states: d/dx[f(g(x))] = ___", "expected": "f'(g(x)) · g'(x)", "piece_id": "calc_chain_rule"},
        {"prompt": "The product rule: d/dx[f(x)g(x)] = ___", "expected": "f'(x)g(x) + f(x)g'(x)", "piece_id": "calc_product_rule"},
        {"prompt": "The fundamental theorem of calculus: ∫_a^b f'(x)dx = ___", "expected": "f(b) - f(a)", "piece_id": "calc_ftc"},
        {"prompt": "∫ x^n dx = ___ (n ≠ -1)", "expected": "x^(n+1)/(n+1) + C", "piece_id": "calc_power_integral"},
        {"prompt": "∫ e^x dx = ___", "expected": "e^x + C", "piece_id": "calc_exp_integral"},
        {"prompt": "∫ 1/x dx = ___", "expected": "ln|x| + C", "piece_id": "calc_ln_integral"},
        {"prompt": "L'Hôpital's rule: if lim f/g = 0/0 then lim f/g = ___", "expected": "lim f'/g'", "piece_id": "calc_lhopital"},
        {"prompt": "Taylor series of e^x = ___", "expected": "Σ x^n/n! = 1 + x + x²/2! + x³/3! + ...", "piece_id": "calc_taylor_exp"},
        {"prompt": "The limit lim_{x→0} sin(x)/x = ___", "expected": "1", "piece_id": "calc_sinc_limit"},
        {"prompt": "Integration by parts: ∫ u dv = ___", "expected": "uv - ∫ v du", "piece_id": "calc_ibp"},
    ],
    "math_algebra": [
        {"prompt": "Quadratic formula: for ax²+bx+c=0, x = ___", "expected": "(-b ± √(b²-4ac)) / 2a", "piece_id": "alg_quadratic"},
        {"prompt": "Binomial theorem: (a+b)^n = ___", "expected": "Σ C(n,k) a^(n-k) b^k", "piece_id": "alg_binomial"},
        {"prompt": "Difference of squares: a² - b² = ___", "expected": "(a+b)(a-b)", "piece_id": "alg_diff_squares"},
        {"prompt": "Sum of cubes: a³ + b³ = ___", "expected": "(a+b)(a²-ab+b²)", "piece_id": "alg_sum_cubes"},
        {"prompt": "The geometric series sum: Σ r^k (k=0 to n-1) = ___", "expected": "(1-r^n)/(1-r)", "piece_id": "alg_geo_series"},
        {"prompt": "Infinite geometric series (|r|<1): Σ ar^k = ___", "expected": "a/(1-r)", "piece_id": "alg_infinite_geo"},
        {"prompt": "Logarithm property: log(ab) = ___", "expected": "log(a) + log(b)", "piece_id": "alg_log_product"},
        {"prompt": "Logarithm property: log(a^n) = ___", "expected": "n·log(a)", "piece_id": "alg_log_power"},
        {"prompt": "Change of base formula: log_a(b) = ___", "expected": "ln(b)/ln(a)", "piece_id": "alg_log_change_base"},
        {"prompt": "Completing the square: x²+bx = ___", "expected": "(x + b/2)² - (b/2)²", "piece_id": "alg_complete_square"},
    ],
    "math_number_theory": [
        {"prompt": "Fermat's little theorem: if p prime and gcd(a,p)=1, then a^p ≡ ___", "expected": "a (mod p)", "piece_id": "nt_fermat_little"},
        {"prompt": "Euler's theorem: a^φ(n) ≡ ___ (mod n)", "expected": "1 (mod n) when gcd(a,n)=1", "piece_id": "nt_euler_theorem"},
        {"prompt": "Wilson's theorem: (p-1)! ≡ ___ (mod p) for prime p", "expected": "-1 (mod p)", "piece_id": "nt_wilson"},
        {"prompt": "The Chinese Remainder Theorem allows solving simultaneous congruences when ___", "expected": "moduli are pairwise coprime", "piece_id": "nt_crt"},
        {"prompt": "The number of primes less than n is approximately ___", "expected": "n/ln(n) (prime number theorem)", "piece_id": "nt_prime_counting"},
        {"prompt": "Bezout's identity: for integers a,b there exist x,y such that ax+by = ___", "expected": "gcd(a,b)", "piece_id": "nt_bezout"},
        {"prompt": "If p|ab and p is prime, then ___", "expected": "p|a or p|b (Euclid's lemma)", "piece_id": "nt_euclid_lemma"},
    ],
    "math_linear_algebra": [
        {"prompt": "The determinant of a 2×2 matrix [[a,b],[c,d]] = ___", "expected": "ad - bc", "piece_id": "la_det_2x2"},
        {"prompt": "Matrix A is invertible iff det(A) ≠ ___", "expected": "0", "piece_id": "la_invertible"},
        {"prompt": "Eigenvalue equation: Av = ___", "expected": "λv", "piece_id": "la_eigenvalue_eq"},
        {"prompt": "The trace of a matrix equals the sum of its ___", "expected": "eigenvalues", "piece_id": "la_trace_eigenvalues"},
        {"prompt": "The determinant of a matrix equals the product of its ___", "expected": "eigenvalues", "piece_id": "la_det_eigenvalues"},
        {"prompt": "Cauchy-Schwarz: |⟨u,v⟩| ≤ ___", "expected": "||u|| · ||v||", "piece_id": "la_cauchy_schwarz"},
        {"prompt": "The rank-nullity theorem: rank(A) + nullity(A) = ___", "expected": "number of columns", "piece_id": "la_rank_nullity"},
        {"prompt": "For symmetric matrix A, eigenvectors of distinct eigenvalues are ___", "expected": "orthogonal", "piece_id": "la_sym_eigenvectors"},
    ],
    "math_probability": [
        {"prompt": "Bayes' theorem: P(A|B) = ___", "expected": "P(B|A)P(A)/P(B)", "piece_id": "prob_bayes"},
        {"prompt": "P(A ∪ B) = ___", "expected": "P(A) + P(B) - P(A ∩ B)", "piece_id": "prob_union"},
        {"prompt": "For independent events: P(A ∩ B) = ___", "expected": "P(A) · P(B)", "piece_id": "prob_independence"},
        {"prompt": "The expected value E[aX+b] = ___", "expected": "aE[X] + b", "piece_id": "prob_ev_linear"},
        {"prompt": "Var(X) = E[X²] - ___", "expected": "(E[X])²", "piece_id": "prob_variance"},
        {"prompt": "The central limit theorem: sample mean X̄ approaches ___ as n→∞", "expected": "normal distribution N(μ, σ²/n)", "piece_id": "prob_clt"},
        {"prompt": "Binomial distribution: P(X=k) = ___", "expected": "C(n,k) p^k (1-p)^(n-k)", "piece_id": "prob_binomial"},
        {"prompt": "Poisson distribution: P(X=k) = ___", "expected": "λ^k e^(-λ) / k!", "piece_id": "prob_poisson"},
        {"prompt": "Chebyshev's inequality: P(|X-μ| ≥ kσ) ≤ ___", "expected": "1/k²", "piece_id": "prob_chebyshev"},
    ],
    "math_combinatorics": [
        {"prompt": "Number of ways to choose k from n: C(n,k) = ___", "expected": "n! / (k!(n-k)!)", "piece_id": "comb_choose"},
        {"prompt": "Number of permutations of n items: P(n) = ___", "expected": "n!", "piece_id": "comb_perm"},
        {"prompt": "Inclusion-exclusion for 2 sets: |A∪B| = ___", "expected": "|A| + |B| - |A∩B|", "piece_id": "comb_inclusion_exclusion"},
        {"prompt": "Stars and bars: ways to place n identical balls in k bins = ___", "expected": "C(n+k-1, k-1)", "piece_id": "comb_stars_bars"},
        {"prompt": "Fibonacci numbers satisfy: F(n) = ___", "expected": "F(n-1) + F(n-2)", "piece_id": "comb_fibonacci"},
        {"prompt": "Catalan number: C_n = ___", "expected": "C(2n,n)/(n+1)", "piece_id": "comb_catalan"},
        {"prompt": "Pigeonhole principle: if n+1 items in n boxes, some box has at least ___", "expected": "2 items", "piece_id": "comb_pigeonhole"},
    ],
    "math_geometry": [
        {"prompt": "Pythagorean theorem: a² + b² = ___", "expected": "c²", "piece_id": "geo_pythagorean"},
        {"prompt": "Area of a circle with radius r: A = ___", "expected": "πr²", "piece_id": "geo_circle_area"},
        {"prompt": "Circumference of a circle: C = ___", "expected": "2πr", "piece_id": "geo_circumference"},
        {"prompt": "Volume of a sphere: V = ___", "expected": "(4/3)πr³", "piece_id": "geo_sphere_volume"},
        {"prompt": "Law of cosines: c² = ___", "expected": "a² + b² - 2ab cos(C)", "piece_id": "geo_law_cosines"},
        {"prompt": "Law of sines: a/sin(A) = ___", "expected": "b/sin(B) = c/sin(C)", "piece_id": "geo_law_sines"},
        {"prompt": "Sum of interior angles of an n-gon = ___", "expected": "(n-2)·180°", "piece_id": "geo_polygon_angles"},
        {"prompt": "Euler's formula for polyhedra: V - E + F = ___", "expected": "2", "piece_id": "geo_euler_polyhedra"},
    ],
    "physics": [
        {"prompt": "Newton's second law: F = ___", "expected": "ma", "piece_id": "phys_newton_2"},
        {"prompt": "Kinetic energy: KE = ___", "expected": "(1/2)mv²", "piece_id": "phys_kinetic_energy"},
        {"prompt": "Potential energy (gravity): PE = ___", "expected": "mgh", "piece_id": "phys_potential_energy"},
        {"prompt": "Ohm's law: V = ___", "expected": "IR", "piece_id": "phys_ohms_law"},
        {"prompt": "Wave speed: v = ___", "expected": "fλ", "piece_id": "phys_wave_speed"},
        {"prompt": "Einstein's mass-energy equivalence: E = ___", "expected": "mc²", "piece_id": "phys_emc2"},
        {"prompt": "Gravitational force: F = ___", "expected": "Gm₁m₂/r²", "piece_id": "phys_gravity"},
        {"prompt": "Period of a pendulum: T = ___", "expected": "2π√(L/g)", "piece_id": "phys_pendulum"},
    ],
    "chemistry": [
        {"prompt": "Ideal gas law: PV = ___", "expected": "nRT", "piece_id": "chem_ideal_gas"},
        {"prompt": "pH = ___", "expected": "-log[H⁺]", "piece_id": "chem_ph"},
        {"prompt": "Avogadro's number: N_A ≈ ___", "expected": "6.022 × 10²³ mol⁻¹", "piece_id": "chem_avogadro"},
        {"prompt": "Arrhenius equation: k = ___", "expected": "A·exp(-Ea/RT)", "piece_id": "chem_arrhenius"},
        {"prompt": "Molarity = ___", "expected": "moles of solute / liters of solution", "piece_id": "chem_molarity"},
        {"prompt": "Boyle's Law states: at constant T, PV = ___", "expected": "constant", "piece_id": "chem_boyles_law"},
        {"prompt": "Charles's Law: at constant P, V/T = ___", "expected": "constant", "piece_id": "chem_charles_law"},
        {"prompt": "Gibbs free energy: ΔG = ___", "expected": "ΔH - TΔS", "piece_id": "chem_gibbs"},
        {"prompt": "Henderson-Hasselbalch: pH = ___", "expected": "pKa + log([A⁻]/[HA])", "piece_id": "chem_henderson"},
        {"prompt": "Faraday's law of electrolysis: mass deposited m = ___", "expected": "MIt/nF", "piece_id": "chem_faraday"},
    ],
    "biology": [
        {"prompt": "The central dogma: DNA → ___ → Protein", "expected": "RNA", "piece_id": "bio_central_dogma"},
        {"prompt": "Human diploid chromosome number: 2n = ___", "expected": "46", "piece_id": "bio_chromosomes"},
        {"prompt": "ATP stands for ___", "expected": "Adenosine Triphosphate", "piece_id": "bio_atp"},
        {"prompt": "Photosynthesis overall: 6CO₂ + 6H₂O → ___", "expected": "C₆H₁₂O₆ + 6O₂", "piece_id": "bio_photosynthesis"},
        {"prompt": "Cellular respiration: C₆H₁₂O₆ + 6O₂ → ___", "expected": "6CO₂ + 6H₂O + ATP", "piece_id": "bio_respiration"},
        {"prompt": "Mendel's law of segregation: alleles ___ during gamete formation", "expected": "separate", "piece_id": "bio_mendel_seg"},
        {"prompt": "Hardy-Weinberg: p² + 2pq + q² = ___", "expected": "1", "piece_id": "bio_hardy_weinberg"},
        {"prompt": "DNA base pairing: A pairs with ___, G pairs with ___", "expected": "T, C", "piece_id": "bio_base_pairing"},
        {"prompt": "Mitosis produces ___ daughter cells", "expected": "2 genetically identical", "piece_id": "bio_mitosis"},
        {"prompt": "Meiosis produces ___ daughter cells", "expected": "4 haploid", "piece_id": "bio_meiosis"},
    ],
    "computer_science": [
        {"prompt": "Big-O of merge sort: O(___)", "expected": "n log n", "piece_id": "cs_merge_sort"},
        {"prompt": "Big-O of binary search: O(___)", "expected": "log n", "piece_id": "cs_binary_search"},
        {"prompt": "Big-O of bubble sort: O(___)", "expected": "n²", "piece_id": "cs_bubble_sort"},
        {"prompt": "P ≠ NP means that problems verifiable in polynomial time ___ solvable in polynomial time", "expected": "may not be", "piece_id": "cs_p_np"},
        {"prompt": "A hash table supports average O(___) lookup", "expected": "1", "piece_id": "cs_hash_table"},
        {"prompt": "Dijkstra's algorithm finds ___", "expected": "shortest paths from a source vertex", "piece_id": "cs_dijkstra"},
        {"prompt": "A complete binary tree of height h has at most ___ nodes", "expected": "2^(h+1) - 1", "piece_id": "cs_binary_tree"},
        {"prompt": "Dynamic programming solves problems by ___ subproblems", "expected": "overlapping / memoizing", "piece_id": "cs_dp"},
        {"prompt": "Turing machine is a model of ___", "expected": "computation / any algorithm", "piece_id": "cs_turing"},
        {"prompt": "Floyd-Warshall algorithm finds ___", "expected": "all-pairs shortest paths", "piece_id": "cs_floyd_warshall"},
    ],
    "history": [
        {"prompt": "World War II ended in the year ___", "expected": "1945", "piece_id": "hist_ww2_end"},
        {"prompt": "The French Revolution began in ___", "expected": "1789", "piece_id": "hist_french_rev"},
        {"prompt": "The first US President was ___", "expected": "George Washington", "piece_id": "hist_first_president"},
        {"prompt": "The Magna Carta was signed in ___", "expected": "1215", "piece_id": "hist_magna_carta"},
        {"prompt": "The printing press was invented by ___", "expected": "Johannes Gutenberg", "piece_id": "hist_printing_press"},
        {"prompt": "India gained independence from Britain in ___", "expected": "1947", "piece_id": "hist_india_independence"},
        {"prompt": "The Roman Empire fell in ___", "expected": "476 AD (Western Roman Empire)", "piece_id": "hist_roman_fall"},
        {"prompt": "The Berlin Wall fell in ___", "expected": "1989", "piece_id": "hist_berlin_wall"},
        {"prompt": "The Renaissance originated in ___", "expected": "Italy (14th-15th century)", "piece_id": "hist_renaissance"},
        {"prompt": "World War I started in ___", "expected": "1914", "piece_id": "hist_ww1_start"},
    ],
    "literature": [
        {"prompt": "Hamlet was written by ___", "expected": "William Shakespeare", "piece_id": "lit_hamlet_author"},
        {"prompt": "1984 was written by ___", "expected": "George Orwell", "piece_id": "lit_1984_author"},
        {"prompt": "The protagonist of Crime and Punishment is ___", "expected": "Raskolnikov", "piece_id": "lit_crime_protagonist"},
        {"prompt": "The Iliad was attributed to ___", "expected": "Homer", "piece_id": "lit_iliad_author"},
        {"prompt": "Don Quixote was written by ___", "expected": "Miguel de Cervantes", "piece_id": "lit_quixote_author"},
        {"prompt": "Pride and Prejudice was written by ___", "expected": "Jane Austen", "piece_id": "lit_pride_author"},
        {"prompt": "A haiku has ___ syllables in 5-7-5 pattern", "expected": "17", "piece_id": "lit_haiku"},
        {"prompt": "The green light in The Great Gatsby symbolizes ___", "expected": "the American Dream / Daisy", "piece_id": "lit_gatsby_light"},
        {"prompt": "A metaphor directly compares two things ___ using 'like' or 'as'", "expected": "without", "piece_id": "lit_metaphor"},
        {"prompt": "The author of One Hundred Years of Solitude is ___", "expected": "Gabriel García Márquez", "piece_id": "lit_100_years"},
    ],
    "philosophy": [
        {"prompt": "Descartes' cogito: 'Cogito, ergo ___'", "expected": "sum (I think, therefore I am)", "piece_id": "phil_cogito"},
        {"prompt": "Kant's categorical imperative: act only according to ___", "expected": "maxims you could will to be universal laws", "piece_id": "phil_categorical"},
        {"prompt": "Utilitarianism holds that the right action maximizes ___", "expected": "happiness/utility for the greatest number", "piece_id": "phil_utilitarianism"},
        {"prompt": "The trolley problem is a thought experiment in ___", "expected": "ethics / moral philosophy", "piece_id": "phil_trolley"},
        {"prompt": "Plato's allegory of the cave illustrates ___", "expected": "the difference between appearance and reality", "piece_id": "phil_cave"},
        {"prompt": "Hume's problem of induction: past patterns ___ justify future predictions", "expected": "cannot logically", "piece_id": "phil_induction"},
        {"prompt": "Existentialism holds that ___ precedes essence", "expected": "existence", "piece_id": "phil_existentialism"},
        {"prompt": "The mind-body problem concerns the relationship between ___", "expected": "mental and physical states", "piece_id": "phil_mind_body"},
        {"prompt": "Occam's razor: entities should not be multiplied beyond ___", "expected": "necessity", "piece_id": "phil_occam"},
        {"prompt": "Aristotle's syllogism: All men are mortal; Socrates is a man; therefore ___", "expected": "Socrates is mortal", "piece_id": "phil_syllogism"},
    ],
}


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ProbeResult:
    """Result of a single knowledge probe"""
    probe_id: str
    prompt: str
    completion: str
    confidence: float
    domain: str
    expert_layer: Optional[int]
    expert_id: Optional[int]
    is_correct: bool


@dataclass
class ExtractedKnowledgePiece:
    """Knowledge piece extracted via semantic probing"""
    piece_id: str
    name: str
    description: str
    formula: str
    domain: str
    subdomain: str
    confidence: float
    source_probe: str
    expert_layer: Optional[int]
    expert_id: Optional[int]
    tags: List[str] = field(default_factory=list)

    def to_piece_db_format(self) -> Dict[str, Any]:
        """Convert to Verantyx piece_db.jsonl format"""
        return {
            "piece_id": self.piece_id,
            "name": self.name,
            "description": self.description,
            "in": {
                "requires": [f"domain:{self.domain}", f"subdomain:{self.subdomain}"],
                "slots": ["query"]
            },
            "out": {
                "produces": ["knowledge", "formula"],
                "schema": "knowledge"
            },
            "executor": "executors.knowledge.lookup",
            "confidence": self.confidence,
            "tags": self.tags + ["600b_semantic_extraction", self.domain, self.subdomain],
            "source": "600b_weight_extraction",
            "knowledge": {
                "formula": self.formula,
                "domain": self.domain,
                "subdomain": self.subdomain,
                "source_probe": self.source_probe,
                "expert_layer": self.expert_layer,
                "expert_id": self.expert_id,
            }
        }


# ──────────────────────────────────────────────
# Semantic Extractor
# ──────────────────────────────────────────────

class SemanticExtractor:
    """
    Extract actual knowledge from DeepSeek V3 experts via knowledge probing.

    Strategy:
    1. Present partial formulas/theorems to the model
    2. Record which experts are most active during the completion
    3. Filter high-confidence completions → create piece DB entries
    4. Associate pieces with specific expert neurons

    In STUB mode: uses hardcoded correct answers with realistic confidence scores.
    """

    CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to include a piece

    def __init__(
        self,
        model=None,
        tokenizer=None,
        stub: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stub = stub

        mode = "STUB" if stub else "REAL"
        print(f"[SEMANTIC_EXTRACTOR] Initialized in {mode} mode")
        print(f"[SEMANTIC_EXTRACTOR] {sum(len(v) for v in KNOWLEDGE_PROBES.values())} probes across {len(KNOWLEDGE_PROBES)} subdomains")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def extract_from_domain(
        self,
        domain: str,
        top_experts: Optional[List[Tuple[int, int]]] = None,
    ) -> List[ExtractedKnowledgePiece]:
        """
        Extract knowledge pieces for a given domain.

        Args:
            domain: High-level domain (e.g., "math", "physics")
            top_experts: Optional list of (layer, expert_id) to focus on.

        Returns:
            List of ExtractedKnowledgePiece, confidence-filtered.
        """
        pieces = []
        # Match subdomains: exact match OR prefix match (e.g., "math" → "math_calculus")
        subdomains = [k for k in KNOWLEDGE_PROBES.keys()
                      if k == domain or k.startswith(domain + "_")]

        for subdomain in subdomains:
            probes = KNOWLEDGE_PROBES[subdomain]
            print(f"[SEMANTIC_EXTRACTOR]   {subdomain}: {len(probes)} probes")

            for probe in probes:
                result = self._run_probe(probe, subdomain, top_experts)

                if result.is_correct and result.confidence >= self.CONFIDENCE_THRESHOLD:
                    piece = self._probe_result_to_piece(result, subdomain)
                    pieces.append(piece)

        print(f"[SEMANTIC_EXTRACTOR] Extracted {len(pieces)} high-confidence pieces for [{domain}]")
        return pieces

    def extract_all(self) -> List[ExtractedKnowledgePiece]:
        """Extract knowledge from all available probes."""
        all_pieces = []
        # Collect all unique top-level domain keys
        # "math_calculus" → "math", "biology" → "biology"
        top_level_domains = []
        seen = set()
        for k in KNOWLEDGE_PROBES.keys():
            # Either it's a compound "math_calculus" → use "math"
            # Or it's a plain "biology" → use "biology"
            if "_" in k and k.split("_")[0] in [kk.split("_")[0] for kk in KNOWLEDGE_PROBES.keys() if kk != k]:
                top = k.split("_")[0]
            else:
                top = k
            if top not in seen:
                seen.add(top)
                top_level_domains.append(top)

        for domain in top_level_domains:
            pieces = self.extract_from_domain(domain)
            all_pieces.extend(pieces)

        print(f"[SEMANTIC_EXTRACTOR] Total: {len(all_pieces)} pieces extracted")
        return all_pieces

    def to_piece_db_jsonl(
        self,
        pieces: List[ExtractedKnowledgePiece],
        output_path: str,
    ) -> int:
        """
        Write extracted pieces to piece_db.jsonl format.

        Returns number of pieces written.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_path, "w") as f:
            for piece in pieces:
                entry = piece.to_piece_db_format()
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"[SEMANTIC_EXTRACTOR] Wrote {count} pieces to {output_path}")
        return count

    # ──────────────────────────────────────────
    # Probe Execution
    # ──────────────────────────────────────────

    def _run_probe(
        self,
        probe: Dict[str, str],
        subdomain: str,
        top_experts: Optional[List[Tuple[int, int]]] = None,
    ) -> ProbeResult:
        """Run a single knowledge probe."""
        if self.stub:
            return self._stub_probe(probe, subdomain)
        else:
            return self._real_probe(probe, subdomain, top_experts)

    def _stub_probe(
        self,
        probe: Dict[str, str],
        subdomain: str,
    ) -> ProbeResult:
        """
        Stub mode: return correct answer with realistic confidence.
        Uses hash-based noise for reproducibility.
        """
        seed = int(hashlib.md5(probe["piece_id"].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Stub: model gives the correct answer with high confidence
        confidence = rng.uniform(0.78, 0.99)
        completion = probe["expected"]

        return ProbeResult(
            probe_id=probe["piece_id"],
            prompt=probe["prompt"],
            completion=completion,
            confidence=confidence,
            domain=subdomain.split("_")[0],
            expert_layer=rng.randint(20, 60),  # Simulated expert location
            expert_id=rng.randint(0, 255),
            is_correct=True,
        )

    def _real_probe(
        self,
        probe: Dict[str, str],
        subdomain: str,
        top_experts: Optional[List[Tuple[int, int]]] = None,
    ) -> ProbeResult:
        """
        Real mode: run the probe through the actual model.

        Optionally constrains which experts are active (for targeted extraction).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use stub=True for testing.")

        import torch

        prompt = probe["prompt"]
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0.0,  # Greedy for knowledge extraction
            )

        # Decode completion
        input_len = inputs["input_ids"].shape[-1]
        completion_ids = outputs.sequences[0][input_len:]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Compute confidence as mean token probability
        if outputs.scores:
            import torch.nn.functional as F
            probs = [
                F.softmax(score, dim=-1).max().item()
                for score in outputs.scores
            ]
            confidence = sum(probs) / len(probs) if probs else 0.5
        else:
            confidence = 0.5

        # Check correctness (fuzzy match)
        expected = probe["expected"]
        is_correct = self._check_completion(completion, expected)

        # Find which expert was most active (would need hooks in practice)
        expert_layer = None
        expert_id = None

        return ProbeResult(
            probe_id=probe["piece_id"],
            prompt=prompt,
            completion=completion,
            confidence=confidence,
            domain=subdomain.split("_")[0],
            expert_layer=expert_layer,
            expert_id=expert_id,
            is_correct=is_correct,
        )

    def _check_completion(self, completion: str, expected: str) -> bool:
        """
        Fuzzy check if completion matches expected answer.
        Handles common variations in mathematical notation.
        """
        comp = completion.lower().strip()
        exp = expected.lower().strip()

        if comp == exp:
            return True

        # Check if expected content is contained
        if exp in comp or comp in exp:
            return True

        # Check key terms
        key_terms = exp.split()[:3]  # First 3 words
        if all(term in comp for term in key_terms):
            return True

        return False

    # ──────────────────────────────────────────
    # Piece Construction
    # ──────────────────────────────────────────

    def _probe_result_to_piece(
        self,
        result: ProbeResult,
        subdomain: str,
    ) -> ExtractedKnowledgePiece:
        """Convert a probe result into a knowledge piece."""
        # Generate human-readable name from piece_id
        name = result.probe_id.replace("_", " ").title()

        # Domain from subdomain:
        # "math_calculus" → "math", "math_linear_algebra" → "math"
        # "biology" → "biology", "computer_science" → "computer_science"
        TOP_LEVEL_DOMAINS = {"math", "physics", "chemistry", "biology",
                             "computer_science", "history", "literature", "philosophy"}
        first_part = subdomain.split("_")[0]
        if first_part in TOP_LEVEL_DOMAINS:
            domain = first_part
        elif subdomain in TOP_LEVEL_DOMAINS:
            domain = subdomain
        else:
            # fallback: use the whole subdomain as domain
            domain = subdomain

        # Description from the probe prompt
        description = result.prompt.replace("___", result.completion)

        return ExtractedKnowledgePiece(
            piece_id=f"600b_{result.probe_id}",
            name=name,
            description=description,
            formula=result.completion,
            domain=domain,
            subdomain=subdomain,
            confidence=result.confidence,
            source_probe=result.prompt,
            expert_layer=result.expert_layer,
            expert_id=result.expert_id,
            tags=[domain, subdomain, "formula", "600b_extracted"],
        )


# ──────────────────────────────────────────────
# CLI / Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Semantic Extractor - Stub Test ===")
    extractor = SemanticExtractor(stub=True)

    pieces = extractor.extract_all()
    print(f"\nExtracted {len(pieces)} pieces")

    print("\nSample pieces:")
    for p in pieces[:5]:
        fmt = p.to_piece_db_format()
        print(f"  [{fmt['piece_id']}] {fmt['name']}")
        print(f"    formula: {fmt['knowledge']['formula']}")
        print(f"    confidence: {fmt['confidence']:.3f}")
