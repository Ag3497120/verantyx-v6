"""
Knowledge Gap Analyzer for HLE-2500

Reads all 2500 questions (question text + category ONLY, no answers)
and identifies what types of knowledge/theorems/axioms are needed.

Output: knowledge_gaps.json — structured taxonomy of required knowledge
by domain, suitable for guiding KB expansion in Phase 2.

Usage:
    python3 tools/knowledge_gap_analyzer.py
"""

import json
import re
import collections
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Domain patterns: (pattern_name, regex)
# Only question text is analyzed — answers are NOT read.
# ─────────────────────────────────────────────────────────────────

MATH_PATTERNS = [
    # Algebra & Number Theory
    ("group_theory",         r"\b(group|subgroup|normal subgroup|quotient group|abelian|cyclic group|dihedral|symmetric group|homomorphism|isomorphism|coset|lagrange|sylow|galois|ring|field|ideal|module)\b"),
    ("number_theory",        r"\b(prime|modular|congruence|gcd|lcm|euler totient|fermat|wilson|chinese remainder|quadratic residue|legendre symbol|diophantine|pell equation|perfect number|mersenne)\b"),
    ("combinatorics",        r"\b(combinat|permut|combination|binomial|catalan|bell number|stirling|partition|generating function|inclusion.exclusion|pigeonhole|ramsey|graph coloring|chromatic|planar graph|tree|spanning tree|hamiltonian|eulerian)\b"),
    ("topology",             r"\b(topolog|homeomorphi|homotop|fundamental group|covering space|manifold|knot|braid|fiber bundle|cohomolog|homolog|de rham|simplicial|CW complex|euler characteristic|genus|orientable)\b"),
    ("analysis_calculus",    r"\b(limit|derivative|integral|series|convergence|taylor|fourier|laplace|differential equation|ODE|PDE|cauchy|riemann|lebesgue|measure theory|banach|hilbert space|functional analysis|operator|spectrum)\b"),
    ("linear_algebra",       r"\b(matrix|eigenvalue|eigenvector|determinant|trace|rank|null space|linear transformation|vector space|basis|orthogonal|unitary|hermitian|positive definite|SVD|jordan form|diagonaliz)\b"),
    ("probability_stats",    r"\b(probability|expectation|variance|distribution|markov|random variable|bayes|hypothesis test|confidence interval|regression|entropy|mutual information|central limit|law of large numbers)\b"),
    ("geometry",             r"\b(geometric|triangle|polygon|circle|sphere|angle|area|volume|euclidean|hyperbolic|projective|affine|coordinate|conic|ellipse|parabola|perpendicular|inscribed|circumscribed)\b"),
    ("graph_theory",         r"\b(graph|vertex|edge|path|cycle|connected|bipartite|matching|flow|cut|clique|independent set|chromatic|planarity|tree|forest|degree sequence|adjacency)\b"),
    ("logic_computation",    r"\b(turing|computab|decidab|complexity|NP|polynomial|reduction|halting|automaton|formal language|grammar|propositional|predicate|modal logic|type theory|lambda calculus|proof)\b"),
    ("algebraic_geometry",   r"\b(algebraic variety|scheme|sheaf|cohomolog|divisor|riemann.roch|elliptic curve|modular form|abelian variety|intersection theory|blow.up|resolution)\b"),
    ("differential_geometry",r"\b(riemannian|metric tensor|christoffel|geodesic|curvature|ricci|sectional curvature|lie group|lie algebra|differential form|exterior derivative|stokes theorem|connection|parallel transport)\b"),
    ("number_fields",        r"\b(algebraic number|number field|ring of integers|discriminant|class group|unit group|p-adic|adele|zeta function|l-function|galois representation|iwasawa)\b"),
]

PHYSICS_PATTERNS = [
    ("quantum_mechanics",    r"\b(quantum|wave function|schrödinger|heisenberg|eigenstate|hamiltonian|observable|hilbert space|commutator|uncertainty principle|spin|bra.ket|dirac|pauli|harmonic oscillator|hydrogen atom|angular momentum|quantum number)\b"),
    ("quantum_field_theory", r"\b(field theory|feynman|lagrangian density|path integral|renormalization|gauge theory|symmetry breaking|higgs|standard model|QED|QCD|gluon|quark|lepton|boson|fermion|propagator|vertex|loop diagram)\b"),
    ("statistical_mechanics",r"\b(partition function|boltzmann|entropy|canonical ensemble|grand canonical|phase transition|ising model|critical exponent|renormalization group|free energy|equation of state|maxwell.boltzmann|fermi.dirac|bose.einstein)\b"),
    ("general_relativity",   r"\b(general relativity|einstein|metric|schwarzschild|kerr|geodesic|stress.energy|einstein equation|gravitational wave|black hole|singularity|event horizon|cosmological constant|friedmann|FLRW)\b"),
    ("electromagnetism",     r"\b(maxwell|electric field|magnetic field|gauss law|faraday|ampere|lorentz force|potential|electromagnetic wave|polarization|dielectric|conductor|capacitor|inductor|impedance)\b"),
    ("condensed_matter",     r"\b(crystal|lattice|band structure|fermi level|superconductor|bcs|phonon|magnon|topological insulator|hall effect|semiconductor|resistivity|bloch theorem|brillouin zone)\b"),
    ("nuclear_particle",     r"\b(nucleus|proton|neutron|decay|half.life|binding energy|fission|fusion|cross section|scattering|collider|detector|quark model|hadron|pion|kaon)\b"),
    ("thermodynamics",       r"\b(thermodynami|temperature|pressure|heat|work|entropy|enthalpy|gibbs|helmholtz|carnot|cycle|adiabatic|isothermal|isobaric|isochoric|efficiency|equation of state)\b"),
    ("optics_waves",         r"\b(wave|interference|diffraction|refraction|reflection|polarization|coherence|laser|photon|snell|brewster|fresnel|fabry.perot|optical fiber|refractive index)\b"),
    ("mechanics",            r"\b(newton|force|momentum|energy|conservation|angular momentum|torque|lagrangian|hamiltonian mechanics|phase space|oscillation|pendulum|rigid body|moment of inertia|euler angle)\b"),
]

CHEMISTRY_PATTERNS = [
    ("organic_chemistry",    r"\b(organic|carbon|hydrogen bond|functional group|alkane|alkene|alkyne|aromatic|benzene|reaction mechanism|sn1|sn2|e1|e2|nucleophile|electrophile|carbonyl|ester|amide|chirality|stereochemistry|enantiomer|diastereomer|nmr|ir spectroscopy)\b"),
    ("inorganic_chemistry",  r"\b(inorganic|coordination complex|ligand|oxidation state|crystal field|d-orbital|transition metal|organometallic|acid.base|lewis acid|brønsted|redox|electrode potential|electrolysis|periodic table|group|period)\b"),
    ("physical_chemistry",   r"\b(physical chemistry|kinetics|rate constant|activation energy|arrhenius|reaction order|equilibrium|le chatelier|gibbs energy|enthalpy|entropy|statistical thermodynamics|spectroscopy|born.oppenheimer|molecular orbital)\b"),
    ("quantum_chemistry",    r"\b(quantum chemistry|basis set|hartree.fock|density functional|DFT|molecular orbital theory|HOMO|LUMO|orbital|electron configuration|wave function chemistry|coupled cluster|perturbation theory|slater determinant|scf)\b"),
    ("biochemistry",         r"\b(enzyme|substrate|protein|amino acid|nucleotide|DNA|RNA|ATP|metabolism|glycolysis|krebs cycle|oxidative phosphorylation|photosynthesis|gene|chromosome|codon|transcription|translation|ribosome)\b"),
    ("analytical_chemistry", r"\b(chromatography|spectroscopy|mass spectrometry|hplc|nmr|ir|uv.vis|titration|calibration|detection limit|sensitivity|selectivity|analytical method)\b"),
]

BIOLOGY_PATTERNS = [
    ("molecular_biology",    r"\b(DNA|RNA|protein|gene|genome|chromosome|plasmid|pcr|cloning|restriction enzyme|ligation|sequencing|crispr|mrna|transcription|translation|splicing|intron|exon|promoter|enhancer|operon)\b"),
    ("cell_biology",         r"\b(cell|mitochondria|nucleus|ribosome|golgi|endoplasmic reticulum|cytoskeleton|cell membrane|organelle|mitosis|meiosis|cell cycle|apoptosis|signal transduction|receptor|ligand|kinase|phosphorylation)\b"),
    ("genetics",             r"\b(genetic|allele|dominant|recessive|mendel|genotype|phenotype|mutation|crossing over|linkage|pedigree|hardy.weinberg|inheritance|sex.linked|chromosome|karyotype|snp|qtl|gwas)\b"),
    ("immunology",           r"\b(immune|antibody|antigen|t.cell|b.cell|innate|adaptive|mhc|complement|cytokine|inflammation|vaccine|autoimmune|tolerance|lymphocyte|macrophage|nk cell|interferon)\b"),
    ("neuroscience",         r"\b(neuron|synapse|neurotransmitter|action potential|membrane potential|ion channel|brain|cortex|hippocampus|receptor|dopamine|serotonin|gaba|glutamate|myelination|neuroplasticity)\b"),
    ("ecology_evolution",    r"\b(evolution|natural selection|mutation|genetic drift|speciation|phylogeny|ecology|ecosystem|population|community|food web|niche|competition|predation|symbiosis|adaptation)\b"),
    ("physiology",           r"\b(physiology|organ|system|cardiovascular|respiratory|digestive|renal|endocrine|hormone|homeostasis|feedback|reflex|muscular|skeletal|nervous system)\b"),
]

CS_PATTERNS = [
    ("algorithms",           r"\b(algorithm|complexity|time complexity|space complexity|big.O|sorting|search|dynamic programming|greedy|divide and conquer|backtracking|graph algorithm|shortest path|minimum spanning tree|network flow|NP.complete|approximation)\b"),
    ("data_structures",      r"\b(data structure|array|linked list|tree|heap|hash table|stack|queue|trie|segment tree|balanced tree|amortized|cache|memory)\b"),
    ("machine_learning",     r"\b(machine learning|neural network|gradient descent|backpropagation|loss function|regularization|overfitting|generalization|transformer|attention|cnn|rnn|lstm|gan|diffusion|reinforcement learning|policy|reward|q.learning)\b"),
    ("cryptography",         r"\b(cryptography|encryption|decryption|key|rsa|elliptic curve crypto|hash function|digital signature|diffie.hellman|aes|public.key|private.key|block cipher|stream cipher|authentication)\b"),
    ("programming_languages",r"\b(type system|type inference|lambda calculus|formal semantics|operational semantics|denotational|category theory for CS|monad|functor|continuations|compilers|parsing|automata)\b"),
    ("information_theory",   r"\b(entropy|channel capacity|shannon|mutual information|coding theorem|error correction|huffman|arithmetic coding|source coding|channel coding|kolmogorov complexity)\b"),
    ("computer_architecture",r"\b(processor|cache|memory hierarchy|pipeline|parallelism|gpu|fpga|instruction set|register|compiler|optimization|branch prediction|prefetch)\b"),
]

HUMANITIES_PATTERNS = [
    ("philosophy",           r"\b(epistemology|ontology|metaphysics|ethics|modal logic|possible world|knowledge|belief|truth|valid|sound|argument|logical form|philosophy of mind|consciousness|qualia|intentionality|phenomenology)\b"),
    ("economics",            r"\b(economics|utility|equilibrium|game theory|nash|mechanism design|auction|market|supply|demand|welfare|pareto|social choice|arrow impossibility|voting)\b"),
    ("linguistics",          r"\b(syntax|semantics|phonology|morphology|grammar|parse|language|universal grammar|minimalist|natural language|discourse|pragmatics|word|sentence|clause)\b"),
    ("history_science",      r"\b(history|discovered|invented|first|named after|origin|theorem is|lemma is|conjecture|proved by|century|year)\b"),
]

ALL_PATTERN_GROUPS = {
    "Math":                   MATH_PATTERNS,
    "Physics":                PHYSICS_PATTERNS,
    "Chemistry":              CHEMISTRY_PATTERNS,
    "Biology/Medicine":       BIOLOGY_PATTERNS,
    "Computer Science/AI":    CS_PATTERNS,
    "Humanities/Social Science": HUMANITIES_PATTERNS,
}

# ─────────────────────────────────────────────────────────────────
# Question complexity signals
# ─────────────────────────────────────────────────────────────────

COMPLEXITY_SIGNALS = {
    "multi_step":     r"\b(compute|calculate|find|determine|show|prove|derive|evaluate|solve)\b.{0,100}\b(and|then|given that|using|where|such that)\b",
    "exact_numeric":  r"\b(how many|what is the (value|number|count|sum|product|maximum|minimum)|compute|calculate|find the (largest|smallest|total|exact))\b",
    "proof_required": r"\b(prove|show that|verify|demonstrate|establish|derive)\b",
    "mcq":            r"Answer Choices:\s*\n\s*[A-E]\.",
    "visual":         r"\b(figure|diagram|image|graph|table|matrix|picture|shown|illustrated)\b",
}

# ─────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────

def analyze_question(question: str, category: str) -> dict:
    q_lower = question.lower()
    detected = {}

    # Check all domain patterns
    for group_name, patterns in ALL_PATTERN_GROUPS.items():
        group_hits = {}
        for pattern_name, regex in patterns:
            matches = re.findall(regex, q_lower, re.IGNORECASE)
            if matches:
                group_hits[pattern_name] = len(matches)
        if group_hits:
            detected[group_name] = group_hits

    # Complexity signals
    complexity = {}
    for sig_name, regex in COMPLEXITY_SIGNALS.items():
        if re.search(regex, question, re.IGNORECASE | re.DOTALL):
            complexity[sig_name] = True

    return {
        "detected_domains": detected,
        "complexity": complexity,
        "is_mcq": bool(re.search(r"Answer Choices:", question)),
        "has_latex": bool(re.search(r"\$[^$]+\$|\\[a-z]+\{", question)),
        "question_length": len(question),
        "category": category,
    }


def run_analysis(hle_path: str, output_path: str):
    questions = []
    with open(hle_path) as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append({
                "id": data.get("id", ""),
                "question": data["question"],
                "category": data.get("category", "Unknown"),
                # Note: 'answer' is deliberately NOT used here
            })

    print(f"Loaded {len(questions)} questions")

    # Per-question analysis
    results = []
    for q in questions:
        analysis = analyze_question(q["question"], q["category"])
        analysis["id"] = q["id"]
        results.append(analysis)

    # ── Aggregate statistics ──
    domain_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    subdomain_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    category_counts = collections.Counter(q["category"] for q in questions)
    mcq_count = sum(1 for r in results if r["is_mcq"])
    latex_count = sum(1 for r in results if r["has_latex"])
    complexity_counts = collections.Counter()
    for r in results:
        for c in r["complexity"]:
            complexity_counts[c] += 1

    for r in results:
        for group, subdomains in r["detected_domains"].items():
            domain_counts[r["category"]][group] += 1
            for sub in subdomains:
                subdomain_counts[group][sub] += 1

    # ── Knowledge gaps: what's most needed ──
    knowledge_gaps = {}
    for group, sub_counts in subdomain_counts.items():
        sorted_subs = sorted(sub_counts.items(), key=lambda x: x[1], reverse=True)
        knowledge_gaps[group] = {
            "total_questions_touching": sum(domain_counts[cat].get(group, 0) for cat in domain_counts),
            "subdomain_frequency": dict(sorted_subs),
            "top_subdomains": [s for s, _ in sorted_subs[:5]],
        }

    # ── Output ──
    output = {
        "summary": {
            "total_questions": len(questions),
            "mcq_count": mcq_count,
            "latex_count": latex_count,
            "category_distribution": dict(category_counts.most_common()),
            "complexity_signals": dict(complexity_counts.most_common()),
        },
        "knowledge_gaps": knowledge_gaps,
        "domain_by_category": {
            cat: dict(sub_counts.most_common())
            for cat, sub_counts in {
                cat: collections.Counter(domain_counts[cat])
                for cat in domain_counts
            }.items()
        },
        "kb_expansion_priority": _compute_kb_priority(knowledge_gaps),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {output_path}")

    # Print summary to console
    print("\n=== Summary ===")
    print(f"  Total: {len(questions)}  MCQ: {mcq_count}  LaTeX: {latex_count}")
    print(f"\n  Category distribution:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:40s} {count:5d} ({100*count/len(questions):.1f}%)")

    print(f"\n=== Top KB Expansion Priorities ===")
    for item in output["kb_expansion_priority"][:15]:
        print(f"  [{item['priority']:3d}] {item['domain']:30s} / {item['subdomain']:30s}  freq={item['frequency']}")


def _compute_kb_priority(knowledge_gaps: dict) -> list:
    """Rank subdomain knowledge needs by frequency across all questions."""
    items = []
    for group, data in knowledge_gaps.items():
        for sub, freq in data["subdomain_frequency"].items():
            items.append({
                "group": group,
                "domain": group,
                "subdomain": sub,
                "frequency": freq,
            })
    items.sort(key=lambda x: x["frequency"], reverse=True)
    for i, item in enumerate(items):
        item["priority"] = i + 1
    return items


if __name__ == "__main__":
    import sys
    workspace = Path(__file__).parent.parent
    hle_path = str(workspace / "hle_2500_eval.jsonl")
    output_path = str(workspace / "knowledge_gaps.json")
    if not Path(hle_path).exists():
        print(f"Error: {hle_path} not found")
        sys.exit(1)
    run_analysis(hle_path, output_path)
