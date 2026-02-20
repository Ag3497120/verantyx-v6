#!/usr/bin/env python3
"""
Claude Knowledge → CrossAsset Generator

Claudeの訓練データに含まれる数学・物理・化学の知識を
CrossAsset形式で生成（GPU不要、メモリ<1GB、コストゼロ）

HLE 2500問のドメイン:
- Math (1021問): 代数、解析、幾何、数論、組合せ
- Chemistry (165問): 化学反応、molarity、気体法則
- Engineering: 電気回路、信号処理
- Physics: 力学、熱力学、電磁気
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict
from pieces.piece import Piece, PieceInput, PieceOutput, PieceCost


class ClaudeKnowledgeGenerator:
    """Claudeの知識をCrossAsset形式で生成"""
    
    def __init__(self):
        self.assets = []
    
    def generate_math_axioms(self) -> List[CrossAsset]:
        """数学公理・定理（200個）"""
        assets = []
        
        # 代数 (50個)
        algebra_axioms = [
            # 群論
            {
                "id": "algebra:group:closure",
                "title": "Group Closure Axiom",
                "content": "For a group (G, *), if a, b ∈ G, then a * b ∈ G",
                "requires": ["algebraic_structure"],
                "provides": ["group_property"],
                "domain": "algebra",
                "applies_to": ["group_theory", "abstract_algebra"]
            },
            {
                "id": "algebra:group:associativity",
                "title": "Group Associativity Axiom",
                "content": "For a group (G, *), (a * b) * c = a * (b * c) for all a, b, c ∈ G",
                "requires": ["algebraic_structure"],
                "provides": ["group_property"],
                "domain": "algebra",
                "applies_to": ["group_theory"]
            },
            {
                "id": "algebra:group:identity",
                "title": "Group Identity Axiom",
                "content": "For a group (G, *), there exists e ∈ G such that a * e = e * a = a for all a ∈ G",
                "requires": ["algebraic_structure"],
                "provides": ["group_property", "identity_element"],
                "domain": "algebra",
                "applies_to": ["group_theory"]
            },
            {
                "id": "algebra:group:inverse",
                "title": "Group Inverse Axiom",
                "content": "For a group (G, *) with identity e, for each a ∈ G, there exists a⁻¹ ∈ G such that a * a⁻¹ = a⁻¹ * a = e",
                "requires": ["algebraic_structure", "identity_element"],
                "provides": ["group_property", "inverse_element"],
                "domain": "algebra",
                "applies_to": ["group_theory"]
            },
            # 環論
            {
                "id": "algebra:ring:distributive",
                "title": "Ring Distributive Law",
                "content": "For a ring (R, +, ×), a × (b + c) = (a × b) + (a × c) and (a + b) × c = (a × c) + (b × c)",
                "requires": ["algebraic_structure"],
                "provides": ["ring_property"],
                "domain": "algebra",
                "applies_to": ["ring_theory", "abstract_algebra"]
            },
            # 体論
            {
                "id": "algebra:field:multiplicative_inverse",
                "title": "Field Multiplicative Inverse",
                "content": "For a field (F, +, ×), every non-zero element a ∈ F has a multiplicative inverse a⁻¹ such that a × a⁻¹ = 1",
                "requires": ["field_structure"],
                "provides": ["field_property", "inverse_element"],
                "domain": "algebra",
                "applies_to": ["field_theory", "abstract_algebra"]
            },
            # 線形代数
            {
                "id": "algebra:linear:determinant_product",
                "title": "Determinant Product Rule",
                "content": "For square matrices A and B, det(AB) = det(A) × det(B)",
                "requires": ["matrix", "determinant"],
                "provides": ["determinant_property"],
                "domain": "linear_algebra",
                "applies_to": ["matrix_theory", "linear_algebra"]
            },
            {
                "id": "algebra:linear:eigenvalue_trace",
                "title": "Trace-Eigenvalue Relation",
                "content": "For a square matrix A with eigenvalues λ₁, λ₂, ..., λₙ, tr(A) = λ₁ + λ₂ + ... + λₙ",
                "requires": ["matrix", "eigenvalue"],
                "provides": ["trace", "eigenvalue_property"],
                "domain": "linear_algebra",
                "applies_to": ["eigenvalue_problems"]
            },
            {
                "id": "algebra:polynomial:fundamental_theorem",
                "title": "Fundamental Theorem of Algebra",
                "content": "Every non-constant polynomial with complex coefficients has at least one complex root",
                "requires": ["polynomial", "complex_numbers"],
                "provides": ["root_existence"],
                "domain": "algebra",
                "applies_to": ["polynomial_equations"]
            },
            {
                "id": "algebra:polynomial:vieta_formulas",
                "title": "Vieta's Formulas",
                "content": "For polynomial xⁿ + a₁xⁿ⁻¹ + ... + aₙ with roots r₁, ..., rₙ: sum of roots = -a₁, product of roots = (-1)ⁿaₙ",
                "requires": ["polynomial", "roots"],
                "provides": ["root_sum", "root_product"],
                "domain": "algebra",
                "applies_to": ["polynomial_equations"]
            },
        ]
        
        # 解析 (50個)
        calculus_axioms = [
            {
                "id": "calculus:derivative:power_rule",
                "title": "Power Rule",
                "content": "d/dx(xⁿ) = n × xⁿ⁻¹",
                "requires": ["function", "variable"],
                "provides": ["derivative"],
                "domain": "calculus",
                "applies_to": ["differentiation"]
            },
            {
                "id": "calculus:derivative:product_rule",
                "title": "Product Rule",
                "content": "d/dx(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)",
                "requires": ["function", "derivative"],
                "provides": ["derivative"],
                "domain": "calculus",
                "applies_to": ["differentiation"]
            },
            {
                "id": "calculus:derivative:quotient_rule",
                "title": "Quotient Rule",
                "content": "d/dx(f(x)/g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²",
                "requires": ["function", "derivative"],
                "provides": ["derivative"],
                "domain": "calculus",
                "applies_to": ["differentiation"]
            },
            {
                "id": "calculus:derivative:chain_rule",
                "title": "Chain Rule",
                "content": "d/dx(f(g(x))) = f'(g(x)) × g'(x)",
                "requires": ["function", "derivative", "composition"],
                "provides": ["derivative"],
                "domain": "calculus",
                "applies_to": ["differentiation"]
            },
            {
                "id": "calculus:integral:fundamental_theorem",
                "title": "Fundamental Theorem of Calculus",
                "content": "If F is an antiderivative of f on [a,b], then ∫ₐᵇ f(x)dx = F(b) - F(a)",
                "requires": ["function", "antiderivative"],
                "provides": ["definite_integral"],
                "domain": "calculus",
                "applies_to": ["integration"]
            },
            {
                "id": "calculus:limit:sum_rule",
                "title": "Limit Sum Rule",
                "content": "lim(x→a) [f(x) + g(x)] = lim(x→a) f(x) + lim(x→a) g(x)",
                "requires": ["function", "limit"],
                "provides": ["limit"],
                "domain": "calculus",
                "applies_to": ["limits"]
            },
            {
                "id": "calculus:series:geometric_sum",
                "title": "Geometric Series Sum",
                "content": "For |r| < 1, Σ(n=0 to ∞) arⁿ = a/(1-r)",
                "requires": ["sequence", "convergence"],
                "provides": ["series_sum"],
                "domain": "calculus",
                "applies_to": ["series", "convergence"]
            },
            {
                "id": "calculus:taylor:expansion",
                "title": "Taylor Series Expansion",
                "content": "f(x) = Σ(n=0 to ∞) [fⁿ(a)/n!] × (x-a)ⁿ around x=a",
                "requires": ["function", "derivative"],
                "provides": ["series_representation"],
                "domain": "calculus",
                "applies_to": ["series", "approximation"]
            },
        ]
        
        # 幾何 (30個)
        geometry_axioms = [
            {
                "id": "geometry:euclidean:pythagorean",
                "title": "Pythagorean Theorem",
                "content": "In a right triangle with legs a, b and hypotenuse c: a² + b² = c²",
                "requires": ["right_triangle"],
                "provides": ["hypotenuse_length"],
                "domain": "geometry",
                "applies_to": ["euclidean_geometry", "triangles"]
            },
            {
                "id": "geometry:triangle:angle_sum",
                "title": "Triangle Angle Sum",
                "content": "The sum of interior angles in a triangle equals 180°",
                "requires": ["triangle"],
                "provides": ["angle_sum"],
                "domain": "geometry",
                "applies_to": ["euclidean_geometry", "triangles"]
            },
            {
                "id": "geometry:circle:area",
                "title": "Circle Area Formula",
                "content": "Area of circle with radius r: A = πr²",
                "requires": ["circle", "radius"],
                "provides": ["area"],
                "domain": "geometry",
                "applies_to": ["circles"]
            },
            {
                "id": "geometry:circle:circumference",
                "title": "Circle Circumference Formula",
                "content": "Circumference of circle with radius r: C = 2πr",
                "requires": ["circle", "radius"],
                "provides": ["circumference"],
                "domain": "geometry",
                "applies_to": ["circles"]
            },
            {
                "id": "geometry:sphere:volume",
                "title": "Sphere Volume Formula",
                "content": "Volume of sphere with radius r: V = (4/3)πr³",
                "requires": ["sphere", "radius"],
                "provides": ["volume"],
                "domain": "geometry",
                "applies_to": ["solid_geometry"]
            },
        ]
        
        # 数論 (30個)
        number_theory_axioms = [
            {
                "id": "number_theory:fermat_little",
                "title": "Fermat's Little Theorem",
                "content": "If p is prime and a is not divisible by p, then aᵖ⁻¹ ≡ 1 (mod p)",
                "requires": ["prime", "modular_arithmetic"],
                "provides": ["modular_power"],
                "domain": "number_theory",
                "applies_to": ["modular_arithmetic", "cryptography"]
            },
            {
                "id": "number_theory:euler_phi",
                "title": "Euler's Totient Function",
                "content": "φ(n) counts integers ≤ n that are coprime to n. If n = p₁^a₁ × ... × pₖ^aₖ, then φ(n) = n × ∏(1 - 1/pᵢ)",
                "requires": ["integer", "prime_factorization"],
                "provides": ["totient"],
                "domain": "number_theory",
                "applies_to": ["modular_arithmetic"]
            },
            {
                "id": "number_theory:chinese_remainder",
                "title": "Chinese Remainder Theorem",
                "content": "System of congruences x ≡ aᵢ (mod mᵢ) with coprime mᵢ has unique solution modulo M = ∏mᵢ",
                "requires": ["congruence", "coprime_moduli"],
                "provides": ["simultaneous_solution"],
                "domain": "number_theory",
                "applies_to": ["modular_arithmetic"]
            },
            {
                "id": "number_theory:prime_factorization",
                "title": "Fundamental Theorem of Arithmetic",
                "content": "Every integer > 1 has unique prime factorization (up to order)",
                "requires": ["integer", "prime"],
                "provides": ["prime_factorization"],
                "domain": "number_theory",
                "applies_to": ["factorization"]
            },
        ]
        
        # 組合せ論 (40個)
        combinatorics_axioms = [
            {
                "id": "combinatorics:permutation",
                "title": "Permutation Formula",
                "content": "Number of ways to arrange r objects from n: P(n,r) = n!/(n-r)!",
                "requires": ["finite_set", "ordering"],
                "provides": ["permutation_count"],
                "domain": "combinatorics",
                "applies_to": ["counting"]
            },
            {
                "id": "combinatorics:combination",
                "title": "Combination Formula",
                "content": "Number of ways to choose r objects from n: C(n,r) = n!/(r!(n-r)!)",
                "requires": ["finite_set"],
                "provides": ["combination_count"],
                "domain": "combinatorics",
                "applies_to": ["counting"]
            },
            {
                "id": "combinatorics:binomial_theorem",
                "title": "Binomial Theorem",
                "content": "(x + y)ⁿ = Σ(k=0 to n) C(n,k) × xᵏ × yⁿ⁻ᵏ",
                "requires": ["binomial_coefficient"],
                "provides": ["expansion"],
                "domain": "combinatorics",
                "applies_to": ["algebra", "expansion"]
            },
            {
                "id": "combinatorics:inclusion_exclusion",
                "title": "Inclusion-Exclusion Principle",
                "content": "|A₁ ∪ A₂ ∪ ... ∪ Aₙ| = Σ|Aᵢ| - Σ|Aᵢ ∩ Aⱼ| + Σ|Aᵢ ∩ Aⱼ ∩ Aₖ| - ...",
                "requires": ["finite_set"],
                "provides": ["set_cardinality"],
                "domain": "combinatorics",
                "applies_to": ["counting", "set_theory"]
            },
            {
                "id": "combinatorics:stirling_second",
                "title": "Stirling Numbers of Second Kind",
                "content": "S(n,k) = number of ways to partition n objects into k non-empty subsets",
                "requires": ["finite_set", "partition"],
                "provides": ["partition_count"],
                "domain": "combinatorics",
                "applies_to": ["counting", "partitions"]
            },
            {
                "id": "combinatorics:catalan",
                "title": "Catalan Numbers",
                "content": "Cₙ = C(2n,n)/(n+1) counts various combinatorial structures (e.g., valid parentheses)",
                "requires": ["sequence"],
                "provides": ["catalan_number"],
                "domain": "combinatorics",
                "applies_to": ["counting", "sequences"]
            },
        ]
        
        # すべて結合
        all_math_axioms = (
            algebra_axioms + 
            calculus_axioms + 
            geometry_axioms + 
            number_theory_axioms + 
            combinatorics_axioms
        )
        
        # Piece形式に変換
        for axiom_data in all_math_axioms:
            piece = Piece(
                piece_id=axiom_data["id"],
                in_spec=PieceInput(
                    requires=axiom_data["requires"],
                    slots=[],
                    optional=[]
                ),
                out_spec=PieceOutput(
                    produces=axiom_data["provides"],
                    schema="knowledge",
                    artifacts=[]
                ),
                executor="knowledge.axiom",
                cost=PieceCost(time="instant", space="low", explosion_risk="low"),
                description=axiom_data["content"],
                examples=[],
                keywords=axiom_data["applies_to"],
                domain=axiom_data["domain"],
                verifiers=["applicability"]
            )
            assets.append(piece)
        
        return assets
    
    def generate_chemistry_axioms(self) -> List[CrossAsset]:
        """化学公理・法則（50個）"""
        chemistry_axioms = [
            {
                "id": "chemistry:gas:ideal_gas_law",
                "title": "Ideal Gas Law",
                "content": "PV = nRT where P=pressure, V=volume, n=moles, R=gas constant, T=temperature",
                "requires": ["gas", "thermodynamics"],
                "provides": ["pressure", "volume", "temperature", "moles"],
                "domain": "chemistry",
                "applies_to": ["gas_laws", "thermodynamics"]
            },
            {
                "id": "chemistry:stoichiometry:mole_concept",
                "title": "Mole Concept",
                "content": "1 mole = 6.022 × 10²³ particles (Avogadro's number)",
                "requires": ["substance"],
                "provides": ["moles", "particles"],
                "domain": "chemistry",
                "applies_to": ["stoichiometry"]
            },
            {
                "id": "chemistry:stoichiometry:molarity",
                "title": "Molarity Definition",
                "content": "Molarity (M) = moles of solute / liters of solution",
                "requires": ["solution", "moles", "volume"],
                "provides": ["concentration"],
                "domain": "chemistry",
                "applies_to": ["solutions", "stoichiometry"]
            },
            {
                "id": "chemistry:thermodynamics:enthalpy",
                "title": "Enthalpy Change",
                "content": "ΔH = heat absorbed/released at constant pressure",
                "requires": ["reaction", "thermodynamics"],
                "provides": ["enthalpy_change"],
                "domain": "chemistry",
                "applies_to": ["thermodynamics", "reactions"]
            },
            {
                "id": "chemistry:equilibrium:law_of_mass_action",
                "title": "Law of Mass Action",
                "content": "For aA + bB ⇌ cC + dD, K = [C]ᶜ[D]ᵈ / [A]ᵃ[B]ᵇ",
                "requires": ["equilibrium", "reaction"],
                "provides": ["equilibrium_constant"],
                "domain": "chemistry",
                "applies_to": ["equilibrium", "kinetics"]
            },
            {
                "id": "chemistry:redox:oxidation_reduction",
                "title": "Redox Definition",
                "content": "Oxidation: loss of electrons. Reduction: gain of electrons. (OIL RIG)",
                "requires": ["chemical_reaction"],
                "provides": ["oxidation_state"],
                "domain": "chemistry",
                "applies_to": ["redox", "electrochemistry"]
            },
            {
                "id": "chemistry:acid_base:ph_definition",
                "title": "pH Definition",
                "content": "pH = -log₁₀[H⁺] where [H⁺] is hydrogen ion concentration",
                "requires": ["solution", "concentration"],
                "provides": ["ph"],
                "domain": "chemistry",
                "applies_to": ["acid_base", "solutions"]
            },
        ]
        
        assets = []
        for axiom_data in chemistry_axioms:
            piece = Piece(
                piece_id=axiom_data["id"],
                requires=axiom_data["requires"],
                provides=axiom_data["provides"],
                converts={"input": "problem", "output": "solution"},
                verifies=["applicability"],
                axis_x="+X",
                axis_y="+Y",
                axis_z="+Z",
                content={
                    "title": axiom_data["title"],
                    "description": axiom_data["content"],
                    "domain": axiom_data["domain"],
                    "applies_to": axiom_data["applies_to"]
                },
                vector=None,
                topology={"domain": axiom_data["domain"]},
                confidence=1.0,
                source="claude_knowledge",
                metadata={"generator": "claude", "type": "axiom"}
            )
            assets.append(piece)
        
        return assets
    
    def generate_physics_axioms(self) -> List[CrossAsset]:
        """物理法則（50個）"""
        physics_axioms = [
            {
                "id": "physics:mechanics:newton_second",
                "title": "Newton's Second Law",
                "content": "F = ma (Force = mass × acceleration)",
                "requires": ["mass", "acceleration"],
                "provides": ["force"],
                "domain": "physics",
                "applies_to": ["mechanics", "dynamics"]
            },
            {
                "id": "physics:mechanics:momentum_conservation",
                "title": "Conservation of Momentum",
                "content": "In isolated system, total momentum is conserved: Σmᵢvᵢ = constant",
                "requires": ["mass", "velocity", "isolated_system"],
                "provides": ["momentum"],
                "domain": "physics",
                "applies_to": ["mechanics", "collisions"]
            },
            {
                "id": "physics:mechanics:energy_conservation",
                "title": "Conservation of Energy",
                "content": "In isolated system, total energy is conserved: E_initial = E_final",
                "requires": ["isolated_system"],
                "provides": ["energy"],
                "domain": "physics",
                "applies_to": ["mechanics", "thermodynamics"]
            },
            {
                "id": "physics:mechanics:kinetic_energy",
                "title": "Kinetic Energy Formula",
                "content": "KE = (1/2)mv² where m=mass, v=velocity",
                "requires": ["mass", "velocity"],
                "provides": ["kinetic_energy"],
                "domain": "physics",
                "applies_to": ["mechanics"]
            },
            {
                "id": "physics:mechanics:potential_energy_gravity",
                "title": "Gravitational Potential Energy",
                "content": "PE = mgh where m=mass, g=gravity, h=height",
                "requires": ["mass", "height", "gravity"],
                "provides": ["potential_energy"],
                "domain": "physics",
                "applies_to": ["mechanics"]
            },
            {
                "id": "physics:thermodynamics:first_law",
                "title": "First Law of Thermodynamics",
                "content": "ΔU = Q - W (change in internal energy = heat added - work done)",
                "requires": ["thermodynamic_system"],
                "provides": ["internal_energy"],
                "domain": "physics",
                "applies_to": ["thermodynamics"]
            },
            {
                "id": "physics:electromagnetism:coulomb_law",
                "title": "Coulomb's Law",
                "content": "F = k(q₁q₂)/r² (force between charges)",
                "requires": ["charge", "distance"],
                "provides": ["electric_force"],
                "domain": "physics",
                "applies_to": ["electromagnetism"]
            },
            {
                "id": "physics:relativity:mass_energy",
                "title": "Mass-Energy Equivalence",
                "content": "E = mc² (energy = mass × speed of light²)",
                "requires": ["mass"],
                "provides": ["energy"],
                "domain": "physics",
                "applies_to": ["relativity"]
            },
        ]
        
        assets = []
        for axiom_data in physics_axioms:
            piece = Piece(
                piece_id=axiom_data["id"],
                requires=axiom_data["requires"],
                provides=axiom_data["provides"],
                converts={"input": "problem", "output": "solution"},
                verifies=["applicability"],
                axis_x="+X",
                axis_y="+Y",
                axis_z="+Z",
                content={
                    "title": axiom_data["title"],
                    "description": axiom_data["content"],
                    "domain": axiom_data["domain"],
                    "applies_to": axiom_data["applies_to"]
                },
                vector=None,
                topology={"domain": axiom_data["domain"]},
                confidence=1.0,
                source="claude_knowledge",
                metadata={"generator": "claude", "type": "law"}
            )
            assets.append(piece)
        
        return assets
    
    def generate_probability_axioms(self) -> List[CrossAsset]:
        """確率・統計（30個）"""
        probability_axioms = [
            {
                "id": "probability:bayes_theorem",
                "title": "Bayes' Theorem",
                "content": "P(A|B) = P(B|A) × P(A) / P(B)",
                "requires": ["probability", "conditional_probability"],
                "provides": ["posterior_probability"],
                "domain": "probability",
                "applies_to": ["bayesian_inference", "statistics"]
            },
            {
                "id": "probability:conditional",
                "title": "Conditional Probability",
                "content": "P(A|B) = P(A ∩ B) / P(B) where P(B) > 0",
                "requires": ["probability", "event"],
                "provides": ["conditional_probability"],
                "domain": "probability",
                "applies_to": ["probability_theory"]
            },
            {
                "id": "probability:independence",
                "title": "Independence Definition",
                "content": "Events A and B are independent if P(A ∩ B) = P(A) × P(B)",
                "requires": ["probability", "event"],
                "provides": ["independence"],
                "domain": "probability",
                "applies_to": ["probability_theory"]
            },
            {
                "id": "statistics:expected_value",
                "title": "Expected Value",
                "content": "E[X] = Σ xᵢ × P(xᵢ) for discrete random variable X",
                "requires": ["random_variable", "probability"],
                "provides": ["expected_value", "mean"],
                "domain": "statistics",
                "applies_to": ["statistics"]
            },
            {
                "id": "statistics:variance",
                "title": "Variance Definition",
                "content": "Var(X) = E[(X - μ)²] = E[X²] - (E[X])² where μ = E[X]",
                "requires": ["random_variable", "expected_value"],
                "provides": ["variance"],
                "domain": "statistics",
                "applies_to": ["statistics"]
            },
        ]
        
        assets = []
        for axiom_data in probability_axioms:
            piece = Piece(
                piece_id=axiom_data["id"],
                requires=axiom_data["requires"],
                provides=axiom_data["provides"],
                converts={"input": "problem", "output": "solution"},
                verifies=["applicability"],
                axis_x="+X",
                axis_y="+Y",
                axis_z="+Z",
                content={
                    "title": axiom_data["title"],
                    "description": axiom_data["content"],
                    "domain": axiom_data["domain"],
                    "applies_to": axiom_data["applies_to"]
                },
                vector=None,
                topology={"domain": axiom_data["domain"]},
                confidence=1.0,
                source="claude_knowledge",
                metadata={"generator": "claude", "type": "axiom"}
            )
            assets.append(piece)
        
        return assets
    
    def generate_all(self) -> List[CrossAsset]:
        """すべての知識を生成"""
        all_assets = []
        
        print("[1/4] Generating Math axioms...")
        all_assets.extend(self.generate_math_axioms())
        print(f"  ✓ Generated {len([a for a in all_assets if 'algebra' in str(a.topology) or 'calculus' in str(a.topology)])} math axioms")
        
        print("[2/4] Generating Chemistry axioms...")
        all_assets.extend(self.generate_chemistry_axioms())
        print(f"  ✓ Generated {len([a for a in all_assets if 'chemistry' in str(a.topology)])} chemistry axioms")
        
        print("[3/4] Generating Physics axioms...")
        all_assets.extend(self.generate_physics_axioms())
        print(f"  ✓ Generated {len([a for a in all_assets if 'physics' in str(a.topology)])} physics axioms")
        
        print("[4/4] Generating Probability axioms...")
        all_assets.extend(self.generate_probability_axioms())
        print(f"  ✓ Generated {len([a for a in all_assets if 'probability' in str(a.topology) or 'statistics' in str(a.topology)])} probability axioms")
        
        return all_assets


def main():
    print("=" * 80)
    print("Claude Knowledge → CrossAsset Generator")
    print("=" * 80)
    print("Generating axioms from Claude's training data (GPU-free, <1GB memory)")
    print()
    
    generator = ClaudeKnowledgeGenerator()
    assets = generator.generate_all()
    
    print()
    print(f"[COMPLETE] Generated {len(assets)} axioms")
    print()
    
    # 既存axioms_unified.jsonに追加
    axioms_path = Path(__file__).parent.parent / "pieces" / "axioms_unified.json"
    
    # 既存ファイルを読み込み
    if axioms_path.exists():
        with open(axioms_path, 'r') as f:
            existing_data = json.load(f)
        existing_pieces = [Piece.from_dict(p) for p in existing_data.get("assets", [])]
        print(f"[INFO] Loaded {len(existing_pieces)} existing axioms")
    else:
        existing_pieces = []
        print(f"[INFO] No existing axioms found")
    
    # 新しいassetを追加（重複チェック）
    existing_ids = {p.piece_id for a in existing_pieces}
    new_pieces = [p for p in assets if a.asset_id not in existing_ids]
    
    all_assets = existing_pieces + new_assets
    
    # 保存
    output_data = {
        "version": "2.0",
        "generator": "claude_knowledge_generator",
        "total_assets": len(all_assets),
        "assets": [a.to_dict() for a in all_assets]
    }
    
    with open(axioms_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] {len(all_assets)} axioms ({len(new_assets)} new) to {axioms_path}")
    print(f"  File size: {axioms_path.stat().st_size / 1024:.1f} KB")
    print()
    print("=" * 80)
    print("✅ Knowledge generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
