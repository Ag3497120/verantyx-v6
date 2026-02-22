"""
Decomposer - 問題文をIRに分解

ルールベース・テンプレートベースで構造抽出
"""

from typing import Dict, Any, Optional, List
import re

from core.ir import IR, TaskType, Domain, AnswerSchema, Entity, Constraint, Query
from decomposer.problem_type_detector import ProblemTypeDetector
from decomposer.latex_normalizer import normalize_latex, detect_answer_schema as detect_answer_schema_latex
from decomposer.knowledge_need_extractor import extract_knowledge_needs

# 600B concept_dirs ブースト（オプション、キャッシュがあれば0ms）
_concept_booster = None
def _get_concept_booster():
    global _concept_booster
    import os
    if os.environ.get("DISABLE_CONCEPT_BOOST") == "1":
        return None
    if _concept_booster is None:
        try:
            from knowledge.concept_boost import get_booster
            _concept_booster = get_booster()
        except Exception:
            _concept_booster = False  # 無効化
    return _concept_booster if _concept_booster is not False else None


class RuleBasedDecomposer:
    """
    ルールベースDecomposer
    
    LLM不使用、完全にルールとテンプレートで分解
    """
    
    def __init__(self):
        # 問題タイプ検出器
        self.type_detector = ProblemTypeDetector()
        
        # キーワード辞書
        self.task_keywords = {
            TaskType.COMPUTE: ["compute", "calculate", "evaluate", "what is", "how much"],
            TaskType.COUNT: ["how many", "count", "number of", "length of", "characters in"],
            TaskType.FIND: ["find", "determine", "identify"],
            TaskType.DECIDE: ["is", "does", "can", "prove", "show that"],
            TaskType.CHOOSE: ["which", "select", "choose"],
            TaskType.CONSTRUCT: ["construct", "build", "create", "sequence"],
            TaskType.OPTIMIZE: ["minimize", "maximize", "optimize", "smallest", "largest", "minimum", "maximum"]
        }
        
        self.domain_keywords = {
            Domain.ARITHMETIC: ["sum", "product", "divide", "calculate", "compute"],
            Domain.ALGEBRA: ["equation", "solve", "variable", "polynomial", "factor", "simplify", "evaluate", "expression"],
            Domain.LOGIC_PROPOSITIONAL: ["->", "→", "&", "|", "~", "¬", "implies", "tautology", "valid", "satisfiable", "p", "q"],
            Domain.LOGIC_MODAL: ["[]", "<>", "□", "◇", "necessary", "possible", "always", "eventually", "kripke"],
            Domain.GRAPH_THEORY: ["graph", "vertex", "vertices", "edge", "edges", "tree", "cycle", "cyclic", "path", "degree", "complete graph", "binary tree",
                                  "planar", "chromatic", "coloring", "bipartite", "connected component", "hamiltonian path", "eulerian", "spanning tree",
                                  "adjacency", "clique", "independent set", "flow", "matching"],
            Domain.CHESS: ["chess", "move", "mate", "checkmate", "board", "piece", "black", "white"],
            Domain.NUMBER_THEORY: ["prime", "divisor", "divisible", "gcd", "lcm", "congruent", "modulo", "mod", "remainder", "factor", "factorial", "!"],
            Domain.COMBINATORICS: ["permutation", "combination", "arrange", "choose", "binomial", "C(n", "P(n", "nCr", "nPr", "ways to", "how many ways",
                                   "counting", "pigeonhole", "surjection", "injection", "bijection"],
            Domain.PROBABILITY: ["probability", "chance", "expected", "random", "coin", "dice", "die", "card", "deck"],
            Domain.GEOMETRY: ["circle", "triangle", "rectangle", "square", "area", "perimeter", "circumference", "radius", "diameter", "hypotenuse", "pythagorean", "angle",
                              "polygon", "hexagon", "pentagon", "octagon", "ellipse", "parabola", "hyperbola", "tangent", "chord", "arc",
                              "congruent", "similar", "volume", "surface area", "cone", "cylinder", "sphere",
                              "coordinate", "slope", "distance formula", "midpoint"],
            Domain.LINEAR_ALGEBRA: ["matrix", "determinant", "inverse", "eigenvalue", "eigenvector", "dot product", "vector", "transpose",
                                    "rank", "null space", "column space", "orthogonal", "projection", "linear transformation",
                                    "singular value", "trace", "symmetric matrix", "positive definite"],
            Domain.CALCULUS: ["derivative", "integral", "limit", "series", "differential", "differentiate", "integrate", "converge", "diverge",
                              "taylor series", "maclaurin", "fourier", "gradient", "partial derivative", "multivariable",
                              "chain rule", "product rule", "quotient rule", "fundamental theorem", "riemann sum",
                              "improper integral", "infinite series", "power series"],
            Domain.MODULAR_ARITHMETIC: ["modulo", "mod", "congruent", "≡", "modular", "euler's totient", "phi", "fermat's little", "chinese remainder"],
            Domain.ADVANCED_PROBABILITY: ["conditional probability", "bayes", "bayes theorem", "binomial distribution", "normal distribution", "poisson", "expected value", "variance", "covariance", "correlation",
                                          "markov chain", "stochastic", "monte carlo", "central limit theorem", "law of large numbers"],
            Domain.STATISTICS: ["mean", "median", "mode", "variance", "standard deviation", "covariance", "correlation coefficient", "sample", "population",
                                "regression", "hypothesis test", "p-value", "confidence interval", "chi-square", "t-test", "anova"],
            Domain.ADVANCED_NUMBER_THEORY: ["prime factorization", "miller-rabin", "primality", "primitive root", "quadratic residue", "legendre symbol", "jacobi", "divisor function", "sigma",
                                            "euler's totient function", "mobius function", "dirichlet", "zeta function", "arithmetic function",
                                            "multiplicative order", "discrete logarithm", "elliptic curve", "modular form"],
            Domain.ADVANCED_COMBINATORICS: ["stirling number", "bell number", "catalan number", "partition function", "derangement", "inclusion-exclusion",
                                            # Group theory keywords (mapped to ADVANCED_COMBINATORICS)
                                            "group theory", "subgroup", "homomorphism", "isomorphism", "coset", "lagrange theorem",
                                            "sylow", "abelian group", "cyclic group", "quotient group", "normal subgroup",
                                            "group action", "orbit", "stabilizer", "galois group", "automorphism",
                                            "ring theory", "ring homomorphism", "ideal", "field extension", "galois field",
                                            # Topology keywords (mapped to ADVANCED_COMBINATORICS)
                                            "topology", "topological", "manifold", "homotopy", "homology", "cohomology",
                                            "fundamental group", "euler characteristic", "genus", "knot theory",
                                            "compact", "hausdorff", "metric space", "open set", "closed set"],
            Domain.STRING: ["string", "length", "character", "word", "palindrome", "substring", "concatenate"],
            Domain.CRYPTOGRAPHY: ["cipher", "encrypt", "decrypt", "caesar", "substitution", "code", "decipher",
                                  "rsa", "public key", "private key", "hash function", "digital signature",
                                  "diffie-hellman", "aes", "block cipher", "stream cipher"],
            Domain.MULTIPLE_CHOICE: ["Answer Choices", "A.", "B.", "C.", "D.", "E.", "select", "choose"],
            # Domains previously missing from keywords dict
            Domain.PHYSICS: ["velocity", "acceleration", "force", "mass", "energy", "momentum", "newton",
                             "quantum", "electron", "proton", "neutron", "photon",
                             "wave", "frequency", "wavelength", "amplitude",
                             "electric field", "magnetic field", "charge", "current", "voltage", "resistance",
                             "thermodynamic", "entropy", "temperature", "heat", "pressure", "work", "power",
                             "torque", "gravity", "gravitational", "relativity", "special relativity", "general relativity",
                             "hamiltonian", "lagrangian", "eigenstate", "wave function", "commutator", "hilbert space",
                             "spin", "orbital", "bohr", "heisenberg", "schrodinger", "planck", "boltzmann",
                             "optics", "refraction", "reflection", "interference", "diffraction",
                             "nuclear", "fission", "fusion", "decay", "half-life", "radioactive"],
            Domain.CHEMISTRY: ["molecule", "atom", "element", "compound", "reaction", "chemical",
                               "mole", "atomic number", "atomic mass", "bond", "ionic bond", "covalent bond",
                               "acid", "base", "ph", "oxidation", "reduction", "redox",
                               "electron configuration", "periodic table", "isotope", "valence",
                               "organic chemistry", "functional group", "alkane", "alkene", "alkyne",
                               "polymer", "protein", "amino acid", "nucleotide", "dna", "rna",
                               "catalyst", "equilibrium constant", "gibbs", "enthalpy", "entropy of reaction",
                               "molar mass", "stoichiometry", "avogadro", "ideal gas", "gas law"],
            Domain.COMPUTER_SCIENCE: ["algorithm", "complexity", "time complexity", "space complexity",
                                      "sorting", "big o", "dynamic programming", "recursion", "memoization",
                                      "binary search", "hash table", "data structure", "queue", "stack",
                                      "linked list", "binary tree", "bst", "heap", "trie",
                                      "depth-first search", "breadth-first search", "dijkstra", "shortest path",
                                      "turing machine", "automaton", "regular expression", "context-free grammar",
                                      "np-hard", "np-complete", "p vs np", "reduction",
                                      "bit", "byte", "binary", "boolean circuit", "logic gate"],
            Domain.PHILOSOPHY: ["argument", "premise", "conclusion", "deductive", "inductive", "abductive",
                                "fallacy", "epistemology", "ontology", "ethics", "metaphysics",
                                "syllogism", "modus ponens", "modus tollens", "dilemma",
                                "utilitarianism", "kantian", "virtue ethics", "consequentialism",
                                "a priori", "a posteriori", "empiricism", "rationalism",
                                "socrates", "plato", "aristotle", "descartes", "kant", "hume",
                                "phenomenology", "existentialism", "pragmatism", "analytic philosophy"],
        }
        
        self.answer_schema_patterns = {
            AnswerSchema.INTEGER: [r"\b(integer|whole number)\b", r"how many", r"count"],
            AnswerSchema.BOOLEAN: [r"\b(true|false|yes|no)\b", r"^Is .+\?$", r"^Does .+\?$"],
            AnswerSchema.OPTION_LABEL: [r"Answer Choices:", r"\bA\.", r"\bB\."],
            AnswerSchema.MOVE_SEQUENCE: [r"move", r"sequence", r"chess"],
            AnswerSchema.EXPRESSION: [r"expression", r"formula", r"polynomial"]
        }
    
    def decompose(self, problem_text: str) -> IR:
        """
        問題文をIRに分解

        Args:
            problem_text: 問題文

        Returns:
            IR
        """
        # LaTeX 正規化（最初に必ず実行）
        original_text = problem_text.strip()
        normalized_text = normalize_latex(original_text)

        # 両方保持（schema判定には元テキスト、処理には正規化テキスト）
        text = normalized_text
        text_lower = normalized_text.lower()

        # 問題タイプ検出（新規）
        problem_type_info = self.type_detector.detect(text)

        # タスクタイプ推定
        task = self._detect_task(text_lower)

        # ドメイン推定（問題タイプからのブーストを適用）
        domain = self._detect_domain(text_lower, problem_type_info=problem_type_info)
        
        # 選択肢抽出
        options = self._extract_options(text)
        
        # Answer Schema推定
        # LaTeX-based answer schema detection (for metadata hint)
        latex_answer_schema = detect_answer_schema_latex(original_text)

        if options:
            answer_schema = AnswerSchema.OPTION_LABEL
        else:
            answer_schema = self._detect_answer_schema(text_lower, domain, task)
        
        # エンティティ抽出
        entities = self._extract_entities(text, domain)
        
        # 制約抽出
        constraints = self._extract_constraints(text, domain)
        
        # クエリ構築
        query = self._build_query(text_lower, task)
        
        # キーワード抽出（ピース選択のヒント）
        keywords = []
        text_lower = text.lower()
        
        # 数論・組み合わせ
        if "permutation" in text_lower:
            keywords.append("permutation")
        if "combination" in text_lower or "binomial coefficient" in text_lower:
            keywords.append("combination")
        # Layer A: N! パターン検出 → factorial キーワード追加
        # Fix [02]: "6!" のような記号表記でも factorial キーワードを生成する
        if "factorial" in text_lower or re.search(r'\d+\s*!', text):
            keywords.append("factorial")

        # Layer A: べき乗パターン検出 → power キーワード追加
        # Fix [06]: "2^10" や "2**10" でも power キーワードを生成する
        if re.search(r'\d+\s*(?:\^|\*\*)\s*\d+', text) or "to the power" in text_lower:
            keywords.append("power")

        if "prime" in text_lower:
            keywords.append("prime")
        if "gcd" in text_lower or "greatest common divisor" in text_lower:
            keywords.append("gcd")
        if "lcm" in text_lower or "least common multiple" in text_lower:
            keywords.append("lcm")
        if "divisor" in text_lower:
            keywords.append("divisor")
        
        # 確率
        if "probability" in text_lower or "chance" in text_lower:
            keywords.append("probability")
        if "coin" in text_lower or "flip" in text_lower:
            keywords.append("coin")
        if "dice" in text_lower or "die" in text_lower:
            keywords.append("dice")
        if "card" in text_lower or "deck" in text_lower:
            keywords.append("card")
        if "expected" in text_lower:
            keywords.append("expected")
        if "random" in text_lower:
            keywords.append("random")
        # 複数イベント検出
        if any(word in text_lower for word in ["twice", "two times", "both", "all", "multiple"]):
            # さらに、"and"や"getting"などで複数の結果を求めている場合
            if "and" in text_lower or "getting" in text_lower:
                keywords.append("multiple")
        
        # 幾何
        if "circle" in text_lower:
            keywords.append("circle")
        if "triangle" in text_lower:
            keywords.append("triangle")
        if "rectangle" in text_lower or "square" in text_lower:
            keywords.append("rectangle")
        if "area" in text_lower:
            keywords.append("area")
        if "perimeter" in text_lower or "circumference" in text_lower:
            keywords.append("perimeter")
            if "circumference" in text_lower:
                keywords.append("circumference")
        if "radius" in text_lower:
            keywords.append("radius")
        if "hypotenuse" in text_lower or "pythagorean" in text_lower:
            keywords.append("pythagorean")
        
        # 代数
        if "solve" in text_lower or "equation" in text_lower:
            keywords.append("solve")
            keywords.append("equation")
        if "simplify" in text_lower:
            keywords.append("simplify")
        if "factor" in text_lower and "factorial" not in text_lower:
            keywords.append("factor")
        if "evaluate" in text_lower:
            keywords.append("evaluate")
        if "polynomial" in text_lower:
            keywords.append("polynomial")
        
        # グラフ理論
        if "graph" in text_lower:
            keywords.append("graph")
        if "vertex" in text_lower or "vertices" in text_lower:
            keywords.append("vertex")
        if "edge" in text_lower or "edges" in text_lower:
            keywords.append("edges")
        if "tree" in text_lower:
            keywords.append("tree")
        if "cyclic" in text_lower or "cycle" in text_lower:
            keywords.append("cyclic")
        if "degree" in text_lower:
            keywords.append("degree")
        if "complete" in text_lower:
            keywords.append("complete")
        if "binary" in text_lower:
            keywords.append("binary")
        
        # 線形代数
        if "matrix" in text_lower or "matrices" in text_lower:
            keywords.append("matrix")
        if "determinant" in text_lower:
            keywords.append("determinant")
        if "inverse" in text_lower and ("matrix" in text_lower or "matrices" in text_lower):
            keywords.append("inverse")
        if "eigenvalue" in text_lower or "eigenvector" in text_lower:
            keywords.append("eigenvalue")
        if "dot product" in text_lower or "inner product" in text_lower:
            keywords.append("dot")
            keywords.append("product")
        if "vector" in text_lower:
            keywords.append("vector")
        
        # 微積分
        if "derivative" in text_lower or "differentiate" in text_lower:
            keywords.append("derivative")
        if "integral" in text_lower or "integrate" in text_lower:
            keywords.append("integral")
        if "limit" in text_lower and ("approaches" in text_lower or "as" in text_lower):
            keywords.append("limit")
        if "series" in text_lower and ("sum" in text_lower or "converge" in text_lower):
            keywords.append("series")
        
        # ─── 600B キーワードブースト（全文平均より discriminative）───
        # キーワード抽出後のこのタイミングで呼ぶことで
        # 「factorial」「derivative」等の専門語だけを使って
        # concept_dirs と照合できる（全文では溶けていた信号を回収）
        domain_confidence = 0.8
        booster_kw = _get_concept_booster()
        if booster_kw is not None and keywords:
            try:
                kw_boost = booster_kw.get_scores_by_keywords(keywords)
                if kw_boost:
                    top_kw_domain_val = max(kw_boost, key=kw_boost.get)
                    top_kw_score = kw_boost[top_kw_domain_val]
                    # 既存のルールベースドメインと比較してより強いシグナルがあれば上書き
                    # スコール域: search_keywords() の normalized score (0-1) × BOOST_FACTOR(8.0)
                    # 実測最大値: ~3.5（combinatorics 0.375×8=3.0, number_theory 0.443×8=3.55）
                    if top_kw_score > 1.5:  # 正規化スコア > 0.19 相当
                        try:
                            kw_domain = Domain(top_kw_domain_val)
                            if domain == Domain.UNKNOWN:
                                # ルールベースが判定できなかった → 600Bの判断を採用
                                domain = kw_domain
                                domain_confidence = 0.65
                            elif domain != kw_domain and top_kw_score > 3.0:
                                # 強いシグナル(0.375+ normalized) → 既存を上書き
                                domain = kw_domain
                                domain_confidence = 0.70
                        except (ValueError, KeyError):
                            pass
            except Exception:
                pass

        # ─── 不足知識ニーズ抽出（鉄の壁設計: 概念名のみ、問題文は含まない）───
        _pre_ir_dict = {
            "domain": domain.value,
            "task": task.value,
            "entities": [{"type": e.type, "name": e.name, "value": e.value} for e in entities],
            "metadata": {"keywords": keywords},
        }
        knowledge_needs = extract_knowledge_needs(_pre_ir_dict, problem_text)
        missing = [
            {
                "concept": kn.concept,
                "kind": kn.kind,
                "domain": kn.domain,
                "relation": kn.relation,
                "scope": kn.scope,
                "context_hint": kn.context_hint,
            }
            for kn in knowledge_needs
        ]

        # concept_extractor_v2 で固有名詞を追加（鉄の壁: 概念名のみ）
        try:
            from decomposer.concept_extractor_v2 import extract_concepts_v2
            extracted = extract_concepts_v2(problem_text)
            seen = {m["concept"] for m in missing}
            for ec in extracted:
                if ec.confidence >= 0.7:
                    concept_key = ec.name.replace(" ", "_").lower()
                    if concept_key not in seen:
                        missing.append({
                            "concept": concept_key,
                            "kind": ec.kind,
                            "domain": ec.domain_hint,
                            "relation": "",
                            "scope": "concise",
                            "context_hint": "",
                        })
                        seen.add(concept_key)
        except Exception:
            pass

        # IR構築（問題タイプ情報を追加）
        ir = IR(
            task=task,
            domain=domain,
            answer_schema=answer_schema,
            entities=entities,
            constraints=constraints,
            query=query,
            options=options,
            metadata={
                "source_text": problem_text,
                "confidence": domain_confidence,
                "keywords": keywords,
                "problem_type": problem_type_info.get('primary_type').value if problem_type_info else "unknown",
                "problem_type_confidence": problem_type_info.get('confidence', 0.0) if problem_type_info else 0.0,
                "latex_answer_schema": latex_answer_schema,  # LaTeX-based answer type hint
                "normalized_text": normalized_text  # Store normalized version
            },
            missing=missing,
        )
        
        return ir
    
    def _detect_task(self, text: str) -> TaskType:
        """タスクタイプ検出"""
        scores = {task: 0 for task in TaskType}
        
        # 高優先度キーワード（より具体的なもの）
        high_priority = {
            TaskType.COUNT: ["how many", "count", "number of", "length of", "characters in"],
            TaskType.OPTIMIZE: ["minimize", "maximize", "smallest", "largest", "minimum", "maximum"],
            TaskType.CHOOSE: ["which", "select", "choose"],
        }
        
        # 高優先度キーワードを先にチェック
        for task, keywords in high_priority.items():
            for keyword in keywords:
                if keyword in text:
                    scores[task] += 3  # 高スコア
        
        # 通常のキーワード
        for task, keywords in self.task_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[task] += 1
        
        # 最高スコアのタスク
        best_task = max(scores, key=scores.get)
        
        if scores[best_task] == 0:
            # キーワードが見つからない場合はCOMPUTEをデフォルト
            return TaskType.COMPUTE
        
        return best_task
    
    def _detect_domain(self, text: str, problem_type_info: Optional[Dict[str, Any]] = None) -> Domain:
        """ドメイン検出（記号優先、コンテキスト考慮、問題タイプブースト）"""
        scores = {domain: 0 for domain in Domain}
        text_lower = text.lower()
        
        # 問題タイプによるブースト（Priority 3の改善）
        if problem_type_info:
            primary_type = problem_type_info.get('primary_type')
            if primary_type:
                boosts = self.type_detector.get_domain_boost(primary_type)
                for domain_str, boost in boosts.items():
                    try:
                        domain_enum = Domain[domain_str.upper()]
                        scores[domain_enum] += boost
                    except (KeyError, AttributeError):
                        pass

        # 600B concept_dirs ブースト（アンカーキーワード max-pooling）
        # 全文 mean より discriminative power が高い
        booster = _get_concept_booster()
        if booster is not None:
            try:
                from knowledge.concept_boost import extract_anchor_kws
                anchor_kws = extract_anchor_kws(text)
                if anchor_kws:
                    concept_scores = booster.get_scores_by_keywords(anchor_kws)
                else:
                    concept_scores = booster.get_scores(text)
                for domain_val, boost in concept_scores.items():
                    try:
                        domain_enum = Domain(domain_val)
                        scores[domain_enum] += boost
                    except (ValueError, KeyError):
                        pass
            except Exception:
                pass
        
        # 高優先度パターン（先に検出）
        import re
        
        # 組み合わせパターン C(n,r) or P(n,r) - 最優先
        if re.search(r'[CP]\(\s*\d+\s*,\s*\d+\s*\)', text):
            scores[Domain.COMBINATORICS] += 25  # 大幅に引き上げ
        
        # 組み合わせキーワード
        if any(kw in text_lower for kw in ['permutation', 'combination', 'binomial coefficient', 'ways to', 'how many ways']):
            scores[Domain.COMBINATORICS] += 20
        
        # nCr, nPr パターン
        if re.search(r'\d+[CP]r?\d+', text):
            scores[Domain.COMBINATORICS] += 20
        
        # 階乗パターン (n!)
        if re.search(r'\d+!', text) or 'factorial' in text_lower:
            scores[Domain.NUMBER_THEORY] += 18
        
        # 数論の文脈検出（primeなど）
        if 'prime' in text_lower:
            scores[Domain.NUMBER_THEORY] += 15
        
        # 多肢選択問題の検出（最高優先度）
        if 'answer choices' in text_lower or 'answer choice' in text_lower:
            scores[Domain.MULTIPLE_CHOICE] += 50  # 大幅に引き上げ
        # 行頭の A. B. C. D. パターン（行頭限定で誤検出防止）
        choice_lines = len(re.findall(r'(?:^|\n)\s*[A-Z][.)]\s+\S', text))
        if choice_lines >= 3:  # 3行以上の選択肢（誤検出防止）
            scores[Domain.MULTIPLE_CHOICE] += 40
        elif choice_lines >= 2:
            scores[Domain.MULTIPLE_CHOICE] += 20
        
        # 代数: solve/equation/roots → ALGEBRA 優先
        if any(kw in text_lower for kw in ['solve', 'roots of', 'solutions of', 'zeros of', 'find x']):
            scores[Domain.ALGEBRA] += 20
        if re.search(r'[a-zA-Z]\s*[\^*]\s*2', text) and ('=' in text or 'solve' in text_lower):
            scores[Domain.ALGEBRA] += 15  # 二次式 + 等号/solve → 強い代数シグナル

        # 微積分の高優先度キーワード
        if any(kw in text_lower for kw in ['derivative', 'integral', 'limit', 'differential', 'differentiate']):
            scores[Domain.CALCULUS] += 25

        # 線形代数の高優先度キーワード
        if any(kw in text_lower for kw in ['matrix', 'determinant', 'eigenvalue', 'dot product']):
            scores[Domain.LINEAR_ALGEBRA] += 25

        # 群論 高優先度キーワード → ADVANCED_COMBINATORICS
        if any(kw in text_lower for kw in ['group theory', 'subgroup', 'normal subgroup', 'quotient group', 'group action',
                                            'homomorphism', 'isomorphism', 'sylow', 'galois group', 'automorphism group',
                                            'abelian group', 'cyclic group', 'dihedral group', 'lagrange theorem',
                                            'order of the group', 'group of order', 'group homomorphism']):
            scores[Domain.ADVANCED_COMBINATORICS] += 25
        # 環・体論
        if any(kw in text_lower for kw in ['ring theory', 'ring homomorphism', 'ideal', 'field extension', 'galois field', 'finite field']):
            scores[Domain.ADVANCED_COMBINATORICS] += 22
        # トポロジー 高優先度キーワード
        if any(kw in text_lower for kw in ['topological space', 'manifold', 'homotopy', 'homology group', 'cohomology',
                                            'fundamental group', 'euler characteristic', 'genus', 'knot theory',
                                            'hausdorff', 'compact space']):
            scores[Domain.ADVANCED_COMBINATORICS] += 22

        # 物理 高優先度キーワード (quantum は eigenvalue より先に処理して LINEAR_ALGEBRA を上書き)
        if any(kw in text_lower for kw in ['quantum mechanics', 'quantum mechanical', 'quantum field', 'quantum state',
                                            'wave function', 'hamiltonian', 'eigenstate', 'hilbert space',
                                            'schrodinger', 'heisenberg', 'commutator bracket', 'spin state',
                                            'special relativity', 'general relativity', 'lorentz', 'spacetime']):
            scores[Domain.PHYSICS] += 35  # eigenvalue +25 を超えるように引き上げ
        if any(kw in text_lower for kw in ['velocity', 'acceleration', 'newton', 'force', 'momentum', 'kinetic energy',
                                            'potential energy', 'electric field', 'magnetic field', 'thermodynamic',
                                            'entropy', 'half-life', 'radioactive decay', 'nuclear fission', 'nuclear fusion']):
            scores[Domain.PHYSICS] += 20

        # 化学 高優先度キーワード
        if any(kw in text_lower for kw in ['electron configuration', 'periodic table', 'oxidation state', 'oxidation number',
                                            'gibbs free energy', 'enthalpy', 'stoichiometry', 'molar mass', 'avogadro',
                                            'organic chemistry', 'functional group', 'reaction mechanism',
                                            'chemical equation', 'combustion', 'thermite', 'redox reaction',
                                            'chemical formula', 'molecular formula', 'balanced equation']):
            scores[Domain.CHEMISTRY] += 28
        if any(kw in text_lower for kw in ['molecule', 'atom', 'chemical bond', 'ionic bond', 'covalent bond', 'acid', 'base',
                                            'ph level', 'catalyst', 'equilibrium constant', 'ideal gas law',
                                            'chemical reaction', 'reactant', 'product', 'yield',
                                            'methane', 'ethanol', 'glucose', 'sodium chloride', 'sulfuric acid']):
            scores[Domain.CHEMISTRY] += 20

        # コンピュータサイエンス 高優先度キーワード
        if any(kw in text_lower for kw in ['time complexity', 'space complexity', 'big-o', 'big o notation',
                                            'dynamic programming', 'np-hard', 'np-complete', 'turing machine',
                                            'context-free grammar', 'regular language', 'automata']):
            scores[Domain.COMPUTER_SCIENCE] += 28
        if any(kw in text_lower for kw in ['algorithm', 'sorting algorithm', 'binary search', 'depth-first search',
                                            'breadth-first search', 'dijkstra', 'hash table', 'binary tree',
                                            'recursion', 'memoization']):
            scores[Domain.COMPUTER_SCIENCE] += 18

        # 哲学 高優先度キーワード
        if any(kw in text_lower for kw in ['modus ponens', 'modus tollens', 'syllogism', 'valid argument', 'sound argument',
                                            'epistemology', 'ontology', 'a priori', 'a posteriori',
                                            'utilitarianism', 'kantian ethics', 'deontology']):
            scores[Domain.PHILOSOPHY] += 25

        # グラフ理論 追加キーワード
        if any(kw in text_lower for kw in ['chromatic number', 'graph coloring', 'planar graph', 'four color',
                                            'hamiltonian cycle', 'eulerian circuit', 'spanning tree', 'minimum spanning',
                                            'maximum flow', 'bipartite graph', 'complete graph k_']):
            scores[Domain.GRAPH_THEORY] += 22
        
        # 文字列操作の高優先度キーワード（string_lengthバグ修正）
        # "length of" は "string" が近くにある場合のみ文字列ドメインと判定
        import re as _re
        _string_len_hit = (
            'string length' in text_lower
            or 'character count' in text_lower
            or _re.search(r'length of (?:the |a |this )?string', text_lower)
            or _re.search(r'length of ["\']', text_lower)
            or (_re.search(r'length of', text_lower) and 'string' in text_lower)
        )
        if _string_len_hit:
            scores[Domain.STRING] += 30
        if 'palindrome' in text_lower or 'reverse string' in text_lower:
            scores[Domain.STRING] += 25
        
        # キーワードベースの検出
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # 記号は高スコア
                    if len(keyword) <= 2 and not keyword.isalpha():
                        scores[domain] += 5
                    else:
                        scores[domain] += 1
        
        # 数式パターン検出（算術）
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text):
            scores[Domain.ARITHMETIC] += 3
        # Layer A: べき乗パターン検出（算術ドメインを強化）
        if re.search(r'\d+\s*(?:\^|\*\*)\s*\d+', text):
            scores[Domain.ARITHMETIC] += 8
        
        # 論理記号検出（primeなどの数論キーワードがない場合のみ）
        if scores[Domain.NUMBER_THEORY] < 10:
            if any(sym in text for sym in ["->", "→", "&", "|", "~", "¬", "□", "◇"]):
                if any(sym in text for sym in ["[]", "<>", "□", "◇"]):
                    scores[Domain.LOGIC_MODAL] += 10
                else:
                    scores[Domain.LOGIC_PROPOSITIONAL] += 10
        
        best_domain = max(scores, key=scores.get)
        
        if scores[best_domain] == 0:
            return Domain.UNKNOWN
        
        return best_domain
    
    def _detect_answer_schema(self, text: str, domain: Domain, task: TaskType) -> AnswerSchema:
        """Answer Schema検出（ドメイン優先）"""
        # ドメインからの推測（最優先）
        if domain == Domain.MULTIPLE_CHOICE:
            return AnswerSchema.OPTION_LABEL
        elif domain in [Domain.LOGIC_PROPOSITIONAL, Domain.LOGIC_MODAL]:
            return AnswerSchema.BOOLEAN
        elif domain == Domain.CHESS:
            return AnswerSchema.MOVE_SEQUENCE
        elif domain in [Domain.PROBABILITY, Domain.GEOMETRY]:
            # 確率・幾何は小数が多い
            return AnswerSchema.DECIMAL
        elif domain == Domain.LINEAR_ALGEBRA:
            # 線形代数は小数（行列式など）または式
            if "determinant" in text.lower() or "dot product" in text.lower() or "eigenvalue" in text.lower():
                return AnswerSchema.DECIMAL
            return AnswerSchema.EXPRESSION
        elif domain == Domain.CALCULUS:
            # 微積分は式（導関数など）または小数（極限など）
            if "derivative" in text.lower() or "integral" in text.lower():
                return AnswerSchema.EXPRESSION
            if "limit" in text.lower():
                return AnswerSchema.DECIMAL
            return AnswerSchema.EXPRESSION
        
        # パターンマッチング
        for schema, patterns in self.answer_schema_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return schema
        
        # taskベースの推定
        if task == TaskType.DECIDE:
            return AnswerSchema.BOOLEAN
        elif task in [TaskType.COMPUTE, TaskType.COUNT]:
            # ドメインによって分岐
            if domain in [Domain.PROBABILITY, Domain.GEOMETRY]:
                return AnswerSchema.DECIMAL
            else:
                return AnswerSchema.INTEGER
        elif task == TaskType.FIND:
            if domain == Domain.ALGEBRA:
                return AnswerSchema.EXPRESSION
            elif domain in [Domain.PROBABILITY, Domain.GEOMETRY]:
                return AnswerSchema.DECIMAL
            else:
                return AnswerSchema.INTEGER
        
        # ドメインベースの推測
        if domain in [Domain.ARITHMETIC, Domain.COMBINATORICS, Domain.NUMBER_THEORY]:
            return AnswerSchema.INTEGER
        elif domain == Domain.ALGEBRA:
            return AnswerSchema.EXPRESSION
        elif domain in [Domain.PROBABILITY, Domain.GEOMETRY]:
            return AnswerSchema.DECIMAL
        
        # デフォルト
        return AnswerSchema.INTEGER
    
    def _extract_options(self, text: str) -> List[str]:
        """選択肢抽出"""
        options = []
        
        # "Answer Choices:" パターン
        if "Answer Choices:" in text:
            # A. ... B. ... C. ... パターン
            matches = re.findall(r'([A-Z])\.\s+([^\n]+)', text)
            for label, content in matches:
                options.append(f"{label}. {content.strip()}")
        
        return options
    
    def _extract_entities(self, text: str, domain: Domain) -> List[Entity]:
        """エンティティ抽出（数式・論理式・数値）"""
        entities = []
        
        # 組み合わせパターン抽出（最優先）
        if domain == Domain.COMBINATORICS:
            # P(n, r) または C(n, r) パターン
            comb_matches = re.findall(r'[PC]\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
            if comb_matches:
                for n, r in comb_matches:
                    entities.append(Entity(type="number", value=int(n), name="n"))
                    entities.append(Entity(type="number", value=int(r), name="r"))
                return entities  # 早期リターン
            
            # nCr, nPr パターン
            npr_matches = re.findall(r'(\d+)[Pp](\d+)', text)
            ncr_matches = re.findall(r'(\d+)[Cc](\d+)', text)
            if npr_matches or ncr_matches:
                for n, r in (npr_matches + ncr_matches):
                    entities.append(Entity(type="number", value=int(n), name="n"))
                    entities.append(Entity(type="number", value=int(r), name="r"))
                return entities  # 早期リターン
            
            # "arrange n items from m" パターン
            arrange_matches = re.findall(r'arrange\s+(\d+)\s+items?\s+from\s+(\d+)', text, re.IGNORECASE)
            if arrange_matches:
                for r, n in arrange_matches:  # 注: 順序が逆
                    entities.append(Entity(type="number", value=int(n), name="n"))
                    entities.append(Entity(type="number", value=int(r), name="r"))
                return entities
            
            # "choose r items from n" パターン
            choose_matches = re.findall(r'choose\s+(\d+)\s+items?\s+from\s+(\d+)', text, re.IGNORECASE)
            if choose_matches:
                for r, n in choose_matches:  # 注: 順序が逆
                    entities.append(Entity(type="number", value=int(n), name="n"))
                    entities.append(Entity(type="number", value=int(r), name="r"))
                return entities

            # ────────────────────────────────────────────────────────────
            # Layer B: "all permutations of N" / "permutations of N elements"
            # Fix [07]: r が明示されない全順列 → r=n を推定
            # ────────────────────────────────────────────────────────────
            all_perm_match = re.search(
                r'(?:all\s+)?permutations?\s+of\s+(\d+)\s*(?:elements?|items?|objects?|things?|letters?)?',
                text, re.IGNORECASE
            )
            if all_perm_match:
                n_val = int(all_perm_match.group(1))
                entities.append(Entity(type="number", value=n_val, name="n"))
                entities.append(Entity(type="number", value=n_val, name="r"))  # r=n (P(n,n))
                return entities
        
        # 階乗パターン抽出（数論ドメイン、組み合わせより後）
        if domain == Domain.NUMBER_THEORY:
            # "n!" パターン
            factorial_matches = re.findall(r'(\d+)!', text)
            if factorial_matches:
                for match in factorial_matches:
                    entities.append(Entity(type="number", value=int(match), name="n"))
                return entities  # 早期リターン
            
            # "n factorial" パターン
            factorial_matches2 = re.findall(r'(\d+)\s+factorial', text, re.IGNORECASE)
            if factorial_matches2:
                for match in factorial_matches2:
                    entities.append(Entity(type="number", value=int(match), name="n"))
                return entities
        
        # 論理式抽出（logicドメインの場合）
        if domain in [Domain.LOGIC_PROPOSITIONAL, Domain.LOGIC_MODAL]:
            # 論理記号を含む部分文字列を探す
            logic_symbols = ['->',  '→', '&', '∧', '|', '∨', '~', '¬', '[]', '<>', '□', '◇']
            
            # 論理変数パターン (p, q, r等)
            var_pattern = r'\b[p-zP-Z]\b'
            
            # 論理式候補を抽出
            # "Is p -> p" → "p -> p"
            # "Is (p & q) -> p" → "(p & q) -> p"
            formula_patterns = [
                # "Is ... tautology/valid?" パターン
                r'Is\s+([^?]+?)\s+(?:a\s+)?(?:tautology|valid|satisfiable)',
                # "... tautology?" パターン
                r'([^?]+?)\s+(?:tautology|valid|satisfiable)',
                # 論理記号を含む部分（最後の手段）
                r'([p-z\s\(\)&|~\->\[\]<>□◇→∧∨¬]+)',
            ]
            
            for pattern in formula_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    clean_formula = match.strip()
                    # 論理記号を含むかチェック
                    if any(sym in clean_formula for sym in logic_symbols):
                        # 論理変数も含むかチェック
                        if re.search(var_pattern, clean_formula):
                            entities.append(Entity(type="formula", value=clean_formula))
                            # 論理変数（atoms）も抽出
                            atoms = re.findall(var_pattern, clean_formula)
                            unique_atoms = list(set(atoms))
                            if unique_atoms:
                                entities.append(Entity(type="atoms", value=unique_atoms))
                            break
                if entities:
                    break
        
        # ────────────────────────────────────────────────────────────────
        # Layer A: べき乗パターン抽出（arithmetic）
        # "2^10", "2**10", "2 to the power of 10" など
        # Fix [06]: ^/** を relations として保存し base/exponent エンティティを生成
        # ────────────────────────────────────────────────────────────────
        if domain in [Domain.ARITHMETIC, Domain.ALGEBRA]:
            # "N^M" / "N**M"
            power_match = re.search(
                r'(\d+(?:\.\d+)?)\s*(?:\^|\*\*)\s*(\d+(?:\.\d+)?)', text
            )
            if not power_match:
                # "N to the (power of) M"
                power_match2 = re.search(
                    r'(\d+(?:\.\d+)?)\s+to\s+(?:the\s+)?(?:power\s+(?:of\s+))?(\d+(?:\.\d+)?)',
                    text, re.IGNORECASE
                )
                if power_match2:
                    power_match = power_match2

            if power_match:
                base_val = power_match.group(1)
                exp_val  = power_match.group(2)
                try:
                    base_num = int(base_val) if '.' not in base_val else float(base_val)
                    exp_num  = int(exp_val)  if '.' not in exp_val  else float(exp_val)
                    entities.append(Entity(type="number", name="base",     value=base_num))
                    entities.append(Entity(type="number", name="exponent", value=exp_num))
                    return entities   # 早期リターン：表現が確定したので以降不要
                except ValueError:
                    pass

        # 数式抽出（arithmeticの場合）
        if not entities and domain in [Domain.ARITHMETIC, Domain.ALGEBRA]:
            # 数式パターン: "1 + 1", "5 * 6" 等
            # "What is" や "Calculate" の後ろから抽出
            expr_patterns = [
                # Layer A: ^ も許可してべき乗式を式文字列として取れるようにする
                r'(?:what is|calculate|compute|evaluate)\s+([\d\+\-\*\/\(\)\.\s\^]+?)(?:\?|$)',
                r'([\d\+\-\*\/\(\)\.\^]+)',  # 単純な数式（べき乗含む）
            ]

            for pattern in expr_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    clean_expr = match.strip()
                    # 演算子を含むかチェック（^ も演算子として扱う）
                    if any(op in clean_expr for op in ['+', '-', '*', '/', '(', ')', '^']):
                        # ^ を ** に変換して Python 評価可能にする
                        clean_expr = clean_expr.replace('^', '**')
                        entities.append(Entity(type="expression", value=clean_expr))
                        break  # 最初の式のみ
                if entities:
                    break
        
        # ─── SymPy solver 用: 数式文字列エンティティの抽出 ───────────────
        # 微積分: "derivative/integral of <expr>" → Entity(name="expr")
        if domain == Domain.CALCULUS and not any(e.name == "expr" for e in entities):
            # 微分
            diff_match = re.search(
                r'(?:derivative|differentiate|d/dx?)\s+(?:of\s+)?([\w\s\^\*\+\-\(\)\/]+?)(?:\s+with\s+respect|\s+at\s+x\s*=|\?|$)',
                text, re.IGNORECASE
            )
            if diff_match:
                expr_str = diff_match.group(1).strip().rstrip(', ')
                entities.append(Entity(type="expression", name="expr", value=expr_str))

            # 積分
            int_match = re.search(
                r'(?:integral|integrate)\s+(?:of\s+)?([\w\s\^\*\+\-\(\)\/]+?)(?:\s+from\s+|\s+with\s+|\s+dx|\?|$)',
                text, re.IGNORECASE
            )
            if int_match and not any(e.name == "expr" for e in entities):
                expr_str = int_match.group(1).strip().rstrip(', ')
                entities.append(Entity(type="expression", name="expr", value=expr_str))

            # 積分の下限/上限
            bounds_match = re.search(r'from\s+([\d\.\-]+)\s+to\s+([\d\.inf]+)', text, re.IGNORECASE)
            if bounds_match:
                try:
                    lo_val = float(bounds_match.group(1))
                    hi_str = bounds_match.group(2)
                    hi_val = float('inf') if 'inf' in hi_str.lower() else float(hi_str)
                    entities.append(Entity(type="number", name="lo", value=lo_val))
                    entities.append(Entity(type="number", name="hi", value=hi_val))
                except ValueError:
                    pass

            # 変数名（デフォルト "x"）
            if not any(e.name == "variable" for e in entities):
                var_match = re.search(r'with\s+respect\s+to\s+([a-z])', text, re.IGNORECASE)
                if var_match:
                    entities.append(Entity(type="symbol", name="variable", value=var_match.group(1)))
                else:
                    entities.append(Entity(type="symbol", name="variable", value="x"))

        # 代数: "solve <equation>" → Entity(name="equation")
        # domain == ALGEBRA or 問題文に solve/roots キーワードがある場合
        _has_solve_kw = any(kw in text.lower() for kw in ['solve', 'roots', 'solutions', 'zeros', 'find x'])
        if (domain in (Domain.ALGEBRA, Domain.ARITHMETIC, Domain.UNKNOWN) or _has_solve_kw) \
                and not any(e.name == "equation" for e in entities):
            eq_match = re.search(
                r'(?:solve|find(?:\s+the\s+roots?\s+of)?|what\s+(?:are|is)\s+the\s+(?:roots?|solutions?|zeros?)(?:\s+of)?)\s+([\w\s\^\*\+\-\(\)\/\=]+?)(?:\?|$)',
                text, re.IGNORECASE
            )
            if eq_match:
                eq_str = eq_match.group(1).strip()
                # 方程式として有効か確認（=か変数を含む）
                if '=' in eq_str or re.search(r'[a-z]', eq_str, re.IGNORECASE):
                    entities.append(Entity(type="equation", name="equation", value=eq_str))
                    entities.append(Entity(type="symbol", name="variable", value="x"))

        # 数値抽出（一般）
        if not entities:
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            for num in numbers[:5]:  # 最大5個
                try:
                    if '.' in num:
                        value = float(num)
                    else:
                        value = int(num)
                    entities.append(Entity(type="number", value=value))
                except:
                    pass
        
        return entities
    
    def _extract_constraints(self, text: str, domain: Domain) -> List[Constraint]:
        """制約抽出"""
        constraints = []
        
        # 等式パターン
        eq_matches = re.findall(r'(\w+)\s*=\s*(\d+)', text)
        for var, value in eq_matches:
            constraints.append(Constraint(
                type="equals",
                var=var,
                rhs=int(value)
            ))
        
        # 範囲パターン
        range_pattern = r'(\w+)\s*(?:between|from)\s*(\d+)\s*(?:to|and)\s*(\d+)'
        range_matches = re.findall(range_pattern, text, re.IGNORECASE)
        for var, min_val, max_val in range_matches:
            constraints.append(Constraint(
                type="range",
                var=var,
                min=int(min_val),
                max=int(max_val)
            ))
        
        return constraints
    
    def _build_query(self, text: str, task: TaskType) -> Query:
        """クエリ構築"""
        # プロパティ検出
        property_map = {
            "smallest": "minimal",
            "largest": "maximal",
            "minimum": "minimal",
            "maximum": "maximal",
            "least": "minimal",
            "greatest": "maximal"
        }
        
        query_property = "value"  # デフォルト
        for keyword, prop in property_map.items():
            if keyword in text:
                query_property = prop
                break
        
        return Query(
            type=task.value,
            property=query_property
        )
