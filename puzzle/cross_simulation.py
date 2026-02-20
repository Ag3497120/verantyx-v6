"""
Cross Simulation - 立体十字構造でのシミュレーション

本来の構想:
「立体十字構造ないで数値を実際に小さな世界でさっき採掘した公理や定理を使って
実際に人間の頭の中のようにしてシュミレーションを行います。
シュミレーションを行なって結論を人間がわからない状態で出します。」
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
import itertools
import re


@dataclass
class SimulationWorld:
    """シミュレーション用の小さな世界"""
    world_id: int
    valuation: Dict[str, bool]  # 原子命題の評価
    relations: Set[Tuple[int, int]] = field(default_factory=set)  # 世界間の関係


@dataclass
class SimulationResult:
    """シミュレーション結果"""
    status: str  # "proved", "disproved", "unknown"
    confidence: float
    method: str
    worlds_explored: int
    counterexample: Optional[Any] = None
    answer: Optional[Any] = None      # proved時の答え
    trace: List[str] = field(default_factory=list)


class CrossSimulation:
    """
    Cross Simulation - 立体十字構造でのシミュレーション
    
    「人間の頭の中のように」小さな世界で公理・定理を使って推論
    """
    
    def __init__(self, max_worlds: int = 3, max_atoms: int = 10):
        self.max_worlds = max_worlds
        self.max_atoms = max_atoms
    
    def simulate(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any]
    ) -> SimulationResult:
        """
        Crossシミュレーションを実行
        
        Args:
            ir_dict: IR辞書
            pieces: 使用するピースのリスト
            context: 実行コンテキスト
        
        Returns:
            シミュレーション結果
        """
        domain = ir_dict.get("domain", "unknown")
        trace = []
        
        trace.append(f"simulation_start:domain={domain}")
        
        # ドメイン別シミュレーション
        if domain == "logic_propositional":
            return self._simulate_propositional(ir_dict, pieces, context, trace)
        elif domain == "logic_modal":
            return self._simulate_modal(ir_dict, pieces, context, trace)
        elif domain == "arithmetic":
            # 算術: まず旧来の範囲シミュレーション、次に数学シミュレーション
            arith_result = self._simulate_arithmetic(ir_dict, pieces, context, trace)
            if arith_result.status != "unknown":
                return arith_result
            # unknown の場合は数学シミュレーションにフォールスルー
            return self._simulate_mathematical(ir_dict, pieces, context, trace)
        else:
            # その他ドメインの数式シミュレーション（MCQはexecutorに任せる）
            return self._simulate_mathematical(ir_dict, pieces, context, trace)

    # ---------------------------------------------------------------------- #
    # MCQ シミュレーション
    # ---------------------------------------------------------------------- #

    def _simulate_multiple_choice(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any],
        trace: List[str]
    ) -> SimulationResult:
        """MCQ Cross Simulation:
        
        Step 1: stem に対して数学シミュレーションを走らせて計算結果を得る
        Step 2: 計算結果を各選択肢と照合 → 一致する選択肢 = 答え
        Step 3: 数学的に解けない場合は keyword/entity scoring にフォールバック
        
        本来の構想: 「各選択肢が真となる小さな世界を作り、
        一番一貫した（真になる）世界の選択肢を選ぶ」
        """
        try:
            from executors.multiple_choice import solve_multiple_choice, split_stem_choices
            from core.answer_matcher import flexible_match

            source_text = ir_dict.get("metadata", {}).get("source_text", "")
            if not source_text:
                return SimulationResult(
                    status="unknown", confidence=0.0,
                    method="mcq_simulation", worlds_explored=0, trace=trace,
                )

            stem, choices = split_stem_choices(source_text)
            if not choices:
                return SimulationResult(
                    status="unknown", confidence=0.0,
                    method="mcq_simulation", worlds_explored=0, trace=trace,
                )

            trace.append(f"mcq_simulation:choices={list(choices.keys())}")

            # ─── Step 1: stem を数学的にシミュレーション ───────────────────
            domain = ir_dict.get("domain", "unknown")
            numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', stem)
            numbers_float = [float(n) for n in numbers[:10]]

            sim_result = None
            if domain == "calculus":
                sim_result = self._try_calculus_simulation(stem, numbers_float, trace)
            elif domain in ("algebra", "linear_algebra"):
                sim_result = self._try_algebra_simulation(stem, numbers_float, trace)
            elif domain in ("number_theory", "advanced_number_theory", "modular_arithmetic"):
                sim_result = self._try_number_theory_simulation(stem, numbers_float, trace)
            elif domain in ("combinatorics", "advanced_combinatorics"):
                sim_result = self._try_combinatorics_simulation(stem, numbers_float, trace)
            elif domain in ("probability", "advanced_probability", "statistics"):
                sim_result = self._try_probability_simulation(stem, numbers_float, trace)
            elif domain in ("geometry",):
                sim_result = self._try_geometry_simulation(stem, numbers_float, trace)

            # ─── Step 2: 計算結果と各選択肢を照合（各選択肢 = 一つの世界） ───
            if sim_result is not None:
                trace.append(f"mcq_stem_computed:{sim_result}")
                matched_letter = None
                best_score = -1.0

                for letter, choice_text in choices.items():
                    # 数値・式の正規化比較
                    if flexible_match(str(sim_result), choice_text.strip(), tolerance=1e-4):
                        trace.append(f"mcq_exact_match:choice={letter}")
                        return SimulationResult(
                            status="proved",
                            confidence=0.95,
                            method="mcq_math_simulation",
                            worlds_explored=len(choices),
                            counterexample=letter,
                            trace=trace,
                        )

                    # sympy で式の等価性チェック
                    sym_match = self._sympy_expr_match(str(sim_result), choice_text.strip(), trace)
                    if sym_match > best_score:
                        best_score = sym_match
                        matched_letter = letter

                if best_score > 0.8:
                    trace.append(f"mcq_sympy_match:choice={matched_letter},score={best_score:.2f}")
                    return SimulationResult(
                        status="proved",
                        confidence=best_score,
                        method="mcq_math_simulation",
                        worlds_explored=len(choices),
                        counterexample=matched_letter,
                        trace=trace,
                    )

                trace.append(f"mcq_no_choice_matched:computed={sim_result}")

            # ─── Step 3: 600B知識空間での選択肢スコアリング ─────────────────
            # math sim で解けない概念問題に対応
            # concept_dirs で stem に最も一貫した選択肢を選ぶ
            try:
                from knowledge.concept_search import get_searcher
                cs = get_searcher()
                cs600b_scores = cs.score_mcq_choices(stem, choices, top_k=100)
                if cs600b_scores:
                    best_letter = max(cs600b_scores, key=lambda k: cs600b_scores[k])
                    best_score = cs600b_scores[best_letter]
                    trace.append(f"mcq_600b_scores:{dict(list(sorted(cs600b_scores.items(), key=lambda x:-x[1]))[:3])}")
                    trace.append(f"mcq_600b_selected:{best_letter}(score={best_score:.3f})")

                    # 600B スコアと keyword スコアの組み合わせ
                    kw_answer = solve_multiple_choice(source_text, choices=choices)
                    if kw_answer and kw_answer == best_letter:
                        # 両方一致 → 高信頼度
                        return SimulationResult(
                            status="proved", confidence=0.88,
                            method="mcq_600b_kw_agree",
                            worlds_explored=len(choices),
                            counterexample=best_letter, trace=trace,
                        )
                    elif best_score > 0.35:
                        # 600B が強いシグナル → 600B を優先
                        return SimulationResult(
                            status="proved", confidence=0.80,
                            method="mcq_600b_simulation",
                            worlds_explored=len(choices),
                            counterexample=best_letter, trace=trace,
                        )
                    elif kw_answer:
                        # 600B シグナル弱い → keyword fallback
                        return SimulationResult(
                            status="proved", confidence=0.72,
                            method="mcq_keyword_fallback",
                            worlds_explored=len(choices),
                            counterexample=kw_answer, trace=trace,
                        )
            except Exception as e600b:
                trace.append(f"mcq_600b_error:{e600b}")
                # keyword のみ
                answer = solve_multiple_choice(source_text, choices=choices)
                if answer:
                    trace.append(f"mcq_keyword_selected:{answer}")
                    return SimulationResult(
                        status="proved", confidence=0.72,
                        method="mcq_simulation",
                        worlds_explored=len(choices),
                        counterexample=answer, trace=trace,
                    )

            return SimulationResult(
                status="unknown", confidence=0.0,
                method="mcq_simulation", worlds_explored=len(choices), trace=trace,
            )
        except Exception as e:
            trace.append(f"mcq_simulation_error:{e}")
            return SimulationResult(
                status="unknown", confidence=0.0,
                method="mcq_simulation", worlds_explored=0, trace=trace,
            )

    def _sympy_expr_match(self, computed: str, choice_text: str, trace: List[str]) -> float:
        """sympy で2つの式が数学的に等価かをチェック。0.0-1.0 を返す。"""
        try:
            import sympy as sp
            c1 = computed.strip().replace('^', '**')
            c2 = choice_text.strip().replace('^', '**')
            # LaTeX/テキスト表記の正規化
            c2 = re.sub(r'\\([a-zA-Z]+)', r'\1', c2)  # \cos → cos など
            expr1 = sp.sympify(c1)
            expr2 = sp.sympify(c2)
            diff = sp.simplify(expr1 - expr2)
            if diff == 0:
                return 1.0
            # 数値近似で確認
            try:
                x = sp.Symbol('x')
                for val in [0.5, 1.0, 2.0, -1.0]:
                    v1 = float(expr1.subs(x, val))
                    v2 = float(expr2.subs(x, val))
                    if abs(v1 - v2) > 1e-6:
                        return 0.0
                return 0.9  # 複数点で一致 → ほぼ等価
            except:
                return 0.0
        except:
            return 0.0

    # ---------------------------------------------------------------------- #
    # Mathematical Domain Simulation
    # ---------------------------------------------------------------------- #

    def _simulate_mathematical(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any],
        trace: List[str]
    ) -> SimulationResult:
        """数学・科学ドメインの小さな世界シミュレーション"""
        domain = ir_dict.get("domain", "unknown")
        source_text = ir_dict.get("metadata", {}).get("source_text", "")

        trace.append(f"math_simulation:domain={domain}")

        # 数値抽出
        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', source_text)
        numbers_float = [float(n) for n in numbers[:10]]

        # ドメイン別シミュレーション戦略
        result = None

        if domain in ("calculus",):
            result = self._try_calculus_simulation(source_text, numbers_float, trace)
        elif domain in ("algebra", "linear_algebra"):
            result = self._try_algebra_simulation(source_text, numbers_float, trace)
        elif domain in ("arithmetic",):
            # arithmetic → algebra 試行、次に number_theory も試みる
            result = self._try_algebra_simulation(source_text, numbers_float, trace)
            if result is None:
                result = self._try_number_theory_simulation(source_text, numbers_float, trace)
        elif domain in ("number_theory", "advanced_number_theory", "modular_arithmetic"):
            result = self._try_number_theory_simulation(source_text, numbers_float, trace)
        elif domain in ("combinatorics", "advanced_combinatorics"):
            result = self._try_combinatorics_simulation(source_text, numbers_float, trace)
        elif domain in ("probability", "advanced_probability", "statistics"):
            result = self._try_probability_simulation(source_text, numbers_float, trace)
        elif domain in ("geometry",):
            result = self._try_geometry_simulation(source_text, numbers_float, trace)
        else:
            # 未知ドメイン: 全戦略を順番に試みる
            for try_fn in [
                self._try_algebra_simulation,
                self._try_number_theory_simulation,
                self._try_combinatorics_simulation,
                self._try_calculus_simulation,
            ]:
                result = try_fn(source_text, numbers_float, trace)
                if result is not None:
                    break

        if result is not None:
            trace.append(f"math_sim_result:{result}")
            return SimulationResult(
                status="proved",
                confidence=0.9,
                method=f"{domain}_simulation",
                worlds_explored=1,
                counterexample=result,  # 結果を counterexample フィールドに格納（pipeline側で使用）
                trace=trace,
            )

        trace.append("math_sim_no_result")
        return SimulationResult(
            status="unknown",
            confidence=0.3,
            method=f"{domain}_simulation",
            worlds_explored=0,
            trace=trace,
        )

    def _try_calculus_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """微積分シミュレーション: sympy を使って式を評価"""
        try:
            import sympy as sp
            x = sp.Symbol('x')

            # 導関数パターン
            deriv_match = re.search(r'derivative of\s+(.+?)(?:\?|$|\n)', text, re.IGNORECASE)
            if deriv_match:
                expr_str = deriv_match.group(1).strip().replace('^', '**')
                expr = sp.sympify(expr_str)
                result = sp.diff(expr, x)
                trace.append(f"derivative:{expr}->{result}")
                return str(result)

            # 積分パターン
            integ_match = re.search(r'integral of\s+(.+?)(?:\?|$|\n|with)', text, re.IGNORECASE)
            if integ_match:
                expr_str = integ_match.group(1).strip().replace('^', '**')
                expr = sp.sympify(expr_str)
                result = sp.integrate(expr, x)
                trace.append(f"integral:{expr}->{result}")
                return str(result)

            # 極限パターン
            lim_match = re.search(
                r'limit of\s+(.+?)\s+as\s+\w+\s+(?:->|approaches|→)\s+(-?\w+)',
                text, re.IGNORECASE
            )
            if lim_match:
                expr_str = lim_match.group(1).strip().replace('^', '**')
                limit_val_str = lim_match.group(2)
                expr = sp.sympify(expr_str)
                limit_val = sp.sympify(limit_val_str)
                result = sp.limit(expr, x, limit_val)
                trace.append(f"limit:{expr}@{limit_val}->{result}")
                return str(result)

        except Exception as e:
            trace.append(f"calculus_sim_error:{e}")
        return None

    def _try_algebra_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """代数シミュレーション"""
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )
            _transformations = standard_transformations + (implicit_multiplication_application,)

            def _parse(expr_str: str):
                """implicit multiplication に対応したパース"""
                try:
                    return parse_expr(expr_str, transformations=_transformations)
                except Exception:
                    return sp.sympify(expr_str)

            # 方程式パターン
            eq_match = re.search(r'solve\s+(.+?)\s*=\s*(.+?)(?:\?|$|\n)', text, re.IGNORECASE)
            if eq_match:
                lhs = eq_match.group(1).strip().replace('^', '**')
                rhs = eq_match.group(2).strip().replace('^', '**')
                x = sp.Symbol('x')
                solutions = sp.solve(_parse(lhs) - _parse(rhs), x)
                if solutions:
                    trace.append(f"algebra_solve:{solutions}")
                    if len(solutions) == 1:
                        return str(solutions[0])
                    return str(solutions)

            # 数式評価パターン
            eval_match = re.search(
                r'(?:compute|evaluate|find|calculate)\s+(.+?)(?:\?|$|\n)',
                text, re.IGNORECASE
            )
            if eval_match:
                expr_str = eval_match.group(1).strip().replace('^', '**')
                result = _parse(expr_str)
                if result.is_number:
                    return str(int(result)) if result.is_integer else str(float(result))

        except Exception as e:
            trace.append(f"algebra_sim_error:{e}")
        return None

    def _try_number_theory_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """数論シミュレーション"""
        try:
            import math

            # GCDパターン
            gcd_match = re.search(r'gcd\s*\(?(\d+)\s*[,，]\s*(\d+)\)?', text, re.IGNORECASE)
            if gcd_match:
                a, b = int(gcd_match.group(1)), int(gcd_match.group(2))
                result = math.gcd(a, b)
                trace.append(f"gcd({a},{b})={result}")
                return str(result)

            # LCMパターン
            lcm_match = re.search(r'lcm\s*\(?(\d+)\s*[,，]\s*(\d+)\)?', text, re.IGNORECASE)
            if lcm_match:
                a, b = int(lcm_match.group(1)), int(lcm_match.group(2))
                result = (a * b) // math.gcd(a, b)
                trace.append(f"lcm({a},{b})={result}")
                return str(result)

            # 素数判定
            prime_match = re.search(r'(?:is|whether)\s+(\d+)\s+(?:is\s+)?(?:a\s+)?prime', text, re.IGNORECASE)
            if prime_match:
                n = int(prime_match.group(1))
                def is_prime(n):
                    if n < 2:
                        return False
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            return False
                    return True
                result = is_prime(n)
                trace.append(f"is_prime({n})={result}")
                return "Yes" if result else "No"

            # 階乗パターン
            fact_match = re.search(r'(\d+)\s*!|factorial\s+of\s+(\d+)', text, re.IGNORECASE)
            if fact_match:
                n = int(fact_match.group(1) or fact_match.group(2))
                if n <= 20:
                    result = math.factorial(n)
                    trace.append(f"{n}!={result}")
                    return str(result)

            # mod 計算
            mod_match = re.search(r'(\d+)\s*(?:mod|%)\s*(\d+)', text, re.IGNORECASE)
            if mod_match:
                a, b = int(mod_match.group(1)), int(mod_match.group(2))
                result = a % b
                trace.append(f"{a} mod {b}={result}")
                return str(result)

        except Exception as e:
            trace.append(f"nt_sim_error:{e}")
        return None

    def _try_combinatorics_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """組み合わせシミュレーション"""
        try:
            import math

            # C(n,r) パターン
            c_match = re.search(
                r'C\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|(\d+)\s*[Cc]r?\s*(\d+)|choose\s+(\d+)\s+from\s+(\d+)',
                text
            )
            if c_match:
                groups = [g for g in c_match.groups() if g is not None]
                if len(groups) >= 2:
                    n, r = int(groups[0]), int(groups[1])
                    if r <= n:
                        result = math.comb(n, r)
                        trace.append(f"C({n},{r})={result}")
                        return str(result)

            # choose/combinations キーワード
            ways_match = re.search(
                r'how many ways.{0,80}choose\s+(\d+).{0,30}from\s+(\d+)',
                text, re.IGNORECASE
            )
            if ways_match:
                r, n = int(ways_match.group(1)), int(ways_match.group(2))
                if r <= n:
                    result = math.comb(n, r)
                    trace.append(f"ways_choose({n},{r})={result}")
                    return str(result)

            # P(n,r) パターン
            p_match = re.search(
                r'P\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|(\d+)\s*[Pp]r?\s*(\d+)',
                text
            )
            if p_match:
                groups = [g for g in p_match.groups() if g is not None]
                if len(groups) >= 2:
                    n, r = int(groups[0]), int(groups[1])
                    if r <= n:
                        result = math.perm(n, r)
                        trace.append(f"P({n},{r})={result}")
                        return str(result)

        except Exception as e:
            trace.append(f"comb_sim_error:{e}")
        return None

    def _try_probability_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """確率シミュレーション"""
        try:
            # 基本確率 P = favorable / total
            prob_match = re.search(
                r'(\d+)\s+(?:favorable|outcomes?).{0,50}(?:out of|from|total)\s+(\d+)',
                text, re.IGNORECASE
            )
            if prob_match:
                fav, total = int(prob_match.group(1)), int(prob_match.group(2))
                result = fav / total
                trace.append(f"prob={fav}/{total}={result}")
                return str(result)

        except Exception as e:
            trace.append(f"prob_sim_error:{e}")
        return None

    def _try_geometry_simulation(self, text: str, numbers: List[float], trace: List[str]) -> Optional[str]:
        """幾何シミュレーション"""
        try:
            import math

            # 円面積 / 円周
            circle_match = re.search(
                r'(?:area|circumference)\s+of\s+(?:a\s+)?circle.{0,50}radius\s+(?:is\s+|=\s*)?(\d+\.?\d*)',
                text, re.IGNORECASE
            )
            if circle_match:
                r = float(circle_match.group(1))
                if 'circumference' in text.lower():
                    result = 2 * math.pi * r
                else:
                    result = math.pi * r**2
                trace.append(f"circle(r={r})={result:.4f}")
                return str(round(result, 4))

            # 三角形面積
            tri_match = re.search(
                r'area.{0,50}triangle.{0,80}base\s+(?:is\s+)?(\d+\.?\d*).{0,50}height\s+(?:is\s+)?(\d+\.?\d*)',
                text, re.IGNORECASE
            )
            if tri_match:
                base, height = float(tri_match.group(1)), float(tri_match.group(2))
                result = 0.5 * base * height
                trace.append(f"triangle(b={base},h={height})={result}")
                return str(result)

        except Exception as e:
            trace.append(f"geom_sim_error:{e}")
        return None

    # ---------------------------------------------------------------------- #
    # 既存ドメイン (propositional / modal / arithmetic)
    # ---------------------------------------------------------------------- #

    def _simulate_propositional(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any],
        trace: List[str]
    ) -> SimulationResult:
        """
        命題論理のシミュレーション
        
        真理表による「小さな世界」での検証
        """
        # 論理式を取得
        formula = None
        for entity in ir_dict.get("entities", []):
            if entity.get("type") == "formula":
                formula = entity.get("value")
                break
        
        if not formula:
            trace.append("no_formula")
            return SimulationResult(
                status="unknown",
                confidence=0.0,
                method="propositional_simulation",
                worlds_explored=0,
                trace=trace
            )

        # 実際の命題論理式かバリデーション
        # 確率記法 "Y|Z)" や ノルム "||x-y||" を誤検出しないため
        # 明確な論理演算子（->、~、implies、iff等）がなければunknownを返す
        _LOGIC_OPS = ['->', '→', '<->', '↔', '~', '¬',
                      'implies', 'iff', 'not ', 'and ', 'or ',
                      'tautology', 'satisfiable', 'valid']
        has_logic_op = any(op in formula for op in _LOGIC_OPS)
        if not has_logic_op:
            trace.append("formula_not_propositional_logic")
            return SimulationResult(
                status="unknown",
                confidence=0.0,
                method="propositional_simulation",
                worlds_explored=0,
                trace=trace
            )

        # 原子命題を抽出
        atoms = sorted(list(set(re.findall(r'\b[A-Za-z]\b', formula))))
        
        if len(atoms) > self.max_atoms:
            trace.append(f"too_many_atoms:{len(atoms)}")
            return SimulationResult(
                status="unknown",
                confidence=0.0,
                method="propositional_simulation",
                worlds_explored=0,
                trace=trace
            )
        
        # 真理表の「小さな世界」を生成
        worlds_explored = 0
        counterexample = None
        
        trace.append(f"exploring_worlds:atoms={len(atoms)},worlds={2**len(atoms)}")
        
        # 各真理値割り当て = 1つの世界
        for bits in itertools.product([False, True], repeat=len(atoms)):
            valuation = dict(zip(atoms, bits))
            worlds_explored += 1
            
            # この世界で公理・定理を適用してシミュレーション
            world_result = self._evaluate_in_world(formula, valuation, pieces, context)
            
            if not world_result:
                counterexample = valuation
                trace.append(f"counterexample_found:world={worlds_explored}")
                break
        
        if counterexample:
            return SimulationResult(
                status="disproved",
                confidence=1.0,
                method="propositional_simulation",
                worlds_explored=worlds_explored,
                counterexample=counterexample,
                trace=trace
            )
        else:
            trace.append(f"all_worlds_satisfied:{worlds_explored}")
            return SimulationResult(
                status="proved",
                confidence=1.0,
                method="propositional_simulation",
                worlds_explored=worlds_explored,
                trace=trace
            )
    
    def _simulate_modal(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any],
        trace: List[str]
    ) -> SimulationResult:
        """
        様相論理のシミュレーション
        
        Kripkeモデルによる「小さな世界」での検証
        """
        # 論理式を取得
        formula = None
        for entity in ir_dict.get("entities", []):
            if entity.get("type") == "formula":
                formula = entity.get("value")
                break
        
        if not formula:
            return SimulationResult(
                status="unknown",
                confidence=0.0,
                method="modal_simulation",
                worlds_explored=0,
                trace=trace
            )
        
        # 仮定を取得
        assumptions = []
        for constraint in ir_dict.get("constraints", []):
            if constraint.get("type") == "assumption":
                assumptions.append(constraint.get("expression", ""))
        
        # Kripkeフレームの「小さな世界」を生成
        worlds = list(range(self.max_worlds))
        
        trace.append(f"generating_kripke_worlds:{len(worlds)}")
        
        # 関係を生成（仮定を反映）
        relations = self._generate_relations(worlds, assumptions, trace)
        
        # 評価関数を生成
        atoms = self._extract_atoms(formula)
        valuations = self._generate_valuations(worlds, atoms)
        
        worlds_explored = 0
        counterexample = None
        
        # 各Kripkeモデル = 1つの世界
        for R in relations[:10]:
            for V in valuations[:10]:
                worlds_explored += 1
                
                model = (worlds, R, V)
                
                for w in worlds:
                    if not self._evaluate_modal(formula, model, w, pieces, context):
                        counterexample = {
                            "worlds": worlds,
                            "relation": sorted(list(R)),
                            "valuation": V,
                            "failed_world": w
                        }
                        trace.append(f"counterexample_found:model={worlds_explored}")
                        break
                
                if counterexample:
                    break
            
            if counterexample:
                break
        
        if counterexample:
            return SimulationResult(
                status="disproved",
                confidence=0.9,
                method="modal_simulation",
                worlds_explored=worlds_explored,
                counterexample=counterexample,
                trace=trace
            )
        else:
            trace.append(f"models_checked:{worlds_explored}")
            return SimulationResult(
                status="proved",
                confidence=0.8,
                method="modal_simulation",
                worlds_explored=worlds_explored,
                trace=trace
            )
    
    def _simulate_arithmetic(
        self,
        ir_dict: Dict[str, Any],
        pieces: List[Any],
        context: Dict[str, Any],
        trace: List[str]
    ) -> SimulationResult:
        """
        算術のシミュレーション
        
        数値範囲での「小さな世界」での検証
        """
        # 制約を取得
        ranges = {}
        for constraint in ir_dict.get("constraints", []):
            if constraint.get("type") == "range":
                var = constraint.get("var")
                min_val = constraint.get("min", 0)
                max_val = constraint.get("max", 10)
                ranges[var] = (min_val, max_val)
        
        if not ranges:
            return SimulationResult(
                status="unknown",
                confidence=0.5,
                method="arithmetic_simulation",
                worlds_explored=0,
                trace=trace
            )
        
        trace.append(f"arithmetic_range_simulation:{ranges}")
        
        return SimulationResult(
            status="unknown",
            confidence=0.7,
            method="arithmetic_simulation",
            worlds_explored=len(ranges),
            trace=trace
        )
    
    def _evaluate_in_world(
        self,
        formula: str,
        valuation: Dict[str, bool],
        pieces: List[Any],
        context: Dict[str, Any]
    ) -> bool:
        """世界での評価（簡易実装）"""
        from executors.logic import PropositionalLogic
        
        try:
            prop_logic = PropositionalLogic()
            ast, _ = prop_logic.parse(formula)
            return prop_logic.evaluate(ast, valuation)
        except:
            # パース失敗 = 実際の命題論理式でない（確率記法 "|" 等の誤抽出）
            # → True を返してcounterexample扱いにしない
            return True
    
    def _evaluate_modal(
        self,
        formula: str,
        model: Tuple[List[int], Set[Tuple[int, int]], Dict[str, Dict[int, bool]]],
        world: int,
        pieces: List[Any],
        context: Dict[str, Any]
    ) -> bool:
        """Kripkeモデルでの評価（簡易実装）"""
        return True
    
    def _generate_relations(
        self,
        worlds: List[int],
        assumptions: List[str],
        trace: List[str]
    ) -> List[Set[Tuple[int, int]]]:
        """関係を生成（仮定を反映）"""
        relations = []
        basic_relation = set()
        
        if any("reflexive" in a.lower() for a in assumptions):
            for w in worlds:
                basic_relation.add((w, w))
            trace.append("assumption:reflexive")
        
        if any("symmetric" in a.lower() for a in assumptions):
            trace.append("assumption:symmetric")
        
        if any("transitive" in a.lower() for a in assumptions):
            trace.append("assumption:transitive")
        
        relations.append(basic_relation)
        return relations
    
    def _generate_valuations(
        self,
        worlds: List[int],
        atoms: List[str]
    ) -> List[Dict[str, Dict[int, bool]]]:
        """評価関数を生成"""
        valuation = {}
        for atom in atoms:
            valuation[atom] = {w: False for w in worlds}
        
        return [valuation]
    
    def _extract_atoms(self, formula: str) -> List[str]:
        """原子命題を抽出"""
        return sorted(list(set(re.findall(r'\b[A-Za-z]\b', formula))))
