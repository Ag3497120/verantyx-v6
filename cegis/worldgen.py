"""
WorldGen - 有限モデル生成器 (Finite Model Generator)

CEGISループの「世界生成」担当。
LLM不使用で、反例探索・候補フィルタリングのための小世界を構築する。

生成できる構造:
  group          - 有限群 (Z_n, S_n, V4, D_n)
  graph          - 有限グラフ (完全, 空, パス, サイクル, 二部)
  ring           - 有限環 (Z_n)
  sequence       - 数列 (等差, 等比, フィボナッチ, 素数列)
  number         - 整数・有理数サンプル
  propositional  - 命題論理の真偽割り当て
  set            - 有限集合・部分集合
  function       - 有限集合上の関数
  permutation    - 置換
  matrix         - 有限行列サンプル
  polynomial     - 多項式サンプル
  substitution   - 変数代入の小世界 (恒等式・多項式検証用)
  finite_field   - 有限体 GF(p)
  finite_group   - 有限群 (巡回群・対称群の強化版)
  modular        - 剰余演算の世界 (合同式・Fermat小定理)
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple


@dataclass
class FiniteModel:
    """有限モデル - Crossシミュレーターが操作する「小さな世界」"""
    domain: str                                       # "group", "graph", ...
    size: int                                         # モデルのサイズ
    elements: List[Any]                               # 要素のリスト
    relations: Dict[str, Any] = field(default_factory=dict)   # 関係・演算
    properties: Dict[str, bool] = field(default_factory=dict) # 性質フラグ
    generator_params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        props = ", ".join(k for k, v in self.properties.items() if v)
        return f"FiniteModel({self.domain}, size={self.size}, [{props}])"


class WorldGenerator:
    """
    有限モデル生成器のファクトリ

    使い方:
        gen = WorldGenerator(max_size=8, max_worlds=50)
        worlds = gen.generate("group", {"type": "abelian"})
    """

    def __init__(self, max_size: int = 8, max_worlds: int = 50):
        self.max_size = max_size
        self.max_worlds = max_worlds
        self._rng = random.Random(42)  # 再現性のため固定シード

    def generate(
        self,
        domain: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[FiniteModel]:
        """ドメインに応じた有限モデルリストを生成"""
        params = params or {}
        gen_map: Dict[str, Any] = {
            "group":         self._gen_groups,
            "graph":         self._gen_graphs,
            "ring":          self._gen_rings,
            "sequence":      self._gen_sequences,
            "number":        self._gen_number_samples,
            "propositional": self._gen_truth_assignments,
            "set":           self._gen_sets,
            "function":      self._gen_functions,
            "permutation":   self._gen_permutations,
            "matrix":        self._gen_matrices,
            "polynomial":    self._gen_polynomials,
            "substitution":  self._gen_substitution,
            "finite_field":  self._gen_finite_field,
            "finite_group":  self._gen_finite_group,
            "modular":       self._gen_modular,
        }
        gen_fn = gen_map.get(domain, self._gen_number_samples)
        return list(itertools.islice(gen_fn(params), self.max_worlds))

    # ────────────────────────────────────────────────────────────────────
    # 有限群
    # ────────────────────────────────────────────────────────────────────

    def _gen_groups(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限群を生成: 巡回群 Z_n, Klein V4, 二面体群 D_n, 対称群 S_3"""

        # 巡回群 Z_n (n=2..max_size)
        for n in range(2, self.max_size + 1):
            elements = list(range(n))
            op = {(i, j): (i + j) % n for i in elements for j in elements}
            yield FiniteModel(
                domain="group", size=n, elements=elements,
                relations={"op": op, "identity": 0, "type": f"Z_{n}"},
                properties={
                    "abelian": True, "cyclic": True,
                    "order": n, "simple": self._is_prime(n),
                },
                generator_params={"type": "cyclic", "n": n},
            )

        # Klein 四元群 V4
        els = [0, 1, 2, 3]
        v4_op = {
            (0,0):0,(0,1):1,(0,2):2,(0,3):3,
            (1,0):1,(1,1):0,(1,2):3,(1,3):2,
            (2,0):2,(2,1):3,(2,2):0,(2,3):1,
            (3,0):3,(3,1):2,(3,2):1,(3,3):0,
        }
        yield FiniteModel(
            domain="group", size=4, elements=els,
            relations={"op": v4_op, "identity": 0, "type": "V4"},
            properties={"abelian": True, "cyclic": False, "order": 4, "simple": False},
            generator_params={"type": "klein4"},
        )

        # 二面体群 D_n (n=3,4,5)
        for n in range(3, 6):
            # r=回転, s=反射; 乗算テーブルは簡略化
            els_d = list(range(2 * n))  # 0..n-1=回転, n..2n-1=反射
            yield FiniteModel(
                domain="group", size=2*n, elements=els_d,
                relations={"type": f"D_{n}"},
                properties={
                    "abelian": n == 2, "cyclic": False,
                    "order": 2*n, "simple": False,
                },
                generator_params={"type": "dihedral", "n": n},
            )

        # 対称群 S_3
        s3_elements = list(itertools.permutations(range(3)))
        def s3_mul(a, b):
            return tuple(a[b[i]] for i in range(3))
        s3_op = {(i, j): s3_elements.index(s3_mul(s3_elements[i], s3_elements[j]))
                 for i in range(6) for j in range(6)}
        yield FiniteModel(
            domain="group", size=6, elements=list(range(6)),
            relations={"op": s3_op, "identity": 0, "type": "S_3"},
            properties={"abelian": False, "cyclic": False, "order": 6, "simple": False},
            generator_params={"type": "symmetric", "n": 3},
        )

    # ────────────────────────────────────────────────────────────────────
    # 有限グラフ
    # ────────────────────────────────────────────────────────────────────

    def _gen_graphs(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限グラフを生成: 完全, 空, パス, サイクル, 二部, ランダム"""
        n_range = params.get("n_range", range(3, min(self.max_size + 1, 8)))

        for n in n_range:
            verts = list(range(n))

            # 完全グラフ K_n
            edges_k = list(itertools.combinations(verts, 2))
            yield FiniteModel(
                domain="graph", size=n, elements=verts,
                relations={"edges": edges_k, "type": f"K_{n}"},
                properties={
                    "connected": True, "complete": True,
                    "planar": n <= 4, "bipartite": n <= 2,
                    "eulerian": n % 2 == 1,   # K_n: 各頂点次数 n-1
                    "hamiltonian": n >= 3,
                },
                generator_params={"type": "complete", "n": n},
            )

            # 空グラフ
            yield FiniteModel(
                domain="graph", size=n, elements=verts,
                relations={"edges": [], "type": f"empty_{n}"},
                properties={
                    "connected": n == 1, "complete": n <= 1,
                    "planar": True, "bipartite": True,
                    "eulerian": False, "hamiltonian": n <= 1,
                },
                generator_params={"type": "empty", "n": n},
            )

            # パスグラフ P_n
            path_edges = [(i, i+1) for i in range(n-1)]
            yield FiniteModel(
                domain="graph", size=n, elements=verts,
                relations={"edges": path_edges, "type": f"P_{n}"},
                properties={
                    "connected": True, "complete": n <= 2,
                    "planar": True, "bipartite": True,
                    "eulerian": n == 1, "hamiltonian": True,
                },
                generator_params={"type": "path", "n": n},
            )

            # サイクルグラフ C_n
            if n >= 3:
                cycle_edges = [(i, (i+1) % n) for i in range(n)]
                yield FiniteModel(
                    domain="graph", size=n, elements=verts,
                    relations={"edges": cycle_edges, "type": f"C_{n}"},
                    properties={
                        "connected": True, "complete": n <= 3,
                        "planar": True, "bipartite": n % 2 == 0,
                        "eulerian": True, "hamiltonian": True,
                    },
                    generator_params={"type": "cycle", "n": n},
                )

    # ────────────────────────────────────────────────────────────────────
    # 有限環
    # ────────────────────────────────────────────────────────────────────

    def _gen_rings(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限環 Z_n を生成"""
        for n in range(2, self.max_size + 1):
            elements = list(range(n))
            add = {(i, j): (i + j) % n for i in elements for j in elements}
            mul = {(i, j): (i * j) % n for i in elements for j in elements}

            # 整域チェック (零因子なし)
            integral = all(
                mul[(i, j)] != 0
                for i in elements for j in elements
                if i != 0 and j != 0
            )
            # 体チェック (全非零元が乗法逆元を持つ)
            is_field = integral and all(
                any(mul[(i, j)] == 1 % n for j in elements)
                for i in elements if i != 0
            )
            yield FiniteModel(
                domain="ring", size=n, elements=elements,
                relations={"add": add, "mul": mul, "zero": 0, "one": 1 % n, "type": f"Z_{n}"},
                properties={
                    "commutative": True,
                    "integral_domain": integral,
                    "field": is_field,
                    "prime_field": is_field and self._is_prime(n),
                },
                generator_params={"type": "Z_n", "n": n},
            )

    # ────────────────────────────────────────────────────────────────────
    # 数列
    # ────────────────────────────────────────────────────────────────────

    def _gen_sequences(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """数列を生成: 等差, 等比, フィボナッチ, 素数列, 平方数列"""
        length = params.get("length", 8)

        # 等差数列 (d=1..4, start=0..4)
        for d in range(1, 5):
            for start in range(0, 5):
                seq = [start + i * d for i in range(length)]
                yield FiniteModel(
                    domain="sequence", size=length, elements=seq,
                    relations={"type": "arithmetic", "d": d, "first": start},
                    properties={"arithmetic": True, "geometric": d == 0},
                    generator_params={"type": "arithmetic", "d": d, "start": start},
                )

        # 等比数列 (r=2,3,1/2)
        for r in [2, 3, 0.5]:
            for start in [1, 2]:
                seq = [start * (r ** i) for i in range(length)]
                yield FiniteModel(
                    domain="sequence", size=length, elements=seq,
                    relations={"type": "geometric", "r": r, "first": start},
                    properties={"arithmetic": False, "geometric": True},
                    generator_params={"type": "geometric", "r": r, "start": start},
                )

        # フィボナッチ数列
        fib = [0, 1]
        for _ in range(length - 2):
            fib.append(fib[-1] + fib[-2])
        yield FiniteModel(
            domain="sequence", size=length, elements=fib,
            relations={"type": "fibonacci"},
            properties={"arithmetic": False, "geometric": False, "recurrence": True},
            generator_params={"type": "fibonacci"},
        )

        # 素数列
        primes = [n for n in range(2, 100) if self._is_prime(n)][:length]
        yield FiniteModel(
            domain="sequence", size=len(primes), elements=primes,
            relations={"type": "primes"},
            properties={"prime": True},
            generator_params={"type": "primes"},
        )

        # 平方数列
        squares = [i * i for i in range(length)]
        yield FiniteModel(
            domain="sequence", size=length, elements=squares,
            relations={"type": "squares"},
            properties={"square": True},
            generator_params={"type": "squares"},
        )

    # ────────────────────────────────────────────────────────────────────
    # 数値サンプル
    # ────────────────────────────────────────────────────────────────────

    def _gen_number_samples(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """整数・有理数・実数サンプルを生成"""
        lo = params.get("lo", -10)
        hi = params.get("hi", 10)
        for n in range(lo, hi + 1):
            yield FiniteModel(
                domain="number", size=1, elements=[n],
                relations={"value": n},
                properties={
                    "positive": n > 0, "negative": n < 0, "zero": n == 0,
                    "even": n % 2 == 0, "odd": n % 2 != 0,
                    "prime": self._is_prime(n),
                    "perfect_square": self._is_perfect_square(n),
                },
                generator_params={"type": "integer", "value": n},
            )

        # 有理数サンプル
        for p in range(-5, 6):
            for q in range(1, 6):
                from fractions import Fraction
                frac = Fraction(p, q)
                yield FiniteModel(
                    domain="number", size=1, elements=[frac],
                    relations={"value": frac, "numerator": p, "denominator": q},
                    properties={
                        "rational": True, "integer": q == 1,
                        "positive": frac > 0, "negative": frac < 0,
                    },
                    generator_params={"type": "rational", "p": p, "q": q},
                )

    # ────────────────────────────────────────────────────────────────────
    # 命題論理の真偽割り当て
    # ────────────────────────────────────────────────────────────────────

    def _gen_truth_assignments(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """命題論理: 全ての真偽値割り当てを生成"""
        atoms = params.get("atoms", ["p", "q", "r"])
        for assignment in itertools.product([False, True], repeat=len(atoms)):
            valuation = dict(zip(atoms, assignment))
            yield FiniteModel(
                domain="propositional", size=len(atoms), elements=atoms,
                relations={"valuation": valuation},
                properties=valuation,
                generator_params={"atoms": atoms, "assignment": list(assignment)},
            )

    # ────────────────────────────────────────────────────────────────────
    # 有限集合
    # ────────────────────────────────────────────────────────────────────

    def _gen_sets(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限集合・部分集合を生成"""
        max_n = params.get("max_n", 5)
        for n in range(0, max_n + 1):
            universe = set(range(n))
            for r in range(n + 1):
                for subset_tuple in itertools.combinations(range(n), r):
                    subset = set(subset_tuple)
                    complement = universe - subset
                    yield FiniteModel(
                        domain="set", size=n, elements=list(range(n)),
                        relations={
                            "subset": subset, "complement": complement,
                            "universe": universe, "size": len(subset),
                        },
                        properties={
                            "empty": len(subset) == 0, "full": subset == universe,
                            "proper": subset != universe,
                        },
                        generator_params={"n": n, "subset": list(subset)},
                    )

    # ────────────────────────────────────────────────────────────────────
    # 関数
    # ────────────────────────────────────────────────────────────────────

    def _gen_functions(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限集合上の関数を生成"""
        n = min(params.get("n", 3), 4)  # n=4 で 4^4=256 通りまで
        domain_set = list(range(n))
        codomain = list(range(n))
        for mapping_tuple in itertools.product(codomain, repeat=n):
            fn = dict(zip(domain_set, mapping_tuple))
            injective  = len(set(mapping_tuple)) == n
            surjective = set(mapping_tuple) == set(codomain)
            yield FiniteModel(
                domain="function", size=n, elements=domain_set,
                relations={"mapping": fn},
                properties={
                    "injective": injective, "surjective": surjective,
                    "bijective": injective and surjective,
                    "identity": all(fn[i] == i for i in domain_set),
                },
                generator_params={"n": n, "mapping": list(mapping_tuple)},
            )

    # ────────────────────────────────────────────────────────────────────
    # 置換
    # ────────────────────────────────────────────────────────────────────

    def _gen_permutations(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """置換を生成"""
        n = min(params.get("n", 4), 5)
        for perm in itertools.permutations(range(n)):
            # 偶奇判定（転置回数）
            inversions = sum(
                1 for i in range(n) for j in range(i+1, n) if perm[i] > perm[j]
            )
            is_even = inversions % 2 == 0
            yield FiniteModel(
                domain="permutation", size=n, elements=list(perm),
                relations={"perm": perm, "inversions": inversions},
                properties={
                    "even": is_even, "odd": not is_even,
                    "identity": all(perm[i] == i for i in range(n)),
                    "derangement": all(perm[i] != i for i in range(n)),
                },
                generator_params={"n": n},
            )

    # ────────────────────────────────────────────────────────────────────
    # 行列
    # ────────────────────────────────────────────────────────────────────

    def _gen_matrices(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """小さな行列サンプルを生成"""
        dim = params.get("dim", 2)
        val_range = params.get("val_range", range(-2, 3))

        for vals in itertools.islice(
            itertools.product(val_range, repeat=dim*dim), 200
        ):
            mat = [list(vals[i*dim:(i+1)*dim]) for i in range(dim)]
            det = self._det2(mat) if dim == 2 else None
            yield FiniteModel(
                domain="matrix", size=dim, elements=mat,
                relations={"matrix": mat, "det": det, "dim": dim},
                properties={
                    "invertible": det is not None and det != 0,
                    "zero_matrix": all(v == 0 for v in vals),
                    "identity": mat == [[1 if i==j else 0 for j in range(dim)] for i in range(dim)],
                },
                generator_params={"dim": dim},
            )

    # ────────────────────────────────────────────────────────────────────
    # 多項式
    # ────────────────────────────────────────────────────────────────────

    def _gen_polynomials(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """多項式サンプルを生成"""
        max_deg = params.get("max_deg", 3)
        coeff_range = params.get("coeff_range", range(-2, 3))

        for deg in range(0, max_deg + 1):
            for coeffs in itertools.islice(
                itertools.product(coeff_range, repeat=deg+1), 30
            ):
                if coeffs[-1] == 0 and deg > 0:  # 最高次係数は非零
                    continue
                # 根を求める（次数2まで）
                roots = []
                if deg == 1:
                    a, b = coeffs[1], coeffs[0]
                    if a != 0:
                        roots = [-b / a]
                elif deg == 2:
                    a, b, c = coeffs[2], coeffs[1], coeffs[0]
                    disc = b*b - 4*a*c
                    if disc >= 0 and a != 0:
                        roots = [(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)]
                yield FiniteModel(
                    domain="polynomial", size=deg+1, elements=list(coeffs),
                    relations={"coeffs": list(coeffs), "degree": deg, "roots": roots},
                    properties={
                        "monic": coeffs[-1] == 1 if coeffs else False,
                        "has_real_roots": len(roots) > 0,
                        "constant": deg == 0,
                    },
                    generator_params={"degree": deg},
                )

    # ────────────────────────────────────────────────────────────────────
    # ユーティリティ
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def _is_perfect_square(n: int) -> bool:
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    @staticmethod
    def _det2(mat: List[List]) -> Optional[float]:
        """2x2 行列式"""
        if len(mat) == 2 and len(mat[0]) == 2:
            return mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0]
        return None

    # ────────────────────────────────────────────────────────────────────
    # 変数代入の小世界 (恒等式・多項式検証用)
    # ────────────────────────────────────────────────────────────────────

    def _gen_substitution(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """変数代入の小世界 — 多項式恒等式の反例探索用"""
        count = params.get("count", 30)
        var_names = params.get("vars", ["x", "y", "n", "k", "a", "b"])

        # 戦略的な値のセット
        seed_values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17]
        # 有理数も追加
        from fractions import Fraction
        seed_values.extend([Fraction(1, 2), Fraction(1, 3), Fraction(2, 3),
                           Fraction(-1, 2), Fraction(3, 2)])

        generated = 0
        for _ in range(count * 3):  # 多めに試行
            if generated >= count:
                break

            # ランダムに変数を選択（1〜4個）
            num_vars = self._rng.randint(1, min(4, len(var_names)))
            selected_vars = self._rng.sample(var_names, num_vars)

            # 値を割り当て
            assignment = {v: self._rng.choice(seed_values) for v in selected_vars}

            yield FiniteModel(
                domain="substitution",
                size=len(assignment),
                elements=list(assignment.values()),
                relations={"assignment": assignment},
                properties=assignment,
                generator_params={"vars": selected_vars, "type": "substitution"},
            )
            generated += 1

    # ────────────────────────────────────────────────────────────────────
    # 有限体 GF(p)
    # ────────────────────────────────────────────────────────────────────

    def _gen_finite_field(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限体 GF(p) を生成"""
        primes = params.get("primes", [2, 3, 5, 7, 11, 13])

        for p in primes:
            if not self._is_prime(p):
                continue

            elements = list(range(p))
            add_table = {(a, b): (a + b) % p for a in elements for b in elements}
            mul_table = {(a, b): (a * b) % p for a in elements for b in elements}

            # 乗法逆元を計算
            inv_table = {}
            for a in elements:
                if a == 0:
                    continue
                for b in elements:
                    if mul_table[(a, b)] == 1:
                        inv_table[a] = b
                        break

            # primitive root を探索
            primitive_root = None
            for g in range(1, p):
                powers = set()
                for k in range(1, p):
                    powers.add(pow(g, k, p))
                if len(powers) == p - 1:  # 全ての非零元を生成
                    primitive_root = g
                    break

            yield FiniteModel(
                domain="finite_field",
                size=p,
                elements=elements,
                relations={
                    "add": add_table,
                    "mul": mul_table,
                    "inv": inv_table,
                    "zero": 0,
                    "one": 1,
                    "primitive_root": primitive_root,
                },
                properties={
                    "p": p,
                    "characteristic": p,
                    "size": p,
                    "field": True,
                    "prime_field": True,
                    "has_primitive_root": primitive_root is not None,
                },
                generator_params={"type": "GF(p)", "p": p},
            )

    # ────────────────────────────────────────────────────────────────────
    # 有限群（強化版） - 巡回群・対称群
    # ────────────────────────────────────────────────────────────────────

    def _gen_finite_group(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """有限群を生成（既存の _gen_groups を補完）"""
        orders = params.get("orders", [2, 3, 4, 5, 6, 7, 8, 9, 10, 12])

        for n in orders:
            if n > self.max_size and n > 12:
                continue

            # 巡回群 Z/nZ
            elements = list(range(n))
            op = {(i, j): (i + j) % n for i in elements for j in elements}

            # 生成元を見つける
            generators = []
            for g in range(n):
                generated = set()
                current = 0
                for _ in range(n):
                    generated.add(current)
                    current = (current + g) % n
                if len(generated) == n:
                    generators.append(g)

            yield FiniteModel(
                domain="finite_group",
                size=n,
                elements=elements,
                relations={
                    "op": op,
                    "identity": 0,
                    "generators": generators,
                    "type": f"Z_{n}",
                },
                properties={
                    "abelian": True,
                    "cyclic": True,
                    "order": n,
                    "simple": self._is_prime(n),
                    "prime_order": self._is_prime(n),
                },
                generator_params={"type": "cyclic", "n": n},
            )

    # ────────────────────────────────────────────────────────────────────
    # 剰余演算の世界 (mod p)
    # ────────────────────────────────────────────────────────────────────

    def _gen_modular(self, params: Dict) -> Generator[FiniteModel, None, None]:
        """剰余演算の世界 — 合同式・Fermat小定理の検証用"""
        moduli = params.get("moduli", [2, 3, 5, 7, 11, 13, 17, 19, 23])

        for m in moduli:
            if m > 30:  # 大きすぎる modulus は避ける
                continue

            elements = list(range(m))

            # ユニット群 (Z/mZ)* の要素（m と互いに素）
            units = [a for a in elements if math.gcd(a, m) == 1]

            # Euler の totient 関数
            phi = len(units)

            # Fermat の小定理の検証用: a^(p-1) ≡ 1 (mod p) for prime p
            fermat_holds = True
            if self._is_prime(m) and m > 1:
                for a in range(1, m):
                    if pow(a, m - 1, m) != 1:
                        fermat_holds = False
                        break
            else:
                fermat_holds = False

            # primitive root の存在（素数の場合）
            primitive_root = None
            if self._is_prime(m):
                for g in range(1, m):
                    powers = set()
                    for k in range(1, m):
                        powers.add(pow(g, k, m))
                    if len(powers) == m - 1:
                        primitive_root = g
                        break

            yield FiniteModel(
                domain="modular",
                size=m,
                elements=elements,
                relations={
                    "mod": m,
                    "units": units,
                    "phi": phi,
                    "primitive_root": primitive_root,
                },
                properties={
                    "mod": m,
                    "prime": self._is_prime(m),
                    "composite": m > 1 and not self._is_prime(m),
                    "phi": phi,
                    "fermat_holds": fermat_holds,
                    "has_primitive_root": primitive_root is not None,
                },
                generator_params={"type": "Z/mZ", "mod": m},
            )
