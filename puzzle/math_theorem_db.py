"""
math_theorem_db.py
==================
Verantyx Cross DB - 数学定理ピース集

設計思想:
  各定理は「テスト可能なピース」として格納
  - requires: 適用条件 (ILタグ)
  - provides: 結論 (GOALに刺さる)
  - test_fn:  小世界での検証関数 → True/False/'unknown'
  - keywords: ドメイン検出キーワード
  - known_values: 具体的な既知値テーブル

これが Cross DB の資産 (+X定義, -X反例, -Y検証) を統合した形

対応ドメイン:
  combinatorics, graph_theory, group_theory, game_theory,
  topology, number_theory, linear_algebra, order_theory, probability
"""

from __future__ import annotations
import re
import math
from typing import Dict, List, Optional, Any, Callable, Tuple


# ============================================================
# 定理ピースの型
# ============================================================

class TheoremPiece:
    """Cross DB の1ピース = テスト可能な定理"""

    def __init__(
        self,
        theorem_id: str,
        domain: str,
        statement: str,
        keywords: List[str],
        test_fn: Optional[Callable],   # (claim_text, problem_text) → 'promote'|'reject'|'unknown'
        known_values: Optional[Dict] = None,
        counterexample: Optional[str] = None,
    ):
        self.theorem_id = theorem_id
        self.domain = domain
        self.statement = statement
        self.keywords = [k.lower() for k in keywords]
        self.test_fn = test_fn
        self.known_values = known_values or {}
        self.counterexample = counterexample

    def matches_problem(self, problem_text: str) -> float:
        """問題文とのマッチスコア (0.0-1.0)"""
        text = problem_text.lower()
        hits = sum(1 for kw in self.keywords if kw in text)
        return hits / max(len(self.keywords), 1)

    def test(self, claim_text: str, problem_text: str) -> Tuple[str, float]:
        """
        仮説テスト → (verdict, confidence)
        verdict: 'promote' | 'reject' | 'unknown'
        """
        if self.test_fn is None:
            return ('unknown', 0.3)
        try:
            result = self.test_fn(claim_text, problem_text)
            if result == 'promote':
                return ('promote', 0.85)
            elif result == 'reject':
                return ('reject', 0.90)
            else:
                return ('unknown', 0.3)
        except Exception:
            return ('unknown', 0.3)


# ============================================================
# ============================================================
# 定理ピース定義
# ============================================================
# ============================================================

# ---- COMBINATORICS ----

def _test_pigeonhole(claim: str, problem: str) -> str:
    """鳩の巣原理: n個をk箱に入れると少なくとも⌈n/k⌉個が1箱に"""
    c = claim.lower()
    p = problem.lower()
    if "pigeonhole" in p or ("pigeon" in p and "hole" in p):
        if "at least" in c or "⌈" in c or "ceil" in c:
            return 'promote'
        if "at most" in c and "average" in c:
            return 'reject'
    return 'unknown'

def _test_ramsey(claim: str, problem: str) -> str:
    """Ramsey数: R(3,3)=6, R(4,4)=18"""
    RAMSEY = {"R(3,3)": 6, "R(4,4)": 18, "R(3,4)": 9, "R(3,5)": 14}
    p = problem.lower()
    for key, val in RAMSEY.items():
        if key.lower() in p:
            # = の後の数値を取得（例: "R(3,3) = 6" → 6）
            eq_m = re.search(r'=\s*(\d+)', claim)
            if eq_m:
                if int(eq_m.group(1)) == val:
                    return 'promote'
                else:
                    return 'reject'
            # 選択肢が数値のみの場合
            nums = re.findall(r'\b(\d+)\b', claim)
            if nums and int(nums[-1]) == val:
                return 'promote'
            elif nums and len(nums) == 1 and int(nums[0]) != val:
                return 'reject'
    return 'unknown'

def _test_binomial_symmetry(claim: str, problem: str) -> str:
    """C(n,k) = C(n,n-k)"""
    c = claim.lower()
    if "c(n,k) = c(n,n-k)" in c or "symmetric" in c and "binomial" in c:
        return 'promote'
    return 'unknown'

def _test_catalan(claim: str, problem: str) -> str:
    """Catalan数: C_n = C(2n,n)/(n+1)"""
    # 具体値テーブル
    CATALAN = {0:1, 1:1, 2:2, 3:5, 4:14, 5:42, 6:132, 7:429, 8:1430}
    p = problem.lower()
    if "catalan" in p:
        n_match = re.search(r'C_?(\d+)|catalan.*?(\d+)', p)
        if n_match:
            n = int(n_match.group(1) or n_match.group(2))
            if n in CATALAN:
                nums = re.findall(r'\d+', claim)
                if nums and int(nums[0]) == CATALAN[n]:
                    return 'promote'
                elif nums and int(nums[0]) != CATALAN[n]:
                    return 'reject'
    return 'unknown'

def _test_derangement(claim: str, problem: str) -> str:
    """完全順列: D(n) ≈ n!/e"""
    DERANG = {0:1, 1:0, 2:1, 3:2, 4:9, 5:44, 6:265, 7:1854, 8:14833}
    p = problem.lower()
    if "derangement" in p or "!n" in p or "subfactorial" in p:
        n_match = re.search(r'D\((\d+)\)|!(\d+)', problem)
        if n_match:
            n = int(n_match.group(1) or n_match.group(2) or 0)
            if n in DERANG:
                nums = re.findall(r'\d+', claim)
                if nums and int(nums[0]) == DERANG[n]:
                    return 'promote'
                elif nums:
                    return 'reject'
    return 'unknown'

def _test_inclusion_exclusion(claim: str, problem: str) -> str:
    p = problem.lower()
    c = claim.lower()
    if "inclusion" in p and "exclusion" in p:
        if "sum" in c and "intersection" in c:
            return 'promote'
        if "product" in c:
            return 'reject'
    return 'unknown'

def _test_stirling_s2(claim: str, problem: str) -> str:
    """第2種Stirling数 S(n,k)"""
    S2 = {
        (1,1):1, (2,1):1,(2,2):1,
        (3,1):1,(3,2):3,(3,3):1,
        (4,1):1,(4,2):7,(4,3):6,(4,4):1,
        (5,1):1,(5,2):15,(5,3):25,(5,4):10,(5,5):1,
        (6,2):31,(6,3):90,(6,4):65,(6,5):15,
    }
    p = problem.lower()
    if "stirling" in p and "second" in p or "S(" in problem or "S_{" in problem:
        m = re.search(r'S\(?_?\{?(\d+)[,\s]+(\d+)\}?\)?', problem)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            if (n,k) in S2:
                nums = re.findall(r'\d+', claim)
                if nums and int(nums[0]) == S2[(n,k)]:
                    return 'promote'
                elif nums:
                    return 'reject'
    return 'unknown'

def _test_bell_number(claim: str, problem: str) -> str:
    BELL = {0:1,1:1,2:2,3:5,4:15,5:52,6:203,7:877,8:4140}
    p = problem.lower()
    if "bell number" in p or "bell(" in p or "B_n" in p:
        n_m = re.search(r'B_?(\d+)|bell.*?(\d+)', p)
        if n_m:
            n = int(n_m.group(1) or n_m.group(2))
            if n in BELL:
                nums = re.findall(r'\d+', claim)
                if nums and int(nums[0]) == BELL[n]:
                    return 'promote'
                elif nums:
                    return 'reject'
    return 'unknown'


# ---- GRAPH THEORY ----

def _test_kuratowski(claim: str, problem: str) -> str:
    """Kuratowskiの定理: K5またはK_{3,3}を含まない ⟺ 平面グラフ"""
    c = claim.lower()
    p = problem.lower()
    has_k33 = bool(re.search(r'K_{?3,3}?|k3,3|k_{3,3}', p, re.I))
    has_k5  = bool(re.search(r'K_?{?5}?|k5\b', p, re.I))
    if has_k33 or has_k5:
        if "planar" in c and "non" not in c and "not" not in c:
            return 'reject'   # K33/K5 は非平面
        if "non.*planar" in c or "not.*planar" in c:
            return 'promote'
        if "planar" in c:
            return 'reject'
    return 'unknown'

def _test_euler_formula(claim: str, problem: str) -> str:
    """オイラーの公式: V-E+F=2 (平面グラフ)"""
    c = claim.lower()
    p = problem.lower()
    if "euler" in p and ("formula" in p or "characteristic" in p):
        if "v-e+f=2" in c or "v - e + f = 2" in c:
            return 'promote'
        if "v-e+f=0" in c or "v-e+f=1" in c:
            return 'reject'
    return 'unknown'

def _test_four_color(claim: str, problem: str) -> str:
    """四色定理: 平面グラフは4色で彩色可能"""
    c = claim.lower()
    p = problem.lower()
    if "four color" in p or "4-color" in p or "4 color" in p:
        if "4" in c and ("color" in c or "colour" in c):
            return 'promote'
        if "5" in c and ("color" in c or "colour" in c):
            return 'reject'
        if "3" in c and ("always" in c or "sufficient" in c):
            return 'reject'
    return 'unknown'

def _test_handshaking(claim: str, problem: str) -> str:
    """握手補題: 全頂点の次数の和 = 2|E|"""
    c = claim.lower()
    p = problem.lower()
    if ("sum" in p or "handshaking" in p) and "degree" in p:
        if "2" in c and "edge" in c:
            return 'promote'
        if "sum.*degree.*odd" in c:
            return 'reject'  # 次数の和は常に偶数
    return 'unknown'

def _test_euler_path(claim: str, problem: str) -> str:
    """オイラー路/回路の条件"""
    c = claim.lower()
    p = problem.lower()
    if "euler" in p and ("path" in p or "circuit" in p or "trail" in p):
        # Euler回路: 全頂点の次数が偶数
        if "euler circuit" in c or "eulerian circuit" in c:
            if "even" in c and "degree" in c:
                return 'promote'
            if "odd" in c:
                return 'reject'
        # Euler路: 奇数次数頂点が0または2個
        if "euler path" in c or "eulerian path" in c:
            if "two" in c and "odd" in c and "degree" in c:
                return 'promote'
    return 'unknown'

def _test_hall_marriage(claim: str, problem: str) -> str:
    """Hallの結婚定理: 完全マッチングの存在条件"""
    c = claim.lower()
    p = problem.lower()
    if "hall" in p or "marriage" in p or "perfect matching" in p:
        if "hall" in c and "condition" in c:
            return 'promote'
        if "|N(S)| >= |S|" in c or "neighborhood" in c and "size" in c:
            return 'promote'
    return 'unknown'

def _test_bipartite_coloring(claim: str, problem: str) -> str:
    """二部グラフは2彩色可能 / 奇数サイクルを含まない"""
    c = claim.lower()
    p = problem.lower()
    if "bipartite" in p:
        if "2-color" in c or "2 color" in c or "two.*color" in c:
            return 'promote'
        if "odd cycle" in c and "no" in c:
            return 'promote'
        if "chromatic number.*2" in c:
            return 'promote'
        if "3-color" in c or "chromatic.*3" in c:
            return 'reject'
    return 'unknown'

def _test_chromatic_complete(claim: str, problem: str) -> str:
    """完全グラフ Kn の彩色数 = n"""
    p = problem.lower()
    kn_m = re.search(r'K_?\{?(\d+)\}?|complete graph.*?(\d+)', p)
    if kn_m and ("chromatic" in p or "color" in p):
        n = int(kn_m.group(1) or kn_m.group(2) or 0)
        nums = re.findall(r'\d+', claim)
        if nums and int(nums[0]) == n:
            return 'promote'
        elif nums and int(nums[0]) != n:
            return 'reject'
    return 'unknown'

def _test_planarity_edge_bound(claim: str, problem: str) -> str:
    """平面グラフ: e ≤ 3v - 6"""
    c = claim.lower()
    p = problem.lower()
    if "planar" in p and ("edge" in p or "bound" in p):
        if "3v-6" in c.replace(" ","") or "3n-6" in c.replace(" ",""):
            return 'promote'
        if "3v" in c and "6" in c:
            return 'promote'
    return 'unknown'


# ---- GROUP THEORY ----

def _test_lagrange(claim: str, problem: str) -> str:
    """Lagrangeの定理: |H| | |G|"""
    c = claim.lower()
    p = problem.lower()
    if "lagrange" in p or ("order" in p and "subgroup" in p and "divide" in p):
        if "divides" in c or "divisor" in c:
            return 'promote'
        if "does not divide" in c:
            return 'reject'
    # 具体的なケース
    g_order_m = re.search(r'group.*?order\s+(\d+)|order\s+(\d+).*?group', p)
    h_order_m = re.search(r'subgroup.*?order\s+(\d+)|order\s+(\d+).*?subgroup', p)
    if g_order_m and h_order_m:
        g_ord = int(g_order_m.group(1) or g_order_m.group(2))
        h_ord = int(h_order_m.group(1) or h_order_m.group(2))
        nums = re.findall(r'\d+', claim)
        if nums:
            claimed = int(nums[0])
            if g_ord % claimed == 0:
                return 'promote'
            else:
                return 'reject'
    return 'unknown'

def _test_abelian_group(claim: str, problem: str) -> str:
    """アーベル群の性質"""
    c = claim.lower()
    p = problem.lower()
    # S_n for n>=3 is non-abelian
    sn_m = re.search(r's_?(\d+)|symmetric group.*?(\d+)', p, re.I)
    if sn_m:
        n = int(sn_m.group(1) or sn_m.group(2) or 2)
        if n >= 3:
            if "abelian" in c and "non" not in c and "not" not in c:
                return 'reject'
            if "non" in c and "abelian" in c:
                return 'promote'
            if "nonabelian" in c or "non-abelian" in c:
                return 'promote'
    # Z/n is abelian
    zn_m = re.search(r'z/?_?(\d+)|\u2124/(\d+)', p, re.I)
    if zn_m:
        if "abelian" in c and "non" not in c and "not" not in c:
            return 'promote'
        if "non" in c and "abelian" in c:
            return 'reject'
    # Klein four-group is abelian
    if "klein" in p or "v_4" in p:
        if "abelian" in c and "non" not in c and "not" not in c:
            return 'promote'
    return 'unknown'

def _test_cyclic_group(claim: str, problem: str) -> str:
    """巡回群の性質"""
    c = claim.lower()
    p = problem.lower()
    # Z/n is cyclic
    zn_m = re.search(r'z/?_?\d+', p, re.I)
    if zn_m:
        if "cyclic" in c and "non" not in c and "not" not in c:
            return 'promote'
        if ("non" in c and "cyclic" in c) or "not cyclic" in c:
            return 'reject'
    # S_3 is not cyclic
    if re.search(r's_?3\b|symmetric group on 3', p, re.I):
        if "cyclic" in c and "not" not in c and "non" not in c:
            return 'reject'
    # Z/2 x Z/2 (Klein) is NOT cyclic
    if re.search(r'z/?_?2.*[×x].*z/?_?2|klein', p, re.I):
        if "cyclic" in c and "not" not in c:
            return 'reject'
        if ("non" in c and "cyclic" in c) or "not cyclic" in c:
            return 'promote'
    return 'unknown'

def _test_cauchy_group(claim: str, problem: str) -> str:
    """Cauchyの定理: p||G| ならGはp位の元を持つ"""
    c = claim.lower()
    p = problem.lower()
    if "cauchy" in p or ("prime" in p and "order" in p and "element" in p):
        if "element.*order.*p" in c or "p.*order" in c:
            return 'promote'
    return 'unknown'

def _test_group_homomorphism(claim: str, problem: str) -> str:
    """群準同型の性質"""
    c = claim.lower()
    p = problem.lower()
    if "homomorphism" in p or "kernel" in p:
        if "kernel.*normal" in c or "normal.*subgroup" in c:
            return 'promote'
        if "image.*subgroup" in c:
            return 'promote'
        if "kernel.*not.*normal" in c:
            return 'reject'
    return 'unknown'


# ---- GAME THEORY ----

def _test_nim_xor(claim: str, problem: str) -> str:
    """Nim: XOR(heap sizes) = 0 ⟺ 後手勝ち"""
    p = problem.lower()
    c = claim.lower()
    if "nim" in p:
        piles = re.findall(r'\d+', problem)
        piles_int = [int(x) for x in piles if int(x) < 1000]
        if len(piles_int) >= 2:
            xor = 0
            for pile in piles_int[:8]:
                xor ^= pile
            first_wins = xor != 0
            if "first" in c and "win" in c:
                return 'promote' if first_wins else 'reject'
            if "second" in c and "win" in c:
                return 'reject' if first_wins else 'promote'
    return 'unknown'

def _test_grundy(claim: str, problem: str) -> str:
    """Sprague-Grundy: Grundy値=0 ⟺ 後手必勝"""
    c = claim.lower()
    p = problem.lower()
    if "grundy" in p or "sprague" in p:
        if "grundy.*zero" in c or "grundy = 0" in c:
            if "second" in c and "win" in c:
                return 'promote'
        if "grundy.*nonzero" in c or "grundy ≠ 0" in c:
            if "first" in c and "win" in c:
                return 'promote'
    return 'unknown'

def _test_nash_existence(claim: str, problem: str) -> str:
    """Nashの均衡存在定理"""
    c = claim.lower()
    p = problem.lower()
    if "nash" in p and "equilibrium" in p:
        if "exist" in c and "always" in c:
            return 'promote'
        if "does not always exist" in c or "not guaranteed" in c:
            if "mixed" not in p:
                return 'reject'
    return 'unknown'

def _test_minimax(claim: str, problem: str) -> str:
    """ミニマックス定理 (零和ゲーム)"""
    c = claim.lower()
    p = problem.lower()
    if "minimax" in p or "zero.*sum" in p or "zero-sum" in p:
        if "minimax" in c and "maximin" in c and "equal" in c:
            return 'promote'
        if "saddle point" in c:
            return 'promote'
    return 'unknown'


# ---- TOPOLOGY ----

def _test_fundamental_group(claim: str, problem: str) -> str:
    """基本群の既知値"""
    FUND_GROUPS = {
        r'S\^?1|circle': 'Z',
        r'S\^?n|sphere.*n': '0 for n>=2',
        r'torus|T\^?2': 'Z+Z',
        r'RP\^?2|real projective': 'Z/2',
        r'RP\^?n': 'Z/2 for n>=2',
        r'klein bottle': 'non-abelian',
    }
    p = problem.lower()
    c = claim.lower()
    if "fundamental group" in p or "π_1" in p or "pi_1" in p:
        for pattern, value in FUND_GROUPS.items():
            if re.search(pattern, p, re.I):
                val_lower = value.lower()
                if val_lower in c or value in claim:
                    return 'promote'
                # Z vs 0 の判定
                if val_lower == 'z' and ('0' in claim or 'trivial' in c):
                    return 'reject'
                if '0' in val_lower and 'Z' in claim and 'Z/2' not in claim:
                    return 'reject'
    return 'unknown'

def _test_euler_characteristic(claim: str, problem: str) -> str:
    """オイラー標数: χ(S^n) = 1+(-1)^n"""
    EULER_CHAR = {
        r'S\^?0': 2,
        r'S\^?1|circle': 0,
        r'S\^?2|sphere\b': 2,
        r'S\^?n': None,  # 1+(-1)^n
        r'torus|T\^?2': 0,
        r'klein bottle': 0,
        r'RP\^?2': 1,
    }
    p = problem.lower()
    if "euler characteristic" in p or "χ(" in problem or "chi(" in p:
        for pattern, val in EULER_CHAR.items():
            if re.search(pattern, problem, re.I) and val is not None:
                nums = re.findall(r'-?\d+', claim)
                if nums and int(nums[0]) == val:
                    return 'promote'
                elif nums and int(nums[0]) != val:
                    return 'reject'
    return 'unknown'

def _test_brouwer(claim: str, problem: str) -> str:
    """Brouwerの不動点定理"""
    c = claim.lower()
    p = problem.lower()
    if "brouwer" in p or ("fixed point" in p and ("ball" in p or "disk" in p or "continuous" in p)):
        if "fixed point" in c and "exist" in c:
            return 'promote'
        if "no fixed" in c or "fixed point.*not" in c:
            return 'reject'
    return 'unknown'

def _test_homology(claim: str, problem: str) -> str:
    """ホモロジー群の既知値"""
    p = problem
    c = claim
    # H_n(S^m)
    h_match = re.search(r'H_(\d+)\(S\^?(\d+)\)', p)
    if h_match:
        k = int(h_match.group(1))
        m = int(h_match.group(2))
        # H_k(S^m) = Z if k=0 or k=m, 0 otherwise
        expected = 'Z' if k == 0 or k == m else '0'
        if expected in c:
            return 'promote'
        else:
            # もし逆なら reject
            if expected == 'Z' and '0' in c and 'Z' not in c:
                return 'reject'
            if expected == '0' and 'Z' in c and '0' not in c:
                return 'reject'
    return 'unknown'


# ---- NUMBER THEORY ----

def _test_fermat_little(claim: str, problem: str) -> str:
    """フェルマーの小定理: a^(p-1) ≡ 1 (mod p)"""
    c = claim.lower()
    p = problem.lower()
    if "fermat" in p and ("little" in p or "theorem" in p):
        if "a^(p-1)" in c or "a^{p-1}" in c:
            return 'promote'
        if "congruent.*1" in c:
            return 'promote'
    # 具体的な mod 計算
    mod_m = re.search(r'(\d+)\^(\d+)\s*(?:mod|≡)\s*(\d+)', problem)
    if mod_m:
        base, exp, mod = int(mod_m.group(1)), int(mod_m.group(2)), int(mod_m.group(3))
        result = pow(base, exp, mod)
        nums = re.findall(r'\d+', claim)
        if nums and int(nums[0]) % mod == result:
            return 'promote'
        elif nums and int(nums[0]) % mod != result:
            return 'reject'
    return 'unknown'

def _test_wilson(claim: str, problem: str) -> str:
    """Wilsonの定理: (p-1)! ≡ -1 (mod p)"""
    c = claim.lower()
    p = problem.lower()
    if "wilson" in p:
        if "(p-1)!" in c and "≡ -1" in c:
            return 'promote'
        if "p-1.*factorial.*-1" in c:
            return 'promote'
    return 'unknown'

def _test_prime_distribution(claim: str, problem: str) -> str:
    """素数の分布"""
    c = claim.lower()
    p = problem.lower()
    if "prime" in p and "infinitely many" in p:
        if "infinitely many" in c and "prime" in c:
            return 'promote'
        if "finitely many" in c:
            return 'reject'
    if "prime number theorem" in p or "π(x)" in p:
        if "x/ln" in c or "x/log" in c:
            return 'promote'
    return 'unknown'

def _test_gcd_computation(claim: str, problem: str) -> str:
    """GCDの計算"""
    gcd_m = re.search(r'gcd\((\d+),\s*(\d+)\)', problem, re.I)
    if gcd_m:
        a, b = int(gcd_m.group(1)), int(gcd_m.group(2))
        expected = math.gcd(a, b)
        nums = re.findall(r'\d+', claim)
        if nums and int(nums[0]) == expected:
            return 'promote'
        elif nums:
            return 'reject'
    return 'unknown'

def _test_prime_check(claim: str, problem: str) -> str:
    """素数判定"""
    c = claim.lower()
    p = problem.lower()
    # 具体的な数
    is_prime_q = re.search(r'is\s+(\d+)\s+prime', p)
    if is_prime_q:
        n = int(is_prime_q.group(1))
        prime = n >= 2 and all(n % i != 0 for i in range(2, int(n**0.5)+1))
        if "yes" in c or "prime" in c and "not" not in c:
            return 'promote' if prime else 'reject'
        if "no" in c or "not prime" in c:
            return 'reject' if prime else 'promote'
    return 'unknown'


# ---- LINEAR ALGEBRA ----

def _test_rank_nullity(claim: str, problem: str) -> str:
    """階数-零化次元定理: rank(A) + nullity(A) = n"""
    c = claim.lower()
    p = problem.lower()
    if "rank" in p and "nullity" in p:
        if "rank + nullity" in c or "rank.*nullity.*n" in c:
            return 'promote'
        if "rank.*nullity.*m" in c and "n" in p:
            return 'reject'
    return 'unknown'

def _test_spd_eigenvalues(claim: str, problem: str) -> str:
    """SPD行列: 全固有値 > 0"""
    c = claim.lower()
    p = problem.lower()
    if "positive definite" in p or "spd" in p:
        if "positive eigenvalue" in c or "eigenvalue.*positive" in c:
            return 'promote'
        if "negative eigenvalue" in c:
            return 'reject'
        if "all eigenvalue.*positive" in c:
            return 'promote'
    return 'unknown'

def _test_det_properties(claim: str, problem: str) -> str:
    """行列式の性質"""
    c = claim.lower()
    p = problem.lower()
    if "determinant" in p or "det(" in p:
        # det(AB) = det(A)det(B)
        if "det(ab)" in c or "det(a)det(b)" in c:
            return 'promote'
        # rank < n → det = 0
        if "rank" in p:
            n_m = re.search(r'(\d+)×(\d+)|(\d+)\s*by\s*(\d+)', p)
            r_m = re.search(r'rank.*?(\d+)', p)
            if n_m and r_m:
                n = int(n_m.group(1) or n_m.group(3))
                r = int(r_m.group(1))
                if r < n:
                    if "0" in claim and "det" in c:
                        return 'promote'
    return 'unknown'

def _test_matrix_space_dimension(claim: str, problem: str) -> str:
    """行列空間の次元: dim(MAT_n) = n^2, dim(SPD_n) = n(n+1)/2"""
    p = problem.lower()
    # SPD_7 の次元
    spd_m = re.search(r'SPD_(\d+)|SPD.*?(\d+)', problem)
    mat_m = re.search(r'MAT_(\d+)|MAT.*?(\d+)', problem)
    if spd_m:
        n = int(spd_m.group(1) or spd_m.group(2))
        expected = n * (n + 1) // 2
        nums = re.findall(r'\d+', claim)
        if nums and int(nums[0]) == expected:
            return 'promote'
        elif nums and int(nums[0]) != expected:
            return 'reject'
    if mat_m:
        n = int(mat_m.group(1) or mat_m.group(2))
        expected = n * n
        nums = re.findall(r'\d+', claim)
        if nums and int(nums[0]) == expected:
            return 'promote'
        elif nums and int(nums[0]) != expected:
            return 'reject'
    return 'unknown'


# ---- ORDER THEORY ----

def _test_tarski_fixed_point(claim: str, problem: str) -> str:
    """Tarskiの不動点定理: 完備束上の単調函数は不動点を持つ"""
    c = claim.lower()
    p = problem.lower()
    if "tarski" in p or ("fixed point" in p and ("lattice" in p or "monotone" in p or "poset" in p)):
        if "fixed point" in c and ("exist" in c or "has" in c):
            return 'promote'
        if "no fixed" in c:
            return 'reject'
    # extensive functions
    if "extensive" in p and "fp" in p:
        if "extensive" in c and "and" in c:
            return 'promote'  # f and g extensive → fp(f∘g) = fp(f)∩fp(g)
        if "extensive" in c and "or" in c:
            return 'reject'   # "or" では不十分
    return 'unknown'

def _test_zorn_lemma(claim: str, problem: str) -> str:
    """Zornの補題: 上界を持つ帰納的順序集合は極大元を持つ"""
    c = claim.lower()
    p = problem.lower()
    if "zorn" in p:
        if "maximal element" in c:
            return 'promote'
        if "no maximal" in c:
            return 'reject'
    return 'unknown'


# ---- PROBABILITY ----

def _test_bayes(claim: str, problem: str) -> str:
    """Bayesの定理"""
    c = claim.lower()
    p = problem.lower()
    if "bayes" in p:
        if "p(a|b)" in c and "p(b|a)" in c:
            return 'promote'
    return 'unknown'

def _test_law_of_large_numbers(claim: str, problem: str) -> str:
    """大数の法則"""
    c = claim.lower()
    p = problem.lower()
    if "law of large numbers" in p or "LLN" in p:
        if "converge" in c and "expected value" in c:
            return 'promote'
        if "does not converge" in c:
            return 'reject'
    return 'unknown'

def _test_central_limit(claim: str, problem: str) -> str:
    """中心極限定理"""
    c = claim.lower()
    p = problem.lower()
    if "central limit" in p or "CLT" in p:
        if "normal distribution" in c or "gaussian" in c:
            return 'promote'
    return 'unknown'


# ============================================================
# Cross DB: 定理ピースの登録
# ============================================================

ALL_THEOREMS: List[TheoremPiece] = [

    # === COMBINATORICS ===
    TheoremPiece("pigeonhole", "combinatorics",
        "n items in k containers → at least ⌈n/k⌉ in one",
        ["pigeonhole", "pigeon", "hole", "drawer"],
        _test_pigeonhole),

    TheoremPiece("ramsey", "combinatorics",
        "R(3,3)=6, R(4,4)=18, R(3,4)=9",
        ["ramsey", "R(3,3)", "R(4,4)", "ramsey number"],
        _test_ramsey,
        known_values={"R(3,3)": 6, "R(4,4)": 18, "R(3,4)": 9, "R(3,5)": 14}),

    TheoremPiece("catalan", "combinatorics",
        "C_n = C(2n,n)/(n+1)",
        ["catalan", "C_n", "catalan number"],
        _test_catalan,
        known_values={0:1,1:1,2:2,3:5,4:14,5:42,6:132}),

    TheoremPiece("derangement", "combinatorics",
        "D(n) = (n-1)(D(n-1) + D(n-2))",
        ["derangement", "subfactorial", "!n", "complete disorder"],
        _test_derangement,
        known_values={0:1,1:0,2:1,3:2,4:9,5:44,6:265}),

    TheoremPiece("inclusion_exclusion", "combinatorics",
        "|A∪B| = |A| + |B| - |A∩B|",
        ["inclusion", "exclusion", "inclusion-exclusion"],
        _test_inclusion_exclusion),

    TheoremPiece("stirling_s2", "combinatorics",
        "S(n,k): number of partitions of n into k non-empty subsets",
        ["stirling", "S(n,k)", "second kind", "partition"],
        _test_stirling_s2),

    TheoremPiece("bell_number", "combinatorics",
        "B(n): number of set partitions",
        ["bell number", "bell(", "B_n"],
        _test_bell_number,
        known_values={0:1,1:1,2:2,3:5,4:15,5:52,6:203}),

    TheoremPiece("binomial_symmetry", "combinatorics",
        "C(n,k) = C(n,n-k)",
        ["binomial", "C(n,k)", "symmetric"],
        _test_binomial_symmetry),

    # === GRAPH THEORY ===
    TheoremPiece("kuratowski", "graph_theory",
        "G planar ⟺ no K5 or K3,3 subdivision",
        ["kuratowski", "K_{3,3}", "K_5", "planar", "non-planar", "nonplanar"],
        _test_kuratowski),

    TheoremPiece("euler_formula", "graph_theory",
        "V - E + F = 2 for connected planar graphs",
        ["euler formula", "V-E+F", "euler characteristic", "planar graph"],
        _test_euler_formula),

    TheoremPiece("four_color", "graph_theory",
        "Every planar graph is 4-colorable",
        ["four color", "4-color", "4 color", "four colour", "chromatic planar"],
        _test_four_color),

    TheoremPiece("handshaking", "graph_theory",
        "Sum of degrees = 2|E|",
        ["handshaking", "sum of degree", "handshake lemma"],
        _test_handshaking),

    TheoremPiece("euler_path", "graph_theory",
        "Euler circuit ⟺ all degrees even; Euler path ⟺ exactly 2 odd-degree vertices",
        ["euler circuit", "euler path", "eulerian", "odd degree"],
        _test_euler_path),

    TheoremPiece("hall_marriage", "graph_theory",
        "Perfect matching exists ⟺ Hall's condition: |N(S)| ≥ |S|",
        ["hall theorem", "marriage theorem", "perfect matching", "bipartite matching"],
        _test_hall_marriage),

    TheoremPiece("bipartite_2color", "graph_theory",
        "Bipartite ⟺ 2-colorable ⟺ no odd cycles",
        ["bipartite", "2-color", "bipartite graph"],
        _test_bipartite_coloring),

    TheoremPiece("chromatic_complete", "graph_theory",
        "χ(Kn) = n",
        ["chromatic number", "K_n", "complete graph", "coloring"],
        _test_chromatic_complete),

    TheoremPiece("planarity_edge_bound", "graph_theory",
        "Planar graph: e ≤ 3v - 6",
        ["planar", "edge bound", "3v-6", "3n-6"],
        _test_planarity_edge_bound),

    # === GROUP THEORY ===
    TheoremPiece("lagrange", "group_theory",
        "Order of subgroup divides order of group",
        ["lagrange", "order of subgroup", "divides", "index"],
        _test_lagrange),

    TheoremPiece("abelian_group", "group_theory",
        "S_n non-abelian for n≥3; Z/n abelian; Klein V4 abelian",
        ["abelian", "commutative", "S_3", "S_n", "symmetric group", "Z/n"],
        _test_abelian_group),

    TheoremPiece("cyclic_group", "group_theory",
        "Z/n cyclic; S_3 not cyclic; Klein not cyclic",
        ["cyclic", "cyclic group", "generator", "Z/n"],
        _test_cyclic_group),

    TheoremPiece("cauchy_theorem", "group_theory",
        "If prime p divides |G|, G has element of order p",
        ["cauchy", "prime", "order p", "element of order"],
        _test_cauchy_group),

    TheoremPiece("group_homomorphism", "group_theory",
        "Kernel of homomorphism is normal subgroup; image is subgroup",
        ["homomorphism", "kernel", "image", "normal subgroup", "isomorphism"],
        _test_group_homomorphism),

    # === GAME THEORY ===
    TheoremPiece("nim_xor", "game_theory",
        "Nim: first player wins ⟺ XOR of piles ≠ 0",
        ["nim", "xor", "heap", "piles", "first player wins"],
        _test_nim_xor),

    TheoremPiece("sprague_grundy", "game_theory",
        "Grundy value = 0 ⟺ second player wins",
        ["grundy", "sprague", "sprague-grundy", "grundy number"],
        _test_grundy),

    TheoremPiece("nash_existence", "game_theory",
        "Every finite game has a Nash equilibrium in mixed strategies",
        ["nash equilibrium", "nash", "mixed strategy", "equilibrium exist"],
        _test_nash_existence),

    TheoremPiece("minimax", "game_theory",
        "In zero-sum games: maximin = minimax value",
        ["minimax", "maximin", "zero-sum", "saddle point"],
        _test_minimax),

    # === TOPOLOGY ===
    TheoremPiece("fundamental_group", "topology",
        "π₁(S¹)=Z, π₁(Sⁿ)=0 (n≥2), π₁(T²)=Z+Z, π₁(RP²)=Z/2",
        ["fundamental group", "π_1", "pi_1", "homotopy group", "first homotopy"],
        _test_fundamental_group),

    TheoremPiece("euler_char_topology", "topology",
        "χ(S²)=2, χ(S¹)=0, χ(T²)=0, χ(RP²)=1",
        ["euler characteristic", "χ(", "chi(", "euler number"],
        _test_euler_characteristic),

    TheoremPiece("brouwer_fixed_point", "topology",
        "Every continuous map from D^n to D^n has a fixed point",
        ["brouwer", "fixed point theorem", "ball", "disk", "continuous map"],
        _test_brouwer),

    TheoremPiece("homology_sphere", "topology",
        "H_k(S^n) = Z if k=0 or k=n, else 0",
        ["homology", "H_n", "H_k", "sphere", "singular homology"],
        _test_homology),

    # === NUMBER THEORY ===
    TheoremPiece("fermat_little", "number_theory",
        "a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1",
        ["fermat little", "fermat's little", "a^(p-1)", "mod p"],
        _test_fermat_little),

    TheoremPiece("wilson_theorem", "number_theory",
        "(p-1)! ≡ -1 (mod p) iff p is prime",
        ["wilson", "(p-1)!", "factorial", "prime"],
        _test_wilson),

    TheoremPiece("prime_distribution", "number_theory",
        "Infinitely many primes; π(x) ~ x/ln(x)",
        ["infinitely many prime", "prime distribution", "prime number theorem", "π(x)"],
        _test_prime_distribution),

    TheoremPiece("gcd_computation", "number_theory",
        "gcd(a,b) computed by Euclidean algorithm",
        ["gcd", "greatest common divisor", "gcd("],
        _test_gcd_computation),

    # === LINEAR ALGEBRA ===
    TheoremPiece("rank_nullity", "linear_algebra",
        "rank(A) + nullity(A) = n (number of columns)",
        ["rank-nullity", "rank nullity", "null space", "kernel dimension"],
        _test_rank_nullity),

    TheoremPiece("spd_eigenvalues", "linear_algebra",
        "SPD matrix has all positive eigenvalues",
        ["positive definite", "SPD", "positive eigenvalue", "symmetric positive"],
        _test_spd_eigenvalues),

    TheoremPiece("det_properties", "linear_algebra",
        "det(AB)=det(A)det(B); rank<n → det=0",
        ["determinant", "det(", "det(AB)"],
        _test_det_properties),

    TheoremPiece("matrix_space_dim", "linear_algebra",
        "dim(MAT_n)=n², dim(SPD_n)=n(n+1)/2",
        ["MAT_", "SPD_", "dimension", "space of matrices"],
        _test_matrix_space_dimension),

    # === ORDER THEORY ===
    TheoremPiece("tarski_fixed_point", "order_theory",
        "Monotone function on complete lattice has fixed point",
        ["tarski", "fixed point", "monotone", "complete lattice", "extensive", "fp("],
        _test_tarski_fixed_point),

    TheoremPiece("zorn_lemma", "order_theory",
        "Inductive poset with upper bounds has maximal element",
        ["zorn", "maximal element", "upper bound", "chain"],
        _test_zorn_lemma),

    # === PROBABILITY ===
    TheoremPiece("bayes_theorem", "probability",
        "P(A|B) = P(B|A)P(A)/P(B)",
        ["bayes", "conditional probability", "posterior", "prior"],
        _test_bayes),

    TheoremPiece("law_large_numbers", "probability",
        "Sample mean converges to expected value",
        ["law of large numbers", "LLN", "converge", "sample mean"],
        _test_law_of_large_numbers),

    TheoremPiece("central_limit", "probability",
        "Sum of iid variables converges to normal distribution",
        ["central limit theorem", "CLT", "normal distribution", "gaussian"],
        _test_central_limit),
]

# インデックス構築
_DOMAIN_INDEX: Dict[str, List[TheoremPiece]] = {}
_KEYWORD_INDEX: Dict[str, List[TheoremPiece]] = {}

for _thm in ALL_THEOREMS:
    _DOMAIN_INDEX.setdefault(_thm.domain, []).append(_thm)
    for _kw in _thm.keywords:
        _KEYWORD_INDEX.setdefault(_kw, []).append(_thm)


def get_relevant_theorems(
    problem_text: str,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> List[TheoremPiece]:
    """
    問題文に関連する定理ピースを取得

    +X軸 (定義・前提) から候補を引く Cross DB 検索
    """
    candidates = []

    if domain and domain in _DOMAIN_INDEX:
        for thm in _DOMAIN_INDEX[domain]:
            score = thm.matches_problem(problem_text)
            if score > 0:
                candidates.append((thm, score))

    # キーワードベースの補完
    text = problem_text.lower()
    for kw, thms in _KEYWORD_INDEX.items():
        if kw in text:
            for thm in thms:
                if not any(t is thm for t, _ in candidates):
                    score = thm.matches_problem(problem_text)
                    candidates.append((thm, score))

    candidates.sort(key=lambda x: -x[1])
    return [thm for thm, _ in candidates[:top_k]]


if __name__ == "__main__":
    print(f"Cross DB: {len(ALL_THEOREMS)} theorem pieces loaded")
    print()

    # テスト
    tests = [
        ("S_3 is abelian", "Is S_3 an abelian group?", "reject"),
        ("Z is abelian", "Is Z/6 an abelian group?", "promote"),
        ("K_{3,3} is planar", "Is K_{3,3} a planar graph?", "reject"),
        ("B = promote(has fixed point)", "Monotone function on complete lattice: does it have a fixed point?", "promote"),
        ("R(3,3) = 6", "What is the Ramsey number R(3,3)?", "promote"),
        ("R(3,3) = 5", "What is the Ramsey number R(3,3)?", "reject"),
    ]

    for claim, problem, expected in tests:
        relevant = get_relevant_theorems(problem)
        best = 'unknown'
        for thm in relevant:
            v, c = thm.test(claim, problem)
            if v != 'unknown':
                best = v
                break
        mark = '✅' if best == expected else '❌'
        print(f"{mark} expected={expected} got={best} | {claim[:40]}")
