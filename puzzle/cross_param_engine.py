"""
cross_param_engine.py
=====================
Cross Simulator のパラメータ抽出・実計算エンジン

設計思想 (kofdaiの構想):
  問題文 → パラメータ抽出 → 小世界で実際に計算 → 選択肢と照合

  テキストマッチではなく、「ソルバーとして機能するCross Simulator」を実現する。
  
  類似Crossを探して同じ方法で推論 = 問題タイプを識別し、
  そのタイプのテンプレートで計算を実行する。

フロー:
  1. identify_problem_type(text) → ProblemType (enum)
  2. extract_params(text, type)   → dict of params
  3. build_small_world(params)    → MicroWorld
  4. compute(world, params)       → value/verdict
  5. match_to_choices(value, choices) → (label, conf)

対応済みタイプ:
  - NIM_GAME          : Nim山からXOR計算
  - COMBINATORICS_CNK : C(n,k) 二項係数
  - COMBINATORICS_STIRLING : S(n,k) 第2種Stirling数
  - COMBINATORICS_CATALAN  : Catalan数
  - COMBINATORICS_BELL     : Bell数
  - COMBINATORICS_DERANGE  : 完全順列
  - COMBINATORICS_RAMSEY   : Ramsey数 (テーブル参照)
  - GROUP_PROPERTY    : 群の性質 (位数, 可換性, 巡回性)
  - GRAPH_PROPERTY    : グラフの性質 (彩色数, 平面性, ...)
  - MODULAR_ARITH     : modular arithmetic
  - GCD_LCM           : GCD/LCM計算
  - MATRIX_DIMENSION  : 行列空間の次元
  - GAME_COIN         : コインゲーム (取り除き系)
  - LINEAR_RECURRENCE : 漸化式の値
"""

from __future__ import annotations
import re, math, itertools
from typing import Any, Dict, List, Optional, Tuple
from fractions import Fraction


# ============================================================
# 問題タイプ識別
# ============================================================

class ProblemType:
    NIM_GAME             = "nim_game"
    COMBINATORICS_CNK    = "comb_cnk"
    COMBINATORICS_STIRLING = "comb_stirling"
    COMBINATORICS_CATALAN  = "comb_catalan"
    COMBINATORICS_BELL     = "comb_bell"
    COMBINATORICS_DERANGE  = "comb_derange"
    COMBINATORICS_RAMSEY   = "comb_ramsey"
    COMBINATORICS_PERM     = "comb_perm"
    GROUP_ORDER            = "group_order"
    GROUP_ABELIAN          = "group_abelian"
    GROUP_CYCLIC           = "group_cyclic"
    GRAPH_CHROMATIC        = "graph_chromatic"
    GRAPH_PLANARITY        = "graph_planarity"
    GRAPH_EULER            = "graph_euler"
    MODULAR_ARITH          = "modular_arith"
    GCD_COMPUTE            = "gcd_compute"
    LCM_COMPUTE            = "lcm_compute"
    MATRIX_DIM             = "matrix_dim"
    MATRIX_DET             = "matrix_det"
    MATRIX_TRACE           = "matrix_trace"
    GAME_COIN_ENDPOINT     = "game_coin_endpoint"
    LINEAR_RECURRENCE      = "linear_recurrence"
    SEQUENCE_VALUE         = "sequence_value"
    EULER_TOTIENT          = "euler_totient"
    PRIME_CHECK            = "prime_check"
    # Phase 3 additions
    FIBONACCI_SEQ          = "fibonacci_seq"
    LUCAS_SEQ              = "lucas_seq"
    INT_PARTITION          = "int_partition"
    POWER_MOD              = "power_mod"
    CAYLEY_TREE_COUNT      = "cayley_tree"
    GRAPH_EDGE_COUNT       = "graph_edge_count"
    REGULAR_POLYGON_ANGLE  = "polygon_angle"
    COUNTING_SURJECTIONS   = "surjections"
    MULT_ORDER             = "mult_order"
    ARITH_SUM              = "arith_sum"
    MOEBIUS_FUNCTION       = "moebius_fn"
    DISCRETE_LOG           = "discrete_log"
    MATRIX_DET             = "matrix_det"
    COMB_MULTINOMIAL       = "multinomial"
    PRIME_COUNTING         = "prime_counting"
    GROUP_DIHEDRAL_ORDER   = "dihedral_order"
    HURWITZ_GENUS          = "hurwitz_genus"
    EULER_CHAR_SURFACE     = "euler_char_surface"
    SUM_OF_DIVISORS        = "sum_divisors"
    UNKNOWN                = "unknown"


def identify_problem_type(text: str) -> str:
    """問題文から問題タイプを識別"""
    t = text.lower()
    
    # Nim ゲーム — word boundary required to avoid "minimal", "minimum", "denominator"
    if re.search(r'\bnim\b', t) and ("pile" in t or "heap" in t or "stone" in t or "token" in t):
        return ProblemType.NIM_GAME
    
    # 二項係数 C(n,k)
    if re.search(r'\bC\s*\(\s*\d+\s*,\s*\d+\s*\)|\bChoose\b|\(\s*\d+\s*\\\\choose\s*\d+', text, re.I):
        return ProblemType.COMBINATORICS_CNK
    if re.search(r'\d+\s+choose\s+\d+', t):
        return ProblemType.COMBINATORICS_CNK
    if re.search(r'\\binom\{\d+\}\{\d+\}', text):
        return ProblemType.COMBINATORICS_CNK
    
    # Stirling数
    if "stirling" in t and "second" in t:
        return ProblemType.COMBINATORICS_STIRLING
    if re.search(r'S\s*\(\s*\d+\s*,\s*\d+\s*\)', text):
        return ProblemType.COMBINATORICS_STIRLING
    
    # Catalan数
    if "catalan" in t:
        return ProblemType.COMBINATORICS_CATALAN
    
    # Bell数
    if "bell number" in t or re.search(r'\bB_\d+\b|\bbell\s*\(\s*\d+', t):
        return ProblemType.COMBINATORICS_BELL
    
    # 完全順列 (Derangement) — require "derangement" or "subfactorial" keyword; avoid false matches on !\d+ in general math
    if "derangement" in t or "subfactorial" in t or "no fixed point" in t or "no element in its original" in t:
        return ProblemType.COMBINATORICS_DERANGE
    
    # Ramsey数
    if "ramsey" in t or re.search(r'R\s*\(\s*\d+\s*,\s*\d+\s*\)', text):
        return ProblemType.COMBINATORICS_RAMSEY
    
    # 順列 P(n,k)
    if re.search(r'\bP\s*\(\s*\d+\s*,\s*\d+\s*\)', text) or "permutation" in t:
        return ProblemType.COMBINATORICS_PERM
    
    # Nim ゲーム (コイン取り除き)
    if "coin" in t and ("take" in t or "remove" in t or "pick" in t):
        return ProblemType.GAME_COIN_ENDPOINT
    
    # GCD
    if re.search(r'gcd\s*\(\s*\d+', t) or "greatest common divisor" in t:
        return ProblemType.GCD_COMPUTE
    
    # LCM
    if re.search(r'lcm\s*\(\s*\d+', t) or "least common multiple" in t:
        return ProblemType.LCM_COMPUTE
    
    # Multiplicative order (must check before modular_arith)
    if re.search(r'order\s+of\s+\d+\s+mod(?:ulo)?\s+\d+|multiplicative order', t):
        return ProblemType.MULT_ORDER

    # Power mod: a^b mod m (must check before generic modular_arith)
    if re.search(r'\d+\s*\^\s*\d+\s*(?:mod|\\bmod\b)\s*\d+|\d+\s+to the\s+\d+.*mod\s+\d+', t):
        return ProblemType.POWER_MOD

    # Modular arithmetic
    if re.search(r'\d+\s*\\?mod\s*\d+|\d+\s*≡|\bmod\b', t):
        return ProblemType.MODULAR_ARITH
    
    # オイラーのφ関数 — require explicit phi(N) with number AND euler/totient context
    if re.search(r'φ\s*\(\s*\d+\s*\)|euler.*phi\s*\(\s*\d+|euler.*totient', t):
        return ProblemType.EULER_TOTIENT
    if re.search(r'(?:what\s+is|find|compute|calculate)\s+(?:the\s+)?(?:euler(?:\'s)?\s+)?totient\s+of\s+\d+|totient\s+of\s+\d+|totient\s+function\s+of\s+\d+', t):
        return ProblemType.EULER_TOTIENT
    # phi(N) only if "totient" or "euler" context nearby
    if re.search(r'\bphi\s*\(\s*\d+\s*\)', t) and re.search(r'totient|euler(?:\'?s)?\s+phi|number theory', t):
        return ProblemType.EULER_TOTIENT
    
    # 素数判定
    if re.search(r'is\s+\d+\s+(?:a\s+)?prime|prime\s*(?:number)?\s*\?', t):
        return ProblemType.PRIME_CHECK
    
    # 行列次元
    if re.search(r'SPD_\d+|MAT_\d+|dimension.*matri|matri.*dimension', t):
        return ProblemType.MATRIX_DIM
    
    # 行列の跡 — require direct "what is the trace" question with concrete matrix data
    if re.search(r'(?:what\s+is|find|compute|calculate)\s+the\s+trace\s+of|trace\s+of\s+(?:the\s+)?(?:matrix|A\b)', t) and re.search(r'\d', t):
        if not re.search(r'trace\s+of\s+(?:a\s+)?(?:disease|element|fossil|record|history)', t):
            return ProblemType.MATRIX_TRACE
    
    # 二面体群の位数 (check BEFORE generic group_order to avoid preemption)
    # Exclude: semidihedral, subgroup-counting, image/figure questions, multi-answer questions
    if re.search(r'\bdihedral\b', t) and re.search(r'\d', t) and not re.search(r'semidihedral|quasi-?dihedral', t):
        if re.search(r'order of.*d_?\d+|d_?\d+.*order|order of.*dihedral\s+group\s+D_?\d+|what is.*\|D_?\d+\||find.*order.*dihedral\s+group', t):
            if not re.search(r'how many.*subgroup|number of.*subgroup|subgroup.*dihedral|attach|figure|image|visualiz|grid|V_\d+', t):
                return ProblemType.GROUP_DIHEDRAL_ORDER

    # 群の位数 — require direct question about group order, not just "group" and "order" in same text
    if re.search(r'what is the order of.*group|find the order of.*group|order of the (?:group|subgroup)|order of (?:s_?\d+|a_?\d+|z_?\d+|gl|sl|psl|pgl)', t):
        return ProblemType.GROUP_ORDER
    
    # 群の可換性 — require direct question about abelianness, not just "abelian" in general math text
    if re.search(r'is\s+\w.*\babelian\b|is.*group\s+abelian|is.*abelian\s+group\?|which.*groups?\s+(?:is|are)\s+abelian|prove.*abelian|show.*group.*abelian', t):
        return ProblemType.GROUP_ABELIAN

    # 群の巡回性 — require direct question about cyclic property
    if re.search(r'is\s+\w.*\bcyclic\b|is.*group\s+cyclic|is.*cyclic\s+group\?|which.*groups?\s+(?:is|are)\s+cyclic|prove.*cyclic|show.*group.*cyclic|every group of order.*cyclic|group of order.*is cyclic', t):
        return ProblemType.GROUP_CYCLIC
    
    # グラフの彩色数 - specific named graph with explicit chromatic number question
    if "chromatic" in t and ("number" in t or "coloring" in t or "colouring" in t):
        if re.search(r'chromatic number of|what is the chromatic|find the chromatic', t):
            if re.search(r'K_?\{?\d+\}?|complete graph K_?\d+|cycle C_?\d+|petersen|bipartite\s+K_?\{?\d+', t):
                return ProblemType.GRAPH_CHROMATIC

    # 平面グラフ - 特定グラフ名 + planarity の組み合わせが必要
    if "planar" in t:
        if re.search(r'K_?\{?\d+[\},]|complete graph|K_{3,3}|K_5\b|petersen', t, re.I):
            return ProblemType.GRAPH_PLANARITY

    # ── Phase 3 additions ──

    # Fibonacci number — require "nth Fibonacci number" or "fibonacci(n)" or "F_n="
    # Avoid: Fibonacci heap, Fibonacci polynomial, formula/identity questions, image questions,
    #        combination/counting questions that merely mention Fibonacci numbers as constraints
    if re.search(r'\bfibonacci\b', t) and not re.search(r'fibonacci\s+heap|fibonacci\s+poly|fibonacci\s+sequence.*formula', t):
        # Must be asking for a specific numeric value — not image questions, not symbolic results
        if not re.search(r'attach|image|figure|diagram|show(?:n|ing)|photo|picture|illustrat', t):
            # Exclude: combination/counting queries that use Fibonacci as constraints
            if not re.search(r'how many.*combin|how many.*ways|count.*combin|combination.*fibonacci.*sum|fibonacci.*combin', t):
                if re.search(r'(?:the\s+)?\d+(?:th|st|nd|rd)\s+fibonacci\s+number|fibonacci\s*\(\s*\d+\s*\)|\bF_\d+\b\s*=|\bfind\s+the\s+\d+(?:th|st|nd|rd)\s+fibonacci', t):
                    if not re.search(r'chess|move|checkmate|board|pawn|knight|bishop|rook|queen|king|go\s+board|polynomial|partial diff', t):
                        return ProblemType.FIBONACCI_SEQ

    # Lucas number — require explicit "nth Lucas number" or "L_n =" with clear Lucas context
    if re.search(r'(?:the\s+)?\d+(?:th|st|nd|rd)?\s+lucas\s+number|\blucas\s+number.*?\d+', t, re.I):
        if not re.search(r'lucas.*theorem|lucas.*lemma|line.*segment|lucas.*test|laplacian|simplicial|chain complex|homology', t):
            return ProblemType.LUCAS_SEQ
    # "L_n =" pattern only if "lucas" also appears nearby
    if re.search(r'\bL_\d+\b\s*=', t, re.I) and re.search(r'\blucas\b', t, re.I):
        if not re.search(r'laplacian|simplicial|chain complex|homology|lucas.*theorem', t):
            return ProblemType.LUCAS_SEQ

    # Integer partition p(n) — require explicit "number of partitions" or "partition of n"
    if re.search(r'number of partitions? of\s+\d+|\bpartition function p\s*\(|\bp\s*\(\s*\d+\s*\)\s*=', t):
        return ProblemType.INT_PARTITION

    # Power mod: a^b mod m (fast exponentiation)
    if re.search(r'\d+\s*\^\s*\d+\s*(?:mod|\\bmod\b)\s*\d+|\d+\s+to the\s+\d+.*mod\s+\d+', t):
        return ProblemType.POWER_MOD

    # Cayley's tree formula n^(n-2)
    if re.search(r'labeled tree|spanning tree.*complete|number of.*tree.*\d+\s+(?:labeled\s+)?(?:vertices?|nodes?)', t):
        return ProblemType.CAYLEY_TREE_COUNT

    # Graph edge count: K_n, K_{m,n}, C_n
    if re.search(r'(?:edges?\s+(?:in|of)\s+K_?\{?\d+\}?|K_?\{?\d+\}?\s+has\s+.*edge|how many edges)', t):
        return ProblemType.GRAPH_EDGE_COUNT

    # Regular polygon interior/exterior angle — require explicit number of sides or named polygon
    if re.search(r'(?:interior|exterior)\s+angle\s+of\s+(?:a\s+)?(?:regular\s+)?\w*gon|angle\s+sum\s+of\s+(?:a\s+)?(?:regular\s+)?\w*gon|regular\s+polygon\s+with\s+\d+\s+sides?|interior\s+angle.*regular\s+\d+|exterior\s+angle.*\d+-?gon', t):
        if re.search(r'\d', t) and not re.search(r'atomic|crystal|molecular|bond|lattice', t):
            return ProblemType.REGULAR_POLYGON_ANGLE

    # Counting surjections (onto functions)
    if re.search(r'surject|onto function|surjective|number of onto|number of surject', t):
        return ProblemType.COUNTING_SURJECTIONS

    # Multiplicative order of a mod n
    if re.search(r'order of\s+\d+\s+mod(?:ulo)?\s+\d+|multiplicative order', t):
        return ProblemType.MULT_ORDER

    # Arithmetic/geometric series sum
    if re.search(r'sum\s+of\s+(?:the\s+)?first\s+\d+\s+(?:terms?|integers?|natural|positive|odd|even)|sum\s+of\s+first\s+\d+|arithmetic\s+(?:series|progression)\s+sum|1\s*\+\s*2\s*\+.*\+\s*n', t):
        return ProblemType.ARITH_SUM

    # Mobius function
    if re.search(r'mobius|möbius|mu\s*\(\s*\d+\s*\)', t):
        return ProblemType.MOEBIUS_FUNCTION

    # Discrete log: find x such that a^x ≡ b (mod p)
    if re.search(r'discrete\s+log|dlog|find\s+x.*a\s*\^\s*x|log_\d+\s*\(\s*\d+\s*\)\s*mod', t):
        return ProblemType.DISCRETE_LOG

    # Matrix determinant
    if re.search(r'determinant|det\s*\(|det\s*\[', t) and not "eigenvalue" in t:
        return ProblemType.MATRIX_DET

    # Multinomial coefficient
    if re.search(r'multinomial|arrangements.*repeated|n!/\(\w+!\)', t):
        return ProblemType.COMB_MULTINOMIAL

    # Dihedral group order
    if re.search(r'dihedral\s+group\s+D_?\{?(\d+)\}?|order\s+of.*dihedral', t):
        return ProblemType.GROUP_DIHEDRAL_ORDER

    # Euler characteristic of surface — require "euler characteristic" explicitly,
    # not just "genus" which is common in unrelated questions
    if re.search(r'euler\s+characteristic\s+of|compute.*euler\s+characteristic|find.*euler\s+characteristic', t):
        return ProblemType.EULER_CHAR_SURFACE

    # Sum of divisors sigma(n)
    if re.search(r'sum\s+of\s+divisors|sigma\s*\(\s*\d+\s*\)|perfect\s+number.*divisors?', t):
        return ProblemType.SUM_OF_DIVISORS

    return ProblemType.UNKNOWN


# ============================================================
# パラメータ抽出
# ============================================================

def extract_params(text: str, prob_type: str) -> Dict[str, Any]:
    """問題文から計算パラメータを抽出"""
    
    if prob_type == ProblemType.NIM_GAME:
        return _extract_nim_params(text)
    
    if prob_type == ProblemType.COMBINATORICS_CNK:
        return _extract_cnk_params(text)
    
    if prob_type == ProblemType.COMBINATORICS_STIRLING:
        return _extract_stirling_params(text)
    
    if prob_type == ProblemType.COMBINATORICS_CATALAN:
        return _extract_n_param(text, "catalan")
    
    if prob_type == ProblemType.COMBINATORICS_BELL:
        return _extract_n_param(text, "bell")
    
    if prob_type == ProblemType.COMBINATORICS_DERANGE:
        return _extract_n_param(text, "derangement")
    
    if prob_type == ProblemType.COMBINATORICS_RAMSEY:
        return _extract_ramsey_params(text)
    
    if prob_type == ProblemType.GCD_COMPUTE:
        return _extract_gcd_params(text)
    
    if prob_type == ProblemType.LCM_COMPUTE:
        return _extract_lcm_params(text)
    
    if prob_type == ProblemType.MODULAR_ARITH:
        return _extract_mod_params(text)
    
    if prob_type == ProblemType.EULER_TOTIENT:
        return _extract_single_n(text, "phi")
    
    if prob_type == ProblemType.PRIME_CHECK:
        return _extract_single_n(text, "prime")
    
    if prob_type == ProblemType.MATRIX_DIM:
        return _extract_matrix_dim_params(text)
    
    if prob_type == ProblemType.GROUP_ORDER:
        return _extract_group_params(text)
    
    if prob_type == ProblemType.GROUP_ABELIAN:
        return _extract_group_params(text)
    
    if prob_type == ProblemType.GROUP_CYCLIC:
        return _extract_group_params(text)
    
    if prob_type == ProblemType.GRAPH_CHROMATIC:
        return _extract_graph_params(text)
    
    if prob_type == ProblemType.GRAPH_PLANARITY:
        return _extract_graph_params(text)

    # ── Phase 3 additions ──
    if prob_type == ProblemType.FIBONACCI_SEQ:
        return _extract_single_n(text, "fibonacci")
    if prob_type == ProblemType.LUCAS_SEQ:
        return _extract_single_n(text, "lucas")
    if prob_type == ProblemType.INT_PARTITION:
        return _extract_single_n(text, "partition")
    if prob_type == ProblemType.POWER_MOD:
        return _extract_power_mod_params(text)
    if prob_type == ProblemType.CAYLEY_TREE_COUNT:
        return _extract_single_n(text, "tree")
    if prob_type == ProblemType.GRAPH_EDGE_COUNT:
        return _extract_graph_params(text)
    if prob_type == ProblemType.REGULAR_POLYGON_ANGLE:
        return _extract_polygon_params(text)
    if prob_type == ProblemType.COUNTING_SURJECTIONS:
        return _extract_surjection_params(text)
    if prob_type == ProblemType.MULT_ORDER:
        return _extract_mult_order_params(text)
    if prob_type == ProblemType.ARITH_SUM:
        return _extract_arith_sum_params(text)
    if prob_type == ProblemType.MOEBIUS_FUNCTION:
        return _extract_single_n(text, "moebius")
    if prob_type == ProblemType.DISCRETE_LOG:
        return _extract_dlog_params(text)
    if prob_type == ProblemType.MATRIX_DET:
        return _extract_matrix_det_params(text)
    if prob_type == ProblemType.COMB_MULTINOMIAL:
        return _extract_multinomial_params(text)
    if prob_type == ProblemType.GROUP_DIHEDRAL_ORDER:
        return _extract_single_n(text, "dihedral")
    if prob_type == ProblemType.EULER_CHAR_SURFACE:
        return _extract_genus_params(text)
    if prob_type == ProblemType.SUM_OF_DIVISORS:
        return _extract_single_n(text, "sigma")

    return {}


def _extract_nim_params(text: str) -> Dict:
    piles = re.findall(r'\b(\d+)\s+(?:stone|token|chip|counter|object|coin)', text, re.I)
    if not piles:
        # "pile of 3, pile of 5, pile of 7"
        piles = re.findall(r'pile[^.]*?(\d+)', text, re.I)
    if not piles:
        # "heaps of size 3, 5, 7"
        piles = re.findall(r'(?:heaps?|piles?)\s+(?:of\s+(?:size\s+)?)?(\d+(?:\s*,\s*\d+)*)', text, re.I)
        if piles:
            piles = [p.strip() for p in piles[0].split(',')]
    # 数字のみのカンマ区切りリスト (例: "3, 5, 7")
    if not piles:
        nums = re.findall(r'\b(\d+)\b', text)
        piles = nums[:6] if len(nums) <= 8 else []
    return {"piles": [int(p) for p in piles if 0 < int(p) < 10000]}


def _extract_cnk_params(text: str) -> Dict:
    # C(n,k) or n choose k
    m = re.search(r'C\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    m = re.search(r'(\d+)\s+choose\s+(\d+)', text, re.I)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    m = re.search(r'\\binom\{(\d+)\}\{(\d+)\}', text)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    return {}


def _extract_stirling_params(text: str) -> Dict:
    m = re.search(r'S\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    return {}


def _extract_n_param(text: str, kind: str) -> Dict:
    if kind == "catalan":
        m = re.search(r'[Cc]atalan.*?(\d+)|[Cc]_(\d+)', text)
    elif kind == "bell":
        m = re.search(r'[Bb]ell.*?(\d+)|B_(\d+)', text)
    elif kind == "derangement":
        m = re.search(r'[Dd]erangement.*?(\d+)|D\((\d+)\)|!(\d+)', text)
    else:
        m = None
    if m:
        val = next(v for v in m.groups() if v is not None)
        return {"n": int(val)}
    return {}


def _extract_ramsey_params(text: str) -> Dict:
    m = re.search(r'R\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
    if m:
        return {"a": int(m.group(1)), "b": int(m.group(2))}
    return {}


def _extract_gcd_params(text: str) -> Dict:
    m = re.search(r'gcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "b": int(m.group(2))}
    return {}


def _extract_lcm_params(text: str) -> Dict:
    m = re.search(r'lcm\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "b": int(m.group(2))}
    return {}


def _extract_mod_params(text: str) -> Dict:
    # a^b mod m
    m = re.search(r'(\d+)\s*\^\s*(\d+)\s*(?:\\?mod|≡)\s*(\d+)', text)
    if m:
        return {"a": int(m.group(1)), "exp": int(m.group(2)), "mod": int(m.group(3))}
    # a mod m
    m = re.search(r'(\d+)\s*\\?mod\s*(\d+)', text)
    if m:
        return {"a": int(m.group(1)), "mod": int(m.group(2))}
    return {}


def _extract_single_n(text: str, kind: str) -> Dict:
    if kind == "phi":
        m = re.search(r'φ\((\d+)\)|phi\((\d+)\)|totient.*?(\d+)', text, re.I)
    elif kind == "prime":
        m = re.search(r'is\s+(\d+)\s+prime|(\d+)\s+is\s+prime', text, re.I)
    elif kind == "fibonacci":
        # Must find specific N — ordinal "Nth fibonacci number", F_N=, or fibonacci(N)
        # Do NOT use "fibonacci.*?(\d+)" — picks up context digits like "fibonacci numbers are 1"
        m = re.search(r'\bF_\{?(\d+)\}?\b|fibonacci\s*\(\s*(\d+)\s*\)|(\d+)\s*(?:th|st|nd|rd)\s+fibonacci|fibonacci\s+number\s+(?:at\s+)?(?:position\s+)?(\d+)', text, re.I)
        if not m:
            # Also try "Nth fibonacci" with ordinal before "fibonacci"
            m = re.search(r'the\s+(\d+)(?:th|st|nd|rd)\s+fibonacci|(\d{2,})[- ](?:th|st|nd|rd)?\s*fibonacci', text, re.I)
    elif kind == "lucas":
        m = re.search(r'L_?\{?(\d+)\}?|lucas.*?(\d+)|(\d+)[- ]th lucas', text, re.I)
    elif kind == "partition":
        m = re.search(r'p\s*\(\s*(\d+)\s*\)|partition.*?(\d+)|(\d+)\s+into.*partition', text, re.I)
    elif kind == "tree":
        m = re.search(r'(\d+)\s+(?:labeled\s+)?(?:vertices?|nodes?)|on\s+(\d+)\s+vertices?', text, re.I)
    elif kind == "moebius":
        m = re.search(r'(?:mu|μ|möbius|mobius)\s*\(\s*(\d+)\s*\)|mobius.*?(\d+)', text, re.I)
    elif kind == "sigma":
        m = re.search(r'sigma\s*\(\s*(\d+)\s*\)|sum.*divisors.*?(\d+)|(\d+).*sum.*divisors', text, re.I)
    elif kind == "dihedral":
        m = re.search(r'D_?\{?(\d+)\}?|dihedral.*?(\d+)', text, re.I)
    else:
        m = re.search(r'\b(\d+)\b', text)
    if m:
        val = next((v for v in m.groups() if v is not None), None)
        if val:
            return {"n": int(val)}
    return {}


def _extract_power_mod_params(text: str) -> Dict:
    """Extract a, exp, mod from 'a^exp mod m' patterns"""
    m = re.search(r'(\d+)\s*\^\s*(\d+)\s*(?:mod|\\bmod)\s*(\d+)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "exp": int(m.group(2)), "mod": int(m.group(3))}
    # "a to the power b mod m"
    m = re.search(r'(\d+)\s+to the\s+(\d+).*?mod\s+(\d+)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "exp": int(m.group(2)), "mod": int(m.group(3))}
    return {}


def _extract_polygon_params(text: str) -> Dict:
    """Extract n (sides) and angle type from polygon angle questions"""
    n = None
    angle_type = "interior"
    t = text.lower()
    # regular n-gon
    m = re.search(r'regular\s+(\d+)-?gon|(\d+)-sided|(\d+)\s+sides?', text, re.I)
    if m:
        n = int(next(v for v in m.groups() if v is not None))
    # Named polygons
    named = {"triangle": 3, "quadrilateral": 4, "pentagon": 5, "hexagon": 6,
             "heptagon": 7, "octagon": 8, "nonagon": 9, "decagon": 10,
             "dodecagon": 12}
    for name, sides in named.items():
        if name in t:
            n = sides
            break
    if "exterior" in t:
        angle_type = "exterior"
    elif "sum" in t:
        angle_type = "sum"
    return {"n": n, "angle_type": angle_type} if n else {}


def _extract_surjection_params(text: str) -> Dict:
    """Extract n (domain), k (codomain) from surjection counting problems"""
    # "surjections from an n-element set to a k-element set"
    m = re.search(r'(\d+)[- ]element.*?(\d+)[- ]element', text, re.I)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    m = re.search(r'(\d+).*?onto.*?(\d+)', text, re.I)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    # "f: {1,...,n} -> {1,...,k}"
    m = re.search(r'f\s*:\s*\{[^}]*(\d+)\s*\}\s*->\s*\{[^}]*(\d+)\s*\}', text)
    if m:
        return {"n": int(m.group(1)), "k": int(m.group(2))}
    return {}


def _extract_mult_order_params(text: str) -> Dict:
    """Extract a, mod from 'order of a mod m' patterns"""
    m = re.search(r'order\s+of\s+(\d+)\s+mod(?:ulo)?\s+(\d+)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "mod": int(m.group(2))}
    m = re.search(r'(\d+)\s+has\s+order.*?mod\s+(\d+)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "mod": int(m.group(2))}
    return {}


def _extract_arith_sum_params(text: str) -> Dict:
    """Extract n, a1, d from arithmetic sum problems"""
    t = text.lower()
    # "sum of first n natural numbers"
    m = re.search(r'sum\s+of\s+(?:first\s+)?(\d+)\s+(?:natural\s+)?(?:integers?|numbers?)', t)
    if m:
        return {"n": int(m.group(1)), "a1": 1, "d": 1}
    # "1+2+...+n"
    m = re.search(r'1\s*\+\s*2\s*\+.*?(\d+)', t)
    if m:
        return {"n": int(m.group(1)), "a1": 1, "d": 1}
    # Arithmetic series: first term a1, common difference d, n terms
    m_n = re.search(r'(\d+)\s+terms?', t)
    m_a1 = re.search(r'first\s+term\s+(?:is\s+)?(\d+)', t)
    m_d = re.search(r'common\s+difference\s+(?:is\s+)?(\d+)', t)
    if m_n and m_a1 and m_d:
        return {"n": int(m_n.group(1)), "a1": int(m_a1.group(1)), "d": int(m_d.group(1))}
    return {}


def _extract_dlog_params(text: str) -> Dict:
    """Extract a, b, mod for discrete log: a^x ≡ b (mod m)"""
    m = re.search(r'(\d+)\s*\^\s*x\s*[≡=]\s*(\d+)\s*(?:mod|\\bmod)\s*(\d+)', text, re.I)
    if m:
        return {"a": int(m.group(1)), "b": int(m.group(2)), "mod": int(m.group(3))}
    m = re.search(r'log_(\d+)\s*\(\s*(\d+)\s*\)\s*(?:mod\s*(\d+))?', text, re.I)
    if m:
        return {"a": int(m.group(1)), "b": int(m.group(2)),
                "mod": int(m.group(3)) if m.group(3) else 0}
    return {}


def _extract_matrix_det_params(text: str) -> Dict:
    """Extract 2x2 or 3x3 matrix from text"""
    # Look for [[a,b],[c,d]] or similar patterns
    m = re.search(r'\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]', text)
    if m:
        row1 = [int(x.strip()) for x in m.group(1).split(',') if x.strip().lstrip('-').isdigit()]
        row2 = [int(x.strip()) for x in m.group(2).split(',') if x.strip().lstrip('-').isdigit()]
        if len(row1) == 2 and len(row2) == 2:
            return {"matrix": [row1, row2]}
    return {}


def _extract_multinomial_params(text: str) -> Dict:
    """Extract n and partition [k1,k2,...] for multinomial n!/(k1!k2!...)"""
    m = re.search(r'(\d+)!\s*/\s*\(([^)]+)\)', text)
    if m:
        n = int(m.group(1))
        ks_str = m.group(2).replace('!', '').replace('×', '*').replace('·', '*')
        ks = [int(x.strip()) for x in re.findall(r'\d+', ks_str)]
        return {"n": n, "ks": ks}
    # Arrangements of word with repeated letters
    m = re.search(r'(\d+)\s+(?:objects?|letters?|items?|elements?)', text, re.I)
    if m:
        n = int(m.group(1))
        repeated = re.findall(r'(\d+)\s+(?:identical|repeated|same)', text, re.I)
        if repeated:
            ks = [int(x) for x in repeated]
            remaining = n - sum(ks)
            if remaining > 0:
                ks.append(remaining)
            return {"n": n, "ks": ks}
    return {}


def _extract_genus_params(text: str) -> Dict:
    """Extract genus and orientability from surface description"""
    t = text.lower()
    m = re.search(r'genus[- ](\d+)|genus\s+(\d+)|\bg\s*=\s*(\d+)', t)
    genus = None
    if m:
        genus = int(next(v for v in m.groups() if v is not None))
    orientable = "non-orientable" not in t and "nonorientable" not in t
    return {"genus": genus, "orientable": orientable} if genus is not None else {}


def _extract_matrix_dim_params(text: str) -> Dict:
    # SPD_n: symmetric positive definite n×n
    m = re.search(r'SPD_(\d+)|SPD.*?(\d+)', text)
    if m:
        n = int(m.group(1) or m.group(2))
        return {"type": "spd", "n": n}
    # MAT_n or n×n matrices
    m = re.search(r'MAT_(\d+)|(\d+)\s*[×x]\s*(\d+)', text)
    if m:
        if m.group(1):
            n = int(m.group(1))
            return {"type": "mat", "n": n}
        else:
            n, k = int(m.group(2)), int(m.group(3))
            return {"type": "mat_nk", "n": n, "k": k}
    return {}


def _extract_group_params(text: str) -> Dict:
    result = {}
    # Z/n or Z_n
    m = re.search(r'Z/?_?(\d+)|\\mathbb{Z}/(\d+)', text, re.I)
    if m:
        result["group_type"] = "cyclic"
        result["n"] = int(m.group(1) or m.group(2))
    # S_n
    m = re.search(r'S_?(\d+)\b', text)
    if m:
        result["group_type"] = "symmetric"
        result["n"] = int(m.group(1))
    # Klein / V_4
    if re.search(r'[Kk]lein|V_4', text):
        result["group_type"] = "klein"
        result["n"] = 4
    # D_n (dihedral)
    m = re.search(r'D_?(\d+)\b', text)
    if m:
        result["group_type"] = "dihedral"
        result["n"] = int(m.group(1))
    # A_n (alternating)
    m = re.search(r'A_?(\d+)\b', text)
    if m:
        result["group_type"] = "alternating"
        result["n"] = int(m.group(1))
    # 一般: order
    m = re.search(r'order\s+(\d+)', text, re.I)
    if m:
        result["order"] = int(m.group(1))
    return result


def _extract_graph_params(text: str) -> Dict:
    result = {}
    # K_{n} 完全グラフ
    m = re.search(r'K_\{?(\d+)\}?|complete graph.*?(\d+)', text)
    if m:
        result["graph_type"] = "complete"
        result["n"] = int(m.group(1) or m.group(2))
    # K_{a,b} 完全二部グラフ
    m = re.search(r'K_?\{?(\d+),(\d+)\}?|K_\{(\d+),(\d+)\}', text)
    if m:
        groups = [g for g in m.groups() if g is not None]
        if len(groups) >= 2:
            result["graph_type"] = "bipartite_complete"
            result["a"] = int(groups[0])
            result["b"] = int(groups[1])
    # C_n サイクル
    m = re.search(r'C_?(\d+)\b|cycle.*?(\d+)', text)
    if m:
        result["graph_type"] = "cycle"
        result["n"] = int(m.group(1) or m.group(2))
    # Petersen graph
    if "petersen" in text.lower():
        result["graph_type"] = "petersen"
        result["n"] = 10
    return result


# ============================================================
# 小世界での計算 (実際の演算)
# ============================================================

def compute_in_small_world(prob_type: str, params: Dict) -> Optional[Any]:
    """
    小世界で実際に計算を実行
    Return: 計算された値 (数値/文字列/bool) or None
    """
    
    # === NIM ===
    if prob_type == ProblemType.NIM_GAME:
        piles = params.get("piles", [])
        if not piles:
            return None
        xor = 0
        for p in piles:
            xor ^= p
        return {"first_wins": xor != 0, "xor": xor}
    
    # === 二項係数 ===
    if prob_type == ProblemType.COMBINATORICS_CNK:
        n, k = params.get("n"), params.get("k")
        if n is None or k is None:
            return None
        if n > 100 or k < 0 or k > n:
            return None
        return math.comb(n, k)
    
    # === Stirling数 ===
    if prob_type == ProblemType.COMBINATORICS_STIRLING:
        n, k = params.get("n"), params.get("k")
        if n is None or k is None:
            return None
        return _stirling2(n, k)
    
    # === Catalan数 ===
    if prob_type == ProblemType.COMBINATORICS_CATALAN:
        n = params.get("n")
        if n is None or n > 20:
            return None
        return math.comb(2*n, n) // (n+1)
    
    # === Bell数 ===
    if prob_type == ProblemType.COMBINATORICS_BELL:
        n = params.get("n")
        if n is None or n > 15:
            return None
        return _bell_number(n)
    
    # === 完全順列 ===
    if prob_type == ProblemType.COMBINATORICS_DERANGE:
        n = params.get("n")
        if n is None or n > 20:
            return None
        return _derangement(n)
    
    # === Ramsey数 ===
    if prob_type == ProblemType.COMBINATORICS_RAMSEY:
        RAMSEY = {(3,3):6, (4,4):18, (3,4):9, (4,3):9, (3,5):14, (5,3):14}
        a, b = params.get("a"), params.get("b")
        if a is None or b is None:
            return None
        return RAMSEY.get((a, b))
    
    # === GCD ===
    if prob_type == ProblemType.GCD_COMPUTE:
        a, b = params.get("a"), params.get("b")
        if a is None or b is None:
            return None
        return math.gcd(a, b)
    
    # === LCM ===
    if prob_type == ProblemType.LCM_COMPUTE:
        a, b = params.get("a"), params.get("b")
        if a is None or b is None:
            return None
        return a * b // math.gcd(a, b)
    
    # === Modular arithmetic ===
    if prob_type == ProblemType.MODULAR_ARITH:
        a = params.get("a")
        exp = params.get("exp")
        mod = params.get("mod")
        if a is None or mod is None:
            return None
        if exp is not None:
            return pow(a, exp, mod)
        return a % mod
    
    # === Euler's totient ===
    if prob_type == ProblemType.EULER_TOTIENT:
        n = params.get("n")
        if n is None:
            return None
        return _totient(n)
    
    # === 素数判定 ===
    if prob_type == ProblemType.PRIME_CHECK:
        n = params.get("n")
        if n is None:
            return None
        if n < 2:
            return False
        return all(n % i != 0 for i in range(2, int(n**0.5)+1))
    
    # === 行列空間の次元 ===
    if prob_type == ProblemType.MATRIX_DIM:
        typ = params.get("type", "")
        n = params.get("n")
        if n is None:
            return None
        if typ == "spd":
            # SPD_n: n×n 対称正定値行列の多様体次元 = n(n+1)/2
            return n * (n+1) // 2
        elif typ == "mat":
            # MAT_n: n×n 行列空間の次元 = n^2
            return n * n
        elif typ == "mat_nk":
            k = params.get("k", n)
            return n * k
        return None
    
    # === 群の性質 ===
    if prob_type in (ProblemType.GROUP_ORDER, ProblemType.GROUP_ABELIAN, ProblemType.GROUP_CYCLIC):
        return _compute_group_property(prob_type, params)
    
    # === グラフの彩色数 ===
    if prob_type == ProblemType.GRAPH_CHROMATIC:
        n = params.get("n", 0) or params.get("a", 0) or 0
        if n <= 0:
            return None  # パラメータが抽出できていない → 誤発動防止
        return _compute_graph_chromatic(params)
    
    # === グラフの平面性 ===
    if prob_type == ProblemType.GRAPH_PLANARITY:
        n = params.get("n", 0) or params.get("a", 0) or 0
        if n <= 0:
            return None
        return _compute_graph_planarity(params)

    # ── Phase 3 additions ──

    if prob_type == ProblemType.FIBONACCI_SEQ:
        n = params.get("n")
        if n is None or n < 0 or n > 100:
            return None
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    if prob_type == ProblemType.LUCAS_SEQ:
        n = params.get("n")
        if n is None or n < 0 or n > 80:
            return None
        if n == 0: return 2
        if n == 1: return 1
        a, b = 2, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    if prob_type == ProblemType.INT_PARTITION:
        n = params.get("n")
        if n is None or n < 0 or n > 50:
            return None
        # Dynamic programming for partition function
        dp = [0] * (n + 1)
        dp[0] = 1
        for k in range(1, n + 1):
            for j in range(k, n + 1):
                dp[j] += dp[j - k]
        return dp[n]

    if prob_type == ProblemType.POWER_MOD:
        a = params.get("a")
        exp = params.get("exp")
        mod = params.get("mod")
        if a is None or exp is None or mod is None or mod <= 0:
            return None
        if exp < 0:
            return None
        return pow(int(a), int(exp), int(mod))

    if prob_type == ProblemType.CAYLEY_TREE_COUNT:
        n = params.get("n")
        if n is None or n < 1 or n > 20:
            return None
        if n <= 2:
            return 1
        return n ** (n - 2)

    if prob_type == ProblemType.GRAPH_EDGE_COUNT:
        n = params.get("n")
        m = params.get("m")
        gtype = params.get("graph_type", "")
        if "bipartite" in gtype or (n and m and n != m):
            if n and m:
                return n * m  # K_{n,m}
        if n:
            return n * (n - 1) // 2  # K_n
        return None

    if prob_type == ProblemType.REGULAR_POLYGON_ANGLE:
        n = params.get("n")
        angle_type = params.get("angle_type", "interior")
        if n is None or n < 3:
            return None
        if angle_type == "sum":
            return (n - 2) * 180
        if angle_type == "exterior":
            return 360 // n if 360 % n == 0 else 360 / n
        # interior angle each
        return (n - 2) * 180 / n

    if prob_type == ProblemType.COUNTING_SURJECTIONS:
        n = params.get("n")  # domain size
        k = params.get("k")  # codomain size
        if n is None or k is None or k > n or k <= 0:
            return None
        # Inclusion-exclusion: sum_{j=0}^{k} (-1)^j C(k,j)(k-j)^n
        total = 0
        for j in range(k + 1):
            total += ((-1) ** j) * math.comb(k, j) * ((k - j) ** n)
        return total

    if prob_type == ProblemType.MULT_ORDER:
        a = params.get("a")
        mod = params.get("mod")
        if a is None or mod is None or mod <= 1:
            return None
        if math.gcd(a, mod) != 1:
            return None  # order undefined
        order = 1
        cur = a % mod
        while cur != 1 and order <= mod:
            cur = (cur * a) % mod
            order += 1
        return order if cur == 1 else None

    if prob_type == ProblemType.ARITH_SUM:
        n = params.get("n")
        a1 = params.get("a1", 1)
        d = params.get("d", 1)
        if n is None:
            return None
        return n * (2 * a1 + (n - 1) * d) // 2

    if prob_type == ProblemType.MOEBIUS_FUNCTION:
        n = params.get("n")
        if n is None or n <= 0:
            return None
        if n == 1:
            return 1
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                factors.append(p)
                temp //= p
                if temp % p == 0:
                    return 0  # p^2 divides n
            p += 1
        if temp > 1:
            factors.append(temp)
        return (-1) ** len(factors)

    if prob_type == ProblemType.DISCRETE_LOG:
        # Baby-step giant-step for small moduli
        a = params.get("a")
        b = params.get("b")
        mod = params.get("mod")
        if a is None or b is None or mod is None or mod > 10**6:
            return None
        # brute force for small mod
        cur = 1
        for x in range(mod):
            if cur == b % mod:
                return x
            cur = (cur * a) % mod
        return None

    if prob_type == ProblemType.MATRIX_DET:
        mat = params.get("matrix")
        if mat is None:
            return None
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        if n == 3:
            a = mat
            return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                  - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                  + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
        return None

    if prob_type == ProblemType.COMB_MULTINOMIAL:
        n = params.get("n")
        ks = params.get("ks", [])
        if n is None or not ks:
            return None
        if sum(ks) != n:
            return None
        result = math.factorial(n)
        for k in ks:
            result //= math.factorial(k)
        return result

    if prob_type == ProblemType.GROUP_DIHEDRAL_ORDER:
        n = params.get("n")
        if n is None or n < 1:
            return None
        return 2 * n

    if prob_type == ProblemType.EULER_CHAR_SURFACE:
        genus = params.get("genus")
        orientable = params.get("orientable", True)
        if genus is None:
            return None
        if orientable:
            return 2 - 2 * genus  # orientable surface
        else:
            return 2 - genus  # non-orientable

    if prob_type == ProblemType.SUM_OF_DIVISORS:
        n = params.get("n")
        if n is None or n <= 0 or n > 10**7:
            return None
        total = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                total += i
                if i != n // i:
                    total += n // i
        return total

    return None


def _stirling2(n: int, k: int) -> Optional[int]:
    """第2種Stirling数 S(n,k) - 動的計画法で計算"""
    if k < 0 or k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    if k == 0 or k > n:
        return 0
    # DP: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    if n > 25:
        return None  # 大きすぎる
    dp = [[0] * (n+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, i+1):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    return dp[n][k]


def _bell_number(n: int) -> int:
    """Bell数 B(n) - Bell三角形で計算"""
    if n == 0:
        return 1
    row = [1]
    for _ in range(n):
        new_row = [row[-1]]
        for j in range(len(row)):
            new_row.append(new_row[-1] + row[j])
        row = new_row
    return row[0]


def _derangement(n: int) -> int:
    """完全順列 D(n)"""
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0  # D(0), D(1)
    for i in range(2, n+1):
        a, b = b, (i-1) * (a + b)
    return b


def _totient(n: int) -> int:
    """Euler's totient φ(n)"""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def _compute_group_property(prob_type: str, params: Dict) -> Optional[Any]:
    """群の性質を計算"""
    gtype = params.get("group_type", "")
    n = params.get("n")
    
    if prob_type == ProblemType.GROUP_ORDER:
        if gtype == "cyclic" and n:
            return n           # |Z/n| = n
        if gtype == "symmetric" and n:
            return math.factorial(n)  # |S_n| = n!
        if gtype == "alternating" and n:
            return math.factorial(n) // 2  # |A_n| = n!/2
        if gtype == "dihedral" and n:
            return 2 * n  # |D_n| = 2n
        if gtype == "klein":
            return 4
    
    if prob_type == ProblemType.GROUP_ABELIAN:
        if gtype == "cyclic":
            return True    # Z/n は常に可換
        if gtype == "symmetric" and n and n >= 3:
            return False   # S_n (n≥3) は非可換
        if gtype == "symmetric" and n and n <= 2:
            return True    # S_1, S_2 は可換
        if gtype == "alternating" and n and n >= 4:
            return False   # A_n (n≥4) は非可換
        if gtype == "klein":
            return True    # Klein群は可換
    
    if prob_type == ProblemType.GROUP_CYCLIC:
        if gtype == "cyclic":
            return True    # Z/n は巡回群
        if gtype == "symmetric" and n and n >= 3:
            return False   # S_n (n≥3) は非巡回
        if gtype == "klein":
            return False   # Klein群は非巡回 (Z/2×Z/2)
    
    return None


def _compute_graph_chromatic(params: Dict) -> Optional[int]:
    """グラフの彩色数を計算"""
    gtype = params.get("graph_type", "")
    n = params.get("n")
    a = params.get("a")
    b = params.get("b")
    
    if gtype == "complete" and n:
        return n   # χ(K_n) = n
    
    if gtype == "bipartite_complete" and a and b:
        # χ(K_{a,b}) = 2 (a,b >= 1)
        return 2
    
    if gtype == "cycle" and n:
        # χ(C_n) = 2 if n even, 3 if n odd
        return 2 if n % 2 == 0 else 3
    
    if gtype == "petersen":
        return 3  # Petersen graphの彩色数 = 3
    
    return None


def _compute_graph_planarity(params: Dict) -> Optional[bool]:
    """グラフが平面グラフかどうか"""
    gtype = params.get("graph_type", "")
    n = params.get("n")
    a = params.get("a")
    b = params.get("b")
    
    if gtype == "complete":
        if n and n <= 4:
            return True   # K_1, K_2, K_3, K_4 は平面グラフ
        if n and n >= 5:
            return False  # K_5 以上は非平面
    
    if gtype == "bipartite_complete":
        if a and b:
            if a <= 2 or b <= 2:
                return True   # K_{1,n}, K_{2,n} は平面グラフ
            if a >= 3 and b >= 3:
                return False  # K_{3,3} 以上は非平面 (Kuratowski)
    
    if gtype == "cycle":
        return True  # すべてのサイクルは平面グラフ
    
    if gtype == "petersen":
        return False  # Petersen graph は非平面
    
    return None


# ============================================================
# 選択肢との照合
# ============================================================

def match_value_to_choices(
    value: Any,
    choices: List[Tuple[str, str]],
    prob_type: str,
    params: Dict,
) -> Optional[Tuple[str, float]]:
    """
    計算された値を選択肢と照合
    Returns: (label, confidence) or None
    """
    if value is None:
        return None
    
    # Nim: first/second player wins
    if isinstance(value, dict) and "first_wins" in value:
        first_wins = value["first_wins"]
        # 各選択肢をチェック
        scores = {}
        for lbl, txt in choices:
            t = txt.lower()
            if "first" in t and "win" in t:
                scores[lbl] = (1.0 if first_wins else -1.0)
            elif "second" in t and "win" in t:
                scores[lbl] = (-1.0 if first_wins else 1.0)
            elif "1st" in t and "win" in t:
                scores[lbl] = (1.0 if first_wins else -1.0)
            elif "2nd" in t and "win" in t:
                scores[lbl] = (-1.0 if first_wins else 1.0)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] > 0:
                return (best, 0.88)
        return None
    
    # bool値 (is_prime, is_abelian, is_planar, etc.)
    if isinstance(value, bool):
        return _match_bool_to_choices(value, choices, prob_type)
    
    # 数値
    value_str = str(value)
    value_int = int(value) if isinstance(value, int) else None
    
    # 選択肢を数値として解析して照合
    match_label = None
    for lbl, txt in choices:
        txt_stripped = txt.strip()
        # 直接一致
        if txt_stripped == value_str:
            return (lbl, 0.95)
        # 数値として一致
        if value_int is not None:
            nums = re.findall(r'^-?\d+$|^\d+$', txt_stripped)
            if nums and int(nums[0]) == value_int:
                return (lbl, 0.95)
            # LaTeX の数値
            nums = re.findall(r'\\?(?:boxed)?\{?(-?\d+)\}?$', txt_stripped)
            if nums and int(nums[0]) == value_int:
                return (lbl, 0.92)
        # 部分一致（選択肢が式の場合）
        if value_str in txt_stripped:
            if match_label is None:
                match_label = lbl
    
    if match_label:
        return (match_label, 0.80)
    
    return None


def _match_bool_to_choices(
    value: bool,
    choices: List[Tuple[str, str]],
    prob_type: str,
) -> Optional[Tuple[str, float]]:
    """bool値を選択肢と照合"""
    
    if prob_type == ProblemType.GROUP_ABELIAN:
        for lbl, txt in choices:
            t = txt.lower()
            if value:  # is abelian
                if ("abelian" in t or "commutative" in t) and "non" not in t and "not" not in t:
                    return (lbl, 0.90)
            else:  # not abelian
                if "non" in t and "abelian" in t:
                    return (lbl, 0.90)
                if "not" in t and "abelian" in t:
                    return (lbl, 0.90)
                if "nonabelian" in t or "non-abelian" in t:
                    return (lbl, 0.90)
    
    if prob_type == ProblemType.GROUP_CYCLIC:
        for lbl, txt in choices:
            t = txt.lower()
            if value:  # is cyclic
                if "cyclic" in t and "non" not in t and "not" not in t:
                    return (lbl, 0.90)
            else:  # not cyclic
                if "non" in t and "cyclic" in t:
                    return (lbl, 0.90)
    
    if prob_type == ProblemType.GRAPH_PLANARITY:
        for lbl, txt in choices:
            t = txt.lower()
            if value:  # is planar
                if "planar" in t and "non" not in t and "not" not in t:
                    return (lbl, 0.90)
            else:  # not planar
                if "non" in t and "planar" in t:
                    return (lbl, 0.90)
                if "not" in t and "planar" in t:
                    return (lbl, 0.90)
    
    if prob_type == ProblemType.PRIME_CHECK:
        for lbl, txt in choices:
            t = txt.lower()
            if value:  # is prime
                if "prime" in t and "not" not in t:
                    return (lbl, 0.90)
                if "yes" == t.strip():
                    return (lbl, 0.90)
            else:  # not prime
                if "not prime" in t or "composite" in t:
                    return (lbl, 0.90)
                if "no" == t.strip():
                    return (lbl, 0.90)
    
    return None


# ============================================================
# メインエントリーポイント
# ============================================================

def solve_with_cross_engine(
    problem_text: str,
    choices: List[Tuple[str, str]],
) -> Optional[Tuple[str, float, str]]:
    """
    Cross Paramエンジンで問題を解く
    
    Returns: (label, confidence, reason) or None
    
    フロー:
      問題タイプ識別 → パラメータ抽出 → 小世界で計算 → 選択肢照合
    """
    # 1. 問題タイプ識別
    prob_type = identify_problem_type(problem_text)
    if prob_type == ProblemType.UNKNOWN:
        return None
    
    # 2. パラメータ抽出
    params = extract_params(problem_text, prob_type)
    if not params:
        return None
    
    # 3. 小世界で計算
    value = compute_in_small_world(prob_type, params)
    if value is None:
        return None
    
    # 4. 選択肢と照合
    match = match_value_to_choices(value, choices, prob_type, params)
    if match is None:
        return None
    
    label, conf = match
    reason = f"CrossParamEngine[{prob_type}] params={params} → computed={value}"
    return (label, conf, reason)


# ============================================================
# テスト
# ============================================================

if __name__ == "__main__":
    test_cases = [
        # (problem, choices, expected_label)
        (
            "In a Nim game with piles of 3, 5, and 7 stones. Who wins with optimal play?",
            [("A", "First player wins"), ("B", "Second player wins"), ("C", "It depends")],
            "A"  # XOR = 3^5^7 = 1 ≠ 0 → first player wins
        ),
        (
            "How many ways to choose 3 from 10? (i.e., C(10,3))",
            [("A", "120"), ("B", "30"), ("C", "720"), ("D", "10")],
            "A"  # C(10,3) = 120
        ),
        (
            "Compute S(4,2), the Stirling number of the second kind.",
            [("A", "3"), ("B", "6"), ("C", "7"), ("D", "11")],
            "C"  # S(4,2) = 7
        ),
        (
            "What is the chromatic number of K_5?",
            [("A", "3"), ("B", "4"), ("C", "5"), ("D", "6")],
            "C"  # χ(K_5) = 5
        ),
        (
            "Is K_{3,3} a planar graph?",
            [("A", "Yes, it is planar"), ("B", "No, it is non-planar"), ("C", "Sometimes")],
            "B"  # K_{3,3} is non-planar
        ),
        (
            "What is gcd(48, 18)?",
            [("A", "2"), ("B", "6"), ("C", "9"), ("D", "12")],
            "B"  # gcd(48,18) = 6
        ),
        (
            "Is S_3 an abelian group?",
            [("A", "Yes, abelian"), ("B", "No, non-abelian"), ("C", "Sometimes abelian")],
            "B"  # S_3 is non-abelian
        ),
        (
            "What is the dimension of the manifold of SPD_7 matrices?",
            [("A", "28"), ("B", "49"), ("C", "14"), ("D", "56")],
            "A"  # SPD_7: 7*8/2 = 28
        ),
    ]
    
    print("Cross Param Engine Tests")
    print("=" * 60)
    passed = 0
    for problem, choices, expected in test_cases:
        result = solve_with_cross_engine(problem, choices)
        if result:
            label, conf, reason = result
            ok = label == expected
            mark = "✅" if ok else "❌"
            passed += int(ok)
            print(f"{mark} expected={expected} got={label} conf={conf:.2f}")
            print(f"   {reason[:80]}")
        else:
            print(f"❌ expected={expected} → No result (engine returned None)")
        print()
    
    print(f"Passed: {passed}/{len(test_cases)}")
