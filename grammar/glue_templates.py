"""
Grammar Glue Templates - AnswerSchema → 出力フォーマット辞書

設計哲学:
  「候補=文字列」を全面禁止。
  文字列を出すのはこの層だけ。
  それ以前はすべて構造体（値・行列・集合・証明計画）として扱う。

使い方:
    glue = GrammarGlue()
    output = glue.render(42, "integer")      # → "42"
    output = glue.render(0.5, "rational")    # → "1/2"
    output = glue.render("B", "option_label") # → "B"
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 正規化関数
# ─────────────────────────────────────────────────────────────────────────────


def _norm_integer(v: Any) -> str:
    try:
        f = float(str(v))
        return str(int(round(f)))
    except (TypeError, ValueError):
        return str(v)


def _norm_rational(v: Any) -> str:
    """有理数を最簡分数で表示"""
    try:
        if isinstance(v, (int, float)):
            frac = Fraction(v).limit_denominator(10000)
            return str(frac)
        if isinstance(v, str) and "/" in v:
            a, b = v.split("/", 1)
            frac = Fraction(int(a.strip()), int(b.strip()))
            return str(frac)
        return _norm_integer(v)
    except Exception:
        return str(v)


def _norm_decimal(v: Any) -> str:
    """実数を有効6桁で表示"""
    try:
        f = float(str(v))
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except (TypeError, ValueError):
        return str(v)


def _norm_complex(v: Any) -> str:
    """複素数を a+bi 形式で表示"""
    try:
        if isinstance(v, complex):
            re, im = v.real, v.imag
            if im == 0:
                return _norm_decimal(re)
            sign = "+" if im >= 0 else "-"
            return f"{_norm_decimal(re)}{sign}{_norm_decimal(abs(im))}i"
        return str(v)
    except Exception:
        return str(v)


def _norm_boolean(v: Any) -> str:
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "t", "correct"):
        return "True"
    if s in ("false", "no", "0", "f", "incorrect"):
        return "False"
    return str(v)


def _norm_option_label(v: Any) -> str:
    return str(v).strip().upper()[:1]


def _norm_expression(v: Any) -> str:
    return str(v).strip()


def _norm_formula(v: Any) -> str:
    s = str(v).strip()
    # LaTeX 未囲みなら $...$ で囲む
    if not s.startswith("$") and ("\\" in s or "^" in s or "_" in s):
        return f"${s}$"
    return s


def _norm_sequence(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _norm_set(v: Any) -> str:
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(sorted(str(x) for x in v)) + "}"
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(str(x) for x in v) + "}"
    return str(v)


def _norm_matrix(v: Any) -> str:
    if isinstance(v, list) and v and isinstance(v[0], list):
        rows = "; ".join(" ".join(str(x) for x in row) for row in v)
        return f"[{rows}]"
    return str(v)


def _norm_proof(v: Any) -> str:
    return str(v).strip()


def _norm_text(v: Any) -> str:
    return str(v).strip()


def _norm_move_sequence(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return " ".join(str(m) for m in v)
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# テンプレート辞書
# ─────────────────────────────────────────────────────────────────────────────

GLUE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "integer": {
        "format":    "number",
        "normalize": _norm_integer,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "0",
        "examples":  ["42", "-7", "0", "1000"],
        "description": "整数値",
    },
    "rational": {
        "format":    "fraction",
        "normalize": _norm_rational,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "0",
        "examples":  ["3/4", "1/2", "-2/3", "7"],
        "description": "有理数（最簡分数）",
    },
    "decimal": {
        "format":    "float",
        "normalize": _norm_decimal,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "0",
        "examples":  ["3.14", "2.718", "-0.5", "1e-6"],
        "description": "実数（有効6桁）",
    },
    "complex": {
        "format":    "complex",
        "normalize": _norm_complex,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "0",
        "examples":  ["3+4i", "-1+2i", "5", "2i"],
        "description": "複素数",
    },
    "boolean": {
        "format":    "bool",
        "normalize": _norm_boolean,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "False",
        "examples":  ["True", "False"],
        "description": "真偽値",
    },
    "option_label": {
        "format":    "choice",
        "normalize": _norm_option_label,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["A", "B", "C", "D", "E"],
        "description": "選択肢ラベル（A/B/C/D/E）",
    },
    "expression": {
        "format":    "math_expr",
        "normalize": _norm_expression,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["x^2 + 1", "e^{iπ}", "sin(x)/x"],
        "description": "数学式",
    },
    "formula": {
        "format":    "latex",
        "normalize": _norm_formula,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["$ax^2+bx+c=0$", "$F=ma$", "$E=mc^2$"],
        "description": "LaTeX 数式",
    },
    "text": {
        "format":    "text",
        "normalize": _norm_text,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["The answer is 42.", "因为..."],
        "description": "自由テキスト",
    },
    "sequence": {
        "format":    "list",
        "normalize": _norm_sequence,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["1, 2, 3", "a, b, c"],
        "description": "数列・リスト",
    },
    "set": {
        "format":    "set",
        "normalize": _norm_set,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "∅",
        "examples":  ["{1, 2, 3}", "{a, b}"],
        "description": "集合",
    },
    "graph": {
        "format":    "graph",
        "normalize": lambda v: str(v),
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["V={1,2,3}, E={(1,2),(2,3)}", "K_4"],
        "description": "グラフ表現",
    },
    "matrix": {
        "format":    "matrix",
        "normalize": _norm_matrix,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "[[]]",
        "examples":  ["[[1,0],[0,1]]", "[[2,3],[1,4]]"],
        "description": "行列",
    },
    "proof": {
        "format":    "proof",
        "normalize": _norm_proof,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["Let n be... Then by... ∎"],
        "description": "証明テキスト",
    },
    "move_sequence": {
        "format":    "moves",
        "normalize": _norm_move_sequence,
        "prefix":    "",
        "suffix":    "",
        "fallback":  "",
        "examples":  ["e4 e5 Nf3", "1. d4 Nf6 2. c4"],
        "description": "手順列（チェス等）",
    },
}

# デフォルトテンプレート
_DEFAULT_TEMPLATE = GLUE_TEMPLATES["text"]


# ─────────────────────────────────────────────────────────────────────────────
# GrammarGlue クラス
# ─────────────────────────────────────────────────────────────────────────────


class GrammarGlue:
    """
    Grammar Glue - 構造体 → 文字列変換

    「候補=文字列」禁止原則の最後の砦。
    すべての構造体はここで初めて文字列になる。

    使い方:
        glue = GrammarGlue()
        s = glue.render(42, "integer")         # → "42"
        s = glue.render([1,2,3], "sequence")   # → "1, 2, 3"
        s = glue.render("c", "option_label")   # → "C"
    """

    def __init__(self) -> None:
        self.templates = GLUE_TEMPLATES

    def render(self, value: Any, schema: str) -> str:
        """
        値をスキーマに従って文字列化

        Args:
            value: 構造体の値
            schema: AnswerSchema の値文字列

        Returns:
            人間可読な文字列
        """
        if value is None:
            return self.fallback(schema)

        tmpl = self.templates.get(schema, _DEFAULT_TEMPLATE)
        try:
            normalized = tmpl["normalize"](value)
            return f"{tmpl['prefix']}{normalized}{tmpl['suffix']}"
        except Exception:
            return str(value)

    def fallback(self, schema: str) -> str:
        """スキーマのフォールバック値"""
        tmpl = self.templates.get(schema, _DEFAULT_TEMPLATE)
        return tmpl.get("fallback", "")

    def get_examples(self, schema: str) -> List[str]:
        """スキーマの出力例を返す"""
        tmpl = self.templates.get(schema, _DEFAULT_TEMPLATE)
        return tmpl.get("examples", [])

    def describe(self, schema: str) -> str:
        """スキーマの説明を返す"""
        tmpl = self.templates.get(schema, _DEFAULT_TEMPLATE)
        return tmpl.get("description", schema)

    def infer_schema(self, value: Any) -> str:
        """
        値から最適なスキーマを推定（補助関数）

        Args:
            value: 値

        Returns:
            推定スキーマ名
        """
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "decimal" if value != int(value) else "integer"
        if isinstance(value, complex):
            return "complex"
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], list):
                return "matrix"
            return "sequence"
        if isinstance(value, (set, frozenset)):
            return "set"
        s = str(value).strip()
        if s in ("True", "False"):
            return "boolean"
        if len(s) == 1 and s.upper() in "ABCDE":
            return "option_label"
        try:
            int(s)
            return "integer"
        except ValueError:
            pass
        try:
            float(s)
            return "decimal"
        except ValueError:
            pass
        return "text"

    def render_with_infer(self, value: Any) -> str:
        """スキーマを自動推定してレンダリング"""
        schema = self.infer_schema(value)
        return self.render(value, schema)


# ─────────────────────────────────────────────────────────────────────────────
# モジュールレベルのデフォルトインスタンス
# ─────────────────────────────────────────────────────────────────────────────

_default_glue = GrammarGlue()


def render(value: Any, schema: str) -> str:
    """モジュールレベルのレンダリング関数"""
    return _default_glue.render(value, schema)


def render_auto(value: Any) -> str:
    """スキーマ自動推定版レンダリング"""
    return _default_glue.render_with_infer(value)
