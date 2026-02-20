"""
Calculus Executor - 微積分計算
"""
import re
from typing import Any, Dict, Optional
from sympy import symbols, diff, integrate, limit, series, simplify, sympify, oo
from sympy.parsing.sympy_parser import parse_expr


# =======================================
# Helper functions
# =======================================

def _parse_expression(text: str) -> Optional[str]:
    """
    テキストから数式を抽出
    
    Patterns:
        - "derivative of x^2" → "x^2"
        - "integral of 2x" → "2x"
        - "limit of (x-2)/(x-1) as x approaches" → "(x-2)/(x-1)"
        - "f(x) = x^2" → "x^2"
    """
    text_lower = text.lower()
    
    # "of ..." パターン
    if " of " in text_lower:
        after_of = text[text_lower.index(" of ") + 4:]
        # "with respect to"の前まで
        if "with respect" in after_of.lower():
            expr = after_of[:after_of.lower().index("with respect")].strip()
        # "as x approaches"の前まで（極限の場合）
        elif " as " in after_of.lower():
            expr = after_of[:after_of.lower().index(" as ")].strip()
        else:
            expr = after_of.strip().rstrip(".")
        return expr
    
    # "f(x) = ..." パターン
    if "=" in text:
        expr = text.split("=")[1].strip().rstrip(".")
        return expr
    
    return None


def _parse_variable(text: str) -> str:
    """
    変数名を抽出（デフォルト: x）
    
    Patterns:
        - "with respect to x" → "x"
        - "with respect to t" → "t"
    """
    text_lower = text.lower()
    
    if "with respect to" in text_lower:
        after = text_lower[text_lower.index("with respect to") + 16:].strip()
        # 最初の単語
        match = re.match(r'([a-z])', after)
        if match:
            return match.group(1)
    
    return "x"  # デフォルト


def _fix_implicit_multiplication(expr_str: str) -> str:
    """
    暗黙的な乗算を明示的に変換
    
    Examples:
        "3x" → "3*x"
        "2x^2" → "2*x^2"
        "5x" → "5*x"
    """
    # 数字と変数の間に*を挿入: "3x" → "3*x"
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    
    # 閉じ括弧と変数の間に*を挿入: ")x" → ")*x"
    expr_str = re.sub(r'\)([a-zA-Z])', r')*\1', expr_str)
    
    # 末尾の?や.を除去
    expr_str = expr_str.rstrip('?.!')
    
    return expr_str


# =======================================
# Executor functions (for piece calls)
# =======================================

def derivative(**kwargs) -> Dict[str, Any]:
    """
    導関数を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
        expression: str (例: "x^2")
        variable: str (デフォルト: "x")
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    expr_str = kwargs.get("expression")
    var_str = kwargs.get("variable", "x")
    
    if expr_str is None and source_text:
        expr_str = _parse_expression(source_text)
        var_str = _parse_variable(source_text)
    
    if expr_str is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "expression"
        }
    
    try:
        # 暗黙的乗算を修正
        expr_str = _fix_implicit_multiplication(expr_str)
        
        # ^ を ** に変換
        expr_str = expr_str.replace("^", "**")
        
        var = symbols(var_str)
        expr = parse_expr(expr_str)
        derivative_result = diff(expr, var)
        
        result_str = str(derivative_result)
        
        # 定数の場合は数値化
        if derivative_result.is_number:
            return {
                "success": True,
                "value": float(derivative_result),
                "confidence": 1.0,
                "schema": "expression"
            }
        
        return {
            "success": True,
            "value": result_str,
            "confidence": 1.0,
            "schema": "expression"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "expression"
        }


def integral(**kwargs) -> Dict[str, Any]:
    """
    積分を計算（不定積分、定積分）
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
        expression: str (例: "2x")
        variable: str (デフォルト: "x")
        lower_limit: float (optional)
        upper_limit: float (optional)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    expr_str = kwargs.get("expression")
    var_str = kwargs.get("variable", "x")
    
    if expr_str is None and source_text:
        expr_str = _parse_expression(source_text)
        var_str = _parse_variable(source_text)
    
    if expr_str is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "expression"
        }
    
    try:
        # 暗黙的乗算を修正
        expr_str = _fix_implicit_multiplication(expr_str)
        
        # ^ を ** に変換
        expr_str = expr_str.replace("^", "**")
        
        var = symbols(var_str)
        expr = parse_expr(expr_str)
        
        lower = kwargs.get("lower_limit")
        upper = kwargs.get("upper_limit")
        
        if lower is not None and upper is not None:
            # 定積分
            integral_result = integrate(expr, (var, lower, upper))
        else:
            # 不定積分
            integral_result = integrate(expr, var)
        
        result_str = str(integral_result)
        
        # 定数の場合は数値化
        if integral_result.is_number:
            return {
                "success": True,
                "value": float(integral_result),
                "confidence": 1.0,
                "schema": "expression"
            }
        
        return {
            "success": True,
            "value": result_str,
            "confidence": 1.0,
            "schema": "expression"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "expression"
        }


def limit(**kwargs) -> Dict[str, Any]:
    """
    極限を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
        expression: str (例: "(x^2 - 4)/(x - 2)")
        variable: str (デフォルト: "x")
        point: float (極限を取る点)
        direction: str ("+", "-", "+-") デフォルト: "+-"
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    expr_str = kwargs.get("expression")
    var_str = kwargs.get("variable", "x")
    point = kwargs.get("point")
    
    if expr_str is None and source_text:
        text = source_text
        expr_str = _parse_expression(text)
        var_str = _parse_variable(text)
        
        # "as x approaches 2" から点を抽出
        match = re.search(r'approaches\s+(\d+)', text.lower())
        if match:
            point = float(match.group(1))
    
    if expr_str is None or point is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        # 暗黙的乗算を修正
        expr_str = _fix_implicit_multiplication(expr_str)
        
        # ^ を ** に変換
        expr_str = expr_str.replace("^", "**")
        
        var = symbols(var_str)
        expr = parse_expr(expr_str)
        
        # 方向
        direction = kwargs.get("direction", "+-")
        
        # Import limit from sympy properly
        from sympy import limit as sympy_limit
        
        if direction == "+":
            lim = sympy_limit(expr, var, point, "+")
        elif direction == "-":
            lim = sympy_limit(expr, var, point, "-")
        else:
            lim = sympy_limit(expr, var, point)
        
        result_str = str(lim)
        
        # 数値の場合
        if lim.is_number:
            return {
                "success": True,
                "value": float(lim),
                "confidence": 1.0,
                "schema": "decimal"
            }
        
        return {
            "success": True,
            "value": result_str,
            "confidence": 1.0,
            "schema": "decimal"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }


def series_sum(**kwargs) -> Dict[str, Any]:
    """
    級数和を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
        expression: str (例: "1/2^n")
        variable: str (デフォルト: "n")
        start: int (開始値)
        end: int or "inf" (終了値)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    expr_str = kwargs.get("expression")
    var_str = kwargs.get("variable", "n")
    start = kwargs.get("start", 0)
    end = kwargs.get("end", "inf")
    
    if expr_str is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        # 暗黙的乗算を修正
        expr_str = _fix_implicit_multiplication(expr_str)
        
        # ^ を ** に変換
        expr_str = expr_str.replace("^", "**")
        
        var = symbols(var_str)
        expr = parse_expr(expr_str)
        
        if end == "inf" or end == float('inf'):
            # 無限級数
            sum_result = None
            # 簡単な幾何級数などを試す
            try:
                sum_result = simplify(sum(expr.subs(var, i) for i in range(start, 100)))
            except:
                pass
        else:
            # 有限級数
            sum_result = sum(expr.subs(var, i) for i in range(start, end + 1))
        
        if sum_result is None:
            return {
                "success": False,
                "value": None,
                "confidence": 0.0,
                "schema": "decimal"
            }
        
        result_str = str(sum_result)
        
        # 数値の場合
        if sum_result.is_number:
            return {
                "success": True,
                "value": float(sum_result),
                "confidence": 1.0,
                "schema": "decimal"
            }
        
        return {
            "success": True,
            "value": result_str,
            "confidence": 1.0,
            "schema": "decimal"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }


# =======================================
# Legacy class-based implementation (unused)
# =======================================

class CalculusExecutor:
    """微積分計算の実行"""
    
    def execute(self, func_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        微積分関数を実行
        
        Args:
            func_name: 関数名 (derivative, integral, limit, series_sum)
            params: パラメータ辞書
            
        Returns:
            {
                "success": bool,
                "result": Any,
                "confidence": float,
                "method": str
            }
        """
        try:
            if func_name == "derivative":
                return self._derivative(params)
            elif func_name == "integral":
                return self._integral(params)
            elif func_name == "limit":
                return self._limit(params)
            elif func_name == "series_sum":
                return self._series_sum(params)
            else:
                return {
                    "success": False,
                    "result": None,
                    "confidence": 0.0,
                    "method": f"unknown_function: {func_name}"
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"error: {str(e)}"
            }
    
    def _parse_expression(self, text: str) -> Optional[str]:
        """
        テキストから数式を抽出
        
        Patterns:
            - "derivative of x^2" → "x^2"
            - "integral of 2x" → "2x"
            - "f(x) = x^2" → "x^2"
        """
        text_lower = text.lower()
        
        # "of ..." パターン
        if " of " in text_lower:
            after_of = text[text_lower.index(" of ") + 4:]
            # "with respect to"の前まで
            if "with respect" in after_of.lower():
                expr = after_of[:after_of.lower().index("with respect")].strip()
            else:
                expr = after_of.strip().rstrip(".")
            return expr
        
        # "f(x) = ..." パターン
        if "=" in text:
            expr = text.split("=")[1].strip().rstrip(".")
            return expr
        
        return None
    
    def _parse_variable(self, text: str) -> str:
        """
        変数名を抽出（デフォルト: x）
        
        Patterns:
            - "with respect to x" → "x"
            - "with respect to t" → "t"
        """
        text_lower = text.lower()
        
        if "with respect to" in text_lower:
            after = text_lower[text_lower.index("with respect to") + 16:].strip()
            # 最初の単語
            match = re.match(r'([a-z])', after)
            if match:
                return match.group(1)
        
        return "x"  # デフォルト
    
    def _derivative(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        導関数を計算
        
        Params:
            expression: str (例: "x^2") または
            source_text: str (例: "derivative of x^2 with respect to x")
            variable: str (デフォルト: "x")
        """
        expr_str = params.get("expression")
        var_str = params.get("variable", "x")
        
        if expr_str is None and "source_text" in params:
            expr_str = self._parse_expression(params["source_text"])
            var_str = self._parse_variable(params["source_text"])
        
        if expr_str is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_expression"
            }
        
        try:
            # ^ を ** に変換
            expr_str = expr_str.replace("^", "**")
            
            var = symbols(var_str)
            expr = parse_expr(expr_str)
            derivative = diff(expr, var)
            
            result_str = str(derivative)
            
            # 定数の場合は数値化
            if derivative.is_number:
                return {
                    "success": True,
                    "result": float(derivative),
                    "confidence": 1.0,
                    "method": "sympy_diff"
                }
            
            return {
                "success": True,
                "result": result_str,
                "confidence": 1.0,
                "method": "sympy_diff"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"parse_error: {str(e)}"
            }
    
    def _integral(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        積分を計算（不定積分、定積分）
        
        Params:
            expression: str (例: "2x")
            variable: str (デフォルト: "x")
            lower_limit: float (optional)
            upper_limit: float (optional)
        """
        expr_str = params.get("expression")
        var_str = params.get("variable", "x")
        
        if expr_str is None and "source_text" in params:
            expr_str = self._parse_expression(params["source_text"])
            var_str = self._parse_variable(params["source_text"])
        
        if expr_str is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_expression"
            }
        
        try:
            # ^ を ** に変換
            expr_str = expr_str.replace("^", "**")
            
            var = symbols(var_str)
            expr = parse_expr(expr_str)
            
            lower = params.get("lower_limit")
            upper = params.get("upper_limit")
            
            if lower is not None and upper is not None:
                # 定積分
                integral_result = integrate(expr, (var, lower, upper))
            else:
                # 不定積分
                integral_result = integrate(expr, var)
            
            result_str = str(integral_result)
            
            # 定数の場合は数値化
            if integral_result.is_number:
                return {
                    "success": True,
                    "result": float(integral_result),
                    "confidence": 1.0,
                    "method": "sympy_integrate"
                }
            
            return {
                "success": True,
                "result": result_str,
                "confidence": 1.0,
                "method": "sympy_integrate"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"parse_error: {str(e)}"
            }
    
    def _limit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        極限を計算
        
        Params:
            expression: str (例: "(x^2 - 4)/(x - 2)")
            variable: str (デフォルト: "x")
            point: float (極限を取る点)
            direction: str ("+", "-", "+-") デフォルト: "+-"
        """
        expr_str = params.get("expression")
        var_str = params.get("variable", "x")
        point = params.get("point")
        
        if expr_str is None and "source_text" in params:
            text = params["source_text"]
            expr_str = self._parse_expression(text)
            var_str = self._parse_variable(text)
            
            # "as x approaches 2" から点を抽出
            match = re.search(r'approaches\s+(\d+)', text.lower())
            if match:
                point = float(match.group(1))
        
        if expr_str is None or point is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_expression_or_point"
            }
        
        try:
            # ^ を ** に変換
            expr_str = expr_str.replace("^", "**")
            
            var = symbols(var_str)
            expr = parse_expr(expr_str)
            
            # 方向
            direction = params.get("direction", "+-")
            
            if direction == "+":
                lim = limit(expr, var, point, "+")
            elif direction == "-":
                lim = limit(expr, var, point, "-")
            else:
                lim = limit(expr, var, point)
            
            result_str = str(lim)
            
            # 数値の場合
            if lim.is_number:
                return {
                    "success": True,
                    "result": float(lim),
                    "confidence": 1.0,
                    "method": "sympy_limit"
                }
            
            return {
                "success": True,
                "result": result_str,
                "confidence": 1.0,
                "method": "sympy_limit"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"parse_error: {str(e)}"
            }
    
    def _series_sum(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        級数和を計算
        
        Params:
            expression: str (例: "1/2^n")
            variable: str (デフォルト: "n")
            start: int (開始値)
            end: int or "inf" (終了値)
        """
        expr_str = params.get("expression")
        var_str = params.get("variable", "n")
        start = params.get("start", 0)
        end = params.get("end", "inf")
        
        if expr_str is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_expression"
            }
        
        try:
            # ^ を ** に変換
            expr_str = expr_str.replace("^", "**")
            
            var = symbols(var_str)
            expr = parse_expr(expr_str)
            
            if end == "inf" or end == float('inf'):
                # 無限級数
                sum_result = None
                # 簡単な幾何級数などを試す
                try:
                    sum_result = simplify(sum(expr.subs(var, i) for i in range(start, 100)))
                except:
                    pass
            else:
                # 有限級数
                sum_result = sum(expr.subs(var, i) for i in range(start, end + 1))
            
            if sum_result is None:
                return {
                    "success": False,
                    "result": None,
                    "confidence": 0.0,
                    "method": "series_diverges"
                }
            
            result_str = str(sum_result)
            
            # 数値の場合
            if sum_result.is_number:
                return {
                    "success": True,
                    "result": float(sum_result),
                    "confidence": 1.0,
                    "method": "series_sum"
                }
            
            return {
                "success": True,
                "result": result_str,
                "confidence": 1.0,
                "method": "series_sum"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"parse_error: {str(e)}"
            }
