"""
Arithmetic Executor - 数式評価

verantyx_ios ArithmeticSolverをポート（強化版）
"""

import ast
import operator
import re
from typing import Dict, Any


class SafeEvaluator:
    """安全なAST評価器"""
    
    def __init__(self):
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
    
    def eval(self, node):
        """ASTノードを評価"""
        # Python 3.8+ では ast.Constant のみ
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {type(node.value)}")
        # Python 3.7以前との互換性
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self.eval(node.left)
            right = self.eval(node.right)
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self.eval(node.operand)
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise TypeError(f"Unsupported AST node: {type(node)}")


def evaluate(expression: str = None, ir: Dict = None, context: Dict = None,
             base=None, exponent=None, **kwargs) -> Dict[str, Any]:
    """
    数式を評価

    Args:
        expression: 評価する数式（"1+1"等）
        base:       べき乗の底（arithmetic_power スロット）
        exponent:   べき乗の指数（arithmetic_power スロット）
        ir:         IR辞書
        context:    実行コンテキスト

    Returns:
        実行結果
    """
    evaluator = SafeEvaluator()

    # Layer A Fix [06]: base/exponent スロットから expression を構築
    if expression is None and base is not None and exponent is not None:
        try:
            expression = f"{base}**{exponent}"
        except Exception:
            pass

    # 式の取得
    if expression is None:
        # IRから抽出
        if ir:
            # Layer A: base/exponent エンティティが IR にある場合
            entities = ir.get("entities", [])
            base_ent  = next((e for e in entities if e.get("name") == "base"),     None)
            exp_ent   = next((e for e in entities if e.get("name") == "exponent"), None)
            if base_ent and exp_ent:
                try:
                    expression = f"{base_ent['value']}**{exp_ent['value']}"
                except Exception:
                    pass

        if expression is None and ir:
            # エンティティから
            for entity in ir.get("entities", []):
                if entity.get("type") in ("expression", "number"):
                    val = entity.get("value")
                    if val:
                        expression = str(val)
                        break
            
            # source_textから数式を抽出
            if expression is None:
                source_text = ir.get("metadata", {}).get("source_text", "")
                if source_text:
                    # "What is 1+1?" -> "1+1"
                    # "Calculate 5 * 6" -> "5*6"
                    expr_match = re.search(r'(\d+\.?\d*\s*[\+\-\*\/\^]\s*\d+\.?\d*)', source_text)
                    if expr_match:
                        expression = expr_match.group(1)
        
        # コンテキストから抽出
        if expression is None and context and "artifacts" in context:
            expression = context["artifacts"].get("expression")
    
    if not expression:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No expression found"
        }
    
    # クリーニング
    clean_expr = str(expression).replace(" ", "").strip()
    
    # ^ を ** に変換（冪乗）
    clean_expr = clean_expr.replace("^", "**")
    
    # 数値のみの場合
    try:
        if re.match(r'^-?\d+\.?\d*$', clean_expr):
            result = float(clean_expr) if "." in clean_expr else int(clean_expr)
            schema = "decimal" if isinstance(result, float) else "integer"
            return {
                "value": result,
                "schema": schema,
                "confidence": 1.0,
                "artifacts": {"expression": clean_expr}
            }
    except:
        pass
    
    try:
        # AST解析して評価
        tree = ast.parse(clean_expr, mode='eval')
        result = evaluator.eval(tree.body)
        
        # 結果の型判定
        if isinstance(result, float):
            # 整数かチェック
            if result.is_integer():
                result = int(result)
                schema = "integer"
            else:
                schema = "decimal"
        elif isinstance(result, int):
            schema = "integer"
        else:
            schema = "expression"
        
        return {
            "value": result,
            "schema": schema,
            "confidence": 1.0,
            "artifacts": {
                "expression": clean_expr,
                "result_type": type(result).__name__
            }
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def check_equality(lhs: str = None, rhs: str = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    等式が成り立つかチェック
    
    Args:
        lhs: 左辺
        rhs: 右辺
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    evaluator = SafeEvaluator()
    
    # 制約から等式を抽出
    if lhs is None or rhs is None:
        if ir and "constraints" in ir:
            for constraint in ir["constraints"]:
                if constraint.get("type") == "equals":
                    lhs = constraint.get("lhs")
                    rhs = constraint.get("rhs")
                    break
    
    if lhs is None or rhs is None:
        return {
            "value": False,
            "schema": "boolean",
            "confidence": 0.0,
            "error": "Missing lhs or rhs"
        }
    
    try:
        # 両辺を評価
        lhs_tree = ast.parse(str(lhs), mode='eval')
        lhs_val = evaluator.eval(lhs_tree.body)
        
        rhs_tree = ast.parse(str(rhs), mode='eval')
        rhs_val = evaluator.eval(rhs_tree.body)
        
        # 比較（浮動小数点の許容誤差）
        if isinstance(lhs_val, float) or isinstance(rhs_val, float):
            result = abs(lhs_val - rhs_val) < 1e-9
        else:
            result = lhs_val == rhs_val
        
        return {
            "value": result,
            "schema": "boolean",
            "confidence": 1.0,
            "artifacts": {
                "lhs": lhs,
                "rhs": rhs,
                "lhs_value": lhs_val,
                "rhs_value": rhs_val
            }
        }
    
    except Exception as e:
        return {
            "value": False,
            "schema": "boolean",
            "confidence": 0.0,
            "error": str(e)
        }
