"""
Combinatorics Executor - 組み合わせ論演算
"""

import math
from typing import Dict, Any


def permutation(n: int = None, r: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    順列 P(n, r) = n! / (n-r)!
    
    Args:
        n: 全体の数
        r: 選ぶ数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # 数値の取得
    if (n is None or r is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            n, r = numbers[0], numbers[1]
    
    if n is None or r is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "Need two numbers (n, r)"
        }
    
    try:
        n_int, r_int = int(n), int(r)
        
        if n_int < 0 or r_int < 0:
            return {
                "value": 0,
                "schema": "integer",
                "confidence": 1.0
            }
        
        if r_int > n_int:
            return {
                "value": 0,
                "schema": "integer",
                "confidence": 1.0
            }
        
        # P(n, r) = n! / (n-r)!
        result = math.factorial(n_int) // math.factorial(n_int - r_int)
        
        return {
            "value": result,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"n": n_int, "r": r_int, "formula": "n!/(n-r)!"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def combination(n: int = None, r: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    組み合わせ C(n, r) = n! / (r! * (n-r)!)
    
    Args:
        n: 全体の数
        r: 選ぶ数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (n is None or r is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            n, r = numbers[0], numbers[1]
    
    if n is None or r is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "Need two numbers (n, r)"
        }
    
    try:
        n_int, r_int = int(n), int(r)
        
        if n_int < 0 or r_int < 0:
            return {
                "value": 0,
                "schema": "integer",
                "confidence": 1.0
            }
        
        if r_int > n_int:
            return {
                "value": 0,
                "schema": "integer",
                "confidence": 1.0
            }
        
        # C(n, r) = n! / (r! * (n-r)!)
        result = math.factorial(n_int) // (math.factorial(r_int) * math.factorial(n_int - r_int))
        
        return {
            "value": result,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"n": n_int, "r": r_int, "formula": "n!/(r!*(n-r)!)"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def binomial_coefficient(n: int = None, k: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    二項係数（combinationのエイリアス）
    """
    return combination(n=n, r=k, ir=ir, context=context, **kwargs)
