"""
Number Theory Executor - 数論演算
"""

import math
from typing import Dict, Any, List


def is_prime(number: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    素数判定
    
    Args:
        number: 判定する数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # 数値の取得
    if number is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                number = entity.get("value")
                break
    
    if number is None:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": "No number found"
        }
    
    try:
        n = int(number)
        
        if n < 2:
            result = False
        elif n == 2:
            result = True
        elif n % 2 == 0:
            result = False
        else:
            # Trial division
            result = True
            for i in range(3, int(math.sqrt(n)) + 1, 2):
                if n % i == 0:
                    result = False
                    break
        
        return {
            "value": result,
            "schema": "boolean",
            "confidence": 1.0,
            "artifacts": {"number": n, "method": "trial_division"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": str(e)
        }


def count_divisors(number: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    約数の個数を数える
    
    Args:
        number: 数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if number is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                number = entity.get("value")
                break
    
    if number is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No number found"
        }
    
    try:
        n = int(number)
        
        if n <= 0:
            return {
                "value": 0,
                "schema": "integer",
                "confidence": 1.0
            }
        
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        
        return {
            "value": count,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"number": n}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def gcd(a: int = None, b: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    最大公約数を計算
    
    Args:
        a: 数1
        b: 数2
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # 数値の取得
    if (a is None or b is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
    
    if a is None or b is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "Need two numbers"
        }
    
    try:
        result = math.gcd(int(a), int(b))
        
        return {
            "value": result,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"a": a, "b": b}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def lcm(a: int = None, b: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    最小公倍数を計算
    
    Args:
        a: 数1
        b: 数2
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (a is None or b is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
    
    if a is None or b is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "Need two numbers"
        }
    
    try:
        a_int, b_int = int(a), int(b)
        result = abs(a_int * b_int) // math.gcd(a_int, b_int)
        
        return {
            "value": result,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"a": a, "b": b}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def factorial(n: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    階乗を計算
    
    Args:
        n: 数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if n is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                n = entity.get("value")
                break
    
    if n is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No number found"
        }
    
    try:
        n_int = int(n)
        
        if n_int < 0:
            return {
                "value": None,
                "schema": "integer",
                "confidence": 0.0,
                "error": "Factorial of negative number"
            }
        
        if n_int > 1000:
            return {
                "value": None,
                "schema": "integer",
                "confidence": 0.0,
                "error": "Number too large (>1000)"
            }
        
        result = math.factorial(n_int)
        
        return {
            "value": result,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"n": n_int}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }
