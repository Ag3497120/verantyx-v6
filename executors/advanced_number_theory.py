"""
Advanced Number Theory Executor - 高度な数論
"""
import random
from typing import Any, Dict, List, Optional, Tuple


# =======================================
# Helper functions
# =======================================

def _miller_rabin_test(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin素数判定法
    
    Args:
        n: 判定する数
        k: テスト回数（多いほど精度が高い）
    
    Returns:
        True if probably prime, False if composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # n - 1 = 2^r * d の形に分解
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # k回のテスト
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def _prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    素因数分解
    
    Returns:
        [(prime, exponent), ...]
    """
    if n <= 1:
        return []
    
    factors = []
    d = 2
    
    while d * d <= n:
        exp = 0
        while n % d == 0:
            exp += 1
            n //= d
        if exp > 0:
            factors.append((d, exp))
        d += 1
    
    if n > 1:
        factors.append((n, 1))
    
    return factors


def _is_primitive_root(g: int, p: int) -> bool:
    """
    原始根判定
    
    gがmod pの原始根 ⟺ ord_p(g) = φ(p) = p-1
    """
    if p <= 1 or g <= 0:
        return False
    
    # φ(p) = p - 1 (pが素数の場合)
    phi = p - 1
    
    # φ(p)の素因数で割った値でg^(φ/q) ≡ 1 (mod p) をチェック
    factors = _prime_factorization(phi)
    
    for prime, _ in factors:
        if pow(g, phi // prime, p) == 1:
            return False
    
    return True


def _legendre_symbol(a: int, p: int) -> int:
    """
    ルジャンドル記号 (a/p)
    
    Returns:
        1 if a is quadratic residue mod p
        -1 if a is not quadratic residue mod p
        0 if a ≡ 0 (mod p)
    """
    if a % p == 0:
        return 0
    
    # オイラーの規準: (a/p) ≡ a^((p-1)/2) (mod p)
    result = pow(a, (p - 1) // 2, p)
    
    return -1 if result == p - 1 else result


# =======================================
# Executor functions (for piece calls)
# =======================================

def prime_factorization(**kwargs) -> Dict[str, Any]:
    """
    素因数分解
    
    Params:
        n: int
    
    Returns:
        List[(prime, exponent)] or formatted string
    """
    ir = kwargs.get("ir", {})
    
    n = kwargs.get("n")
    
    # IRのentitiesから抽出
    if n is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if numbers:
            n = numbers[0]
    
    if n is None or n <= 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "list"
        }
    
    try:
        n = int(n)
        factors = _prime_factorization(n)
        
        return {
            "success": True,
            "value": factors,
            "confidence": 1.0,
            "schema": "list"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "list"
        }


def miller_rabin(**kwargs) -> Dict[str, Any]:
    """
    Miller-Rabin素数判定
    
    Params:
        n: int (判定する数)
        k: int (テスト回数、デフォルト5)
    
    Returns:
        True if probably prime, False if composite
    """
    ir = kwargs.get("ir", {})
    
    n = kwargs.get("n")
    k = kwargs.get("k", 5)
    
    # IRのentitiesから抽出
    if n is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if numbers:
            n = numbers[0]
            if len(numbers) >= 2:
                k = numbers[1]
    
    if n is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }
    
    try:
        n = int(n)
        k = int(k)
        
        result = _miller_rabin_test(n, k)
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "boolean"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }


def is_primitive_root(**kwargs) -> Dict[str, Any]:
    """
    原始根判定
    
    Params:
        g: int (候補)
        p: int (法、素数)
    
    Returns:
        True if g is primitive root mod p
    """
    ir = kwargs.get("ir", {})
    
    g = kwargs.get("g")
    p = kwargs.get("p")
    
    # IRのentitiesから抽出
    if g is None or p is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            g, p = numbers[0], numbers[1]
    
    if g is None or p is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }
    
    try:
        g = int(g)
        p = int(p)
        
        result = _is_primitive_root(g, p)
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "boolean"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }


def quadratic_residue(**kwargs) -> Dict[str, Any]:
    """
    平方剰余判定
    
    Params:
        a: int
        p: int (素数)
    
    Returns:
        True if a is quadratic residue mod p
    """
    ir = kwargs.get("ir", {})
    
    a = kwargs.get("a")
    p = kwargs.get("p")
    
    # IRのentitiesから抽出
    if a is None or p is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            a, p = numbers[0], numbers[1]
    
    if a is None or p is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }
    
    try:
        a = int(a)
        p = int(p)
        
        # ルジャンドル記号を使用
        symbol = _legendre_symbol(a, p)
        result = (symbol == 1)
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "boolean"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "boolean"
        }


def legendre_symbol(**kwargs) -> Dict[str, Any]:
    """
    ルジャンドル記号 (a/p)
    
    Params:
        a: int
        p: int (素数)
    
    Returns:
        1, -1, or 0
    """
    ir = kwargs.get("ir", {})
    
    a = kwargs.get("a")
    p = kwargs.get("p")
    
    # IRのentitiesから抽出
    if a is None or p is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            a, p = numbers[0], numbers[1]
    
    if a is None or p is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    try:
        a = int(a)
        p = int(p)
        
        result = _legendre_symbol(a, p)
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "integer"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }


def divisor_sum(**kwargs) -> Dict[str, Any]:
    """
    約数の和（σ関数）
    
    σ(n) = Σd (dはnの約数)
    
    Params:
        n: int
    """
    ir = kwargs.get("ir", {})
    
    n = kwargs.get("n")
    
    # IRのentitiesから抽出
    if n is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if numbers:
            n = numbers[0]
    
    if n is None or n <= 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    try:
        n = int(n)
        
        # 素因数分解を使用した高速計算
        # σ(n) = ∏((p^(e+1) - 1) / (p - 1)) for (p, e) in factors
        factors = _prime_factorization(n)
        result = 1
        
        for prime, exp in factors:
            result *= (prime ** (exp + 1) - 1) // (prime - 1)
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "integer"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
