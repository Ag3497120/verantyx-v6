"""
Modular Arithmetic Executor - 剰余演算（高度な数論）
"""
import math
from typing import Any, Dict, List, Tuple, Optional


# =======================================
# Helper functions
# =======================================

def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    拡張ユークリッドの互除法
    
    Returns:
        (gcd, x, y) where gcd = ax + by
    """
    if b == 0:
        return a, 1, 0
    
    gcd, x1, y1 = _extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return gcd, x, y


def _mod_inverse_single(a: int, m: int) -> Optional[int]:
    """
    逆元を求める (a * x ≡ 1 mod m)
    
    Returns:
        x if exists, None otherwise
    """
    gcd, x, _ = _extended_gcd(a, m)
    
    if gcd != 1:
        return None  # 逆元は存在しない
    
    return (x % m + m) % m


def _prime_factors(n: int) -> List[int]:
    """素因数を列挙"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            if not factors or factors[-1] != d:
                factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


# =======================================
# Executor functions (for piece calls)
# =======================================

def mod_power(**kwargs) -> Dict[str, Any]:
    """
    べき乗の余りを計算 (a^b mod m)
    
    Params:
        a: int (底)
        b: int (指数)
        m: int (法)
    
    Uses: 高速べき乗法（繰り返し二乗法）
    """
    ir = kwargs.get("ir", {})
    
    a = kwargs.get("a")
    b = kwargs.get("b")
    m = kwargs.get("m")
    
    # IRのentitiesから抽出
    if a is None or b is None or m is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 3:
            a, b, m = numbers[0], numbers[1], numbers[2]
    
    if a is None or b is None or m is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    try:
        # Pythonの組み込み関数を使用（高速）
        result = pow(int(a), int(b), int(m))
        
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


def mod_inverse(**kwargs) -> Dict[str, Any]:
    """
    逆元を求める (a * x ≡ 1 mod m)
    
    Params:
        a: int
        m: int (法)
    
    Returns:
        x such that a*x ≡ 1 (mod m)
    """
    ir = kwargs.get("ir", {})
    
    a = kwargs.get("a")
    m = kwargs.get("m")
    
    # IRのentitiesから抽出
    if a is None or m is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            a, m = numbers[0], numbers[1]
    
    if a is None or m is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    try:
        result = _mod_inverse_single(int(a), int(m))
        
        if result is None:
            return {
                "success": False,
                "value": None,
                "confidence": 0.0,
                "schema": "integer"
            }
        
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


def chinese_remainder(**kwargs) -> Dict[str, Any]:
    """
    中国剰余定理
    
    Params:
        remainders: List[int] (余り)
        moduli: List[int] (法)
    
    Solve:
        x ≡ r1 (mod m1)
        x ≡ r2 (mod m2)
        ...
    
    Returns:
        x (最小の非負整数解)
    """
    ir = kwargs.get("ir", {})
    
    remainders = kwargs.get("remainders")
    moduli = kwargs.get("moduli")
    
    if remainders is None or moduli is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    if len(remainders) != len(moduli):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "integer"
        }
    
    try:
        # 中国剰余定理のアルゴリズム
        total = 0
        prod = 1
        for m in moduli:
            prod *= m
        
        for r, m in zip(remainders, moduli):
            p = prod // m
            inv = _mod_inverse_single(p, m)
            if inv is None:
                return {
                    "success": False,
                    "value": None,
                    "confidence": 0.0,
                    "schema": "integer"
                }
            total += r * p * inv
        
        result = total % prod
        
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


def euler_phi(**kwargs) -> Dict[str, Any]:
    """
    オイラーのφ関数（トーシェント関数）
    
    φ(n) = nと互いに素な正の整数（≤n）の個数
    
    Params:
        n: int
    
    Formula:
        φ(n) = n * ∏(1 - 1/p) for all prime p | n
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
        result = n
        
        # 素因数で割る
        i = 2
        while i * i <= n:
            if n % i == 0:
                # iで割り切れる間割る
                while n % i == 0:
                    n //= i
                # φ(n) = φ(n) * (1 - 1/i) = φ(n) * (i-1)/i
                result -= result // i
            i += 1
        
        # 残りが素数の場合
        if n > 1:
            result -= result // n
        
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


def fermat_little(**kwargs) -> Dict[str, Any]:
    """
    フェルマーの小定理を利用した計算
    
    a^(p-1) ≡ 1 (mod p) (pが素数、gcd(a,p)=1)
    
    Params:
        a: int
        p: int (素数)
        mode: str ("verify" or "compute")
    
    Mode:
        - verify: a^(p-1) ≡ 1 (mod p) を検証
        - compute: a^(p-1) mod p を計算
    """
    ir = kwargs.get("ir", {})
    
    a = kwargs.get("a")
    p = kwargs.get("p")
    mode = kwargs.get("mode", "compute")
    
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
            "schema": "integer" if mode == "compute" else "boolean"
        }
    
    try:
        a = int(a)
        p = int(p)
        
        # a^(p-1) mod p を計算
        result = pow(a, p - 1, p)
        
        if mode == "verify":
            # 1と等しいか検証
            return {
                "success": True,
                "value": (result == 1),
                "confidence": 1.0,
                "schema": "boolean"
            }
        else:
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
            "schema": "integer" if mode == "compute" else "boolean"
        }
