"""
Advanced Probability Executor - 高度な確率論
"""
import math
from typing import Any, Dict, List, Optional


# =======================================
# Helper functions
# =======================================

def _combination(n: int, k: int) -> int:
    """組み合わせ C(n, k)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _factorial(n: int) -> int:
    """階乗"""
    if n <= 1:
        return 1
    return math.factorial(n)


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    正規分布の累積分布関数（CDF）
    
    scipy使わない近似版
    """
    # 標準化
    z = (x - mu) / sigma
    
    # 誤差関数の近似（Abramowitz and Stegun）
    sign = 1 if z >= 0 else -1
    z = abs(z)
    
    # erf(z)の近似
    t = 1.0 / (1.0 + 0.3275911 * z)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = 1 - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * math.exp(-z * z)
    
    return 0.5 * (1 + sign * erf)


# =======================================
# Executor functions (for piece calls)
# =======================================

def conditional_probability(**kwargs) -> Dict[str, Any]:
    """
    条件付き確率 P(A|B) = P(A∩B) / P(B)
    
    Params:
        p_a_and_b: float (P(A∩B))
        p_b: float (P(B))
    
    または:
        p_a: float (P(A))
        p_b: float (P(B))
        p_a_given_b: float (P(A|B)) - 既知の場合
    """
    ir = kwargs.get("ir", {})
    
    p_a_and_b = kwargs.get("p_a_and_b")
    p_b = kwargs.get("p_b")
    
    if p_a_and_b is None or p_b is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if p_b == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        result = float(p_a_and_b) / float(p_b)
        
        return {
            "success": True,
            "value": result,
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


def bayes_theorem(**kwargs) -> Dict[str, Any]:
    """
    ベイズの定理
    
    P(A|B) = P(B|A) * P(A) / P(B)
    
    Params:
        p_b_given_a: float (P(B|A))
        p_a: float (P(A))
        p_b: float (P(B))
    """
    ir = kwargs.get("ir", {})
    
    p_b_given_a = kwargs.get("p_b_given_a")
    p_a = kwargs.get("p_a")
    p_b = kwargs.get("p_b")
    
    if p_b_given_a is None or p_a is None or p_b is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if p_b == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        result = float(p_b_given_a) * float(p_a) / float(p_b)
        
        return {
            "success": True,
            "value": result,
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


def binomial_distribution(**kwargs) -> Dict[str, Any]:
    """
    二項分布
    
    P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
    
    Params:
        n: int (試行回数)
        k: int (成功回数)
        p: float (成功確率)
    """
    ir = kwargs.get("ir", {})
    
    n = kwargs.get("n")
    k = kwargs.get("k")
    p = kwargs.get("p")
    
    # IRのentitiesから抽出
    if n is None or k is None or p is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 3:
            n, k, p = int(numbers[0]), int(numbers[1]), float(numbers[2])
        elif len(numbers) >= 2:
            n, k = int(numbers[0]), int(numbers[1])
            # pはデフォルト0.5
            p = 0.5
    
    if n is None or k is None or p is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        n = int(n)
        k = int(k)
        p = float(p)
        
        if k > n or k < 0:
            return {
                "success": True,
                "value": 0.0,
                "confidence": 1.0,
                "schema": "decimal"
            }
        
        # C(n, k) * p^k * (1-p)^(n-k)
        comb = _combination(n, k)
        result = comb * (p ** k) * ((1 - p) ** (n - k))
        
        return {
            "success": True,
            "value": result,
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


def normal_distribution(**kwargs) -> Dict[str, Any]:
    """
    正規分布（ガウス分布）
    
    Params:
        x: float (値)
        mu: float (平均、デフォルト0)
        sigma: float (標準偏差、デフォルト1)
        mode: str ("pdf" or "cdf")
    
    Returns:
        - pdf: 確率密度関数の値
        - cdf: 累積分布関数の値
    """
    ir = kwargs.get("ir", {})
    
    x = kwargs.get("x")
    mu = kwargs.get("mu", 0.0)
    sigma = kwargs.get("sigma", 1.0)
    mode = kwargs.get("mode", "cdf")
    
    # IRのentitiesから抽出
    if x is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if numbers:
            x = float(numbers[0])
            if len(numbers) >= 2:
                mu = float(numbers[1])
            if len(numbers) >= 3:
                sigma = float(numbers[2])
    
    if x is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        x = float(x)
        mu = float(mu)
        sigma = float(sigma)
        
        if mode == "pdf":
            # 確率密度関数
            coefficient = 1.0 / (sigma * math.sqrt(2 * math.pi))
            exponent = -0.5 * ((x - mu) / sigma) ** 2
            result = coefficient * math.exp(exponent)
        else:
            # 累積分布関数
            result = _normal_cdf(x, mu, sigma)
        
        return {
            "success": True,
            "value": result,
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


def poisson_distribution(**kwargs) -> Dict[str, Any]:
    """
    ポアソン分布
    
    P(X = k) = (λ^k * e^(-λ)) / k!
    
    Params:
        k: int (事象の発生回数)
        lambda_: float (平均発生率λ)
    """
    ir = kwargs.get("ir", {})
    
    k = kwargs.get("k")
    lambda_ = kwargs.get("lambda")
    
    # IRのentitiesから抽出
    if k is None or lambda_ is None:
        entities = ir.get("entities", [])
        numbers = [e.get("value") for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            k, lambda_ = int(numbers[0]), float(numbers[1])
    
    if k is None or lambda_ is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        k = int(k)
        lambda_ = float(lambda_)
        
        # (λ^k * e^(-λ)) / k!
        result = (lambda_ ** k) * math.exp(-lambda_) / _factorial(k)
        
        return {
            "success": True,
            "value": result,
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


def expected_value_advanced(**kwargs) -> Dict[str, Any]:
    """
    期待値の計算（分布ベース）
    
    Params:
        values: List[float] (値のリスト)
        probabilities: List[float] (確率のリスト)
    
    Returns:
        E[X] = Σ(x * P(X=x))
    """
    ir = kwargs.get("ir", {})
    
    values = kwargs.get("values")
    probabilities = kwargs.get("probabilities")
    
    if values is None or probabilities is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(values) != len(probabilities):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        result = sum(v * p for v, p in zip(values, probabilities))
        
        return {
            "success": True,
            "value": result,
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
