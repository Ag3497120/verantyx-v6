"""
Statistics Executor - 統計学
"""
import math
from typing import Any, Dict, List, Optional


# =======================================
# Helper functions
# =======================================

def _mean(data: List[float]) -> float:
    """平均値"""
    if not data:
        return 0.0
    return sum(data) / len(data)


# =======================================
# Executor functions (for piece calls)
# =======================================

def variance(**kwargs) -> Dict[str, Any]:
    """
    分散
    
    Var(X) = E[(X - μ)^2] = Σ(x - μ)^2 / n
    
    Params:
        data: List[float] (データ)
        population: bool (母集団分散か標本分散か、デフォルトTrue)
    """
    ir = kwargs.get("ir", {})
    
    data = kwargs.get("data")
    population = kwargs.get("population", True)
    
    if data is None or not isinstance(data, list):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(data) == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        data = [float(x) for x in data]
        mean = _mean(data)
        
        # 分散の計算
        squared_diffs = [(x - mean) ** 2 for x in data]
        
        if population:
            # 母集団分散
            result = sum(squared_diffs) / len(data)
        else:
            # 標本分散（不偏分散）
            if len(data) == 1:
                return {
                    "success": False,
                    "value": None,
                    "confidence": 0.0,
                    "schema": "decimal"
                }
            result = sum(squared_diffs) / (len(data) - 1)
        
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


def standard_deviation(**kwargs) -> Dict[str, Any]:
    """
    標準偏差
    
    σ = √Var(X)
    
    Params:
        data: List[float] (データ)
        population: bool (母集団か標本か、デフォルトTrue)
    """
    ir = kwargs.get("ir", {})
    
    data = kwargs.get("data")
    population = kwargs.get("population", True)
    
    # 分散を計算
    var_result = variance(data=data, population=population)
    
    if not var_result["success"]:
        return var_result
    
    try:
        var_value = var_result["value"]
        result = math.sqrt(var_value)
        
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


def covariance(**kwargs) -> Dict[str, Any]:
    """
    共分散
    
    Cov(X, Y) = E[(X - μX)(Y - μY)]
    
    Params:
        data_x: List[float] (Xのデータ)
        data_y: List[float] (Yのデータ)
        population: bool (母集団か標本か、デフォルトTrue)
    """
    ir = kwargs.get("ir", {})
    
    data_x = kwargs.get("data_x")
    data_y = kwargs.get("data_y")
    population = kwargs.get("population", True)
    
    if data_x is None or data_y is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(data_x) != len(data_y) or len(data_x) == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        data_x = [float(x) for x in data_x]
        data_y = [float(y) for y in data_y]
        
        mean_x = _mean(data_x)
        mean_y = _mean(data_y)
        
        # 共分散の計算
        products = [(x - mean_x) * (y - mean_y) for x, y in zip(data_x, data_y)]
        
        if population:
            # 母集団共分散
            result = sum(products) / len(data_x)
        else:
            # 標本共分散
            if len(data_x) == 1:
                return {
                    "success": False,
                    "value": None,
                    "confidence": 0.0,
                    "schema": "decimal"
                }
            result = sum(products) / (len(data_x) - 1)
        
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


def correlation(**kwargs) -> Dict[str, Any]:
    """
    相関係数（ピアソン）
    
    ρ = Cov(X, Y) / (σX * σY)
    
    Params:
        data_x: List[float] (Xのデータ)
        data_y: List[float] (Yのデータ)
    """
    ir = kwargs.get("ir", {})
    
    data_x = kwargs.get("data_x")
    data_y = kwargs.get("data_y")
    
    if data_x is None or data_y is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        # 共分散を計算
        cov_result = covariance(data_x=data_x, data_y=data_y, population=True)
        if not cov_result["success"]:
            return cov_result
        
        cov_value = cov_result["value"]
        
        # 標準偏差を計算
        std_x_result = standard_deviation(data=data_x, population=True)
        std_y_result = standard_deviation(data=data_y, population=True)
        
        if not std_x_result["success"] or not std_y_result["success"]:
            return {
                "success": False,
                "value": None,
                "confidence": 0.0,
                "schema": "decimal"
            }
        
        std_x = std_x_result["value"]
        std_y = std_y_result["value"]
        
        if std_x == 0 or std_y == 0:
            return {
                "success": False,
                "value": None,
                "confidence": 0.0,
                "schema": "decimal"
            }
        
        # 相関係数
        result = cov_value / (std_x * std_y)
        
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


def mean_calculation(**kwargs) -> Dict[str, Any]:
    """
    平均値の計算
    
    Params:
        data: List[float] (データ)
    """
    ir = kwargs.get("ir", {})
    
    data = kwargs.get("data")
    
    if data is None or not isinstance(data, list):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(data) == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        data = [float(x) for x in data]
        result = _mean(data)
        
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


def median_calculation(**kwargs) -> Dict[str, Any]:
    """
    中央値の計算
    
    Params:
        data: List[float] (データ)
    """
    ir = kwargs.get("ir", {})
    
    data = kwargs.get("data")
    
    if data is None or not isinstance(data, list):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(data) == 0:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        data = sorted([float(x) for x in data])
        n = len(data)
        
        if n % 2 == 0:
            # 偶数個の場合、中央2つの平均
            result = (data[n // 2 - 1] + data[n // 2]) / 2
        else:
            # 奇数個の場合、中央の値
            result = data[n // 2]
        
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
