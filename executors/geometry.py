"""
Geometry Executor - 幾何計算
"""

import math
from typing import Dict, Any


def triangle_area(base: float = None, height: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    三角形の面積 A = (base * height) / 2
    
    Args:
        base: 底辺
        height: 高さ
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (base is None or height is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            base, height = numbers[0], numbers[1]
    
    if base is None or height is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need base and height"
        }
    
    try:
        result = (float(base) * float(height)) / 2
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"base": base, "height": height, "formula": "(base*height)/2"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def circle_area(radius: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    円の面積 A = π * r^2
    
    Args:
        radius: 半径
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if radius is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                radius = entity.get("value")
                break
    
    if radius is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need radius"
        }
    
    try:
        result = math.pi * float(radius) ** 2
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"radius": radius, "formula": "π*r²"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def circle_circumference(radius: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    円周 C = 2 * π * r
    
    Args:
        radius: 半径
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if radius is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                radius = entity.get("value")
                break
    
    if radius is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need radius"
        }
    
    try:
        result = 2 * math.pi * float(radius)
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"radius": radius, "formula": "2πr"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def rectangle_area(width: float = None, height: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    長方形の面積 A = width * height
    
    Args:
        width: 幅
        height: 高さ
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (width is None or height is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            width, height = numbers[0], numbers[1]
    
    if width is None or height is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need width and height"
        }
    
    try:
        result = float(width) * float(height)
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"width": width, "height": height}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def rectangle_perimeter(length: float = None, width: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    長方形の周囲長 P = 2 * (length + width)
    
    Args:
        length: 長さ
        width: 幅
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (length is None or width is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            length, width = numbers[0], numbers[1]
    
    if length is None or width is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need length and width"
        }
    
    try:
        result = 2 * (float(length) + float(width))
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"length": length, "width": width, "formula": "2*(length+width)"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def pythagorean(a: float = None, b: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    ピタゴラスの定理 c = √(a² + b²)
    
    Args:
        a: 辺a
        b: 辺b
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
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need two sides (a, b)"
        }
    
    try:
        result = math.sqrt(float(a) ** 2 + float(b) ** 2)
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"a": a, "b": b, "formula": "√(a²+b²)"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def sphere_volume(radius: float = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    球の体積 V = (4/3) * π * r^3
    
    Args:
        radius: 半径
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if radius is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                radius = entity.get("value")
                break
    
    if radius is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need radius"
        }
    
    try:
        result = (4/3) * math.pi * float(radius) ** 3
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"radius": radius, "formula": "(4/3)πr³"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }
