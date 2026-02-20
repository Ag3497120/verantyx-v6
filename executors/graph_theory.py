"""
Graph Theory Executor - グラフ理論計算
"""

import math
from typing import Dict, Any


def complete_graph_edges(n: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    完全グラフの辺の数: C(n, 2) = n(n-1)/2
    
    Args:
        n: 頂点数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if n is None and ir:
        entities = ir.get("entities", [])
        for entity in entities:
            if entity.get("type") == "number":
                n = int(entity.get("value"))
                break
    
    if n is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No vertex count found"
        }
    
    try:
        # 完全グラフの辺数: C(n, 2) = n(n-1)/2
        edges = n * (n - 1) // 2
        
        return {
            "value": edges,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"vertices": n, "formula": "n(n-1)/2"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def tree_minimum_edges(n: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    木の最小辺数: n - 1
    
    Args:
        n: 頂点数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if n is None and ir:
        entities = ir.get("entities", [])
        for entity in entities:
            if entity.get("type") == "number":
                n = int(entity.get("value"))
                break
    
    if n is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No vertex count found"
        }
    
    try:
        # 木の辺数: n - 1
        edges = n - 1
        
        return {
            "value": edges,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"vertices": n, "formula": "n-1"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def is_cyclic(vertices: int = None, edges: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    グラフがサイクルを持つか判定
    
    Args:
        vertices: 頂点数
        edges: 辺数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (vertices is None or edges is None) and ir:
        entities = ir.get("entities", [])
        numbers = [int(e.get("value")) for e in entities if e.get("type") == "number"]
        if len(numbers) >= 2:
            vertices = numbers[0]
            edges = numbers[1]
    
    if vertices is None or edges is None:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": "Missing vertices or edges count"
        }
    
    try:
        # 連結グラフの場合、辺数 >= 頂点数 ならサイクルあり
        # 木の辺数は n-1、それより多ければサイクルあり
        has_cycle = edges >= vertices
        
        return {
            "value": has_cycle,
            "schema": "boolean",
            "confidence": 1.0,
            "artifacts": {"vertices": vertices, "edges": edges}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": str(e)
        }


def degree_sum(edges: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    次数の和: 2 * 辺数（握手補題）
    
    Args:
        edges: 辺数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if edges is None and ir:
        entities = ir.get("entities", [])
        for entity in entities:
            if entity.get("type") == "number":
                edges = int(entity.get("value"))
                break
    
    if edges is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No edge count found"
        }
    
    try:
        # 握手補題: Σdeg(v) = 2|E|
        degree_sum_value = 2 * edges
        
        return {
            "value": degree_sum_value,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"edges": edges, "formula": "2*edges"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }


def binary_tree_vertices(height: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    完全二分木の頂点数: 2^(h+1) - 1
    
    Args:
        height: 高さ
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if height is None and ir:
        entities = ir.get("entities", [])
        for entity in entities:
            if entity.get("type") == "number":
                height = int(entity.get("value"))
                break
    
    if height is None:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": "No height found"
        }
    
    try:
        # 完全二分木の頂点数: 2^(h+1) - 1
        vertices = (2 ** (height + 1)) - 1
        
        return {
            "value": vertices,
            "schema": "integer",
            "confidence": 1.0,
            "artifacts": {"height": height, "formula": "2^(h+1)-1"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "integer",
            "confidence": 0.0,
            "error": str(e)
        }
