"""
Probability Executor - 確率計算
"""

from typing import Dict, Any
import math


def basic_probability(favorable: int = None, total: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    基本確率 P = favorable / total
    
    Args:
        favorable: 有利な場合の数
        total: 全体の場合の数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if (favorable is None or total is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            favorable, total = numbers[0], numbers[1]
    
    if favorable is None or total is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need two numbers (favorable, total)"
        }
    
    try:
        fav_int, tot_int = int(favorable), int(total)
        
        if tot_int == 0:
            return {
                "value": None,
                "schema": "decimal",
                "confidence": 0.0,
                "error": "Total cannot be zero"
            }
        
        result = fav_int / tot_int
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"favorable": fav_int, "total": tot_int}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def coin_flip_probability(flips: int = 1, heads: int = 1, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    コイン投げの確率（公平なコイン）
    
    Args:
        flips: コイン投げの回数
        heads: 表の回数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    try:
        # 単純なケース: 1回のコイン投げ
        if flips == 1 and heads == 1:
            return {
                "value": 0.5,
                "schema": "decimal",
                "confidence": 1.0,
                "artifacts": {"flips": 1, "heads": 1}
            }
        
        # 複数回のコイン投げ（二項分布）
        from math import comb
        prob = comb(flips, heads) * (0.5 ** flips)
        
        return {
            "value": prob,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"flips": flips, "heads": heads}
        }
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def coin_flip_multiple(ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    複数回コイン投げの確率（テキストから自動抽出）
    
    "flip a coin twice" → flips=2
    "getting two heads" → heads=2
    "both heads" → heads=2
    
    Args:
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if not ir:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "IR required"
        }
    
    import re
    text = ir.get("metadata", {}).get("source_text", "").lower()
    
    # デフォルト値
    flips = 1
    heads = 1
    
    # "twice" → 2回
    if "twice" in text:
        flips = 2
    # "two times" → 2回
    elif "two times" in text:
        flips = 2
    # "three times" → 3回
    elif "three times" in text:
        flips = 3
    # 数値パターン "N times"
    else:
        match = re.search(r'(\d+)\s+times', text)
        if match:
            flips = int(match.group(1))
    
    # "two heads" / "both heads" → 全て表
    if "two heads" in text or "both heads" in text:
        heads = 2
    elif "three heads" in text or "all heads" in text:
        heads = 3
    # 数値パターン "N heads"
    else:
        match = re.search(r'(\d+)\s+heads?', text)
        if match:
            heads = int(match.group(1))
        elif "getting" in text and "heads" in text:
            # "getting heads" で特に数が指定されていない場合、flipsと同じと仮定
            heads = flips
    
    # coin_flip_probabilityを呼び出す
    return coin_flip_probability(flips=flips, heads=heads, ir=ir, context=context)


def dice_probability(sides: int = 6, target: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    サイコロの確率
    
    Args:
        sides: サイコロの面数
        target: 目標の出目
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # デフォルト: 標準的な6面サイコロ
    if sides is None:
        sides = 6
    
    # ir_dictから数値を抽出（targetとして使用）
    if target is None and ir:
        entities = ir.get("entities", [])
        for entity in entities:
            if entity.get("type") == "number":
                target = int(entity.get("value"))
                break
    
    # target指定なしの場合は1/sides
    if target is None:
        prob = 1.0 / sides
    else:
        # 特定の目が出る確率
        prob = 1.0 / sides if 1 <= target <= sides else 0.0
    
    return {
        "value": prob,
        "schema": "decimal",
        "confidence": 1.0,
        "artifacts": {"sides": sides, "target": target}
    }


def card_probability(total_cards: int = 52, target_cards: int = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    カード引きの確率
    
    Args:
        total_cards: 全カード数
        target_cards: 目標のカード数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # デフォルト: 標準デッキ52枚、1スート13枚
    if total_cards is None:
        total_cards = 52
    
    if target_cards is None:
        # ハート（1スート）の確率
        target_cards = 13
    
    try:
        prob = target_cards / total_cards
        
        return {
            "value": prob,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"total_cards": total_cards, "target_cards": target_cards}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def multiple_events(p1: float = None, p2: float = None, independent: bool = True, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    複数イベントの確率（独立事象）
    
    Args:
        p1: イベント1の確率
        p2: イベント2の確率
        independent: 独立かどうか
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    if p1 is None or p2 is None:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": "Need two probabilities"
        }
    
    try:
        if independent:
            # 独立事象: P(A and B) = P(A) * P(B)
            result = p1 * p2
        else:
            # 非独立の場合は単純な積（実装簡略化）
            result = p1 * p2
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"p1": p1, "p2": p2, "independent": independent}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }


def expected_value(sides: int = 6, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    サイコロの期待値
    
    Args:
        sides: サイコロの面数
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # デフォルト: 6面サイコロ
    if sides is None:
        sides = 6
    
    try:
        # サイコロの期待値: (1+2+...+n)/n = (n+1)/2
        result = (sides + 1) / 2.0
        
        return {
            "value": result,
            "schema": "decimal",
            "confidence": 1.0,
            "artifacts": {"sides": sides, "formula": "(n+1)/2"}
        }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "decimal",
            "confidence": 0.0,
            "error": str(e)
        }
