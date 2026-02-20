"""
Enumerate Executor - 範囲列挙・選択肢生成（強化版）
"""

from typing import Dict, Any, List
import re


def integer_range(min: int = 0, max: int = 10, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    整数範囲を列挙
    
    Args:
        min: 最小値
        max: 最大値
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # IRから範囲を抽出
    if ir and "constraints" in ir:
        for constraint in ir["constraints"]:
            if constraint.get("type") == "range":
                min = constraint.get("min", min)
                max = constraint.get("max", max)
                break
    
    # 範囲チェック
    if max - min > 1000:
        return {
            "value": None,
            "schema": "sequence",
            "confidence": 0.0,
            "error": f"Range too large: {max - min}"
        }
    
    # 列挙
    values = list(range(min, max + 1))
    
    return {
        "value": values,
        "schema": "sequence",
        "confidence": 1.0,
        "artifacts": {
            "min": min,
            "max": max,
            "count": len(values)
        }
    }


def options(ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    選択肢を生成（全ての選択肢を列挙）
    
    Args:
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果（全選択肢のリスト）
    """
    # IRから選択肢を取得
    option_list = []
    
    if ir and "options" in ir:
        option_list = ir["options"]
    
    if not option_list:
        return {
            "value": None,
            "schema": "option_label",
            "confidence": 0.0,
            "error": "No options found"
        }
    
    # 選択肢ラベルを抽出（A, B, C, ...）
    labels = []
    for opt in option_list:
        # "A. ..." 形式から "A" を抽出
        if isinstance(opt, str) and len(opt) > 0:
            # 最初の文字がA-Jなら
            first_char = opt[0].upper()
            if first_char in "ABCDEFGHIJ":
                labels.append(first_char)
            else:
                # "A." で始まるか探す
                match = re.match(r'^([A-J])[.\s]', opt)
                if match:
                    labels.append(match.group(1))
    
    if not labels:
        # ラベルがない場合は順番に生成
        labels = [chr(65 + i) for i in range(min(len(option_list), 10))]
    
    # 最初の選択肢をデフォルトとして返す（Verifier/スコアリングで選択）
    # 注: 全ての選択肢をartifactsに含める
    return {
        "value": labels[0] if labels else None,
        "schema": "option_label",
        "confidence": 1.0 / len(labels) if labels else 0.0,
        "artifacts": {
            "all_options": labels,
            "option_texts": option_list,
            "selected_index": 0
        }
    }
