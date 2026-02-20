"""
Advanced Logic Executors

avh_mathの完全なロジックソルバーを統合
- Propositional Logic (命題論理)
- Modal Logic (様相論理)
"""
from typing import Dict, Any, Optional
import re

# avh_mathの完全なソルバーをインポート
try:
    from executors import propositional_logic_solver
    from executors import modal_logic_solver
    
    is_tautology = propositional_logic_solver.is_tautology
    is_satisfiable = propositional_logic_solver.is_satisfiable
    
    # modal_logic_solverには is_valid_in_* 関数がないので、check_axiom_validityを使う
    check_axiom_validity = modal_logic_solver.check_axiom_validity
    ModalFormula = modal_logic_solver.ModalFormula
    KripkeFrame = modal_logic_solver.KripkeFrame
    KripkeModel = modal_logic_solver.KripkeModel
    
    SOLVERS_AVAILABLE = True
except ImportError as e:
    SOLVERS_AVAILABLE = False
    print(f"[WARN] Advanced logic solvers not available: {e}")


def prop_tautology_check(formula: str = None, **kwargs) -> Dict[str, Any]:
    """
    命題論理のトートロジー判定
    
    Args:
        formula: 論理式（例: "((A -> B) & A) -> B"）
    
    Returns:
        実行結果
    """
    if not SOLVERS_AVAILABLE:
        return {"success": False, "error": "Solvers not available"}
    
    # formulaの取得
    if formula is None:
        source_text = kwargs.get('source_text', '')
        # 式を抽出（引用符内または記号を含む部分）
        match = re.search(r'["\']([^"\']+)["\']', source_text)
        if match:
            formula = match.group(1)
        else:
            # 論理記号を含む部分を探す
            match = re.search(r'([A-Z][\s\->|&~()]+[A-Z)])', source_text)
            if match:
                formula = match.group(1)
    
    if not formula:
        return {"success": False, "error": "No formula found"}
    
    try:
        is_taut, counterexample = is_tautology(formula)
        
        answer = "Yes" if is_taut else "No"
        
        return {
            "success": True,
            "answer": answer,
            "value": is_taut,
            "artifacts": {
                "method": "propositional_tautology",
                "formula": formula,
                "counterexample": counterexample if not is_taut else None
            },
            "confidence": 1.0
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def prop_satisfiability_check(formula: str = None, **kwargs) -> Dict[str, Any]:
    """
    命題論理の充足可能性判定
    
    Args:
        formula: 論理式
    
    Returns:
        実行結果
    """
    if not SOLVERS_AVAILABLE:
        return {"success": False, "error": "Solvers not available"}
    
    if formula is None:
        source_text = kwargs.get('source_text', '')
        match = re.search(r'["\']([^"\']+)["\']', source_text)
        if match:
            formula = match.group(1)
    
    if not formula:
        return {"success": False, "error": "No formula found"}
    
    try:
        is_sat, assignment = is_satisfiable(formula)
        
        answer = "Satisfiable" if is_sat else "Unsatisfiable"
        
        return {
            "success": True,
            "answer": answer,
            "value": is_sat,
            "artifacts": {
                "method": "propositional_satisfiability",
                "formula": formula,
                "assignment": assignment if is_sat else None
            },
            "confidence": 1.0
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def modal_validity_check(formula: str = None, system: str = "K", **kwargs) -> Dict[str, Any]:
    """
    様相論理の妥当性判定
    
    Args:
        formula: 様相論理式（例: "[]A -> A"）
        system: 論理体系（K, T, S4, S5）
    
    Returns:
        実行結果
    """
    if not SOLVERS_AVAILABLE:
        return {"success": False, "error": "Solvers not available"}
    
    if formula is None:
        source_text = kwargs.get('source_text', '')
        match = re.search(r'["\']([^"\']+)["\']', source_text)
        if match:
            formula = match.group(1)
    
    if not formula:
        return {"success": False, "error": "No formula found"}
    
    # システムの判定
    if system is None:
        text_lower = kwargs.get('source_text', '').lower()
        if 's5' in text_lower:
            system = 'S5'
        elif 's4' in text_lower:
            system = 'S4'
        elif ' t ' in text_lower or 'system t' in text_lower:
            system = 'T'
        else:
            system = 'K'
    
    try:
        # check_axiom_validityを使用
        # システムに対応する公理名を取得
        axiom_map = {
            'T': 'T',
            'S4': '4',
            'S5': '5',
            'K': 'K'
        }
        axiom_name = axiom_map.get(system.upper(), 'K')
        
        is_valid, explanation = check_axiom_validity(axiom_name, formula)
        
        answer = "Valid" if is_valid else "Invalid"
        
        return {
            "success": True,
            "answer": answer,
            "value": is_valid,
            "artifacts": {
                "method": f"modal_{system}",
                "formula": formula,
                "system": system,
                "explanation": explanation
            },
            "confidence": 1.0
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def logic_solver_auto(formula: str = None, **kwargs) -> Dict[str, Any]:
    """
    自動論理ソルバー（式の種類を自動判定）
    
    Args:
        formula: 論理式
    
    Returns:
        実行結果
    """
    if not SOLVERS_AVAILABLE:
        return {"success": False, "error": "Solvers not available"}
    
    if formula is None:
        source_text = kwargs.get('source_text', '')
        match = re.search(r'["\']([^"\']+)["\']', source_text)
        if match:
            formula = match.group(1)
    
    if not formula:
        return {"success": False, "error": "No formula found"}
    
    # 様相演算子を含むか確認
    if any(op in formula for op in ['[]', '<>', '□', '◇', 'box', 'diamond']):
        # 様相論理として処理
        return modal_validity_check(formula=formula, **kwargs)
    
    # 「tautology」「satisfiable」などのキーワードで判定
    source_text = kwargs.get('source_text', '').lower()
    
    if 'satisf' in source_text:
        return prop_satisfiability_check(formula=formula, **kwargs)
    elif 'tautolog' in source_text or 'valid' in source_text or 'always true' in source_text:
        return prop_tautology_check(formula=formula, **kwargs)
    else:
        # デフォルトはトートロジー判定
        return prop_tautology_check(formula=formula, **kwargs)
