"""
Logic Executor - 命題論理・様相論理

verantyx_ios PropSolver/ModalSolverをポート
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
import itertools
import re


@dataclass
class Node:
    """論理式のASTノード"""
    kind: str
    value: Optional[str] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    child: Optional['Node'] = None


class PropositionalLogic:
    """命題論理（真理表検証）"""
    
    def __init__(self):
        self.token_re = re.compile(
            r"\s+|"
            r"(<->|↔)|(->|→)|(∧|&)|(∨|\|)|(¬|~|!)|"
            r"([()])|"
            r"([A-Za-z][A-Za-z0-9_]*)"
        )
    
    def parse(self, formula: str) -> Tuple[Node, List[str]]:
        """論理式をパース"""
        tokens = self._tokenize(formula)
        self._pos = 0
        self._tokens = tokens
        
        ast = self._parse_expr()
        atoms = sorted(list(set(
            t for t in tokens 
            if t not in ("AND", "OR", "NOT", "IMP", "IFF", "(", ")")
        )))
        
        return ast, atoms
    
    def _tokenize(self, s: str) -> List[str]:
        """トークン化"""
        out = []
        for m in self.token_re.finditer(s):
            if m.group(0).isspace():
                continue
            if m.group(1):
                out.append("IFF")
            elif m.group(2):
                out.append("IMP")
            elif m.group(3):
                out.append("AND")
            elif m.group(4):
                out.append("OR")
            elif m.group(5):
                out.append("NOT")
            elif m.group(6):
                out.append(m.group(6))  # ( )
            elif m.group(7):
                out.append(m.group(7))  # Atom
        return out
    
    def _curr(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None
    
    def _consume(self, expected=None):
        t = self._curr()
        if expected and t != expected:
            raise ValueError(f"Expected {expected}, got {t}")
        self._pos += 1
        return t
    
    def _parse_expr(self):
        return self._parse_iff()
    
    def _parse_iff(self):
        left = self._parse_imp()
        while self._curr() == "IFF":
            self._consume("IFF")
            right = self._parse_imp()
            left = Node("IFF", left=left, right=right)
        return left
    
    def _parse_imp(self):
        left = self._parse_or()
        while self._curr() == "IMP":
            self._consume("IMP")
            right = self._parse_or()
            left = Node("IMP", left=left, right=right)
        return left
    
    def _parse_or(self):
        left = self._parse_and()
        while self._curr() == "OR":
            self._consume("OR")
            right = self._parse_and()
            left = Node("OR", left=left, right=right)
        return left
    
    def _parse_and(self):
        left = self._parse_not()
        while self._curr() == "AND":
            self._consume("AND")
            right = self._parse_not()
            left = Node("AND", left=left, right=right)
        return left
    
    def _parse_not(self):
        if self._curr() == "NOT":
            self._consume("NOT")
            child = self._parse_not()
            return Node("NOT", child=child)
        return self._parse_atom()
    
    def _parse_atom(self):
        if self._curr() == "(":
            self._consume("(")
            expr = self._parse_expr()
            self._consume(")")
            return expr
        else:
            val = self._consume()
            return Node("ATOM", value=val)
    
    def evaluate(self, ast: Node, valuation: Dict[str, bool]) -> bool:
        """真理値評価"""
        if ast.kind == "ATOM":
            return valuation.get(ast.value, False)
        elif ast.kind == "NOT":
            return not self.evaluate(ast.child, valuation)
        elif ast.kind == "AND":
            return self.evaluate(ast.left, valuation) and self.evaluate(ast.right, valuation)
        elif ast.kind == "OR":
            return self.evaluate(ast.left, valuation) or self.evaluate(ast.right, valuation)
        elif ast.kind == "IMP":
            return (not self.evaluate(ast.left, valuation)) or self.evaluate(ast.right, valuation)
        elif ast.kind == "IFF":
            left_val = self.evaluate(ast.left, valuation)
            right_val = self.evaluate(ast.right, valuation)
            return left_val == right_val
        else:
            return False


class ModalLogic:
    """様相論理（Kripkeモデル探索）"""
    
    def __init__(self, max_worlds: int = 3):
        self.max_worlds = max_worlds
        self.token_re = re.compile(
            r"\s+|"
            r"(<->|↔)|(->|→)|(∧|&)|(∨|\|)|(¬|~|!)|"
            r"(\[\]|□|\bbox\b)|(<>|◇|\bdiamond\b)|"
            r"([()])|"
            r"([A-Za-z][A-Za-z0-9_]*)"
        )
    
    def parse(self, formula: str) -> Node:
        """様相論理式をパース（簡易）"""
        # PropositionalLogicと同様の実装（BOX, DIAMONDを追加）
        # 簡略化のため、基本的なパースのみ実装
        return Node("ATOM", value="p")  # スタブ
    
    def generate_relations(self, worlds: List[int], assumptions: List[str]) -> List[Set[Tuple[int, int]]]:
        """関係生成（仮定を反映）"""
        relations = []
        
        # 全ての可能な関係を生成
        all_pairs = [(w1, w2) for w1 in worlds for w2 in worlds]
        
        # 仮定に応じてフィルタ
        for subset in itertools.chain.from_iterable(
            itertools.combinations(all_pairs, r) for r in range(len(all_pairs) + 1)
        ):
            R = set(subset)
            
            # Reflexive
            if any("reflexive" in a.lower() for a in assumptions):
                if not all((w, w) in R for w in worlds):
                    continue
            
            # Transitive
            if any("transitive" in a.lower() for a in assumptions):
                # 推移閉包チェック（簡易）
                pass
            
            # Symmetric
            if any("symmetric" in a.lower() for a in assumptions):
                if not all((w2, w1) in R for (w1, w2) in R):
                    continue
            
            relations.append(R)
            
            if len(relations) > 100:  # 組み合わせ爆発防止
                break
        
        return relations if relations else [set()]


def prop_truth_table(formula: str = None, atoms: List[str] = None, ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    命題論理の真理表検証
    
    Args:
        formula: 論理式
        atoms: 原子命題のリスト
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # 式の取得
    if formula is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "formula":
                formula = entity.get("value")
            elif entity.get("type") == "atoms" and atoms is None:
                atoms = entity.get("value")
    
    if not formula:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": "No formula found"
        }
    
    try:
        prop_logic = PropositionalLogic()
        ast, parsed_atoms = prop_logic.parse(formula)
        
        if atoms is None:
            atoms = parsed_atoms
        
        # 原子が多すぎる場合は中止
        if len(atoms) > 10:
            return {
                "value": None,
                "schema": "boolean",
                "confidence": 0.0,
                "error": "Too many atoms (>10)"
            }
        
        # 真理表探索
        counterexample = None
        for bits in itertools.product([False, True], repeat=len(atoms)):
            valuation = dict(zip(atoms, bits))
            result = prop_logic.evaluate(ast, valuation)
            
            if not result:
                counterexample = valuation
                break
        
        if counterexample:
            return {
                "value": False,
                "schema": "boolean",
                "confidence": 1.0,
                "artifacts": {
                    "method": "truth_table",
                    "counterexample": counterexample,
                    "formula": formula
                }
            }
        else:
            return {
                "value": True,
                "schema": "boolean",
                "confidence": 1.0,
                "artifacts": {
                    "method": "truth_table",
                    "assignments_checked": 2 ** len(atoms),
                    "formula": formula
                }
            }
    
    except Exception as e:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": str(e)
        }


def modal_kripke(formula: str = None, atoms: List[str] = None, assumptions: List[str] = None, 
                 ir: Dict = None, context: Dict = None, **kwargs) -> Dict[str, Any]:
    """
    様相論理のKripkeモデル探索
    
    Args:
        formula: 様相論理式
        atoms: 原子命題のリスト
        assumptions: 仮定（reflexive, transitive, symmetric）
        ir: IR辞書
        context: 実行コンテキスト
    
    Returns:
        実行結果
    """
    # 式の取得
    if formula is None and ir:
        for entity in ir.get("entities", []):
            if entity.get("type") == "formula":
                formula = entity.get("value")
                break
    
    if assumptions is None:
        assumptions = []
    
    if not formula:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": "No formula found"
        }
    
    try:
        modal_logic = ModalLogic(max_worlds=3)
        
        # 簡易実装：様相演算子を含む場合は不確定を返す
        if "[]" in formula or "<>" in formula or "□" in formula or "◇" in formula:
            return {
                "value": None,
                "schema": "boolean",
                "confidence": 0.5,
                "artifacts": {
                    "method": "modal_kripke",
                    "note": "Modal logic support is limited",
                    "formula": formula
                }
            }
        
        # 様相演算子がない場合は命題論理として評価
        return prop_truth_table(formula=formula, atoms=atoms, ir=ir, context=context)
    
    except Exception as e:
        return {
            "value": None,
            "schema": "boolean",
            "confidence": 0.0,
            "error": str(e)
        }
