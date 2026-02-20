"""
Executor - ピース実行エンジン

ピース経路を実行して構造化候補を生成
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import importlib
import inspect
import re

from pieces.piece import Piece


@dataclass
class StructuredCandidate:
    """
    構造化候補
    
    文字列ではなく、スキーマ付きの構造体
    """
    schema: str  # "integer", "move_sequence", etc.
    fields: Dict[str, Any]  # {"value": 42} or {"moves": ["e4", "e5"]}
    evidence: List[str]  # 使用したpiece_idのリスト
    confidence: float = 1.0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "schema": self.schema,
            "fields": self.fields,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "artifacts": self.artifacts
        }


class Executor:
    """
    Executor - ピース実行エンジン
    """
    
    def __init__(self):
        self.module_cache = {}
        self._garbage_skip_reason: Optional[str] = None  # A+ garbage guard

    # ──────────────────────────────────────────────────────────────────
    # A+ Step 1: Garbage entity guard
    # LaTeX解析失敗・壊れた式・意味のない文字列を検出して止める
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_expression(expr: str) -> tuple:
        """
        式が「計算可能な式」かどうか判定する。
        Returns: (is_valid: bool, reason: str)
        """
        if not expr:
            return False, "empty"

        # Rule 1: 長さが 3 未満（例: ")" や "**2" 単体）
        stripped = expr.strip()
        if len(stripped) < 3:
            return False, f"too_short({len(stripped)})"

        # Rule 2: 許可文字以外が多すぎる（LaTeX残骸 \mathbb, \frac など）
        #   許可: 0-9, a-z, A-Z, +, -, *, /, (, ), ., ^, **, space, %
        NON_MATH_RE = re.compile(r'[\\{}_\[\]@#$\'",;?!&]')
        non_math_chars = NON_MATH_RE.findall(stripped)
        if len(non_math_chars) > 2:
            return False, f"latex_residue({len(non_math_chars)} bad chars)"

        # Rule 3: 括弧バランス不正（( と ) の差が ±1以上）
        open_parens  = stripped.count('(')
        close_parens = stripped.count(')')
        if abs(open_parens - close_parens) > 1:
            return False, f"unbalanced_parens({open_parens}vs{close_parens})"

        # Rule 4: 演算子密度異常（数字/変数がほぼ無い）
        operator_chars  = re.findall(r'[+\-*/^]', stripped)
        alphanum_chars  = re.findall(r'[0-9a-zA-Z]', stripped)
        if len(operator_chars) > 0 and len(alphanum_chars) == 0:
            return False, "no_operands"
        if len(stripped) > 5 and len(operator_chars) > 0:
            op_density = len(operator_chars) / max(len(alphanum_chars), 1)
            if op_density > 3.0:
                return False, f"op_density_too_high({op_density:.1f})"

        # Rule 5: sympify で評価できない / Symbol 1個だけ（式になっていない）
        try:
            import sympy
            py_expr = stripped.replace('^', '**')
            parsed = sympy.sympify(py_expr, evaluate=False)
            # Symbol 1個だけ → 式ではなく変数名
            if isinstance(parsed, sympy.Symbol):
                return False, "single_symbol"
        except Exception:
            return False, "sympify_failed"

        return True, "ok"
    
    def execute_path(
        self,
        pieces: List[Piece],
        ir_dict: Dict[str, Any]
    ) -> Optional[StructuredCandidate]:
        """
        ピース経路を実行
        """
        context = {
            "ir": ir_dict,
            "artifacts": {},
            "confidence": 1.0
        }
        
        for piece in pieces:
            result = self._execute_piece(piece, context)
            
            if result is None:
                return None
            
            context["artifacts"].update(result.get("artifacts", {}))
            context["confidence"] *= result.get("confidence", 1.0)
            
            if "value" in result:
                context["final_value"] = result["value"]
            if "schema" in result:
                context["final_schema"] = result["schema"]
        
        if "final_value" not in context:
            return None
        
        # If final_value is None, treat as execution failure
        if context["final_value"] is None:
            return None
        
        final_piece = pieces[-1]
        
        candidate = StructuredCandidate(
            schema=final_piece.out_spec.schema,
            fields={"value": context["final_value"]},
            evidence=[p.piece_id for p in pieces],
            confidence=context["confidence"],
            artifacts=context["artifacts"]
        )
        
        return candidate
    
    def _execute_piece(
        self,
        piece: Piece,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        単一ピースを実行
        """
        try:
            module_name, func_name = piece.executor.rsplit(".", 1)
            
            if module_name not in self.module_cache:
                try:
                    module = importlib.import_module(module_name)
                    self.module_cache[module_name] = module
                except ImportError:
                    return self._execute_stub(piece, context)
            else:
                module = self.module_cache[module_name]
            
            if not hasattr(module, func_name):
                return self._execute_stub(piece, context)
            
            func = getattr(module, func_name)
            
            params = self._prepare_params(piece, context, func)
            clean_params = {k: v for k, v in params.items() if k not in ['ir', 'context']}
            
            result = func(**clean_params)
            
            if result is not None and not isinstance(result, dict):
                result = {
                    "value": result,
                    "schema": piece.out_spec.schema,
                    "confidence": 1.0
                }
            
            return result
            
        except Exception as e:
            import traceback
            print(f"[EXECUTOR] Error executing {piece.piece_id}: {e}")
            traceback.print_exc()
            return None
    
    def _prepare_params(
        self,
        piece: Piece,
        context: Dict[str, Any],
        func=None
    ) -> Dict[str, Any]:
        """
        実行パラメータを準備（強化版）
        
        piece.in_spec.slots と context から必要な値を抽出。
        スロットが空の場合は関数シグネチャをもとに自動マッピング。
        """
        params = {}
        ir = context["ir"]
        entities = ir.get("entities", [])
        source_text = ir.get("metadata", {}).get("source_text", "")
        
        # 整数エンティティを順番に取得
        int_entities = [e for e in entities
                        if e.get("type") in ("number", "integer", "float")
                        and e.get("value") is not None]
        
        # スロットを処理
        int_entity_ptr = [0]  # mutable counter for sequential assignment

        def fill_slot(slot_name: str, slot_type: str, slot_idx: int) -> bool:
            """1スロットを埋める。埋めたら True を返す"""
            # 1. IRから直接取得
            if slot_name in ir:
                params[slot_name] = ir[slot_name]
                return True
            
            # 2. Artifactsから取得
            if slot_name in context["artifacts"]:
                params[slot_name] = context["artifacts"][slot_name]
                return True
            
            # 3a. 名前付きエンティティ
            for entity in entities:
                if entity.get("name") == slot_name:
                    params[slot_name] = entity.get("value")
                    return True
            
            # 3b. タイプ一致エンティティ
            for entity in entities:
                if entity.get("type") == slot_name:
                    params[slot_name] = entity.get("value")
                    return True
            
            # 3c. 数値スロット（a, b, n, k, m, r, p, q, x など）
            NUMERIC_SLOTS = {"number", "a", "b", "c", "n", "k", "m", "r",
                             "p", "q", "x", "y", "z", "lhs", "rhs"}
            if slot_name in NUMERIC_SLOTS:
                if int_entities:
                    idx = int_entity_ptr[0]
                    if idx < len(int_entities):
                        v = int_entities[idx].get("value")
                        try:
                            params[slot_name] = int(v) if slot_name in {"n", "k", "m", "r"} else float(v)
                        except (TypeError, ValueError):
                            params[slot_name] = v
                        int_entity_ptr[0] += 1
                        return True
            
            # 3d. 文字列スロット（text, question, ciphertext, s1, s2）
            STRING_SLOTS = {"text", "question", "ciphertext", "s1", "s2"}
            if slot_name in STRING_SLOTS:
                target_type = "string"
                for entity in entities:
                    if entity.get("type") == target_type:
                        params[slot_name] = entity.get("value")
                        return True
                if source_text:
                    if slot_name == "text":
                        # 引用符内の文字列を優先抽出
                        quoted = re.search(r"['\"](.+?)['\"]", source_text)
                        params[slot_name] = quoted.group(1) if quoted else source_text
                    else:
                        params[slot_name] = source_text
                    return True
            
            # 3e. equation スロット
            if slot_name == "equation":
                eq_match = re.search(
                    r'([\w\s\+\-\*/\(\)\^]+=[^,\n\.]+)',
                    source_text
                )
                if eq_match:
                    params[slot_name] = eq_match.group(1).strip()
                elif "=" in source_text:
                    params[slot_name] = source_text
                else:
                    params[slot_name] = source_text
                return True
            
            # 3f. expression / formula スロット
            if slot_name in ("expression", "formula"):
                # ──────────────────────────────────────────────────────
                # A+ Garbage guard: 式候補を検証してからセット
                # ──────────────────────────────────────────────────────
                def _try_set_expr(candidate: str) -> bool:
                    """式候補を検証してOKならparams[slot_name]にセット"""
                    if not candidate or len(candidate) > 200:
                        return False
                    # 数値のみ（Decomposerが数値をexpressionとして渡した場合）は許可
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', candidate.strip()):
                        params[slot_name] = candidate.strip()
                        return True
                    is_valid, reason = self.__class__._validate_expression(candidate)
                    if is_valid:
                        params[slot_name] = candidate
                        return True
                    else:
                        # サイドチャネルに記録（pipeline が EXECUTOR_SKIP タグを付けるため）
                        self._garbage_skip_reason = reason
                        return False

                # IRのformulaフィールドを優先
                formula = ir.get("formula") or ir.get("metadata", {}).get("formula")
                if formula and _try_set_expr(formula):
                    return True
                # f(x) = EXPR [at/when/for...] パターン（at以降を除く）
                fx = re.search(
                    r'f\s*\(\s*[a-zA-Z]\s*\)\s*=\s*([^\n,]+?)(?:\s+(?:at|when|for|where)\b|$)',
                    source_text
                )
                if fx and _try_set_expr(fx.group(1).strip()):
                    return True
                # LaTeX式を抽出（短いもののみ）
                latex = re.search(r'\$([^$]{1,100})\$', source_text)
                if latex and _try_set_expr(latex.group(1)):
                    return True
                # source_textが短い場合のみフォールバック
                if len(source_text) <= 200 and _try_set_expr(source_text):
                    return True
                # 長いsource_textは流さない / 全候補がゴミだった → スキップ
                return False
            
            # 3g. x スロット（変数値）
            if slot_name == "x":
                # "at x = N" や "when x = N" パターン
                x_match = re.search(r'[xX]\s*=\s*(-?\d+(?:\.\d+)?)', source_text)
                if x_match:
                    params[slot_name] = float(x_match.group(1))
                    return True
                # エンティティから
                if int_entities and int_entity_ptr[0] < len(int_entities):
                    v = int_entities[int_entity_ptr[0]].get("value", 0)
                    try:
                        params[slot_name] = float(v)
                    except (TypeError, ValueError):
                        params[slot_name] = 0.0
                    int_entity_ptr[0] += 1
                    return True
                params[slot_name] = 0.0
                return True
            
            # 3h. variable スロット（デフォルト "x"）
            if slot_name == "variable":
                var_match = re.search(r'for\s+([a-zA-Z])\b', source_text)
                params[slot_name] = var_match.group(1) if var_match else "x"
                return True
            
            return False  # 未解決

        # スロットが定義されている場合
        for idx, slot_obj in enumerate(piece.in_spec.slots):
            slot_name = slot_obj.get("name") if isinstance(slot_obj, dict) else str(slot_obj)
            slot_type = slot_obj.get("type", "") if isinstance(slot_obj, dict) else ""
            fill_slot(slot_name, slot_type, idx)
        
        # スロットが空（slots: []）の場合 → 関数シグネチャを検査
        if not piece.in_spec.slots and func is not None:
            params = self._infer_from_signature(func, ir, entities, source_text, int_entities)
        
        # 最終フォールバック: スロット1つ & 未解決の場合
        if len(piece.in_spec.slots) == 1 and not params:
            slot_name = (piece.in_spec.slots[0].get("name")
                         if isinstance(piece.in_spec.slots[0], dict)
                         else str(piece.in_spec.slots[0]))
            if source_text:
                params[slot_name] = source_text
        
        return params
    
    def _infer_from_signature(
        self,
        func,
        ir: Dict[str, Any],
        entities: List[Dict],
        source_text: str,
        int_entities: List[Dict]
    ) -> Dict[str, Any]:
        """
        関数シグネチャを検査してパラメータを自動マッピング。
        slots: [] のピース向け。
        """
        params = {}
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return params
        
        int_ptr = [0]
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'args', 'kwargs'):
                continue
            # デフォルト値あり → スキップ（任意パラメータ）
            if param.default is not inspect.Parameter.empty:
                continue
            
            # n, k, m, r → 整数（順番に取得）
            if param_name in ('n', 'k', 'm', 'r'):
                if int_ptr[0] < len(int_entities):
                    v = int_entities[int_ptr[0]].get("value")
                    try:
                        params[param_name] = int(float(v))
                    except (TypeError, ValueError):
                        params[param_name] = 2
                    int_ptr[0] += 1
                else:
                    params[param_name] = 2
            
            # a, b, c → 数値（順番に取得）
            elif param_name in ('a', 'b', 'c', 'p', 'q'):
                if int_ptr[0] < len(int_entities):
                    v = int_entities[int_ptr[0]].get("value")
                    try:
                        params[param_name] = float(v)
                    except (TypeError, ValueError):
                        params[param_name] = 1.0
                    int_ptr[0] += 1
                else:
                    params[param_name] = 1.0
            
            # x → 変数値
            elif param_name == 'x':
                x_match = re.search(r'[xX]\s*=\s*(-?\d+(?:\.\d+)?)', source_text)
                if x_match:
                    params[param_name] = float(x_match.group(1))
                elif int_ptr[0] < len(int_entities):
                    v = int_entities[int_ptr[0]].get("value", 0)
                    try:
                        params[param_name] = float(v)
                    except (TypeError, ValueError):
                        params[param_name] = 0.0
                    int_ptr[0] += 1
                else:
                    params[param_name] = 0.0
            
            # expression / formula
            elif param_name in ('expression', 'formula'):
                formula = ir.get("formula") or ir.get("metadata", {}).get("formula")
                if formula and len(formula) <= 200:
                    params[param_name] = formula
                else:
                    # f(x) = EXPR [at x=N / when x=N] を抽出（at/when以降を除く）
                    fx = re.search(
                        r'f\s*\(\s*[a-zA-Z]\s*\)\s*=\s*([^\n,]+?)(?:\s+(?:at|when|for|where)\b|$)',
                        source_text
                    )
                    if fx:
                        expr = fx.group(1).strip()
                        if len(expr) <= 200:
                            params[param_name] = expr
                        else:
                            continue  # 長すぎる → スキップ
                    else:
                        # LaTeX $...$ を試みる（短いもののみ）
                        latex = re.search(r'\$([^$]{1,100})\$', source_text)
                        if latex:
                            params[param_name] = latex.group(1)
                        elif len(source_text) <= 200:
                            params[param_name] = source_text
                        else:
                            continue  # 長すぎる → スキップ（このパラメータは未設定のまま）
            
            # equation
            elif param_name == 'equation':
                eq_match = re.search(
                    r'([\w\s\+\-\*/\(\)\^]+=[^,\n\.]+)',
                    source_text
                )
                if eq_match:
                    params[param_name] = eq_match.group(1).strip()
                else:
                    params[param_name] = source_text
            
            # variable
            elif param_name == 'variable':
                var_match = re.search(r'for\s+([a-zA-Z])\b', source_text)
                params[param_name] = var_match.group(1) if var_match else "x"
            
            # question / text
            elif param_name in ('question', 'text'):
                if param_name == 'text':
                    quoted = re.search(r"['\"](.+?)['\"]", source_text)
                    params[param_name] = quoted.group(1) if quoted else source_text
                else:
                    params[param_name] = source_text
            
            # その他のstring系
            elif param_name in ('s', 's1', 's2', 'ciphertext'):
                params[param_name] = source_text
        
        return params
    
    def _execute_stub(
        self,
        piece: Piece,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        スタブ実行（実際のexecutorが未実装の場合）
        """
        print(f"[EXECUTOR] Stub execution for {piece.piece_id}")
        
        dummy_values = {
            "integer": 0,
            "boolean": True,
            "option_label": "A",
            "move_sequence": ["e4"],
            "expression": "x"
        }
        
        return {
            "value": dummy_values.get(piece.out_spec.schema, ""),
            "schema": piece.out_spec.schema,
            "confidence": 0.1,
            "artifacts": {"stub": True}
        }
