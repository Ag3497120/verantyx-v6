"""
Verifier API — 統一検証インターフェース

verify(candidate, spec) -> Verdict
Verdict = PASS | FAIL(counterexample) | UNKNOWN(reason)

原則A: 外部ツールは "判定器" のみ
  - Z3: sat/unsat + model (反例/充足例)
  - SymPy: numeric/symbolic check
  - 列挙器: small-world sample + check
  最終答えは Verantyx が生成。外部は「候補を殺す/支える」だけ。

原則C: 複数 Verifier で交差検証
  SymPy → Z3 → 列挙 の順で試す。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────

class VerdictStatus(Enum):
    PASS    = "PASS"     # 候補は正しい（反例なし）
    FAIL    = "FAIL"     # 候補は誤り（反例あり）
    UNKNOWN = "UNKNOWN"  # 判定不能（タイムアウト / スコープ外）


@dataclass
class Verdict:
    """
    検証結果。

    FAIL の場合は counterexample を worldgen に渡して次候補を改善する。
    UNKNOWN の場合は reason を見て別 Verifier に切り替えるか skip する。
    """
    status: VerdictStatus
    verifier: str               # 使用した検証器名
    counterexample: Optional[Dict[str, Any]] = None   # FAIL 時の反例
    witness: Optional[Dict[str, Any]] = None           # PASS 時の証拠
    reason: Optional[str] = None                       # UNKNOWN 時の理由
    details: List[str] = field(default_factory=list)   # デバッグ情報

    def __bool__(self) -> bool:
        return self.status == VerdictStatus.PASS

    @classmethod
    def pass_(cls, verifier: str, witness: Optional[Dict] = None, **kw) -> "Verdict":
        return cls(status=VerdictStatus.PASS, verifier=verifier, witness=witness, **kw)

    @classmethod
    def fail(cls, verifier: str, counterexample: Optional[Dict] = None, **kw) -> "Verdict":
        return cls(status=VerdictStatus.FAIL, verifier=verifier, counterexample=counterexample, **kw)

    @classmethod
    def unknown(cls, verifier: str, reason: str = "", **kw) -> "Verdict":
        return cls(status=VerdictStatus.UNKNOWN, verifier=verifier, reason=reason, **kw)


# ─────────────────────────────────────────────────────────────────────
# VerifySpec — 検証仕様
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VerifySpec:
    """
    どう検証するか・どう反例を作るかの仕様。
    Piece の verify/worldgen フィールドから生成される。
    """
    kind: str               # "numeric" | "symbolic" | "smt" | "enumeration" | "cross_check"
    method: str             # "z3_sat" | "sympy_eval" | "sample_check" | "double_eval"
    expected_value: Optional[Any] = None    # 期待する計算結果（cross_check 用）
    constraints: List[str] = field(default_factory=list)   # SMT 制約式
    worldgen_params: Optional[Dict[str, Any]] = None       # worldgen パラメータ
    timeout_ms: int = 2000                                 # タイムアウト（ms）
    type_check: Optional[str] = None        # "integer" | "decimal" | "boolean"
    range: Optional[Dict[str, float]] = None # {"lo": -1e9, "hi": 1e9}

    @classmethod
    def from_piece_dict(cls, d: Dict[str, Any]) -> "VerifySpec":
        """Piece の verify フィールドから生成"""
        return cls(
            kind=d.get("kind", "numeric"),
            method=d.get("method", "double_eval"),
            expected_value=d.get("expected_value"),
            constraints=d.get("constraints", []),
            worldgen_params=d.get("params", {}).get("worldgen"),
            timeout_ms=d.get("timeout_ms", 2000),
            type_check=d.get("params", {}).get("type_check"),
            range=d.get("params", {}).get("range"),
        )


# ─────────────────────────────────────────────────────────────────────
# BaseVerifier — 抽象基底クラス
# ─────────────────────────────────────────────────────────────────────

class BaseVerifier(ABC):
    """
    検証器の基底クラス。
    全ての検証器はこれを継承し verify() を実装する。
    """
    name: str = "base"

    @abstractmethod
    def verify(self, candidate_value: Any, spec: VerifySpec, context: Dict[str, Any]) -> Verdict:
        """
        候補値を検証する。

        Args:
            candidate_value: Executor が返した値（数値・式・真偽値など）
            spec: 検証仕様
            context: IR dict や entities などのコンテキスト

        Returns:
            Verdict
        """
        ...

    def can_handle(self, spec: VerifySpec) -> bool:
        """この検証器が spec を処理できるか"""
        return True


# ─────────────────────────────────────────────────────────────────────
# verify() — メインエントリポイント（Verifier のカスケード呼び出し）
# ─────────────────────────────────────────────────────────────────────

def verify(
    candidate_value: Any,
    spec: VerifySpec,
    context: Optional[Dict[str, Any]] = None,
    verifiers: Optional[List[BaseVerifier]] = None,
) -> Verdict:
    """
    候補値を検証する。複数 Verifier で交差検証（原則C）。

    SymPy → Z3 → 列挙器 の順で試す。
    最初に PASS/FAIL が出たら返す。全部 UNKNOWN なら UNKNOWN を返す。

    Args:
        candidate_value: 検証対象の値
        spec: 検証仕様
        context: IRや問題情報
        verifiers: 使用する検証器のリスト（None の場合はデフォルト一覧）

    Returns:
        Verdict
    """
    if context is None:
        context = {}

    if verifiers is None:
        verifiers = _get_default_verifiers()

    last_unknown: Optional[Verdict] = None

    for verifier in verifiers:
        if not verifier.can_handle(spec):
            continue
        try:
            verdict = verifier.verify(candidate_value, spec, context)
            if verdict.status in (VerdictStatus.PASS, VerdictStatus.FAIL):
                return verdict
            last_unknown = verdict
        except Exception as e:
            last_unknown = Verdict.unknown(verifier.name, reason=f"exception:{e}")

    if last_unknown is not None:
        return last_unknown
    return Verdict.unknown("cascade", reason="no_verifier_handled")


# ─────────────────────────────────────────────────────────────────────
# Default verifier registry
# ─────────────────────────────────────────────────────────────────────

_default_verifiers: Optional[List[BaseVerifier]] = None


def _get_default_verifiers() -> List[BaseVerifier]:
    """デフォルト検証器リスト（遅延ロード）"""
    global _default_verifiers
    if _default_verifiers is not None:
        return _default_verifiers

    verifiers: List[BaseVerifier] = []

    # SymPy検証器（数値・代数）
    try:
        from verifiers.sympy_verifier import SympyVerifier
        verifiers.append(SympyVerifier())
    except ImportError:
        pass

    # Z3検証器（論理・制約・SMT）
    try:
        from verifiers.z3_verifier import Z3Verifier
        verifiers.append(Z3Verifier())
    except ImportError:
        pass

    # 列挙器（small-world）
    try:
        from verifiers.enum_verifier import EnumVerifier
        verifiers.append(EnumVerifier())
    except ImportError:
        pass

    _default_verifiers = verifiers
    return verifiers


def reset_verifier_cache() -> None:
    """テスト用: verifier キャッシュをリセット"""
    global _default_verifiers
    _default_verifiers = None
