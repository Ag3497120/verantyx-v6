"""
Verifiers — 統一検証器 API

外部ツール（Z3, SymPy, 列挙器）を「検証器（oracle）」として統一インターフェースで扱う。
外部ツールは「答えを出す箱」ではなく「候補を落とす/支える判定器」。

使い方:
    from verifiers import verify, Verdict, VerdictStatus
    verdict = verify(candidate, spec)
    if verdict.status == VerdictStatus.FAIL:
        counterexample = verdict.counterexample  # worldgen に渡す
"""
from verifiers.api import Verdict, VerdictStatus, VerifySpec, BaseVerifier, verify
