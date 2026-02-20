"""
CEGIS - CounterExample Guided Inductive Synthesis

外部ソルバー不要で PhD 級問題に対応するための核心モジュール。

構成:
  certificate.py  - 証明書型定義・検査
  worldgen.py     - 有限モデル生成器（群・グラフ・環・数列 etc.）
  cegis_loop.py   - CEGIS メインループ（生成→反例テスト→精緻化）

哲学:
  「解く」とは、正しい候補が生き残るまで生成と検証を回すこと。
  反例で落とす（高速）→ 残ったら証明書生成（重い）→ 無理なら高信頼タグ（設計上の決断）
"""

from .certificate import Certificate, CertKind, CertificateChecker
from .worldgen import WorldGenerator, FiniteModel
from .cegis_loop import CEGISLoop, CEGISResult, Candidate

__all__ = [
    "Certificate", "CertKind", "CertificateChecker",
    "WorldGenerator", "FiniteModel",
    "CEGISLoop", "CEGISResult", "Candidate",
]
