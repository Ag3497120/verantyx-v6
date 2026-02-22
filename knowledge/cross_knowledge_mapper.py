"""
cross_knowledge_mapper.py
Phase 4: SanitizedFact → Cross Piece（Verantyx内部表現）への変換

SanitizedFact は LLM 由来の検疫済み知識。
それを Cross Piece（探索可能な構造）に変換し、PieceDB に投入できる形にする。

変換の方針:
  - definition  → pattern: symbol が出現したら適用候補
  - theorem     → pattern: preconditions が全部満たされたら consequences を導出
  - criterion   → pattern: 判定条件 → bool結果
  - checkable_examples → worldgen_spec 自動生成（CEGIS/CrossSim 用）
"""

from dataclasses import dataclass, field
from typing import Any
from knowledge.knowledge_sanitizer import SanitizedFact


# ---------------------------------------------------------------------------
# 1. Cross Piece の型定義
# ---------------------------------------------------------------------------

@dataclass
class CrossPiece:
    """
    Verantyx パズル推論が使うピースの形式。
    既存 piece_db.py の Piece クラスに合わせて調整する。
    """
    piece_id: str
    domain: str
    kind: str                              # "definition" | "theorem" | "criterion"
    symbol: str
    pattern: dict                          # 適用条件（トリガー）
    transform: dict                        # 何を導くか
    verify_spec: dict                      # CEGIS / CrossSim 用検証仕様
    worldgen_spec: dict                    # 小世界生成仕様
    grammar_binding: dict                  # 文法層への接着
    metadata: dict = field(default_factory=dict)  # 元の fact_id, confidence, etc.


# ---------------------------------------------------------------------------
# 2. 変換ロジック（Fact → Piece）
# ---------------------------------------------------------------------------

class CrossKnowledgeMapper:
    """
    SanitizedFact のリストを CrossPiece のリストに変換する。
    既存 PieceDB の形式に合わせて `to_piece_dict()` を差し替える。
    """

    def map_all(self, facts: list[SanitizedFact]) -> list[CrossPiece]:
        pieces = []
        for fact in facts:
            piece = self._map_single(fact)
            if piece:
                pieces.append(piece)
        return pieces

    def _map_single(self, fact: SanitizedFact) -> CrossPiece | None:
        kind = fact.kind.lower()

        # --- pattern（適用トリガー）---
        pattern = self._build_pattern(fact)

        # --- transform（何を導くか）---
        transform = self._build_transform(fact)

        # --- verify_spec（検証仕様）---
        verify_spec = self._build_verify_spec(fact)

        # --- worldgen_spec（小世界）---
        worldgen_spec = self._build_worldgen_spec(fact)

        # --- grammar_binding（文法接着）---
        grammar_binding = {
            "template": "By the {kind} of {symbol}: {statement_plain}",
            "bindings": {
                "kind": kind,
                "symbol": fact.symbol,
                "statement_plain": fact.statement_plain,
            },
        }

        return CrossPiece(
            piece_id=f"llm_{fact.domain}_{fact.symbol}_{fact.fact_id}",
            domain=fact.domain,
            kind=kind,
            symbol=fact.symbol,
            pattern=pattern,
            transform=transform,
            verify_spec=verify_spec,
            worldgen_spec=worldgen_spec,
            grammar_binding=grammar_binding,
            metadata={
                "fact_id": fact.fact_id,
                "confidence": fact.confidence,
                "llm_origin": True,
                "source_hint": fact.source_hint,
                "gate_tags": fact.gate_tags,
            },
        )

    # ------------------------------------------------------------------
    # pattern ビルダー
    # ------------------------------------------------------------------

    def _build_pattern(self, fact: SanitizedFact) -> dict:
        """
        ピースの適用条件。
        - definition: symbol がクエリに出現
        - theorem: preconditions が全部満たされている
        - criterion: symbol がクエリに出現 + 判定対象が特定できる
        """
        if fact.kind == "definition":
            return {
                "trigger": "symbol_present",
                "symbol": fact.symbol,
                "domain": fact.domain,
            }
        elif fact.kind in ("theorem", "lemma"):
            return {
                "trigger": "preconditions_satisfied",
                "symbol": fact.symbol,
                "preconditions": fact.preconditions,
                "domain": fact.domain,
            }
        elif fact.kind == "criterion":
            return {
                "trigger": "symbol_present_and_input_typed",
                "symbol": fact.symbol,
                "domain": fact.domain,
                "preconditions": fact.preconditions,
            }
        else:
            return {
                "trigger": "symbol_present",
                "symbol": fact.symbol,
                "domain": fact.domain,
            }

    # ------------------------------------------------------------------
    # transform ビルダー
    # ------------------------------------------------------------------

    def _build_transform(self, fact: SanitizedFact) -> dict:
        """
        ピースを適用したときに何が導かれるか。
        consequences から生成する。
        """
        if fact.consequences:
            return {
                "type": "consequence",
                "outputs": fact.consequences,
                "formal": fact.statement_formal,
            }
        elif fact.kind == "definition":
            return {
                "type": "definition_expansion",
                "formal": fact.statement_formal,
                "plain": fact.statement_plain,
            }
        else:
            return {
                "type": "unknown",
                "formal": fact.statement_formal,
                "plain": fact.statement_plain,
            }

    # ------------------------------------------------------------------
    # verify_spec ビルダー
    # ------------------------------------------------------------------

    def _build_verify_spec(self, fact: SanitizedFact) -> dict:
        """
        CEGIS / CrossSim に渡せる verify_spec を生成する。
        checkable_examples が있으면それを constraints に変換する。
        """
        if not fact.checkable_examples:
            # examples がない場合は弱い spec（oracle は formal から推測）
            oracle = fact.statement_formal if fact.statement_formal else f"verify_{fact.symbol}(x)"
            return {
                "oracle": oracle,
                "constraints": [c for c in fact.preconditions if c],
                "confidence": "low",  # examples がないので弱い
            }

        # examples から constraints を生成
        constraints = []
        for ex in fact.checkable_examples:
            inp = str(ex.get("input", "")).strip()
            out = str(ex.get("expected_output", "")).strip()
            if inp and out:
                constraints.append(f"{fact.symbol}({inp}) == {out}")

        # preconditions も追加
        constraints.extend([c for c in fact.preconditions if c])

        oracle = fact.statement_formal if fact.statement_formal else f"{fact.symbol}(x)"
        return {
            "oracle": oracle,
            "constraints": constraints,
            "examples": fact.checkable_examples,
            "confidence": fact.confidence,
        }

    # ------------------------------------------------------------------
    # worldgen_spec ビルダー
    # ------------------------------------------------------------------

    def _build_worldgen_spec(self, fact: SanitizedFact) -> dict:
        """
        小世界生成仕様。checkable_examples があれば (input, output) ペアで世界を作れる。
        """
        if not fact.checkable_examples:
            return {
                "profile": f"wg_{fact.symbol}_generic",
                "world_type": "finite_small",
                "size": 10,
                "generator": "enumerate_small_inputs",
            }

        # examples から seed worlds を作る
        seed_worlds = []
        for ex in fact.checkable_examples:
            inp = ex.get("input", "")
            out = ex.get("expected_output", "")
            if inp:
                seed_worlds.append({"input": inp, "expected": out})

        return {
            "profile": f"wg_{fact.symbol}_examples",
            "world_type": "example_seeded",
            "seed_worlds": seed_worlds,
            "generator": "example_based_cross_sim",
        }

    # ------------------------------------------------------------------
    # PieceDB 用のdict変換（差し替えポイント）
    # ------------------------------------------------------------------

    def to_piece_dict(self, piece: CrossPiece) -> dict:
        """
        既存の PieceDB が受け取れる形式に変換する。
        piece_db.py の Piece.from_dict() などに合わせて調整する。
        """
        return {
            "id": piece.piece_id,
            "domain": piece.domain,
            "kind": piece.kind,
            "symbol": piece.symbol,
            "pattern": piece.pattern,
            "transform": piece.transform,
            "verify_spec": piece.verify_spec,
            "worldgen_spec": piece.worldgen_spec,
            "grammar_binding": piece.grammar_binding,
            "meta": piece.metadata,
        }
