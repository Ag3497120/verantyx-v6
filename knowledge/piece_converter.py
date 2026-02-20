"""
Piece Converter

Converts extracted weight knowledge (WeightKnowledgePiece) and
semantic extractions (ExtractedKnowledgePiece) into valid
Verantyx piece_db.jsonl entries.

Also provides utilities for merging with the existing piece_db.jsonl,
deduplication, and validation.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass


# ──────────────────────────────────────────────
# Piece Schema Validation
# ──────────────────────────────────────────────

REQUIRED_FIELDS = {"piece_id", "name", "description", "in", "out", "executor", "confidence"}

VALID_EXECUTORS = {
    "executors.arithmetic.evaluate",
    "executors.algebra.solve",
    "executors.calculus.derivative",
    "executors.calculus.integral",
    "executors.linear_algebra.compute",
    "executors.number_theory.compute",
    "executors.probability.compute",
    "executors.combinatorics.compute",
    "executors.geometry.compute",
    "executors.logic.evaluate",
    "executors.knowledge.lookup",      # Generic knowledge lookup
    "executors.knowledge.formula",     # Formula application
    "executors.knowledge.theorem",     # Theorem verification
}

DOMAIN_TO_EXECUTOR = {
    "arithmetic": "executors.arithmetic.evaluate",
    "algebra": "executors.algebra.solve",
    "calculus": "executors.calculus.derivative",
    "math_calculus": "executors.calculus.derivative",
    "math_algebra": "executors.algebra.solve",
    "math_number_theory": "executors.number_theory.compute",
    "math_linear_algebra": "executors.linear_algebra.compute",
    "math_probability": "executors.probability.compute",
    "math_combinatorics": "executors.combinatorics.compute",
    "math_geometry": "executors.geometry.compute",
    "math": "executors.knowledge.lookup",
    "physics": "executors.knowledge.lookup",
    "chemistry": "executors.knowledge.lookup",
    "biology": "executors.knowledge.lookup",
    "computer_science": "executors.knowledge.lookup",
    "history": "executors.knowledge.lookup",
    "literature": "executors.knowledge.lookup",
    "philosophy": "executors.knowledge.lookup",
}


# ──────────────────────────────────────────────
# Converter
# ──────────────────────────────────────────────

class PieceConverter:
    """
    Converts extracted knowledge objects into valid piece_db.jsonl entries.

    Supports:
    - WeightKnowledgePiece (from weight_extractor.py)
    - ExtractedKnowledgePiece (from semantic_extractor.py)
    - Raw dicts from routing analysis
    """

    def __init__(self, piece_db_path: Optional[str] = None):
        """
        Args:
            piece_db_path: Path to existing piece_db.jsonl to check for duplicates
        """
        self.existing_ids: set = set()

        if piece_db_path and Path(piece_db_path).exists():
            self._load_existing_ids(piece_db_path)
            print(f"[PIECE_CONVERTER] Loaded {len(self.existing_ids)} existing piece IDs")
        else:
            print("[PIECE_CONVERTER] No existing piece_db loaded (fresh start)")

    def _load_existing_ids(self, path: str):
        """Load existing piece IDs to avoid duplicates."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        self.existing_ids.add(obj.get("piece_id", ""))
                    except json.JSONDecodeError:
                        pass

    # ──────────────────────────────────────────
    # Main Converters
    # ──────────────────────────────────────────

    def weight_knowledge_to_piece(self, wk) -> Dict[str, Any]:
        """
        Convert WeightKnowledgePiece → Verantyx piece format.

        Args:
            wk: WeightKnowledgePiece instance

        Returns:
            Dict compatible with piece_db.jsonl schema
        """
        domain_str = wk.domain.value if hasattr(wk.domain, 'value') else str(wk.domain)
        executor = DOMAIN_TO_EXECUTOR.get(domain_str, "executors.knowledge.lookup")

        return {
            "piece_id": wk.id,
            "name": wk.name,
            "description": wk.description,
            "in": {
                "requires": [f"domain:{domain_str}"],
                "slots": []
            },
            "out": {
                "produces": ["knowledge"],
                "schema": "knowledge"
            },
            "executor": executor,
            "confidence": float(wk.confidence),
            "tags": list(wk.tags) + ["weight_extracted"],
            "source": "600b_weight_extraction",
            "metadata": {
                "layer": wk.layer,
                "expert_id": wk.expert_id,
                "coords": list(wk.coords),
                "weight_patterns": wk.weight_patterns,
            }
        }

    def semantic_piece_to_db(self, ep) -> Dict[str, Any]:
        """
        Convert ExtractedKnowledgePiece → Verantyx piece format.

        Args:
            ep: ExtractedKnowledgePiece instance

        Returns:
            Dict compatible with piece_db.jsonl schema
        """
        return ep.to_piece_db_format()

    def raw_dict_to_piece(
        self,
        piece_id: str,
        name: str,
        description: str,
        domain: str,
        formula: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        executor: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Convert raw data into a piece_db entry.

        Useful for manually specified knowledge entries.
        """
        executor = executor or DOMAIN_TO_EXECUTOR.get(domain, "executors.knowledge.lookup")
        tags = tags or [domain, "manual"]

        piece = {
            "piece_id": piece_id,
            "name": name,
            "description": description,
            "in": {
                "requires": [f"domain:{domain}"],
                "slots": []
            },
            "out": {
                "produces": ["knowledge"],
                "schema": "knowledge"
            },
            "executor": executor,
            "confidence": float(confidence),
            "tags": tags + ["600b_extracted"],
            "source": "600b_weight_extraction",
            "knowledge": {
                "formula": formula,
                "domain": domain,
            }
        }

        if extra:
            piece.update(extra)

        return piece

    # ──────────────────────────────────────────
    # Batch Operations
    # ──────────────────────────────────────────

    def convert_batch(
        self,
        items: List[Any],
        item_type: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Convert a batch of knowledge items to piece_db format.

        Args:
            items: List of WeightKnowledgePiece, ExtractedKnowledgePiece, or dicts
            item_type: "weight", "semantic", "dict", or "auto"

        Returns:
            List of piece_db dicts (deduplicated, validated)
        """
        pieces = []
        skipped = 0

        for item in items:
            try:
                if item_type == "weight" or (item_type == "auto" and hasattr(item, "weight_patterns")):
                    piece = self.weight_knowledge_to_piece(item)
                elif item_type == "semantic" or (item_type == "auto" and hasattr(item, "formula")):
                    piece = self.semantic_piece_to_db(item)
                elif isinstance(item, dict):
                    piece = item
                else:
                    print(f"[PIECE_CONVERTER] Unknown item type: {type(item)}")
                    skipped += 1
                    continue

                # Validate
                if not self.validate(piece):
                    skipped += 1
                    continue

                # Dedup check
                pid = piece.get("piece_id", "")
                if pid in self.existing_ids:
                    skipped += 1
                    continue

                self.existing_ids.add(pid)
                pieces.append(piece)

            except Exception as e:
                print(f"[PIECE_CONVERTER] Error converting item: {e}")
                skipped += 1

        print(f"[PIECE_CONVERTER] Converted {len(pieces)} pieces, skipped {skipped}")
        return pieces

    def validate(self, piece: Dict[str, Any]) -> bool:
        """
        Validate a piece against the required schema.

        Returns True if valid, False otherwise.
        """
        for field in REQUIRED_FIELDS:
            if field not in piece:
                return False

        if not isinstance(piece.get("piece_id"), str) or not piece["piece_id"]:
            return False

        if not (0.0 <= float(piece.get("confidence", -1)) <= 1.0):
            return False

        return True

    def write_jsonl(
        self,
        pieces: List[Dict[str, Any]],
        output_path: str,
        mode: str = "w",  # "w" = overwrite, "a" = append
    ) -> int:
        """
        Write pieces to a JSONL file.

        Args:
            pieces: List of piece dicts
            output_path: Output file path
            mode: File open mode ("w" or "a")

        Returns:
            Number of pieces written
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(output_path, mode, encoding="utf-8") as f:
            for piece in pieces:
                f.write(json.dumps(piece, ensure_ascii=False) + "\n")
                count += 1

        print(f"[PIECE_CONVERTER] Wrote {count} pieces to {output_path}")
        return count

    def merge_into_existing(
        self,
        new_pieces: List[Dict[str, Any]],
        existing_path: str,
        output_path: Optional[str] = None,
        skip_duplicates: bool = True,
    ) -> int:
        """
        Merge new pieces into an existing piece_db.jsonl.

        Args:
            new_pieces: New pieces to add
            existing_path: Path to existing piece_db.jsonl
            output_path: Output path (None = modify in-place)
            skip_duplicates: Skip pieces with IDs already in existing

        Returns:
            Number of pieces added
        """
        output_path = output_path or existing_path

        # Load existing
        existing = []
        existing_ids = set()
        if Path(existing_path).exists():
            with open(existing_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            existing.append(obj)
                            existing_ids.add(obj.get("piece_id", ""))
                        except json.JSONDecodeError:
                            pass

        # Filter new pieces
        added = 0
        for piece in new_pieces:
            pid = piece.get("piece_id", "")
            if skip_duplicates and pid in existing_ids:
                continue
            existing.append(piece)
            existing_ids.add(pid)
            added += 1

        # Write merged result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for piece in existing:
                f.write(json.dumps(piece, ensure_ascii=False) + "\n")

        print(f"[PIECE_CONVERTER] Merged: {added} new pieces added to {output_path}")
        print(f"[PIECE_CONVERTER] Total pieces in DB: {len(existing)}")
        return added

    def generate_piece_id(self, content: str, prefix: str = "600b") -> str:
        """Generate a stable piece_id from content hash."""
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{prefix}_{h}"

    def stats(self, pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics about a set of pieces."""
        domains = {}
        executors = {}
        confidences = []

        for p in pieces:
            # Domain from tags
            for tag in p.get("tags", []):
                if not tag.startswith("600b"):
                    domains[tag] = domains.get(tag, 0) + 1

            # Executor
            exc = p.get("executor", "unknown")
            executors[exc] = executors.get(exc, 0) + 1

            # Confidence
            confidences.append(float(p.get("confidence", 0)))

        return {
            "total": len(pieces),
            "domains": domains,
            "executors": executors,
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
        }


# ──────────────────────────────────────────────
# CLI / Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Piece Converter - Quick Test ===")

    converter = PieceConverter()

    # Test raw dict conversion
    piece = converter.raw_dict_to_piece(
        piece_id="test_power_rule",
        name="Power Rule",
        description="The derivative of x^n is nx^(n-1)",
        domain="math_calculus",
        formula="d/dx[x^n] = nx^(n-1)",
        confidence=0.99,
        tags=["calculus", "derivative"],
    )

    print(f"\nSample piece:")
    print(json.dumps(piece, indent=2))

    # Validate
    assert converter.validate(piece), "Validation failed!"
    print("\n✓ Piece validated successfully")
