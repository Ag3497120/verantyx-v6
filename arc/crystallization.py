"""
arc/crystallization.py — 結晶化: 成功ProgramTreeの再利用

成功した変換パターンを抽象化して保存し、類似タスクに即適用する。
ConceptSignatureと紐づけることで、新タスクに対する初期候補を高速に生成。
"""

from __future__ import annotations
import json
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field, asdict

Grid = List[List[int]]


@dataclass
class Crystal:
    """結晶化された変換パターン"""
    name: str
    rule_type: str  # 'single', 'sequence', 'condition', 'loop'
    description: str
    concept_features: Dict[str, float] = field(default_factory=dict)
    success_count: int = 0
    fail_count: int = 0

    @property
    def confidence(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        return self.success_count / total


class CrystalStore:
    """結晶ストア: 成功パターンの保存・検索"""

    def __init__(self, store_path: Optional[str] = None):
        self.crystals: List[Crystal] = []
        self.store_path = store_path
        if store_path and os.path.exists(store_path):
            self._load()

    def add(self, crystal: Crystal):
        """結晶を追加"""
        self.crystals.append(crystal)
        if self.store_path:
            self._save()

    def find_matching(self, concept_features: Dict[str, float],
                      top_k: int = 5) -> List[Crystal]:
        """概念特徴が類似する結晶を検索"""
        import numpy as np
        if not self.crystals:
            return []

        target = np.array([concept_features.get(k, 0.0)
                          for k in sorted(concept_features.keys())])

        scored = []
        for crystal in self.crystals:
            crystal_vec = np.array([crystal.concept_features.get(k, 0.0)
                                   for k in sorted(concept_features.keys())])
            if np.linalg.norm(target) == 0 or np.linalg.norm(crystal_vec) == 0:
                continue
            sim = float(np.dot(target, crystal_vec) /
                       (np.linalg.norm(target) * np.linalg.norm(crystal_vec)))
            scored.append((crystal, sim))

        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored[:top_k]]

    def record_success(self, name: str):
        """成功を記録"""
        for c in self.crystals:
            if c.name == name:
                c.success_count += 1
                break
        if self.store_path:
            self._save()

    def record_failure(self, name: str):
        """失敗を記録"""
        for c in self.crystals:
            if c.name == name:
                c.fail_count += 1
                break
        if self.store_path:
            self._save()

    def _save(self):
        with open(self.store_path, 'w') as f:
            json.dump([asdict(c) for c in self.crystals], f)

    def _load(self):
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            self.crystals = [Crystal(**d) for d in data]
        except Exception:
            self.crystals = []
