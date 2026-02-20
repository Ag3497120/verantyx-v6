"""
Crystallizer - 過去解答の結晶化

verantyx_ios構想: 高信頼度の過去解答を即座に再利用
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import hashlib


@dataclass
class Crystal:
    """結晶化された解答"""
    signature: str  # 問題の構造シグネチャ
    ir_pattern: Dict[str, Any]  # IRパターン
    answer: Any  # 答え
    schema: str  # 答えのスキーマ
    confidence: float  # 信頼度（0.8以上で即答）
    evidence: List[str] = field(default_factory=list)  # 証拠
    metadata: Dict[str, Any] = field(default_factory=dict)
    # B3: 汚染ガード
    quarantined: bool = False          # True なら読み取りスキップ（隔離タグ）
    verified: bool = False             # CEGIS / 外部検証で確認済みか
    write_source: str = "unknown"      # "cegis_proved" | "cegis_verified" | "legacy"

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "signature": self.signature,
            "ir_pattern": self.ir_pattern,
            "answer": self.answer,
            "schema": self.schema,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "quarantined": self.quarantined,
            "verified": self.verified,
            "write_source": self.write_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Crystal':
        """辞書から復元"""
        return cls(
            signature=data["signature"],
            ir_pattern=data["ir_pattern"],
            answer=data["answer"],
            schema=data["schema"],
            confidence=data["confidence"],
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
            quarantined=data.get("quarantined", False),
            verified=data.get("verified", False),
            write_source=data.get("write_source", "legacy"),
        )


class Crystallizer:
    """
    Crystallizer - 過去解答の結晶化エンジン
    
    使用パターン:
    1. 問題を解く
    2. VERIFIED結果を結晶化（crystallize）
    3. 次回同じパターンの問題で即答（query_crystal）
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.crystals: List[Crystal] = []
        self.load()
    
    def load(self):
        """JSONLからロード"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.crystals.append(Crystal.from_dict(data))
        except FileNotFoundError:
            pass
    
    def save(self):
        """JSONLに保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            for crystal in self.crystals:
                f.write(json.dumps(crystal.to_dict(), ensure_ascii=False) + '\n')
    
    def query_crystal(self, ir_dict: Dict[str, Any], confidence_threshold: float = 0.8) -> Optional[Crystal]:
        """
        IRに合う結晶を検索（B3: 隔離済みは除外）

        Args:
            ir_dict: IR辞書
            confidence_threshold: 信頼度閾値（デフォルト0.8）

        Returns:
            マッチした結晶、見つからない場合None
        """
        signature = self._compute_signature(ir_dict)

        for crystal in self.crystals:
            # B3: 隔離タグが付いていればスキップ
            if crystal.quarantined:
                continue
            if crystal.signature == signature and crystal.confidence >= confidence_threshold:
                if self._matches_pattern(ir_dict, crystal.ir_pattern):
                    return crystal

        return None

    def crystallize(
        self,
        ir_dict: Dict[str, Any],
        answer: Any,
        schema: str,
        confidence: float,
        evidence: List[str],
        metadata: Dict[str, Any] = None,
        # B3: 書き込みガード
        verified: bool = False,
        write_source: str = "unknown",
    ) -> Optional["Crystal"]:
        """
        解答を結晶化（B3: verified=True の場合のみ書き込む）

        Args:
            ir_dict:      IR辞書
            answer:       答え
            schema:       答えのスキーマ
            confidence:   信頼度
            evidence:     証拠（使用したpiece_id等）
            metadata:     メタデータ
            verified:     外部検証済みか（Falseなら書き込みスキップ）
            write_source: 書き込み元タグ（"cegis_proved" 等）

        Returns:
            Crystal（書き込みスキップの場合は None）
        """
        # B3: 未検証は結晶化しない
        if not verified:
            return None

        signature = self._compute_signature(ir_dict)
        ir_pattern = self._extract_pattern(ir_dict)

        crystal = Crystal(
            signature=signature,
            ir_pattern=ir_pattern,
            answer=answer,
            schema=schema,
            confidence=confidence,
            evidence=evidence,
            metadata=metadata or {},
            quarantined=False,
            verified=True,
            write_source=write_source,
        )

        # 既存の結晶と重複チェック（隔離済み含む全件対象）
        existing = None
        for c in self.crystals:
            if c.signature == signature and self._matches_pattern(ir_dict, c.ir_pattern):
                existing = c
                break

        if existing:
            if crystal.confidence > existing.confidence:
                self.crystals.remove(existing)
                self.crystals.append(crystal)
        else:
            self.crystals.append(crystal)

        return crystal

    def quarantine(self, signature: str) -> int:
        """
        B3: シグネチャに一致する全結晶を隔離タグで封印

        削除ではなく隔離（再学習防止・原因調査のため保持）。

        Returns:
            隔離した件数
        """
        count = 0
        for crystal in self.crystals:
            if crystal.signature == signature and not crystal.quarantined:
                crystal.quarantined = True
                count += 1
        return count

    def quarantine_all_unverified(self) -> int:
        """
        B3: verified=False の全結晶を隔離（汚染データの一括封印）

        Returns:
            隔離した件数
        """
        count = 0
        for crystal in self.crystals:
            if not crystal.verified and not crystal.quarantined:
                crystal.quarantined = True
                count += 1
        return count
    
    def _compute_signature(self, ir_dict: Dict[str, Any]) -> str:
        """
        IRから構造シグネチャを計算（B3強化版）

        エンティティの値も含めることで「3+5」と「7*8」が
        同じシグネチャにならないようにする。
        """
        key_parts = [
            ir_dict.get("task", ""),
            ir_dict.get("domain", ""),
            ir_dict.get("answer_schema", ""),
        ]

        # エンティティ：型 + 値（値があれば含める）
        entity_sigs = []
        for e in ir_dict.get("entities", []):
            etype = e.get("type", "")
            evalue = e.get("value")
            ename  = e.get("name", "")
            if evalue is not None:
                entity_sigs.append(f"{etype}:{evalue}")
            elif ename:
                entity_sigs.append(f"{etype}:{ename}")
            else:
                entity_sigs.append(etype)
        key_parts.extend(sorted(entity_sigs))

        # 制約の型
        constraint_types = sorted([
            c.get("type", "") for c in ir_dict.get("constraints", [])
        ])
        key_parts.extend(constraint_types)

        # メタデータのキーワードも含める（より高精度化）
        keywords = ir_dict.get("metadata", {}).get("keywords", [])
        if keywords:
            key_parts.extend(sorted(str(k) for k in keywords[:3]))

        key_str = "|".join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _extract_pattern(self, ir_dict: Dict[str, Any]) -> Dict[str, Any]:
        """IRからパターンを抽出"""
        return {
            "task": ir_dict.get("task"),
            "domain": ir_dict.get("domain"),
            "answer_schema": ir_dict.get("answer_schema"),
            "entity_types": [e.get("type") for e in ir_dict.get("entities", [])],
            "constraint_types": [c.get("type") for c in ir_dict.get("constraints", [])]
        }
    
    def _matches_pattern(self, ir_dict: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """IRがパターンにマッチするかチェック"""
        # 基本フィールド
        if ir_dict.get("task") != pattern.get("task"):
            return False
        if ir_dict.get("domain") != pattern.get("domain"):
            return False
        if ir_dict.get("answer_schema") != pattern.get("answer_schema"):
            return False
        
        # エンティティの型
        entity_types = [e.get("type") for e in ir_dict.get("entities", [])]
        if sorted(entity_types) != sorted(pattern.get("entity_types", [])):
            return False
        
        # 制約の型
        constraint_types = [c.get("type") for c in ir_dict.get("constraints", [])]
        if sorted(constraint_types) != sorted(pattern.get("constraint_types", [])):
            return False
        
        return True
