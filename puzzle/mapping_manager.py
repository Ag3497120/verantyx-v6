"""
MappingManager - 構造パターンマッチング

verantyx_ios構想: テキスト構造から推論テンプレートを検索
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json


@dataclass
class ReasoningTemplate:
    """推論テンプレート"""
    core_formula_pattern: Optional[str] = None
    required_assumptions: List[str] = field(default_factory=list)
    suggested_pieces: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class Mapping:
    """構造マッピング"""
    mapping_id: str
    shape_signature: List[str]  # 構造シグネチャ（順序付き）
    domain: str
    reasoning_template: ReasoningTemplate
    confidence: float
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "mapping_id": self.mapping_id,
            "shape_signature": self.shape_signature,
            "domain": self.domain,
            "reasoning_template": {
                "core_formula_pattern": self.reasoning_template.core_formula_pattern,
                "required_assumptions": self.reasoning_template.required_assumptions,
                "suggested_pieces": self.reasoning_template.suggested_pieces,
                "constraints": self.reasoning_template.constraints
            },
            "confidence": self.confidence,
            "examples": self.examples,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mapping':
        """辞書から復元"""
        template_data = data.get("reasoning_template", {})
        return cls(
            mapping_id=data["mapping_id"],
            shape_signature=data["shape_signature"],
            domain=data["domain"],
            reasoning_template=ReasoningTemplate(
                core_formula_pattern=template_data.get("core_formula_pattern"),
                required_assumptions=template_data.get("required_assumptions", []),
                suggested_pieces=template_data.get("suggested_pieces", []),
                constraints=template_data.get("constraints", [])
            ),
            confidence=data.get("confidence", 0.5),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {})
        )


class MappingManager:
    """
    MappingManager - 構造パターンマッチングエンジン
    
    使用パターン:
    1. 問題の構造シグネチャを計算
    2. マッピングを検索（find_mapping）
    3. 推論テンプレートを適用
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.mappings: List[Mapping] = []
        self.load()
    
    def load(self):
        """JSONLからロード"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.mappings.append(Mapping.from_dict(data))
        except FileNotFoundError:
            pass
    
    def save(self):
        """JSONLに保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            for mapping in self.mappings:
                f.write(json.dumps(mapping.to_dict(), ensure_ascii=False) + '\n')
    
    def find_mapping(
        self,
        shape_signature: List[str],
        confidence_threshold: float = 0.5
    ) -> Optional[Mapping]:
        """
        構造シグネチャに合うマッピングを検索
        
        Args:
            shape_signature: 構造シグネチャ（順序付き）
            confidence_threshold: 信頼度閾値
        
        Returns:
            マッチしたマッピング、見つからない場合None
        """
        best_mapping = None
        best_score = 0.0
        
        for mapping in self.mappings:
            if mapping.confidence < confidence_threshold:
                continue
            
            score = self._similarity_score(shape_signature, mapping.shape_signature)
            
            if score > best_score:
                best_score = score
                best_mapping = mapping
        
        return best_mapping if best_score >= confidence_threshold else None
    
    def add_mapping(self, mapping: Mapping):
        """マッピングを追加"""
        # 重複チェック
        existing = self.find_mapping(mapping.shape_signature, confidence_threshold=0.0)
        if existing:
            # 信頼度が高い方を保持
            if mapping.confidence > existing.confidence:
                self.mappings.remove(existing)
                self.mappings.append(mapping)
        else:
            self.mappings.append(mapping)
    
    def compute_shape_signature(self, ir_dict: Dict[str, Any]) -> List[str]:
        """
        IRから構造シグネチャを計算
        
        構造シグネチャ = 問題の「形」を表す順序付きトークン列
        """
        signature = []
        
        # Task
        signature.append(f"task:{ir_dict.get('task', 'unknown')}")
        
        # Domain
        signature.append(f"domain:{ir_dict.get('domain', 'unknown')}")
        
        # エンティティの型（順序保持）
        for entity in ir_dict.get("entities", []):
            signature.append(f"entity:{entity.get('type', 'unknown')}")
        
        # 制約の型（順序保持）
        for constraint in ir_dict.get("constraints", []):
            signature.append(f"constraint:{constraint.get('type', 'unknown')}")
        
        # クエリ
        query = ir_dict.get("query", {})
        if query:
            signature.append(f"query:{query.get('type', 'unknown')}")
            if query.get("property"):
                signature.append(f"property:{query.get('property')}")
        
        return signature
    
    def _similarity_score(self, sig1: List[str], sig2: List[str]) -> float:
        """
        2つのシグネチャの類似度を計算
        
        Returns:
            0.0-1.0の類似度スコア
        """
        if not sig1 or not sig2:
            return 0.0
        
        # 完全一致
        if sig1 == sig2:
            return 1.0
        
        # 部分一致（Jaccard類似度）
        set1 = set(sig1)
        set2 = set(sig2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
