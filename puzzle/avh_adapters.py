"""
AVH Math Adapters

avh_mathのコンポーネントをVerantyx V6に適合させるアダプター層
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ILSlots:
    """
    avh_mathのILSlots互換クラス
    
    Verantyx V6のIRから変換
    """
    problem_type: str = "unknown"
    domain: str = "unknown"
    formula: Optional[str] = None
    atoms: Optional[List[str]] = None
    target: Optional[str] = None
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    @classmethod
    def from_ir(cls, ir_dict: Dict[str, Any]) -> 'ILSlots':
        """
        Verantyx V6のIRからILSlotsに変換
        
        Args:
            ir_dict: IRの辞書
        
        Returns:
            ILSlotsインスタンス
        """
        # Formula抽出（entitiesから）
        formula = None
        atoms = []
        
        for entity in ir_dict.get("entities", []):
            if entity.get("type") == "formula":
                formula = entity.get("value")
            elif entity.get("type") == "atom":
                atoms.append(entity.get("value"))
        
        # Formulaがない場合、source_textから抽出を試みる
        if not formula:
            source_text = ir_dict.get("metadata", {}).get("source_text", "")
            # 論理式パターンを探す
            if any(op in source_text for op in ["->", "→", "&", "|", "~", "¬"]):
                # 引用符で囲まれた部分を抽出
                import re
                matches = re.findall(r"['\"]([^'\"]+)['\"]", source_text)
                if matches:
                    formula = matches[0]
        
        return cls(
            problem_type=ir_dict.get("task", "unknown"),
            domain=ir_dict.get("domain", "unknown"),
            formula=formula,
            atoms=atoms if atoms else None,
            target=ir_dict.get("query", {}).get("target"),
            constraints=[c.get("expression", "") for c in ir_dict.get("constraints", [])]
        )


@dataclass
class CrossAsset:
    """
    avh_mathのCrossAsset互換クラス
    
    公理・定理を表現
    """
    asset_id: str
    requires: List[str]
    provides: List[str]
    converts: Dict[str, str]
    verifies: List[str]
    content: Dict[str, Any]
    confidence: float = 1.0
    
    @classmethod
    def from_axiom_dict(cls, axiom_dict: Dict[str, Any]) -> 'CrossAsset':
        """
        公理辞書からCrossAssetに変換
        """
        return cls(
            asset_id=axiom_dict.get("asset_id", "unknown"),
            requires=axiom_dict.get("requires", []),
            provides=axiom_dict.get("provides", []),
            converts=axiom_dict.get("converts", {}),
            verifies=axiom_dict.get("verifies", []),
            content=axiom_dict.get("content", {}),
            confidence=axiom_dict.get("confidence", 1.0)
        )


class CrossDB:
    """
    avh_mathのCrossDB互換クラス
    
    公理データベース
    """
    
    def __init__(self, axiom_file: Optional[str] = None):
        """
        Args:
            axiom_file: 公理JSONファイルのパス
        """
        self.assets = []
        
        if axiom_file:
            self.load_axioms(axiom_file)
    
    def load_axioms(self, axiom_file: str):
        """
        公理ファイルをロード
        
        Args:
            axiom_file: pieces/axioms_unified.json のパス
        """
        path = Path(axiom_file)
        if not path.exists():
            print(f"[CROSS_DB] Axiom file not found: {axiom_file}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assets_data = data.get("assets", [])
        self.assets = [CrossAsset.from_axiom_dict(a) for a in assets_data]
        
        print(f"[CROSS_DB] Loaded {len(self.assets)} axioms")
    
    def search(
        self,
        requires: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        domain: Optional[str] = None
    ) -> List[CrossAsset]:
        """
        公理を検索
        
        Args:
            requires: 必要な前提条件
            provides: 提供する結果
            domain: ドメイン
        
        Returns:
            マッチする公理のリスト
        """
        results = []
        
        for asset in self.assets:
            # requiresでフィルタ
            if requires:
                if not any(r in asset.requires for r in requires):
                    continue
            
            # providesでフィルタ
            if provides:
                if not any(p in asset.provides for p in provides):
                    continue
            
            # domainでフィルタ
            if domain:
                asset_domain = asset.content.get("domain", "")
                if domain.lower() not in asset_domain.lower():
                    continue
            
            results.append(asset)
        
        return results
    
    def get_by_id(self, asset_id: str) -> Optional[CrossAsset]:
        """
        IDで公理を取得
        """
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        return None
    
    def list_all(self) -> List[CrossAsset]:
        """
        全公理を取得
        """
        return self.assets


def convert_ir_to_il_slots(ir_dict: Dict[str, Any]) -> ILSlots:
    """
    Verantyx V6のIR → avh_mathのILSlots変換
    
    Args:
        ir_dict: IRの辞書形式
    
    Returns:
        ILSlotsインスタンス
    """
    return ILSlots.from_ir(ir_dict)


def load_cross_db(axiom_file: str = "pieces/axioms_unified.json") -> CrossDB:
    """
    CrossDBをロード
    
    Args:
        axiom_file: 公理ファイルのパス
    
    Returns:
        CrossDBインスタンス
    """
    return CrossDB(axiom_file)
