"""
Piece - Cross-DBの推論部品

型付き署名を持ち、組み合わせ可能な推論ブロック
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class PieceInput:
    """ピース入力仕様"""
    requires: List[str]  # ["domain:arithmetic", "task:compute"]
    slots: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)


@dataclass
class PieceOutput:
    """ピース出力仕様"""
    produces: List[str]  # ["integer", "derivation:proof"]
    schema: str  # "integer"
    artifacts: List[str] = field(default_factory=list)


@dataclass
class PieceCost:
    """ピース計算コスト"""
    time: str = "medium"  # instant, low, medium, high, extreme
    space: str = "low"
    explosion_risk: str = "low"


@dataclass
class PieceVerify:
    """
    ピース検証仕様 (CEGIS 連携)

    証明書の種類とパラメータを定義し、解の正当性を機械的に検証する。
    kind: certificate.CertKind の値（文字列）
    method: 具体的な検証手法
    params: 検証パラメータ
    """
    kind: str = "high_confidence"   # CertKind の値
    method: str = "none"            # "substitution", "cross_check", "small_world", ...
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PieceWorldGen:
    """
    CEGIS 小世界仕様

    このピースが生成した候補をどのドメインの有限モデルで検証するかを指定する。
    domain: WorldGenerator.generate() の第一引数
    params: WorldGenerator.generate() の第二引数
    constraints: 候補が満たすべき制約のリスト
    """
    domain: str = "number"
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


@dataclass
class Piece:
    """
    Piece - 推論部品 (v2: CEGIS 対応)

    型付き署名を持ち、IRの要求と照合して合成可能。
    verify / worldgen フィールドにより CEGIS ループと統合される。
    """
    piece_id: str
    in_spec: PieceInput
    out_spec: PieceOutput
    executor: str  # "module.function"

    name: Optional[str] = None
    description: Optional[str] = None
    verifiers: List[str] = field(default_factory=list)
    cost: PieceCost = field(default_factory=PieceCost)
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # ── CEGIS 拡張フィールド ──────────────────────────────────────────────
    verify: PieceVerify = field(default_factory=PieceVerify)
    worldgen: PieceWorldGen = field(default_factory=PieceWorldGen)
    
    def matches_ir(self, ir_dict: Dict[str, Any]) -> float:
        """
        IRとマッチするかスコアリング（answer_schema優先）
        
        Returns:
            0.0-1.0のマッチスコア
        """
        matched = 0
        total = len(self.in_spec.requires)
        
        if total == 0:
            # Universal pieces with no requirements get a tiny base score
            # so they can be used as fallback when no specific piece matches
            return 0.05
        
        # answer_schemaボーナス（out_specが一致する場合）
        schema_bonus = 0.0
        if self.out_spec.schema == ir_dict.get("answer_schema"):
            schema_bonus = 1.0  # 100%ボーナス（大幅に引き上げ）
        
        for req in self.in_spec.requires:
            if ":" not in req:
                continue
            
            req_type, req_value = req.split(":", 1)
            
            # IRの該当フィールドをチェック（taskを重視）
            if req_type == "task":
                if ir_dict.get("task") == req_value:
                    matched += 2  # taskマッチは2倍の重み
                    total += 1  # totalも調整
            elif req_type == "domain":
                if ir_dict.get("domain") == req_value:
                    matched += 1
            elif req_type == "answer_schema":
                if ir_dict.get("answer_schema") == req_value:
                    matched += 1.5  # answer_schemaは1.5倍
                    total += 0.5  # totalも調整
            elif req_type == "query":
                if ir_dict.get("query", {}).get("type") == req_value:
                    matched += 1
            elif req_type in ["entity", "constraint"]:
                # エンティティ・制約の存在チェック
                if req_type == "entity":
                    if any(e.get("type") == req_value for e in ir_dict.get("entities", [])):
                        matched += 1
                elif req_type == "constraint":
                    if any(c.get("type") == req_value for c in ir_dict.get("constraints", [])):
                        matched += 1
        
        # Base score: マッチ率
        base_score = matched / total if total > 0 else 0.0
        
        # Specificity bonus: より具体的な（requiresが多い）ピースを優先
        # matched が多いほど高スコア
        specificity_bonus = matched * 0.15
        
        # Keyword bonus: IRのkeywordsとピースのtagsが一致する場合
        keyword_bonus = 0.0
        ir_keywords = ir_dict.get("metadata", {}).get("keywords", [])
        if ir_keywords:
            # 特異的なキーワードには高ボーナス
            high_specificity_keywords = [
                "expected", "permutation", "combination", "factorial", 
                "gcd", "lcm", "prime", "pythagorean", "circumference"
            ]
            for keyword in ir_keywords:
                if keyword in self.tags:
                    if keyword in high_specificity_keywords:
                        keyword_bonus += 1.0  # 特異的キーワードは+1.0
                    else:
                        keyword_bonus += 0.5  # 一般キーワードは+0.5
        
        # Final score: 最大3.0まで
        final_score = min(3.0, base_score + schema_bonus + specificity_bonus + keyword_bonus)
        
        return final_score
    
    def can_connect_to(self, other: 'Piece') -> bool:
        """
        他のピースと接続可能かチェック
        
        self.out_spec.produces が other.in_spec.requires を満たすか
        """
        for produce in self.out_spec.produces:
            for require in other.in_spec.requires:
                if produce == require:
                    return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "piece_id": self.piece_id,
            "name": self.name,
            "description": self.description,
            "in": {
                "requires": self.in_spec.requires,
                "slots": self.in_spec.slots,
                "optional": self.in_spec.optional
            },
            "out": {
                "produces": self.out_spec.produces,
                "schema": self.out_spec.schema,
                "artifacts": self.out_spec.artifacts
            },
            "executor": self.executor,
            "verifiers": self.verifiers,
            "cost": {
                "time": self.cost.time,
                "space": self.cost.space,
                "explosion_risk": self.cost.explosion_risk
            },
            "confidence": self.confidence,
            "tags": self.tags,
            "examples": self.examples,
            "verify": {
                "kind": self.verify.kind,
                "method": self.verify.method,
                "params": self.verify.params,
            },
            "worldgen": {
                "domain": self.worldgen.domain,
                "params": self.worldgen.params,
                "constraints": self.worldgen.constraints,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Piece':
        """辞書から復元"""
        # in/out または in_spec/out_spec に対応
        in_data = data.get("in_spec") or data.get("in")
        out_data = data.get("out_spec") or data.get("out")
        
        return cls(
            piece_id=data["piece_id"],
            name=data.get("name"),
            description=data.get("description"),
            in_spec=PieceInput(
                requires=in_data["requires"],
                slots=in_data.get("slots", []),
                optional=in_data.get("optional", [])
            ),
            out_spec=PieceOutput(
                produces=out_data["produces"],
                schema=out_data["schema"],
                artifacts=out_data.get("artifacts", [])
            ),
            executor=data["executor"],
            verifiers=data.get("verifiers", []),
            cost=PieceCost(**data.get("cost", {})),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
            verify=PieceVerify(**data.get("verify", {})),
            worldgen=PieceWorldGen(**data.get("worldgen", {})),
        )


class PieceDB:
    """ピースデータベース"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pieces: List[Piece] = []
        self.load()
    
    def load(self):
        """JSONLからロード"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.pieces.append(Piece.from_dict(data))
        except FileNotFoundError:
            # DBファイルが存在しない場合は空で初期化
            pass
        
        # Claude公理は無効化（HLEに不適合のため）
        # pieces_claude.json / pieces_claude_remapped.json はロードしない
        pass
    
    def save(self):
        """JSONLに保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            for piece in self.pieces:
                f.write(json.dumps(piece.to_dict(), ensure_ascii=False) + '\n')
    
    def add(self, piece: Piece):
        """ピース追加"""
        self.pieces.append(piece)
    
    def find_by_id(self, piece_id: str) -> Optional[Piece]:
        """ID検索"""
        for piece in self.pieces:
            if piece.piece_id == piece_id:
                return piece
        return None
    
    def search(self, ir_dict: Dict[str, Any], top_k: int = 10) -> List[tuple[Piece, float]]:
        """
        IRにマッチするピースを検索
        
        Returns:
            [(piece, score), ...] のリスト（スコア降順）
        """
        scored = []
        for piece in self.pieces:
            score = piece.matches_ir(ir_dict)
            if score > 0:
                # confidenceを考慮した最終スコア
                final_score = score * piece.confidence
                scored.append((piece, final_score))
        
        # スコア降順でソート
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
