"""
Beam Search - ピース合成経路探索

A*ライクなビームサーチで最適なピース組み合わせを探索
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from queue import PriorityQueue
import time

from pieces.piece import Piece, PieceDB


@dataclass
class SearchNode:
    """探索ノード"""
    pieces: List[Piece]  # 使用したピースのリスト
    current_produces: List[str]  # 現在生成可能な型
    cost: float  # 累積コスト
    confidence: float  # 累積信頼度
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """優先度比較（コストが低く、信頼度が高い方が優先）"""
        return (self.cost - self.confidence) < (other.cost - other.confidence)


class BeamSearch:
    """
    Beam Search アセンブラ
    
    IRの要求を満たすピース経路を探索
    """
    
    def __init__(
        self,
        piece_db: PieceDB,
        beam_width: int = 5,
        max_depth: int = 5,
        timeout_sec: float = 10.0
    ):
        self.piece_db = piece_db
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.timeout_sec = timeout_sec
        
        # コスト変換テーブル
        self.time_cost = {
            "instant": 0.1,
            "low": 1.0,
            "medium": 3.0,
            "high": 10.0,
            "extreme": 100.0
        }
    
    def search(
        self,
        ir_dict: Dict[str, Any],
        target_schema: str
    ) -> Optional[List[Piece]]:
        """
        IRを満たすピース経路を探索
        
        Args:
            ir_dict: IR辞書
            target_schema: 目標のanswerスキーマ
        
        Returns:
            ピースのリスト（順序付き）、見つからない場合None
        """
        start_time = time.time()
        
        # 初期候補ピースを取得
        initial_candidates = self.piece_db.search(ir_dict, top_k=self.beam_width * 2)
        
        if not initial_candidates:
            return None
        
        # 最初の候補が既に目標を満たす場合は即座に返す（高速化）
        best_piece, best_score = initial_candidates[0]
        if target_schema in best_piece.out_spec.produces or target_schema == best_piece.out_spec.schema:
            # 高confidence（>0.9）の場合は単一ピースで完結
            if best_piece.confidence > 0.9:
                return [best_piece]
        
        # 優先度キュー初期化
        queue = PriorityQueue()
        
        for piece, match_score in initial_candidates:
            # 初期ノード作成
            node = SearchNode(
                pieces=[piece],
                current_produces=piece.out_spec.produces[:],
                cost=self._calculate_cost(piece),
                confidence=piece.confidence * match_score
            )
            queue.put(node)
        
        # ビームサーチ
        visited = set()
        
        while not queue.empty():
            # タイムアウトチェック
            if time.time() - start_time > self.timeout_sec:
                break
            
            # 現在のノードを取得
            current = queue.get()
            
            # 目標達成チェック
            if target_schema in current.current_produces:
                return current.pieces
            
            # 深さ制限
            if len(current.pieces) >= self.max_depth:
                continue
            
            # 訪問済みチェック（ピースIDのタプルでハッシュ）
            state_key = tuple(p.piece_id for p in current.pieces)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # 次のピース候補を探索
            next_pieces = self._find_connectable_pieces(current, ir_dict)
            
            for next_piece, connect_score in next_pieces[:self.beam_width]:
                # 新しいノード作成
                new_node = SearchNode(
                    pieces=current.pieces + [next_piece],
                    current_produces=self._merge_produces(
                        current.current_produces,
                        next_piece.out_spec.produces
                    ),
                    cost=current.cost + self._calculate_cost(next_piece),
                    confidence=current.confidence * next_piece.confidence * connect_score,
                    artifacts=current.artifacts.copy()
                )
                
                queue.put(new_node)
        
        # 経路が見つからなかった
        return None
    
    def _find_connectable_pieces(
        self,
        current: SearchNode,
        ir_dict: Dict[str, Any]
    ) -> List[Tuple[Piece, float]]:
        """
        現在のノードに接続可能なピースを検索
        
        Returns:
            [(piece, connect_score), ...] のリスト
        """
        connectable = []
        
        # 最後のピースと接続可能なピースを検索
        last_piece = current.pieces[-1]
        
        for piece in self.piece_db.pieces:
            # すでに使用済みのピースはスキップ
            if piece in current.pieces:
                continue
            
            # 接続可能性チェック
            can_connect = False
            connect_score = 0.0
            
            # last_pieceの出力と接続可能か
            if last_piece.can_connect_to(piece):
                can_connect = True
                connect_score += 0.5
            
            # IRとマッチするか
            ir_match_score = piece.matches_ir(ir_dict)
            if ir_match_score > 0:
                can_connect = True
                connect_score += ir_match_score * 0.5
            
            if can_connect:
                connectable.append((piece, connect_score))
        
        # スコア降順でソート
        connectable.sort(key=lambda x: x[1], reverse=True)
        return connectable
    
    def _calculate_cost(self, piece: Piece) -> float:
        """ピースのコストを計算"""
        time_cost = self.time_cost.get(piece.cost.time, 3.0)
        explosion_penalty = {
            "none": 0.0,
            "low": 0.5,
            "medium": 2.0,
            "high": 5.0
        }.get(piece.cost.explosion_risk, 1.0)
        
        return time_cost + explosion_penalty
    
    def _merge_produces(
        self,
        current: List[str],
        new: List[str]
    ) -> List[str]:
        """生成物リストをマージ（重複除去）"""
        merged = set(current)
        merged.update(new)
        return list(merged)


class GreedyAssembler:
    """
    Greedy Assembler（簡易版）
    
    ビームサーチより高速だが精度は劣る
    """
    
    def __init__(self, piece_db: PieceDB):
        self.piece_db = piece_db
    
    def assemble(
        self,
        ir_dict: Dict[str, Any],
        target_schema: str
    ) -> Optional[List[Piece]]:
        """
        貪欲法でピースを選択
        
        IRに最もマッチするピースを選び、
        target_schemaを生成できるまで繰り返す
        """
        selected = []
        current_produces = []
        max_iterations = 5
        
        for _ in range(max_iterations):
            # 目標達成チェック
            if target_schema in current_produces:
                return selected
            
            # 次のピース候補を取得
            candidates = self.piece_db.search(ir_dict, top_k=10)
            
            if not candidates:
                break
            
            # 既存のピースと接続可能なものを優先
            best_piece = None
            best_score = 0.0
            
            for piece, match_score in candidates:
                # 既に使用済みならスキップ
                if piece in selected:
                    continue
                
                score = match_score
                
                # スロット要件チェック（重要！）
                # エンティティ数とスロット数が一致するか、または名前付きエンティティがあるか
                required_slots = piece.in_spec.slots
                entities = ir_dict.get("entities", [])
                
                # 名前付きスロット（n, r）がある場合、名前付きエンティティが必要
                named_slots = [s for s in required_slots if s in ['n', 'r', 'k', 'm']]
                if named_slots:
                    named_entities = [e for e in entities if e.get("name") in named_slots]
                    # 名前付きスロットが全て満たせるか
                    if len(named_entities) < len(named_slots):
                        score -= 2.0  # 大幅ペナルティ
                
                # 一般的な数値スロット（a, b, number）の場合
                general_slots = [s for s in required_slots if s in ['a', 'b', 'number', 'lhs', 'rhs']]
                if general_slots:
                    number_entities = [e for e in entities if e.get("type") == "number"]
                    # 数値エンティティが足りない場合ペナルティ
                    if len(number_entities) < len(general_slots):
                        score -= 2.0  # 大幅ペナルティ
                
                # 接続可能性ボーナス
                if selected and selected[-1].can_connect_to(piece):
                    score += 0.5
                
                # 目標スキーマを直接生成できるならボーナス
                if target_schema in piece.out_spec.produces:
                    score += 1.0
                
                if score > best_score:
                    best_score = score
                    best_piece = piece
            
            if best_piece is None:
                break
            
            # ピース追加
            selected.append(best_piece)
            current_produces.extend(best_piece.out_spec.produces)
        
        # 目標達成チェック
        if target_schema in current_produces:
            return selected
        
        return None
