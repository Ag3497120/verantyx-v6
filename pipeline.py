"""
Verantyx V6 - メインパイプライン

構想準拠の完全実装：
1. Decomposer (ルールベース)
2. Piece Retrieval (型マッチング)
3. Beam Search Assembly (ピース合成)
4. Executor (決定的実行)
5. Grammar Glue (テンプレート)
"""

from typing import Dict, Any, Optional, List
import os

from core.ir import IR
from pieces.piece import PieceDB
from decomposer.decomposer import RuleBasedDecomposer
from assembler.beam_search import BeamSearch, GreedyAssembler
from assembler.executor import Executor, StructuredCandidate
from grammar.composer import GrammarDB, AnswerComposer


class VerantyxV6:
    """
    Verantyx V6 パイプライン
    
    完全ルールベース、LLM不使用の推論システム
    """
    
    def __init__(
        self,
        piece_db_path: Optional[str] = None,
        grammar_db_path: Optional[str] = None,
        use_beam_search: bool = True
    ):
        # デフォルトパス
        if piece_db_path is None:
            piece_db_path = os.path.join(
                os.path.dirname(__file__),
                "pieces/piece_db.jsonl"
            )
        if grammar_db_path is None:
            grammar_db_path = os.path.join(
                os.path.dirname(__file__),
                "grammar/grammar_db.jsonl"
            )
        
        # コンポーネント初期化
        self.decomposer = RuleBasedDecomposer()
        self.piece_db = PieceDB(piece_db_path)
        self.grammar_db = GrammarDB(grammar_db_path)
        
        self.use_beam_search = use_beam_search
        if use_beam_search:
            self.assembler = BeamSearch(self.piece_db)
        else:
            self.assembler = GreedyAssembler(self.piece_db)
        
        self.executor = Executor()
        self.composer = AnswerComposer(self.grammar_db)
        
        # 統計
        self.stats = {
            "total": 0,
            "ir_extracted": 0,
            "pieces_found": 0,
            "executed": 0,
            "composed": 0,
            "failed": 0
        }
    
    def solve(
        self,
        problem_text: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        問題を解く
        
        Args:
            problem_text: 問題文
            expected_answer: 期待される答え（検証用、オプショナル）
        
        Returns:
            結果辞書
        """
        self.stats["total"] += 1
        trace = []
        
        # Step 1: Decompose（問題文→IR）
        trace.append("step:decompose")
        try:
            ir = self.decomposer.decompose(problem_text)
            self.stats["ir_extracted"] += 1
            trace.append(f"ir:task={ir.task.value},domain={ir.domain.value},schema={ir.answer_schema.value}")
        except Exception as e:
            trace.append(f"decompose_error:{e}")
            self.stats["failed"] += 1
            return {
                "status": "FAILED",
                "error": f"Decompose failed: {e}",
                "trace": trace
            }
        
        # Step 2: Piece Retrieval（ピース検索）
        trace.append("step:retrieve")
        ir_dict = ir.to_dict()
        
        if self.use_beam_search:
            # Beam Search
            pieces = self.assembler.search(ir_dict, ir.answer_schema.value)
        else:
            # Greedy
            pieces = self.assembler.assemble(ir_dict, ir.answer_schema.value)
        
        if pieces is None:
            trace.append("no_pieces_found")
            self.stats["failed"] += 1
            return {
                "status": "FAILED",
                "error": "No suitable pieces found",
                "ir": ir_dict,
                "trace": trace
            }
        
        self.stats["pieces_found"] += 1
        trace.append(f"pieces_found:{len(pieces)}")
        trace.append(f"piece_ids:{[p.piece_id for p in pieces]}")
        
        # Step 3: Execute（ピース実行）
        trace.append("step:execute")
        try:
            candidate = self.executor.execute_path(pieces, ir_dict)
            
            if candidate is None:
                trace.append("execution_failed")
                self.stats["failed"] += 1
                return {
                    "status": "FAILED",
                    "error": "Execution failed",
                    "ir": ir_dict,
                    "pieces": [p.piece_id for p in pieces],
                    "trace": trace
                }
            
            self.stats["executed"] += 1
            trace.append(f"candidate:schema={candidate.schema},confidence={candidate.confidence:.2f}")
        
        except Exception as e:
            trace.append(f"execute_error:{e}")
            self.stats["failed"] += 1
            return {
                "status": "FAILED",
                "error": f"Execution error: {e}",
                "ir": ir_dict,
                "pieces": [p.piece_id for p in pieces],
                "trace": trace
            }
        
        # Step 4: Grammar Glue（最終答え合成）
        trace.append("step:compose")
        try:
            answer = self.composer.compose(candidate)
            
            if answer is None:
                trace.append("compose_failed")
                self.stats["failed"] += 1
                return {
                    "status": "FAILED",
                    "error": "Compose failed",
                    "candidate": candidate.to_dict(),
                    "trace": trace
                }
            
            self.stats["composed"] += 1
            trace.append(f"answer:{answer}")
        
        except Exception as e:
            trace.append(f"compose_error:{e}")
            self.stats["failed"] += 1
            return {
                "status": "FAILED",
                "error": f"Compose error: {e}",
                "candidate": candidate.to_dict(),
                "trace": trace
            }
        
        # Step 5: Validation（検証）
        trace.append("step:validate")
        
        # スキーマ検証
        is_valid = self.composer.validate(answer, candidate.schema)
        if not is_valid:
            trace.append("schema_validation_failed")
        
        # 期待される答えとの比較
        status = "UNKNOWN"
        if expected_answer is not None:
            if str(answer).strip() == str(expected_answer).strip():
                status = "VERIFIED"
                trace.append("match:exact")
            else:
                status = "FAILED"
                trace.append(f"mismatch:{answer}!={expected_answer}")
        else:
            status = "SOLVED"
        
        return {
            "status": status,
            "answer": answer,
            "expected": expected_answer,
            "confidence": candidate.confidence,
            "ir": ir_dict,
            "pieces": [p.piece_id for p in pieces],
            "candidate": candidate.to_dict(),
            "trace": trace
        }
    
    def batch_solve(
        self,
        problems: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        複数問題を一括処理
        
        Args:
            problems: 問題リスト [{"text": "...", "expected": "..."}, ...]
            show_progress: 進捗表示
        
        Returns:
            結果リスト
        """
        results = []
        
        for i, problem in enumerate(problems):
            if show_progress:
                print(f"Solving {i+1}/{len(problems)}...", end="\r")
            
            problem_text = problem.get("text") or problem.get("question")
            expected = problem.get("expected") or problem.get("expected_answer")
            
            result = self.solve(problem_text, expected)
            results.append(result)
        
        if show_progress:
            print()
        
        return results
    
    def print_stats(self):
        """統計を表示"""
        print("\n" + "=" * 80)
        print("Verantyx V6 Statistics")
        print("=" * 80)
        print(f"Total problems: {self.stats['total']}")
        print(f"IR extracted: {self.stats['ir_extracted']} ({100*self.stats['ir_extracted']/max(1,self.stats['total']):.1f}%)")
        print(f"Pieces found: {self.stats['pieces_found']} ({100*self.stats['pieces_found']/max(1,self.stats['total']):.1f}%)")
        print(f"Executed: {self.stats['executed']} ({100*self.stats['executed']/max(1,self.stats['total']):.1f}%)")
        print(f"Composed: {self.stats['composed']} ({100*self.stats['composed']/max(1,self.stats['total']):.1f}%)")
        print(f"Failed: {self.stats['failed']} ({100*self.stats['failed']/max(1,self.stats['total']):.1f}%)")
        print("=" * 80)
