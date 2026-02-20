"""
Verantyx V6 + AVH Math Enhanced Pipeline

avh_mathのCross Simulatorを統合した強化版パイプライン
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from pipeline_enhanced import VerantyxV6Enhanced
from puzzle.avh_adapters import CrossDB, ILSlots, convert_ir_to_il_slots
from puzzle.cross_simulator import CrossSimulator, SimulationResult
from core.ir import Domain


@dataclass
class AvhEnhancedResult:
    """AVH統合結果"""
    answer: Any
    status: str
    confidence: float
    reasoning_mode: str  # "verantyx_only", "avh_cross", "verantyx_fallback"
    verified: bool
    axioms_used: List[str]
    trace: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "confidence": self.confidence,
            "reasoning_mode": self.reasoning_mode,
            "verified": self.verified,
            "axioms_used": self.axioms_used,
            "trace": self.trace
        }


class VerantyxAvhEnhanced:
    """
    Verantyx V6 + AVH Math統合パイプライン
    
    推論フロー:
    1. Verantyx V6で基本推論
    2. 論理問題の場合 → AVH Cross Simulator
    3. 失敗時 → Verantyxフォールバック
    """
    
    def __init__(
        self,
        use_avh: bool = True,
        axiom_file: str = "pieces/axioms_unified.json"
    ):
        """
        初期化
        
        Args:
            use_avh: AVH統合を有効化
            axiom_file: 公理ファイルのパス
        """
        # Verantyx V6パイプライン
        self.verantyx = VerantyxV6Enhanced(
            use_beam_search=False,
            use_simulation=False
        )
        
        # AVH統合
        self.use_avh = use_avh
        if self.use_avh:
            self.cross_db = CrossDB(axiom_file)
            self.cross_simulator = CrossSimulator(self.cross_db)
            print(f"[AVH_ENHANCED] Initialized with {len(self.cross_db.assets)} axioms")
        else:
            self.cross_db = None
            self.cross_simulator = None
    
    def solve(self, problem: str) -> AvhEnhancedResult:
        """
        問題を解く
        
        Args:
            problem: 問題文
        
        Returns:
            AvhEnhancedResult
        """
        trace = ["[AVH_ENHANCED] Starting enhanced reasoning"]
        
        # Step 1: Verantyx V6で基本推論
        trace.append("[AVH_ENHANCED] Step 1: Verantyx V6 reasoning")
        v6_result = self.verantyx.solve(problem)
        
        ir = v6_result.get("ir", {})
        domain = ir.get("domain", "unknown")
        confidence = v6_result.get("confidence", 0.0)
        status = v6_result.get("status", "FAILED")
        
        trace.append(f"[AVH_ENHANCED] V6 result: status={status}, confidence={confidence:.2f}, domain={domain}")
        
        # Step 2: 論理問題かつAVH有効の場合
        if self.use_avh and self._is_logic_problem(domain, problem):
            trace.append("[AVH_ENHANCED] Step 2: Logic problem detected, using AVH Cross Simulator")
            
            try:
                # IR → ILSlots変換
                il_slots = convert_ir_to_il_slots(ir)
                
                # AVH Cross Simulation実行
                avh_result = self.cross_simulator.simulate(il_slots)
                
                if avh_result.verified:
                    trace.append(f"[AVH_ENHANCED] AVH verified: {avh_result.conclusion}")
                    trace.extend(avh_result.trace)
                    
                    return AvhEnhancedResult(
                        answer=avh_result.conclusion,
                        status="SOLVED",
                        confidence=avh_result.confidence,
                        reasoning_mode="avh_cross",
                        verified=True,
                        axioms_used=avh_result.axioms_used,
                        trace=trace
                    )
                else:
                    trace.append("[AVH_ENHANCED] AVH simulation did not verify")
                    
            except Exception as e:
                trace.append(f"[AVH_ENHANCED] AVH error: {e}")
        
        # Step 3: Verantyxの結果を返す（フォールバック）
        trace.append("[AVH_ENHANCED] Using Verantyx result")
        
        return AvhEnhancedResult(
            answer=v6_result.get("answer"),
            status=status,
            confidence=confidence,
            reasoning_mode="verantyx_only",
            verified=(status == "VERIFIED"),
            axioms_used=[],
            trace=trace
        )
    
    def _is_logic_problem(self, domain: str, problem: str) -> bool:
        """
        論理問題かどうかを判定
        
        Args:
            domain: ドメイン
            problem: 問題文
        
        Returns:
            True if logic problem
        """
        # ドメインが論理系
        if "logic" in domain.lower():
            return True
        
        # 問題文に論理記号が含まれる
        logic_symbols = ["->", "→", "&", "|", "~", "¬", "□", "◇"]
        if any(sym in problem for sym in logic_symbols):
            return True
        
        # 論理キーワード
        logic_keywords = ["tautology", "satisfiable", "valid", "axiom", "modal"]
        if any(kw in problem.lower() for kw in logic_keywords):
            return True
        
        return False
    
    def batch_solve(
        self,
        problems: List[str],
        verbose: bool = False
    ) -> List[AvhEnhancedResult]:
        """
        バッチ推論
        
        Args:
            problems: 問題リスト
            verbose: 詳細ログ表示
        
        Returns:
            結果リスト
        """
        results = []
        
        for i, problem in enumerate(problems, 1):
            if verbose:
                print(f"\n[AVH_ENHANCED] Solving {i}/{len(problems)}")
                print(f"Problem: {problem[:80]}...")
            
            result = self.solve(problem)
            results.append(result)
            
            if verbose:
                print(f"Answer: {result.answer}")
                print(f"Mode: {result.reasoning_mode}")
                print(f"Verified: {result.verified}")
        
        return results


# Convenience function
def solve_with_avh(problem: str) -> Dict[str, Any]:
    """
    AVH統合パイプラインで問題を解く
    
    Args:
        problem: 問題文
    
    Returns:
        結果辞書
    """
    pipeline = VerantyxAvhEnhanced(use_avh=True)
    result = pipeline.solve(problem)
    return result.to_dict()
