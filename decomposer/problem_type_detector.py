"""
Problem Type Detector - 問題タイプ自動検出

HLE問題の多様な形式を検出し、適切なExecutorを選択
"""
import re
from typing import Optional, List, Dict, Any
from enum import Enum


class ProblemType(Enum):
    """問題タイプ"""
    MULTIPLE_CHOICE = "multiple_choice"
    EQUATION = "equation"
    CALCULATION = "calculation"
    PROOF = "proof"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    OPTIMIZATION = "optimization"
    COUNTING = "counting"
    PROBABILITY = "probability"
    GEOMETRY = "geometry"
    CHESS = "chess"
    CIPHER = "cipher"
    STRING_MANIPULATION = "string_manipulation"
    UNKNOWN = "unknown"


class ProblemTypeDetector:
    """問題タイプ検出器"""
    
    def __init__(self):
        # タイプ別パターン
        self.patterns = {
            ProblemType.MULTIPLE_CHOICE: [
                r'answer choices?:',
                r'(?:\n|\s{2,})[A-E]\.\s+\w+',  # A. text (改行または複数空白の後)
                r'which of the following',
                r'select the correct',
                r'choose (?:one|the)',
            ],
            ProblemType.EQUATION: [
                r'solve\s+(?:for\s+)?[a-z]',
                r'find\s+[a-z]\s+(?:if|when|such that)',
                r'[a-z]\s*=\s*\?',
                r'what is [a-z]',
            ],
            ProblemType.CALCULATION: [
                r'calculate',
                r'compute',
                r'evaluate',
                r'what is\s+[\d\+\-\*/\(\)]+',
            ],
            ProblemType.PROOF: [
                r'prove that',
                r'show that',
                r'demonstrate',
                r'verify that',
            ],
            ProblemType.DEFINITION: [
                r'define',
                r'what is (?:the definition|a)\s+\w+',
                r'explain what',
            ],
            ProblemType.COMPARISON: [
                r'compare',
                r'which is (?:greater|larger|smaller)',
                r'(?:greater|less) than',
            ],
            ProblemType.OPTIMIZATION: [
                r'maximize',
                r'minimize',
                r'optimal',
                r'maximum',
                r'minimum',
                r'largest',
                r'smallest',
            ],
            ProblemType.COUNTING: [
                r'how many',
                r'count',
                r'number of',
            ],
            ProblemType.PROBABILITY: [
                r'probability',
                r'chance',
                r'what is the likelihood',
                r'expected value',
            ],
            ProblemType.GEOMETRY: [
                r'area',
                r'perimeter',
                r'volume',
                r'angle',
                r'triangle',
                r'circle',
                r'square',
                r'rectangle',
            ],
            ProblemType.CHESS: [
                r'chess',
                r'checkmate',
                r'move',
                r'piece',
            ],
            ProblemType.CIPHER: [
                r'cipher',
                r'encrypt',
                r'decrypt',
                r'code',
                r'decipher',
            ],
            ProblemType.STRING_MANIPULATION: [
                r'string',
                r'character',
                r'length',
                r'palindrome',
                r'substring',
            ],
        }
        
        # タイプ別キーワード
        self.keywords = {
            ProblemType.MULTIPLE_CHOICE: ["choice", "select", "option"],
            ProblemType.EQUATION: ["solve", "equation", "variable"],
            ProblemType.CALCULATION: ["calculate", "compute", "evaluate"],
            ProblemType.PROOF: ["prove", "show", "verify"],
            ProblemType.COUNTING: ["count", "how many", "number"],
            ProblemType.PROBABILITY: ["probability", "chance", "expected"],
            ProblemType.GEOMETRY: ["area", "perimeter", "angle"],
        }
    
    def detect(self, question: str) -> Dict[str, Any]:
        """
        問題タイプを検出
        
        Args:
            question: 問題文
        
        Returns:
            {
                "primary_type": ProblemType,
                "confidence": float,
                "secondary_types": List[ProblemType],
                "detected_patterns": List[str]
            }
        """
        question_lower = question.lower()
        
        # 各タイプのスコアリング
        scores = {ptype: 0.0 for ptype in ProblemType}
        detected_patterns = []
        
        # 多肢選択の特別処理（複数選択肢の検出）
        choice_matches = re.findall(r'(?:\n|^)\s*[A-E]\.\s+', question, re.MULTILINE)
        if len(choice_matches) >= 2:
            scores[ProblemType.MULTIPLE_CHOICE] += 3.0  # 強力なシグナル
            detected_patterns.append(f"multiple_choice:choices_count={len(choice_matches)}")
        
        # パターンマッチング
        for ptype, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    scores[ptype] += 1.0
                    detected_patterns.append(f"{ptype.value}:{pattern}")
        
        # キーワードマッチング
        for ptype, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    scores[ptype] += 0.5
        
        # スコアソート
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary type決定
        if sorted_types[0][1] > 0:
            primary_type = sorted_types[0][0]
            confidence = min(1.0, sorted_types[0][1] / 3.0)  # 3パターンで100%
        else:
            primary_type = ProblemType.UNKNOWN
            confidence = 0.0
        
        # Secondary types（スコア > 0）
        secondary_types = [ptype for ptype, score in sorted_types[1:] if score > 0]
        
        return {
            "primary_type": primary_type,
            "confidence": confidence,
            "secondary_types": secondary_types[:3],  # 上位3つ
            "detected_patterns": detected_patterns
        }
    
    def get_recommended_executors(self, problem_type: ProblemType) -> List[str]:
        """
        問題タイプに応じた推奨Executor
        
        Args:
            problem_type: 問題タイプ
        
        Returns:
            推奨executorのリスト
        """
        recommendations = {
            ProblemType.MULTIPLE_CHOICE: [
                "executors.multiple_choice.solve_multiple_choice",
                "executors.enumerate.options"
            ],
            ProblemType.EQUATION: [
                "executors.algebra.solve_linear_equation",
                "executors.algebra.solve_quadratic_equation"
            ],
            ProblemType.CALCULATION: [
                "executors.arithmetic.evaluate",
                "executors.arithmetic.evaluate_expression"
            ],
            ProblemType.COUNTING: [
                "executors.combinatorics.permutation",
                "executors.combinatorics.combination"
            ],
            ProblemType.PROBABILITY: [
                "executors.probability.simple_probability",
                "executors.probability.conditional_probability"
            ],
            ProblemType.GEOMETRY: [
                "executors.geometry.circle_area",
                "executors.geometry.triangle_area"
            ],
            ProblemType.CIPHER: [
                "executors.string_operations.caesar_cipher",
                "executors.string_operations.decode_cipher"
            ],
            ProblemType.STRING_MANIPULATION: [
                "executors.string_operations.string_length",
                "executors.string_operations.reverse_string"
            ]
        }
        
        return recommendations.get(problem_type, [])
    
    def get_domain_boost(self, problem_type: ProblemType) -> Dict[str, float]:
        """
        問題タイプに応じたドメインブースト
        
        Args:
            problem_type: 問題タイプ
        
        Returns:
            {domain: boost_score} の辞書
        """
        boosts = {
            ProblemType.MULTIPLE_CHOICE: {"multiple_choice": 2.0},
            ProblemType.EQUATION: {"algebra": 1.5},
            ProblemType.CALCULATION: {"arithmetic": 1.5},
            ProblemType.COUNTING: {"combinatorics": 1.5, "number_theory": 1.2},
            ProblemType.PROBABILITY: {"probability": 1.5},
            ProblemType.GEOMETRY: {"geometry": 1.5},
            ProblemType.CIPHER: {"cryptography": 2.0, "string": 1.2},
            ProblemType.STRING_MANIPULATION: {"string": 1.5}
        }
        
        return boosts.get(problem_type, {})


def main():
    """テスト実行"""
    detector = ProblemTypeDetector()
    
    test_questions = [
        "What is the capital of France?\nA. London\nB. Paris\nC. Berlin",
        "Solve for x: 2x + 3 = 7",
        "Calculate the area of a circle with radius 5",
        "How many ways can you arrange 5 books?",
        "What is the probability of rolling a 6?",
        "Decrypt the message: KHOOR ZRUOG using Caesar cipher with shift 3",
        "What is the length of the string 'hello world'?",
    ]
    
    print("=" * 80)
    print("Problem Type Detector Test")
    print("=" * 80)
    print()
    
    for i, question in enumerate(test_questions, 1):
        result = detector.detect(question)
        print(f"Test {i}: {question[:60]}...")
        print(f"  Primary Type: {result['primary_type'].value}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if result['secondary_types']:
            print(f"  Secondary: {[t.value for t in result['secondary_types']]}")
        
        # 推奨executor
        executors = detector.get_recommended_executors(result['primary_type'])
        if executors:
            print(f"  Recommended: {executors[0]}")
        print()
    
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
