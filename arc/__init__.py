"""
Verantyx ARC-AGI-2 Module

ARC（Abstraction and Reasoning Corpus）をVerantyxの設計思想で解く:
  1. GridDecomposer: グリッド → GridIR（構造分解）
  2. TransformPieceDB: 原子的変換の辞書
  3. GridCrossSimulator: 変換チェーンを小世界で適用
  4. ARC-CEGIS: 訓練ペアをオラクルとして候補を刈る

LLMは使わない。純粋ルールベース推論。
"""
