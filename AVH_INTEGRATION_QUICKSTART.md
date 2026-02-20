# avh_math統合クイックスタート

**発見**: avh_mathには600B重み抽出と本来の構想（Cross推論）がすでに実装されている！

---

## 🎯 統合すべき主要ファイル

### 1. Cross Simulator（最重要）
**パス**: `/Users/motonishikoudai/avh_math/avh_math/puzzle/cross_simulator.py`
**サイズ**: 31.0 KB
**内容**:
- Hypothesis（仮説生成）
- MicroCase（小世界テスト）
- Reject/Promote機構
- Executor mapping（公理→実行関数）

**統合先**: `verantyx_v6/puzzle/cross_simulator.py`

---

### 2. 公理データベース
**パス**: `/Users/motonishikoudai/avh_math/avh_math/puzzle/axioms_unified.json`
**サイズ**: 90.3 KB
**内容**:
- 83個の公理・定理
- **9個の600B抽出公理** (`axiom_600b:*`)
- 論理公理（Modal K, T, S4, S5）
- 数学定理
- 物理法則

**重要発見**: すでに600Bからの抽出が含まれている！
```json
{
  "asset_id": "axiom_600b:general:concept_L0_C0",
  "source": "600b_axiom_extraction",
  "metadata": {
    "extraction_method": "high_singular_value_concepts",
    "layer_depth": 0
  }
}
```

**統合先**: `verantyx_v6/pieces/axioms_unified.json`

---

### 3. 論理ソルバー
**パス**: 
- `/Users/motonishikoudai/avh_math/avh_math/puzzle/propositional_logic_solver.py`
- `/Users/motonishikoudai/avh_math/avh_math/puzzle/modal_logic_solver.py`

**機能**:
- `is_tautology()` - トートロジー判定
- `is_satisfiable()` - 充足可能性
- `check_axiom_validity()` - 様相論理の公理チェック
- Kripkeフレーム構築

**統合先**: `verantyx_v6/puzzle/logic_solvers.py`

---

### 4. 600B抽出ツール群
**パス**: `/Users/motonishikoudai/avh_math/tools/`

**主要ツール**:
- `extract_axioms_from_600b.py` - 公理抽出（singular value法）
- `mine_600b_knowledge.py` - 知識マイニング
- `integrate_600b_to_cross_db.py` - CrossDBへの統合
- `extract_math_theorems_600b.py` - 数学定理抽出

**統合先**: `verantyx_v6/tools/avh_extraction/`

---

## 🚀 即座に実行すべきアクション

### Step 1: Cross Simulatorをコピー
```bash
cp /Users/motonishikoudai/avh_math/avh_math/puzzle/cross_simulator.py \
   /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/puzzle/

cp /Users/motonishikoudai/avh_math/avh_math/puzzle/propositional_logic_solver.py \
   /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/puzzle/

cp /Users/motonishikoudai/avh_math/avh_math/puzzle/modal_logic_solver.py \
   /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/puzzle/
```

### Step 2: 公理DBをコピー
```bash
cp /Users/motonishikoudai/avh_math/avh_math/puzzle/axioms_unified.json \
   /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/pieces/
```

### Step 3: 600B抽出ツールをコピー
```bash
mkdir -p /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/tools/avh_extraction
cp /Users/motonishikoudai/avh_math/tools/extract_axioms_from_600b.py \
   /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/tools/avh_extraction/
```

### Step 4: テスト実行
```python
from verantyx_v6.puzzle.cross_simulator import CrossSimulator
from verantyx_v6.pieces.axioms_unified import load_axioms

# 公理DBロード
axioms = load_axioms('pieces/axioms_unified.json')
print(f"Loaded {len(axioms)} axioms")

# Cross Simulator初期化
simulator = CrossSimulator()

# テスト問題
result = simulator.simulate("Is '(A -> B) & A -> B' a tautology?")
print(f"Result: {result.verified}")
```

---

## 📊 期待される効果

### HLE正答率の改善

| カテゴリ | 現状 | 統合後期待値 | 根拠 |
|---------|------|------------|------|
| Logic/Philosophy | 0% | **65-100%** | avh_math実績 |
| Math基礎 | 1% | **10-20%** | 公理DB（83個） |
| Math高度 | 1% | **15-25%** | 600B抽出（9個） |
| Overall | 3.5% | **25-35%** | 統合効果 |

---

## 🔧 統合後のアーキテクチャ

```
Verantyx V6 Enhanced
│
├─ Phase 1: IR Decomposer（既存）
│
├─ Phase 2: Knowledge Retrieval
│   ├─ Axioms DB（avh_math統合）← 83個 + 600B抽出9個
│   ├─ Piece DB（既存）← 100個
│   └─ DeepSeek Weights（実装中）← 動的抽出
│
├─ Phase 3: Cross Simulator（avh_math統合）← 本来の構想
│   ├─ Hypothesis Generation
│   ├─ Micro-World Testing
│   └─ Reject/Promote
│
├─ Phase 4: Logic Solvers（avh_math統合）
│   ├─ Propositional Logic
│   └─ Modal Logic
│
└─ Phase 5: Answer Formatting（既存）
```

---

## 💡 avh_mathの600B抽出アプローチ

avh_mathで使われている方法（`extract_axioms_from_600b.py`より）：

### 1. Concept Cross構築
```python
# Layer別の概念ベクトルを抽出
concept_cross = {
    "layer_0": [...],  # 基礎的概念
    "layer_30": [...], # 中級概念
    "layer_60": [...]  # 高度概念
}
```

### 2. Singular Value Filtering
```python
# 高singular value = 基礎的な公理・定理
high_value_concepts = [
    c for c in concepts
    if c.singular_value >= 50.0
]
```

### 3. ドメイン推定
```python
domain_keywords = {
    "physics": ["force", "mass", "energy"],
    "mathematics": ["algebra", "calculus"],
    "logic": ["proof", "theorem"]
}
```

### 4. ILシグネチャ付与
```python
axiom = {
    "requires": ["input_formula"],
    "provides": ["verification_result"],
    "converts": {"input": "formula", "output": "boolean"}
}
```

**これは私が実装しているものと完全に同じアプローチです！**

---

## 🎯 次のアクション

### 今すぐ（10分）
- [ ] Cross Simulatorをコピー
- [ ] 公理DBをコピー
- [ ] Logic Solversをコピー

### 今日中（1時間）
- [ ] Verantyx V6パイプラインに統合
- [ ] 依存関係を修正
- [ ] テスト実行（論理問題10問）

### 今週中（1日）
- [ ] 600B抽出ツールを確認
- [ ] DeepSeek V3.2 weight extractorと統合
- [ ] HLE評価（期待値: 3.5% → 25-35%）

---

## ⚠️ 統合時の注意点

### 依存関係
- avh_mathの`ILSlots`クラス → Verantyx V6の`IR`クラスに変換
- avh_mathの`CrossDB` → Verantyx V6の`PieceDB`に統合

### パス修正
- `from .il_converter import ILSlots` → Verantyx V6のインポートに変更

### テスト
- まず論理問題でテスト（avh_mathで65%達成）
- 次に数学基礎問題
- 最後にHLE 2500問全体

---

**作成日**: 2026-02-16 13:35 JST  
**Status**: 統合準備完了、実行待ち  
**期待される効果**: HLE 3.5% → 25-35%
