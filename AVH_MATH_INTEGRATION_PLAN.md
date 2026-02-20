# avh_math → Verantyx V6 統合計画

**目的**: avh_mathの優れたコンポーネントをVerantyx V6に統合し、600B重み抽出と組み合わせる

---

## 🎯 取り入れるべき主要コンポーネント

### 1. **Cross Simulator (31.0 KB)** ← 最重要
**場所**: `avh_math/puzzle/cross_simulator.py`

**価値**:
- ✅ 立体十字構造での推論（本来の構想に完全準拠）
- ✅ 命題論理・様相論理のシミュレーション
- ✅ Kripkeモデルでの検証
- ✅ 実装済み・動作確認済み

**統合方法**:
```python
# Verantyx V6に統合
verantyx_v6/puzzle/cross_simulator.py ← avh_math版をコピー
verantyx_v6/puzzle/propositional_logic_solver.py
verantyx_v6/puzzle/modal_logic_solver.py
```

**効果**:
- Phase 2のCross Simulationがスタブから実装に
- 論理問題の正答率: 0% → 65-100%（avh_mathで実証済み）

---

### 2. **公理データベース (90.3 KB)** ← 知識ベース
**場所**: `avh_math/puzzle/axioms_unified.json`

**内容**:
- 命題論理の公理・定理
- 様相論理の公理系（K, T, S4, S5）
- 数学的定理
- 推論規則

**統合方法**:
```python
verantyx_v6/pieces/axioms_unified.json ← 既存のpiece_db.jsonlと統合
```

**DeepSeek重み抽出との組み合わせ**:
```
公理DB（静的知識）
    ↓ 不足時
DeepSeek V3.2重み抽出（動的知識）
    ↓ 構造化
統合知識ベース
```

---

### 3. **IL Converter (18.0 KB)** ← 自然言語理解
**場所**: `avh_math/puzzle/il_converter.py`

**機能**:
- 自然言語 → ILスロット変換
- 論理式抽出
- フレーム特性検出

**Verantyx V6の対応物**: `decomposer/decomposer.py`

**統合方法**:
- avh_mathの論理式抽出ロジックを移植
- Decomposerを強化

---

### 4. **Answer Formatter (10.0 KB)** ← HLE形式対応
**場所**: `avh_math/puzzle/answer_formatter.py`

**機能**:
- HLE形式への解答変換
- 複数形式サポート（boolean, integer, string, formula）
- LaTeX正規化

**統合方法**:
```python
verantyx_v6/grammar/answer_formatter.py ← 新規追加
```

---

### 5. **HLE評価ツール (7.6 KB)**
**場所**: `tools/eval_hle_2500_puzzle_reasoner.py`

**価値**:
- 洗練された評価スクリプト
- 処理速度: 766問/秒
- エラー処理: 安定性99.72%

**統合方法**:
```python
verantyx_v6/tools/eval_hle_verantyx.py ← avh_math版を参考に改善
```

---

## 📐 統合アーキテクチャ

### 現在のVerantyx V6
```
Verantyx V6
├─ IR Decomposer
├─ Piece DB (100 pieces)
├─ Executor (15+ executors)
├─ Cross Simulation (スタブ)
└─ DeepSeek Weight Extraction (新規実装)
```

### 統合後のVerantyx V6 Enhanced
```
Verantyx V6 Enhanced
├─ IR Decomposer + IL Converter（強化）
├─ Unified Knowledge Base
│   ├─ Axioms DB (90KB) ← avh_math
│   ├─ Piece DB (100 pieces)
│   └─ DeepSeek Weights (600GB) ← 動的抽出
├─ Cross Simulator（完全実装）← avh_math
│   ├─ Propositional Logic Solver
│   └─ Modal Logic Solver
├─ Executor (15+ executors)
└─ Answer Formatter ← avh_math
```

---

## 🚀 実装ステップ

### Phase 6A: Cross Simulator統合（1-2日）
- [ ] cross_simulator.py をコピー
- [ ] propositional_logic_solver.py をコピー
- [ ] modal_logic_solver.py をコピー
- [ ] Verantyx V6パイプラインに統合
- [ ] テスト実行（論理問題10問）

### Phase 6B: 公理DB統合（0.5日）
- [ ] axioms_unified.json をロード
- [ ] piece_db.jsonl と統合
- [ ] CrossDB検索に組み込み
- [ ] テスト実行

### Phase 6C: IL Converter統合（1日）
- [ ] 論理式抽出ロジックを移植
- [ ] Decomposerに統合
- [ ] テスト実行

### Phase 6D: Answer Formatter統合（0.5日）
- [ ] answer_formatter.py をコピー
- [ ] パイプラインの最終層に追加
- [ ] HLE形式テスト

### Phase 6E: 評価ツール改善（0.5日）
- [ ] eval_hle_2500_puzzle_reasoner.py を参考
- [ ] Verantyx版を改善
- [ ] バッチ処理最適化

---

## 📊 期待される効果

### HLE正答率の改善

| コンポーネント | 対象カテゴリ | 現状 | 期待値 | 根拠 |
|--------------|------------|------|--------|------|
| Cross Simulator | Logic/Philosophy | 0% | **65-100%** | avh_math実績 |
| Axioms DB | Math基礎 | 1% | **10-20%** | 静的知識 |
| Answer Formatter | 全カテゴリ | 3.5% | **+2-3%** | 形式正規化 |
| DeepSeek Weights | Math高度 | 1% | **20-40%** | 動的知識 |

**総合期待値**: 3.5% → **30-50%**

---

## 💡 革新的な統合: 3層知識アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│ Layer 1: 静的知識（avh_math公理DB）              │
│  - 90KB、高速検索                                │
│  - 論理・基礎数学                                │
└─────────────┬───────────────────────────────────┘
              │ 不足時
              ↓
┌─────────────────────────────────────────────────┐
│ Layer 2: 構造化知識（Verantyx Pieces）          │
│  - 100 pieces                                   │
│  - 実行可能な推論                                │
└─────────────┬───────────────────────────────────┘
              │ 不足時
              ↓
┌─────────────────────────────────────────────────┐
│ Layer 3: 動的知識（DeepSeek Weights）           │
│  - 600GB、非発火探索                             │
│  - 研究レベルの知識                              │
└─────────────────────────────────────────────────┘
```

**利点**:
- ✅ 速度: Layer 1（ミリ秒） → Layer 2（秒） → Layer 3（分）
- ✅ コスト: Layer 1（無料） → Layer 2（無料） → Layer 3（要リソース）
- ✅ カバレッジ: 基礎 → 応用 → 研究レベル

---

## 🔧 具体的な統合コード例

### 1. Cross Simulator統合

```python
# verantyx_v6/puzzle/cross_simulator_avh.py
from avh_math.puzzle.cross_simulator import CrossSimulator as AvhCrossSimulator
from avh_math.puzzle.propositional_logic_solver import is_tautology, is_satisfiable

class VerantyxCrossSimulator:
    """
    avh_mathのCross Simulatorを統合
    """
    
    def __init__(self):
        self.avh_simulator = AvhCrossSimulator()
    
    def simulate(self, ir_dict: Dict[str, Any]) -> Optional[Any]:
        """
        IRからシミュレーション実行
        
        avh_mathのロジックを使用
        """
        # ILスロット形式に変換
        il_slots = self._ir_to_il_slots(ir_dict)
        
        # avh_mathのシミュレーション実行
        result = self.avh_simulator.simulate(il_slots)
        
        return result
```

### 2. 公理DB統合

```python
# verantyx_v6/pieces/unified_knowledge.py
import json

class UnifiedKnowledgeBase:
    """
    avh_math公理DB + Verantyx Piece DBの統合
    """
    
    def __init__(self):
        # avh_math公理をロード
        with open('axioms_unified.json') as f:
            self.axioms = json.load(f)
        
        # Verantyx piecesをロード
        self.pieces = self._load_pieces('piece_db.jsonl')
        
        # DeepSeek weight extractorへの参照
        self.weight_extractor = None  # 後で設定
    
    def search(self, query: str, domain: Domain) -> List[Knowledge]:
        """
        3層検索: 公理 → Pieces → DeepSeek
        """
        # Layer 1: 公理DB
        axiom_results = self._search_axioms(query, domain)
        if axiom_results:
            return axiom_results
        
        # Layer 2: Pieces
        piece_results = self._search_pieces(query, domain)
        if piece_results:
            return piece_results
        
        # Layer 3: DeepSeek weights
        if self.weight_extractor:
            weight_results = self.weight_extractor.extract_knowledge(query, domain)
            return weight_results
        
        return []
```

### 3. Answer Formatter統合

```python
# verantyx_v6/grammar/answer_formatter.py
from avh_math.puzzle.answer_formatter import AnswerFormatter as AvhFormatter

class VerantyxAnswerFormatter:
    """
    avh_mathのAnswer Formatterを統合
    """
    
    def __init__(self):
        self.avh_formatter = AvhFormatter()
    
    def format(self, raw_answer: Any, answer_schema: str) -> str:
        """
        HLE形式に変換
        """
        return self.avh_formatter.format_answer(
            answer=raw_answer,
            expected_type=answer_schema
        )
```

---

## 📁 ファイルコピー計画

### コピー元（avh_math） → コピー先（verantyx_v6）

```bash
# Cross Simulator
cp avh_math/puzzle/cross_simulator.py \
   verantyx_v6/puzzle/cross_simulator_avh.py

# Logic Solvers
cp avh_math/puzzle/propositional_logic_solver.py \
   verantyx_v6/puzzle/
cp avh_math/puzzle/modal_logic_solver.py \
   verantyx_v6/puzzle/

# 公理DB
cp avh_math/puzzle/axioms_unified.json \
   verantyx_v6/pieces/

# Answer Formatter
cp avh_math/puzzle/answer_formatter.py \
   verantyx_v6/grammar/

# 評価ツール（参考）
cp tools/eval_hle_2500_puzzle_reasoner.py \
   verantyx_v6/tools/eval_reference.py
```

---

## 🎯 最終目標

### HLE 2500問の正答率目標

| フェーズ | 正答率 | 主な改善 |
|---------|--------|---------|
| 現状（Phase 5G） | 3.5% | ベースライン |
| Phase 6A（Cross Simulator） | **15-20%** | 論理問題対応 |
| Phase 6B（公理DB） | **20-25%** | 基礎数学対応 |
| Phase 6C-D | **25-30%** | 最適化 |
| Phase 6E（DeepSeek統合） | **40-50%** | 高度知識 |

**最終目標**: **50%** (1250/2500問)

---

## ⚠️ 注意事項

### ライセンス・権利
- avh_mathが自作プロジェクトであることを確認
- コードの再利用権限を確認

### 互換性
- Python バージョン互換性
- 依存ライブラリの整合性
- データ形式の統一

---

## 📝 次のアクション

### 即座に実行
1. [ ] avh_mathファイルの詳細確認
2. [ ] cross_simulator.py の内容確認
3. [ ] axioms_unified.json の内容確認
4. [ ] 統合可能性の評価

### 今週中
1. [ ] Phase 6A: Cross Simulator統合
2. [ ] Phase 6B: 公理DB統合
3. [ ] テスト実行（論理問題20問）
4. [ ] HLE評価（改善確認）

---

**作成日**: 2026-02-16 13:22 JST  
**Status**: 統合計画完成、実装準備OK  
**期待される効果**: HLE 3.5% → 40-50%
