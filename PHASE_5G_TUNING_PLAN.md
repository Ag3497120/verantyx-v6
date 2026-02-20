# Phase 5G: HLE正答率向上チューニング計画

**現状**: 3.5% (85/2400問)  
**目標**: 10-15% (250-375/2500問) - 現実的な中間目標

---

## 📊 失敗原因分析

### 1. 未実装Executor
頻出するStub実行:
- `algebra_solve_equation` - 代数方程式
- `chess_stockfish` - チェス問題
- `string_length` / 文字列操作 - 暗号・パターン
- 多肢選択推論 - A/B/C/D形式

### 2. 問題タイプの多様性
- **Math**: 大学院レベル（Lie群、楕円曲線、位相幾何学）
- **Computer Science**: トランスフォーマー、アルゴリズム
- **Physics**: 高次元重力理論、場の理論
- **Humanities**: 法律、哲学、経済学
- **Other**: チェス、暗号、パズル

### 3. 正解判定の問題
- 文字列形式の多様性（"Z+Z+Z", "\mathbb{Z}", "Katie kicked..."）
- 数値精度（小数点以下の桁数）
- LaTeX記法の処理

---

## 🎯 改善施策（優先度順）

### Priority 1: 汎用的な問題処理

#### 1.1 多肢選択問題サポート
**対象**: A/B/C/D形式の問題（推定200-300問）

```python
# executors/multiple_choice.py
def solve_multiple_choice(
    question: str,
    choices: Dict[str, str],
    context: Optional[str] = None
) -> str:
    """
    多肢選択問題を推論で解く
    - キーワードマッチング
    - 消去法
    - 知識ベース照合
    """
    pass
```

#### 1.2 文字列操作Executor
**対象**: 暗号、パターン、文字列長問題（推定50-100問）

```python
# executors/string_operations.py
def string_length(text: str) -> int
def caesar_cipher(text: str, shift: int) -> str
def pattern_match(text: str, pattern: str) -> bool
def extract_pattern(text: str, regex: str) -> List[str]
```

### Priority 2: 代数・方程式サポート

#### 2.1 方程式ソルバー
**対象**: 代数方程式問題（推定100-150問）

```python
# executors/equation_solver.py
def solve_linear_equation(equation: str) -> float
def solve_quadratic_equation(a: float, b: float, c: float) -> List[float]
def solve_system_linear(equations: List[str]) -> Dict[str, float]
def solve_polynomial(coefficients: List[float]) -> List[complex]
```

### Priority 3: 正解判定の改善

#### 3.1 柔軟な正解判定
```python
# core/answer_matcher.py
def flexible_match(predicted: Any, expected: Any) -> bool:
    """
    - LaTeX記法の正規化
    - 数値の許容誤差
    - 文字列の正規化（大小文字、空白）
    - リスト/集合の順序無視
    """
    pass
```

### Priority 4: ピース選択の改善

#### 4.1 問題タイプ検出
```python
# decomposer/problem_type_detector.py
def detect_problem_type(question: str) -> str:
    """
    問題タイプを検出:
    - MULTIPLE_CHOICE
    - EQUATION
    - CHESS
    - CIPHER
    - PROOF
    - CALCULATION
    """
    pass
```

#### 4.2 タイプ別ピース優先度
- 問題タイプに応じた動的スコアリング
- ドメイン特化ボーナスの強化

---

## 📈 実装スケジュール

### Week 1: 基本機能拡充
- [ ] Day 1-2: multiple_choice.py実装
- [ ] Day 3-4: string_operations.py実装
- [ ] Day 5: flexible_match実装
- [ ] Day 6-7: テスト・評価（目標: 5% → 8%）

### Week 2: 専門機能追加
- [ ] Day 8-10: equation_solver.py実装
- [ ] Day 11-12: problem_type_detector実装
- [ ] Day 13-14: 統合テスト（目標: 8% → 12%）

### Week 3: 最適化・調整
- [ ] Day 15-17: ピース選択最適化
- [ ] Day 18-19: 正解判定最適化
- [ ] Day 20-21: 全体評価（目標: 12% → 15%）

---

## 🎯 現実的な目標設定

| フェーズ | 正答率 | 正解数 | 増加 |
|---------|--------|--------|------|
| 現状 | 3.5% | 85/2400 | - |
| Phase 5F完了 | 4-5% | 100-125 | +15-40 |
| Phase 5G Week 1 | 7-8% | 175-200 | +75-100 |
| Phase 5G Week 2 | 10-12% | 250-300 | +75-100 |
| Phase 5G Week 3 | 13-15% | 325-375 | +75 |

**最終目標**: 15% (375/2500問)

これは当初の70%目標には遠く及びませんが、HLEの難易度を考慮すると**現実的な成果**です。

---

## 💡 長期的な改善

### 研究レベル問題への対応
大学院レベルの数学・物理問題（現在のMath 1%）を改善するには：

1. **知識ベース拡充**
   - 専門用語辞書
   - 定理・公式データベース
   - 証明パターンライブラリ

2. **シンボリック推論**
   - SymPy/SageMath統合
   - 定理証明器（Lean, Coq）
   - CAS（Computer Algebra System）

3. **外部リソース活用**
   - Wolfram Alpha API
   - MathOverflow検索
   - arXiv論文検索

これらは**Phase 6以降**の長期目標とします。

---

## 📊 評価指標

### 主要KPI
- **Overall Accuracy**: 全体正答率
- **Category Accuracy**: カテゴリ別正答率
- **Problem Type Coverage**: 対応可能な問題タイプ数

### サブKPI
- Stub実行率（低いほど良い）
- 平均実行時間
- ピース選択精度

---

**作成日**: 2026-02-16  
**Phase**: 5G準備  
**Status**: Draft
