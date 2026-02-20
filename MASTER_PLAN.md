# Verantyx V6 - HLE 70%達成 マスタープラン

**目標**: HLE 1021問中 714問正解（70%）  
**現状**: 36問対応可能（3.5%）  
**必要**: +678問の実装

---

## 📊 現状分析（HLE 1021問）

| 難易度 | 問題数 | 割合 | 現状 | 目標 |
|--------|--------|------|------|------|
| EASY | 162 | 15.9% | 36問 | 162問 ✅ |
| MEDIUM | 372 | 36.4% | 0問 | 372問 ✅ |
| HARD | 37 | 3.6% | 0問 | 37問 ✅ |
| VERY_HARD | 117 | 11.5% | 0問 | 117問 ✅ |
| UNKNOWN | 333 | 32.6% | 0問 | 26問（目標達成） |
| **合計** | **1021** | **100%** | **36** | **714 (70%)** |

**累積カバレッジ**:
- EASY: 15.9%
- EASY + MEDIUM: **52.3%** ← 実装しやすい
- EASY + MEDIUM + HARD: 55.9%
- EASY + MEDIUM + HARD + VERY_HARD: **67.4%**
- + UNKNOWN分析: **70%+** 達成可能

---

## 🎯 実装フェーズ（全8フェーズ）

### **Phase 5-A: 現状**（完了済み）
- **問題数**: 36問（3.5%）
- **内容**: 算術基本、論理基本
- **状態**: ✅ 完了

---

### **Phase 5-B: 数論・組み合わせ** 
- **問題数**: +126問 → 162問（15.9%）
- **推定時間**: 4-8時間
- **優先度**: 最高

#### 実装項目
1. **Executor実装**（完了済み）
   - ✅ `executors/number_theory.py`
   - ✅ `executors/combinatorics.py`

2. **ピース追加**（20個）
   - 素数判定、約数カウント、GCD、LCM、階乗
   - 順列、組み合わせ、二項係数
   - 余り計算、modulo演算

3. **Decomposer強化**
   - 数論キーワード検出
   - 組み合わせパターン認識

4. **テスト・検証**
   - HLE number_theory_basic: 69問
   - HLE combinatorics: 57問

---

### **Phase 5-C: 確率・幾何基本**
- **問題数**: +219問 → 381問（37.3%）
- **推定時間**: 4-8時間
- **優先度**: 高

#### 実装項目
1. **Executor実装**（完了済み）
   - ✅ `executors/probability.py`
   - ✅ `executors/geometry.py`

2. **ピース追加**（30個）
   - 確率計算、期待値、分散
   - 三角形、円、長方形、球の計算
   - 角度計算、ピタゴラス定理

3. **Decomposer強化**
   - 確率キーワード検出
   - 幾何図形認識

4. **テスト・検証**
   - HLE probability_basic: 81問
   - HLE geometry_basic: 138問

---

### **Phase 5-D: 代数基本・グラフ理論**
- **問題数**: +153問 → 534問（52.3%）
- **推定時間**: 8-16時間
- **優先度**: 中

#### 実装項目
1. **Executor実装**
   - `executors/algebra.py`: 多項式展開、因数分解、方程式求解（1次・2次）
   - `executors/graph.py`: 経路探索、最短路、連結性判定

2. **外部ライブラリ統合**
   - SymPy: 記号計算（代数）
   - NetworkX: グラフアルゴリズム

3. **ピース追加**（40個）
   - 多項式演算、方程式求解
   - グラフ探索、最短路、木判定

4. **テスト・検証**
   - HLE algebra_basic: 101問
   - HLE graph_theory: 52問

---

### **Phase 5-E: 線形代数・微積分**
- **問題数**: +37問 → 571問（55.9%）
- **推定時間**: 8-16時間
- **優先度**: 中

#### 実装項目
1. **Executor実装**
   - `executors/linear_algebra.py`: 行列演算、行列式、固有値
   - `executors/calculus.py`: 微分、積分、極限

2. **外部ライブラリ統合**
   - NumPy: 数値計算
   - SciPy: 科学計算
   - SymPy: 記号積分・微分

3. **ピース追加**（30個）
   - 行列演算、固有値計算
   - 微分・積分計算

4. **テスト・検証**
   - HLE linear_algebra: 27問
   - HLE calculus: 10問

---

### **Phase 5-F: 高度な数論・確率**
- **問題数**: +117問 → 688問（67.4%）
- **推定時間**: 16-32時間
- **優先度**: 低

#### 実装項目
1. **Executor実装**
   - `executors/number_theory_advanced.py`: 楕円曲線、代数的整数
   - `executors/probability_advanced.py`: 測度論、確率過程

2. **専門知識ベース**
   - 楕円曲線の性質
   - Kripkeモデルの拡張

3. **ピース追加**（50個）

4. **テスト・検証**
   - HLE VERY_HARD: 117問

---

### **Phase 5-G: UNKNOWN分析・対応**
- **問題数**: +26問 → 714問（70.0%）
- **推定時間**: 8-16時間
- **優先度**: 最終

#### 実装項目
1. **UNKNOWN 333問の詳細分析**
   - 手動分類
   - 対応可能な問題の抽出

2. **追加Executor実装**
   - 分析結果に基づく

3. **テスト・検証**
   - 70%達成確認

---

### **Phase 5-H: 最適化・統合**
- **推定時間**: 4-8時間
- **優先度**: 最終

#### 実装項目
1. **性能最適化**
   - Executor高速化
   - キャッシュ改善

2. **統合テスト**
   - HLE 1021問全体での検証
   - 70%達成確認

3. **ドキュメント整備**

---

## 📝 実装指示書の構造

各フェーズには以下の指示書を作成：

```
PHASE_X_INSTRUCTIONS.md
├── 1. 目標
├── 2. 前提条件（前フェーズの完了状態）
├── 3. 実装手順（ステップバイステップ）
│   ├── 3.1 Executor実装
│   ├── 3.2 ピース追加
│   ├── 3.3 Decomposer強化
│   ├── 3.4 テスト作成
│   └── 3.5 検証実行
├── 4. 完了条件（チェックリスト）
├── 5. 次フェーズへの引き継ぎ
└── 6. トラブルシューティング
```

---

## 🔄 進捗管理システム

### ファイル構成
```
verantyx_v6/
├── MASTER_PLAN.md          # このファイル（マスタープラン）
├── PROGRESS.json           # 進捗状況（JSON）
├── phases/                 # フェーズ別指示書
│   ├── PHASE_5B_INSTRUCTIONS.md
│   ├── PHASE_5C_INSTRUCTIONS.md
│   ├── PHASE_5D_INSTRUCTIONS.md
│   ├── PHASE_5E_INSTRUCTIONS.md
│   ├── PHASE_5F_INSTRUCTIONS.md
│   ├── PHASE_5G_INSTRUCTIONS.md
│   └── PHASE_5H_INSTRUCTIONS.md
├── executors/              # Executor実装
├── pieces/                 # ピースDB
└── tests/                  # フェーズ別テスト
    ├── test_phase_5b.py
    ├── test_phase_5c.py
    └── ...
```

### PROGRESS.json構造
```json
{
  "current_phase": "5B",
  "phases": {
    "5A": {"status": "completed", "problems_covered": 36, "date": "2026-02-15"},
    "5B": {"status": "in_progress", "problems_covered": 0, "date": null},
    "5C": {"status": "not_started", "problems_covered": 0, "date": null}
  },
  "total_problems": 1021,
  "target": 714,
  "current_coverage": 36,
  "percentage": 3.5
}
```

---

## 🚀 セッション継続方法

### 新しいセッションでの開始手順

1. **進捗確認**
   ```bash
   cd ~/.openclaw/workspace/verantyx_v6
   cat PROGRESS.json
   ```

2. **現在フェーズの指示書を読む**
   ```bash
   cat phases/PHASE_<current>_INSTRUCTIONS.md
   ```

3. **実装開始**
   - 指示書に従って実装
   - 各ステップ完了後に PROGRESS.json を更新

4. **完了チェック**
   - フェーズ完了条件を確認
   - テスト実行
   - 次フェーズへ移行

---

## 📊 累積実装時間の見積もり

| フェーズ | 推定時間 | 累積時間 | カバレッジ |
|---------|---------|---------|-----------|
| 5A | 完了 | - | 3.5% |
| 5B | 4-8h | 4-8h | 15.9% |
| 5C | 4-8h | 8-16h | 37.3% |
| 5D | 8-16h | 16-32h | 52.3% |
| 5E | 8-16h | 24-48h | 55.9% |
| 5F | 16-32h | 40-80h | 67.4% |
| 5G | 8-16h | 48-96h | 70.0% |
| 5H | 4-8h | 52-104h | 70.0%+ |

**総推定時間**: 52-104時間（平均78時間）

---

## 🎯 成功基準

1. **HLE 1021問中 714問以上正解** (70%)
2. **各ドメインで適切なカバレッジ**
   - EASY: 100%
   - MEDIUM: 100%
   - HARD: 100%
   - VERY_HARD: 100%
   - UNKNOWN: 8%以上

3. **実装品質**
   - Executor: 単体テスト合格
   - ピース: 統合テスト合格
   - E2E: HLE検証合格

---

## 📝 次のアクション

**今セッション**:
1. ✅ MASTER_PLAN.md作成（このファイル）
2. ⏳ PROGRESS.json作成
3. ⏳ PHASE_5B_INSTRUCTIONS.md作成
4. ⏳ Phase 5Bの実装開始

**次セッション以降**:
- PROGRESS.jsonを確認して継続

---

*作成日: 2026-02-15 16:03 JST*  
*最終更新: 2026-02-15 16:03 JST*
