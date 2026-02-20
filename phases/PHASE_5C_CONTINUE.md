# Phase 5C 継続ガイド（次のセッション用）

**作成日**: 2026-02-15 17:25 JST  
**現状**: テスト30% (3/10) - 目標70%未達  
**推定作業時間**: 1-2時間

---

## 🎯 現状サマリー

### 完了 ✅
- Executor拡充（probability 5関数、geometry 2関数）
- ピース10個追加（48個に）
- Decomposer強化（確率・幾何キーワード）
- Schema統一（float → decimal）
- テストケース作成・実行

### 問題 ⚠️
1. **確率Executor: 全て失敗**
   - coin_flip, dice, card → None返却
   - expected_value のみ成功

2. **幾何Executor: 一部成功**
   - circle_area, triangle_area → 成功 ✅
   - circumference, pythagorean, perimeter → 失敗

3. **テスト結果**
   - 確率: 0/5 (0%)
   - 幾何: 3/5 (60%)
   - 合計: 3/10 (30%)

---

## 📋 次のセッションのTODO（優先順）

### Step 1: 確率Executor改善（30分）

**問題**: coin_flip, dice, cardが全てNone返却

**原因分析**:
```python
# Test 1: Coin flip
# Expected: 0.5
# Result: None

# 予想される問題:
# 1. Executorがパラメータを受け取れていない
# 2. スロット要件不一致
# 3. Executor内でエラー発生
```

**解決策**:

1. **Executor単体テスト**:
```bash
cd ~/.openclaw/workspace/verantyx_v6
python3 -c "
from executors.probability import coin_flip_probability, dice_probability
print('Coin flip:', coin_flip_probability())
print('Dice:', dice_probability(sides=6, target=6))
"
```

2. **パラメータマッピング確認**:
- coin_flip_probability: flips, heads（デフォルト値あり）
- dice_probability: sides, target（デフォルト値あり）
- card_probability: total_cards, target_cards（デフォルト値あり）

→ **全てデフォルト値ありなので、パラメータなしでも動作すべき**

3. **ピーススロット修正**:
```python
# coin_flip, dice, cardのslotsを全て空に設定
# デフォルト動作を優先
```

### Step 2: 幾何Executor改善（20分）

**問題**:
- Test 8: 円周 → 153.9（期待43.98）
- Test 9: Pythagorean → 3（期待5）
- Test 10: Perimeter → 30.0（期待26）

**解決策**:

1. **Executor単体テスト**:
```python
from executors.geometry import circle_circumference, pythagorean, rectangle_perimeter

print('Circumference(7):', circle_circumference(radius=7))
print('Pythagorean(3,4):', pythagorean(a=3, b=4))
print('Perimeter(8,5):', rectangle_perimeter(length=8, width=5))
```

2. **期待値確認**:
- 円周(r=7) = 2πr = 2 * 3.14159 * 7 = **43.98**
- Pythagorean(3,4) = √(9+16) = √25 = **5**
- Perimeter(8,5) = 2*(8+5) = 2*13 = **26**

### Step 3: テスト再実行・検証（10分）

```bash
cd ~/.openclaw/workspace/verantyx_v6
python3 tests/test_phase_5c.py
```

**目標**: 7/10以上（70%）

---

## 🔧 デバッグコマンド

### 確率Executor単体テスト

```bash
python3 -c "
from executors.probability import *

print('=== Probability Executors ===')
print('coin_flip():', coin_flip_probability())
print('dice(6, 6):', dice_probability(sides=6, target=6))
print('card(52, 13):', card_probability(total_cards=52, target_cards=13))
print('expected(6):', expected_value(sides=6))
print('multiple(0.5, 0.5):', multiple_events(p1=0.5, p2=0.5))
"
```

### 幾何Executor単体テスト

```bash
python3 -c "
from executors.geometry import *

print('=== Geometry Executors ===')
print('circle_area(5):', circle_area(radius=5))
print('circle_circumference(7):', circle_circumference(radius=7))
print('triangle_area(10, 6):', triangle_area(base=10, height=6))
print('pythagorean(3, 4):', pythagorean(a=3, b=4))
print('rectangle_perimeter(8, 5):', rectangle_perimeter(length=8, width=5))
"
```

### ピース選択確認

```bash
python3 -c "
from pieces.piece import PieceDB
from decomposer.decomposer import RuleBasedDecomposer

d = RuleBasedDecomposer()
db = PieceDB('pieces/piece_db.jsonl')

ir = d.decompose('What is the probability of flipping heads on a fair coin?')
results = db.search(ir.to_dict(), top_k=5)
for p, score in results[:3]:
    print(f'{score:.3f} - {p.piece_id} (slots: {p.in_spec.slots})')
"
```

---

## 📊 期待される結果

### テスト成功基準

| Test | 問題 | 期待値 | Phase 5C目標 |
|------|------|--------|-------------|
| 1 | Coin flip | 0.5 | ✅ |
| 2 | Dice roll | 0.167 | ✅ |
| 3 | 2 coins | 0.25 | ✅ |
| 4 | Card draw | 0.25 | ✅ |
| 5 | Dice expected | 3.5 | ✅ (既に成功) |
| 6 | Circle area | 78.54 | ✅ (既に成功) |
| 7 | Triangle area | 30 | ✅ (既に成功) |
| 8 | Circumference | 43.98 | ✅ |
| 9 | Pythagorean | 5 | ✅ |
| 10 | Perimeter | 26 | ✅ |

**Step 1完了後の期待**: 5-6/10（50-60%）  
**Step 2完了後の期待**: 7-8/10（70-80%） ✅目標達成

---

## 📁 関連ファイル

- `PROGRESS.json`: 進捗管理
- `tests/test_phase_5c.py`: テストスクリプト
- `executors/probability.py`: 確率Executor（編集対象）
- `executors/geometry.py`: 幾何Executor（編集対象）
- `pieces/piece_db.jsonl`: ピースDB（編集対象）
- `PHASE_5B_COMPLETE.md`: Phase 5B完了レポート（参考）

---

**Status**: Phase 5C継続中（30%）  
**Next milestone**: 70%達成 → Phase 5C完了  
**Estimated time**: 1-2時間

---

*作成: 2026-02-15 17:25 JST*  
*次回セッション: Executor単体テストから開始*
