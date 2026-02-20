# Phase 5B 継続ガイド（次のセッション用）

**作成日**: 2026-02-15 16:57 JST  
**現状**: テスト30-40% (目標70%未達)  
**推定作業時間**: 1-2時間

---

## 🎯 現状サマリー

### 完了 ✅
- Decomposer改善（組み合わせ優先、階乗対応、エンティティ抽出）
- Executor param mapping改善（entities→slots）
- ピーススコアリング改善（specificity bonus）
- BeamSearch→Greedy切り替え

### 問題 ⚠️
1. **汎用ピースが高スコアすぎる**
   - `nt_is_prime_integer` (1.750)
   - `nt_general_integer` (1.750)
   - → 専門ピース（gcd, factorial）が選ばれない

2. **スコアリングバランス不安定**
   - taskを高くする → 汎用ピースが有利
   - answer_schemaを高くする → 不適切ピースが選ばれる

3. **テスト結果不安定**
   - 入力値そのままを返す（Test 2: 48、Test 3: 5）
   - Booleanを返す（期待はInteger）

---

## 📋 次のセッションのTODO（優先順）

### Step 1: 汎用ピース削除/降格（15分）

```bash
cd ~/.openclaw/workspace/verantyx_v6

# 1. nt_is_prime_integerを削除
grep -v 'nt_is_prime_integer' pieces/piece_db.jsonl > pieces/piece_db.jsonl.tmp
mv pieces/piece_db.jsonl.tmp pieces/piece_db.jsonl

# 2. nt_general_integerのconfidenceを0.3に変更
# 手動で編集: "confidence": 0.7 → "confidence": 0.3
```

または、完全に削除：

```bash
grep -v 'nt_is_prime_integer\|nt_general_integer' pieces/piece_db.jsonl > pieces/piece_db.jsonl.tmp
mv pieces/piece_db.jsonl.tmp pieces/piece_db.jsonl
```

### Step 2: スコアリング最終調整（10分）

`pieces/piece.py`の`matches_ir`関数を修正：

```python
# 現状（不安定）:
# task: 3倍, answer_schema: 1.5倍

# 推奨:
# task: 2倍, answer_schema: 1.5倍, domain: 1倍
if req_type == "task":
    if ir_dict.get("task") == req_value:
        matched += 2  # taskは2倍
        total += 1
elif req_type == "answer_schema":
    if ir_dict.get("answer_schema") == req_value:
        matched += 1.5  # answer_schemaは1.5倍
        total += 0.5
```

### Step 3: テスト実行・検証（10分）

```bash
cd ~/.openclaw/workspace/verantyx_v6
python3 tests/test_phase_5b.py
```

**目標**: 7/10以上（70%）

### Step 4: HLE検証（20分）

70%達成後：

```bash
# HLE 126問のサンプルを抽出
python3 -c "
import json
with open('hle_full_analysis.json') as f:
    data = json.load(f)

easy = [q for q in data['problems'] if q['difficulty'] == 'EASY'][:50]
with open('hle_126_sample.json', 'w') as out:
    json.dump(easy, out, indent=2)
"

# 検証実行
python3 run_hle_sample.py hle_126_sample.json
```

### Step 5: Phase 5B完了・Phase 5C開始（10分）

```bash
# PROGRESS.json更新
# Phase 5B: status="completed"
# Phase 5C: status="in_progress"
```

---

## 🔧 デバッグコマンド

### ピース選択確認

```bash
python3 -c "
from pieces.piece import PieceDB
from decomposer.decomposer import RuleBasedDecomposer

d = RuleBasedDecomposer()
db = PieceDB('pieces/piece_db.jsonl')

ir = d.decompose('What is the GCD of 48 and 18?')
results = db.search(ir.to_dict(), top_k=5)
for p, score in results:
    print(f'{score:.3f} - {p.piece_id}')
"
```

### Executor単体テスト

```bash
python3 -c "
from executors.number_theory import gcd, factorial
print('GCD(48, 18):', gcd(a=48, b=18))
print('Factorial(5):', factorial(n=5))
"
```

---

## 📊 期待される結果

### テスト成功基準

| Test | 問題 | 期待値 | 現状 |
|------|------|--------|------|
| 1 | Prime(17) | True | ✅ True |
| 2 | GCD(48,18) | 6 | ❌ 48 |
| 3 | 5! | 120 | ❌ 5 |
| 4 | divisors(12) | 6 | ❌ False |
| 5 | LCM(12,15) | 60 | ❌ 6 |
| 6 | P(5,2) | 20 | ✅ 20 |
| 7 | C(6,2) | 15 | ❌ 30 |
| 8 | C(10,3) | 120 | ❌ 720 |
| 9 | arrange 4 from 6 | 360 | ❌ 15 |
| 10 | choose 3 from 5 | 10 | ✅ 10 |

**Step 1完了後の期待**: 5-6/10（50-60%）  
**Step 2完了後の期待**: 7-8/10（70-80%） ✅目標達成

---

## 📁 関連ファイル

- `PROGRESS.json`: 進捗管理
- `pieces/piece_db.jsonl`: ピースDB（編集対象）
- `pieces/piece.py`: スコアリングロジック（編集対象）
- `tests/test_phase_5b.py`: テストスクリプト
- `PHASE_5B_DEBUG_GUIDE.md`: 詳細デバッグ手順
- `memory/2026-02-15.md`: セッションログ

---

**Status**: Phase 5B継続中  
**Next milestone**: 70%達成 → Phase 5C開始  
**Estimated time**: 1-2時間

---

*作成: 2026-02-15 16:57 JST*  
*次回セッション: Step 1から開始*
