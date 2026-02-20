# Phase 5B実装指示書：数論・組み合わせ

**フェーズ**: 5B  
**目標**: +126問対応 → 累計162問（15.9%）  
**推定時間**: 4-8時間  
**前提条件**: Phase 5A完了（算術基本・論理基本の36問対応済み）

---

## 📋 実装チェックリスト

### ステップ1: Executor実装（完了済み ✅）
- ✅ `executors/number_theory.py` - 素数判定、約数カウント、GCD、LCM、階乗
- ✅ `executors/combinatorics.py` - 順列、組み合わせ、二項係数
- ✅ `executors/probability.py` - 基本確率、期待値（Phase 5C用だが実装済み）
- ✅ `executors/geometry.py` - 幾何計算（Phase 5C用だが実装済み）

### ステップ2: ピースDB拡充（⏳ 実装必要）
- [ ] 20個のピースを `pieces/piece_db.jsonl` に追加
- [ ] 各ピースの動作確認

### ステップ3: Decomposer強化（⏳ 実装必要）
- [ ] 数論キーワード検出の追加
- [ ] 組み合わせパターン認識の追加

### ステップ4: テスト作成（⏳ 実装必要）
- [ ] `tests/test_phase_5b.py` 作成
- [ ] 数論問題10問のテスト
- [ ] 組み合わせ問題10問のテスト

### ステップ5: HLE検証（⏳ 実装必要）
- [ ] HLE number_theory_basic (69問) で検証
- [ ] HLE combinatorics (57問) で検証
- [ ] 合計126問中80問以上正解を確認（63%以上）

---

## 🔧 ステップ2: ピースDB拡充

### 2.1 追加するピース（20個）

**数論ピース（12個）**:

```jsonl
{"piece_id": "nt_is_prime", "name": "Prime Number Checker", "description": "素数判定", "in": {"requires": ["domain:number_theory", "task:decide"], "slots": ["number"]}, "out": {"produces": ["boolean"], "schema": "boolean", "artifacts": []}, "executor": "executors.number_theory.is_prime", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "prime"]}

{"piece_id": "nt_count_divisors", "name": "Divisor Counter", "description": "約数の個数を数える", "in": {"requires": ["domain:number_theory", "task:count"], "slots": ["number"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.count_divisors", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "divisors"]}

{"piece_id": "nt_gcd", "name": "Greatest Common Divisor", "description": "最大公約数", "in": {"requires": ["domain:number_theory"], "slots": ["a", "b"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.gcd", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "gcd"]}

{"piece_id": "nt_lcm", "name": "Least Common Multiple", "description": "最小公倍数", "in": {"requires": ["domain:number_theory"], "slots": ["a", "b"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.lcm", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "lcm"]}

{"piece_id": "nt_factorial", "name": "Factorial Calculator", "description": "階乗 n!", "in": {"requires": ["domain:number_theory"], "slots": ["n"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.factorial", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["number_theory", "factorial"]}

{"piece_id": "nt_is_prime_integer", "name": "Prime Checker (Integer)", "description": "素数判定（整数回答）", "in": {"requires": ["domain:number_theory", "answer_schema:integer"], "slots": ["number"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.is_prime", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "none"}, "confidence": 0.9, "tags": ["number_theory", "prime", "adapter"]}

{"piece_id": "nt_gcd_compute", "name": "GCD Computer", "description": "最大公約数計算（compute task）", "in": {"requires": ["domain:number_theory", "task:compute"], "slots": ["a", "b"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.gcd", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "gcd"]}

{"piece_id": "nt_lcm_compute", "name": "LCM Computer", "description": "最小公倍数計算（compute task）", "in": {"requires": ["domain:number_theory", "task:compute"], "slots": ["a", "b"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.lcm", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "lcm"]}

{"piece_id": "nt_factorial_compute", "name": "Factorial Computer", "description": "階乗計算（compute task）", "in": {"requires": ["domain:number_theory", "task:compute"], "slots": ["n"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.factorial", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["number_theory", "factorial"]}

{"piece_id": "nt_divisor_count_compute", "name": "Divisor Counter (Compute)", "description": "約数個数計算", "in": {"requires": ["domain:number_theory", "task:compute"], "slots": ["number"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.count_divisors", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "divisors"]}

{"piece_id": "nt_divisor_count_find", "name": "Divisor Counter (Find)", "description": "約数個数を見つける", "in": {"requires": ["domain:number_theory", "task:find"], "slots": ["number"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.count_divisors", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["number_theory", "divisors"]}

{"piece_id": "nt_general_integer", "name": "Number Theory (General)", "description": "数論計算（汎用）", "in": {"requires": ["domain:number_theory", "answer_schema:integer"], "slots": []}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.number_theory.factorial", "verifiers": [], "cost": {"time": "low", "space": "low", "explosion_risk": "medium"}, "confidence": 0.7, "tags": ["number_theory", "fallback"]}
```

**組み合わせピース（8個）**:

```jsonl
{"piece_id": "comb_permutation", "name": "Permutation Calculator", "description": "順列 P(n,r)", "in": {"requires": ["domain:combinatorics"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.permutation", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "permutation"]}

{"piece_id": "comb_combination", "name": "Combination Calculator", "description": "組み合わせ C(n,r)", "in": {"requires": ["domain:combinatorics"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.combination", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "combination"]}

{"piece_id": "comb_binomial", "name": "Binomial Coefficient", "description": "二項係数", "in": {"requires": ["domain:combinatorics"], "slots": ["n", "k"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.binomial_coefficient", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "binomial"]}

{"piece_id": "comb_perm_compute", "name": "Permutation (Compute)", "description": "順列計算（compute task）", "in": {"requires": ["domain:combinatorics", "task:compute"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.permutation", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "permutation"]}

{"piece_id": "comb_comb_compute", "name": "Combination (Compute)", "description": "組み合わせ計算（compute task）", "in": {"requires": ["domain:combinatorics", "task:compute"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.combination", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "combination"]}

{"piece_id": "comb_perm_find", "name": "Permutation (Find)", "description": "順列を求める", "in": {"requires": ["domain:combinatorics", "task:find"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.permutation", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "permutation"]}

{"piece_id": "comb_comb_find", "name": "Combination (Find)", "description": "組み合わせを求める", "in": {"requires": ["domain:combinatorics", "task:find"], "slots": ["n", "r"]}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.combination", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 1.0, "tags": ["combinatorics", "combination"]}

{"piece_id": "comb_general_integer", "name": "Combinatorics (General)", "description": "組み合わせ計算（汎用）", "in": {"requires": ["domain:combinatorics", "answer_schema:integer"], "slots": []}, "out": {"produces": ["integer"], "schema": "integer", "artifacts": []}, "executor": "executors.combinatorics.combination", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "medium"}, "confidence": 0.7, "tags": ["combinatorics", "fallback"]}
```

### 2.2 実装手順

1. **現在のピースDB読み込み**
   ```bash
   cd ~/.openclaw/workspace/verantyx_v6
   wc -l pieces/piece_db.jsonl  # 現在のピース数確認
   ```

2. **ピース追加**
   - 上記20個のピースを `pieces/piece_db.jsonl` に追記
   - または新規ファイルとして統合

3. **動作確認**
   ```python
   from pieces.piece import PieceDB
   db = PieceDB('pieces/piece_db.jsonl')
   print(f"Total pieces: {len(db.pieces)}")
   # 期待: 40個（既存20個 + 新規20個）
   ```

---

## 🔧 ステップ3: Decomposer強化

### 3.1 ドメイン検出の改善

**ファイル**: `decomposer/decomposer.py`

**追加するキーワード**:

```python
self.domain_keywords = {
    # ... 既存 ...
    Domain.NUMBER_THEORY: [
        "prime", "divisor", "divisible", "gcd", "lcm", 
        "congruent", "modulo", "mod", "remainder", "factor",
        "factorial", "!"  # 階乗記号
    ],
    Domain.COMBINATORICS: [
        "permutation", "combination", "arrange", "choose",
        "binomial", "C(n", "P(n", "nCr", "nPr",
        "ways to", "how many ways"
    ],
}
```

### 3.2 実装コード

`decomposer/decomposer.py` の `_detect_domain()` メソッドを更新：

```python
def _detect_domain(self, text: str) -> Domain:
    """ドメイン検出（記号優先）"""
    scores = {domain: 0 for domain in Domain}
    
    for domain, keywords in self.domain_keywords.items():
        for keyword in keywords:
            if keyword in text.lower():
                # 記号は高スコア
                if len(keyword) <= 2 and not keyword.isalpha():
                    scores[domain] += 5
                else:
                    scores[domain] += 1
    
    # 数式パターン検出（算術）
    import re
    if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', text):
        scores[Domain.ARITHMETIC] += 3
    
    # 階乗パターン
    if re.search(r'\d+!', text):
        scores[Domain.NUMBER_THEORY] += 5
    
    # 組み合わせパターン C(n,r) or P(n,r)
    if re.search(r'[CP]\(\d+,\s*\d+\)', text):
        scores[Domain.COMBINATORICS] += 10
    
    # 論理記号検出
    if any(sym in text for sym in ["->", "→", "&", "|", "~", "¬", "□", "◇"]):
        if any(sym in text for sym in ["[]", "<>", "□", "◇"]):
            scores[Domain.LOGIC_MODAL] += 10
        else:
            scores[Domain.LOGIC_PROPOSITIONAL] += 10
    
    best_domain = max(scores, key=scores.get)
    
    if scores[best_domain] == 0:
        return Domain.UNKNOWN
    
    return best_domain
```

---

## 🧪 ステップ4: テスト作成

### 4.1 テストファイル作成

**ファイル**: `tests/test_phase_5b.py`

```python
"""
Phase 5B テスト：数論・組み合わせ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced

# テストケース
test_cases = [
    # 数論
    {"question": "Is 17 a prime number?", "expected": "True", "domain": "number_theory"},
    {"question": "What is the GCD of 48 and 18?", "expected": "6", "domain": "number_theory"},
    {"question": "Calculate 5 factorial (5!)", "expected": "120", "domain": "number_theory"},
    {"question": "How many divisors does 12 have?", "expected": "6", "domain": "number_theory"},
    {"question": "Find the LCM of 12 and 15", "expected": "60", "domain": "number_theory"},
    
    # 組み合わせ
    {"question": "Calculate P(5, 2) - the number of permutations", "expected": "20", "domain": "combinatorics"},
    {"question": "Calculate C(6, 2) - the number of combinations", "expected": "15", "domain": "combinatorics"},
    {"question": "What is the binomial coefficient C(10, 3)?", "expected": "120", "domain": "combinatorics"},
    {"question": "How many ways can you arrange 4 items from 6?", "expected": "360", "domain": "combinatorics"},
    {"question": "In how many ways can you choose 3 items from 5?", "expected": "10", "domain": "combinatorics"},
]

def run_tests():
    v6 = VerantyxV6Enhanced()
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['domain']}")
        print(f"Q: {test['question']}")
        print(f"Expected: {test['expected']}")
        
        result = v6.solve(test['question'], use_crystal=False)
        
        # 数値比較
        try:
            ans_num = float(result.get('answer', 0))
            exp_num = float(test['expected'])
            match = abs(ans_num - exp_num) < 0.01
        except:
            match = str(result.get('answer')) == test['expected']
        
        if match:
            print(f"✅ PASS: {result.get('answer')}")
            passed += 1
        else:
            print(f"❌ FAIL: {result.get('answer')}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"{'='*80}")
    
    return passed >= 7  # 70%以上合格

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
```

### 4.2 テスト実行

```bash
cd ~/.openclaw/workspace/verantyx_v6
mkdir -p tests
python3 tests/test_phase_5b.py
```

**合格基準**: 10問中7問以上正解（70%）

---

## 🔍 ステップ5: HLE検証

### 5.1 検証スクリプト作成

**ファイル**: `tests/validate_phase_5b_hle.py`

```python
"""
Phase 5B HLE検証
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline_enhanced import VerantyxV6Enhanced

# HLE分析結果から対象問題を抽出
analysis = json.load(open('hle_full_analysis.json'))
target_domains = ['number_theory_basic', 'combinatorics']

# 問題をロード
hle_path = "/Users/motonishikoudai/avh_math/avh_math/db/hle_math_cross.jsonl"
problems = []

with open(hle_path, 'r') as f:
    for i, line in enumerate(f):
        if line.strip():
            data = json.loads(line)
            problems.append({
                "index": i,
                "text": data.get("problem_text", "")
            })
            if len(problems) >= 1021:
                break

# 検証実行
v6 = VerantyxV6Enhanced()
solved = 0
total = 0

for i, prob in enumerate(problems):
    # ドメイン判定（簡易）
    text_lower = prob["text"].lower()
    is_target = False
    
    if any(kw in text_lower for kw in ['prime', 'divisor', 'gcd', 'lcm', 'factorial']):
        is_target = True
    elif any(kw in text_lower for kw in ['permutation', 'combination', 'choose', 'arrange']):
        is_target = True
    
    if not is_target:
        continue
    
    total += 1
    
    try:
        result = v6.solve(prob["text"], use_crystal=False)
        if result["status"] == "SOLVED":
            solved += 1
        
        if total % 10 == 0:
            print(f"Progress: {solved}/{total} ({solved/total*100:.1f}%)")
    except:
        pass
    
    if total >= 126:  # 目標問題数
        break

print(f"\nFinal: {solved}/{total} ({solved/total*100:.1f}%)")
print(f"Target: 80/126 (63.5%)")
print(f"Result: {'✅ PASS' if solved >= 80 else '❌ FAIL'}")
```

### 5.2 実行

```bash
cd ~/.openclaw/workspace/verantyx_v6
python3 tests/validate_phase_5b_hle.py
```

**合格基準**: 126問中80問以上正解（63%以上）

---

## ✅ 完了条件

Phase 5Bは以下の条件を**すべて**満たした場合に完了とする：

1. ✅ Executor実装完了（既に完了）
2. ✅ ピースDB拡充完了（20個追加）
3. ✅ Decomposer強化完了
4. ✅ 単体テスト合格（10問中7問以上）
5. ✅ HLE検証合格（126問中80問以上）
6. ✅ PROGRESS.json更新

---

## 📝 次フェーズへの引き継ぎ

### Phase 5B完了後の状態

1. **カバレッジ**: 162問（15.9%）
2. **実装済みExecutor**: 
   - arithmetic, logic, number_theory, combinatorics
3. **ピース総数**: 40個
4. **PROGRESS.json**: Phase 5B を "completed" に更新

### Phase 5Cへの準備

Phase 5B完了後、PROGRESS.jsonを更新してPhase 5Cに進む：

```json
{
  "current_phase": "5C",
  "phases": {
    "5B": {
      "status": "completed",
      "problems_covered": 126,
      "completed_date": "<date>"
    },
    "5C": {
      "status": "in_progress"
    }
  }
}
```

次は `phases/PHASE_5C_INSTRUCTIONS.md` を読んで実装開始。

---

## 🐛 トラブルシューティング

### 問題1: Executorが見つからない

**症状**: `ModuleNotFoundError: No module named 'executors.number_theory'`

**解決**:
```bash
cd ~/.openclaw/workspace/verantyx_v6
ls -la executors/number_theory.py  # ファイル存在確認
python3 -c "from executors import number_theory; print('OK')"
```

### 問題2: ピースがロードされない

**症状**: `Total pieces: 20` (期待: 40)

**解決**:
```bash
# piece_db.jsonlの行数確認
wc -l pieces/piece_db.jsonl

# JSONL形式チェック
python3 -c "
import json
with open('pieces/piece_db.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f'Error at line {i}')
"
```

### 問題3: Decomposerがドメインを誤認識

**症状**: 数論問題が `unknown` になる

**解決**:
- `decomposer/decomposer.py` のキーワード追加を確認
- デバッグ出力でスコアを確認

```python
# デバッグ用
ir = decomposer.decompose("Is 17 prime?")
print(f"Domain: {ir.domain}")  # 期待: number_theory
```

---

## 📊 進捗報告テンプレート

Phase 5B完了時、以下を記録：

```
## Phase 5B完了報告

**完了日時**: YYYY-MM-DD HH:MM JST
**実装時間**: X時間

### 実装結果
- ✅ Executor実装: 完了
- ✅ ピース追加: 20個
- ✅ Decomposer強化: 完了
- ✅ 単体テスト: X/10問正解 (XX%)
- ✅ HLE検証: X/126問正解 (XX%)

### 課題・メモ
- （あれば記載）

### 次フェーズ
Phase 5Cに進む
```

---

*作成日: 2026-02-15 16:03 JST*  
*対象: Phase 5B（数論・組み合わせ）*
