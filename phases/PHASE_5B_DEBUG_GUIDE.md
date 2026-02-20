# Phase 5B デバッグガイド

**作成日**: 2026-02-15 16:20 JST  
**状況**: テスト結果 1/10問正解（10%） - デバッグ必要

---

## 🐛 現在の問題

### テスト結果

```
✅ Test 2: GCD計算 - 正解
❌ Test 1: 素数判定 - 失敗（False返却）
❌ Test 3: 階乗計算 - 失敗（5返却、期待120）
❌ Test 4: 約数カウント - 失敗（None返却）
❌ Test 5: LCM計算 - 失敗（3返却、期待60）
❌ Test 6-10: 組み合わせ計算 - 全失敗
```

### 症状分析

1. **GCDのみ成功** → 既存実装が動作
2. **他は全て失敗** → 新規ピースが選択されていないか、Executorが正しく動作していない

---

## 🔍 デバッグ手順

### ステップ1: IR抽出の確認

**目的**: 問題文が正しくドメイン・タスクに分類されているか

```python
from decomposer.decomposer import RuleBasedDecomposer

decomposer = RuleBasedDecomposer()

# テスト1: 素数判定
ir1 = decomposer.decompose("Is 17 a prime number?")
print(f"Domain: {ir1.domain}")  # 期待: number_theory
print(f"Task: {ir1.task}")      # 期待: decide
print(f"Entities: {ir1.entities}")

# テスト3: 階乗
ir3 = decomposer.decompose("Calculate 5 factorial (5!)")
print(f"Domain: {ir3.domain}")  # 期待: number_theory
print(f"Task: {ir3.task}")      # 期待: compute
print(f"Entities: {ir3.entities}")  # 期待: number=5

# テスト6: 順列
ir6 = decomposer.decompose("Calculate P(5, 2) - the number of permutations")
print(f"Domain: {ir6.domain}")  # 期待: combinatorics
print(f"Entities: {ir6.entities}")  # 期待: n=5, r=2
```

**予想される問題**:
- ドメインが `unknown` になっている
- エンティティが抽出されていない（数値が取得できていない）

---

### ステップ2: ピース選択の確認

**目的**: 正しいピースが選択されているか

```python
from pieces.piece import PieceDB
from core.ir import IR, TaskType, Domain, AnswerSchema

db = PieceDB('pieces/piece_db.jsonl')

# Test 1: 素数判定
ir1_dict = {
    "task": "decide",
    "domain": "number_theory",
    "answer_schema": "boolean",
    "entities": [{"type": "number", "value": 17}]
}

results = db.search(ir1_dict, top_k=5)
print("Top 5 pieces:")
for piece, score in results:
    print(f"  {score:.2f} - {piece.piece_id}")

# 期待: number_theory_prime または nt_is_prime が上位
```

**予想される問題**:
- ピースのrequiresがIRと一致していない
- スコアリングが低すぎる

---

### ステップ3: Executor動作確認

**目的**: Executorが正しく呼ばれ、正しい値を返すか

```python
from executors.number_theory import is_prime, factorial
from executors.combinatorics import permutation

# 素数判定
result1 = is_prime(number=17)
print(f"is_prime(17): {result1}")
# 期待: {"value": True, "schema": "boolean", "confidence": 1.0}

# 階乗
result3 = factorial(n=5)
print(f"factorial(5): {result3}")
# 期待: {"value": 120, "schema": "integer", "confidence": 1.0}

# 順列
result6 = permutation(n=5, r=2)
print(f"permutation(5, 2): {result6}")
# 期待: {"value": 20, "schema": "integer", "confidence": 1.0}
```

**予想される問題**:
- Executorがスタブで実装されたまま（コピペミス）
- 引数が正しく渡されていない

---

### ステップ4: エンティティ抽出の改善

**問題**: 「5 factorial (5!)」から数値5が抽出できていない可能性

**解決**: `decomposer.py`の`_extract_entities()`を改善

```python
def _extract_entities(self, text: str, domain: Domain) -> List[Entity]:
    """エンティティ抽出（数値・論理式）"""
    entities = []
    
    # ... 既存コード ...
    
    # 階乗パターン: "5!" または "5 factorial"
    if domain == Domain.NUMBER_THEORY:
        # "n!" パターン
        factorial_matches = re.findall(r'(\d+)!', text)
        for match in factorial_matches:
            entities.append(Entity(type="number", value=int(match)))
        
        # "n factorial" パターン
        factorial_matches2 = re.findall(r'(\d+)\s+factorial', text, re.IGNORECASE)
        for match in factorial_matches2:
            entities.append(Entity(type="number", value=int(match)))
    
    # 組み合わせパターン: "P(n, r)" または "C(n, r)"
    if domain == Domain.COMBINATORICS:
        # P(5, 2) パターン
        perm_matches = re.findall(r'[PC]\((\d+),\s*(\d+)\)', text)
        for n, r in perm_matches:
            entities.append(Entity(type="number", value=int(n), name="n"))
            entities.append(Entity(type="number", value=int(r), name="r"))
    
    return entities
```

---

### ステップ5: ピースのrequires修正

**問題**: ピースのrequiresが厳しすぎる可能性

**例**: `number_theory_prime`

```jsonl
現在:
{"piece_id": "number_theory_prime", "in": {"requires": ["domain:number_theory", "task:decide"], ...}}

修正案:
{"piece_id": "nt_is_prime_general", "in": {"requires": ["domain:number_theory"], ...}}
```

→ taskを必須にせず、ドメインのみでマッチするピースを追加

---

## ✅ 修正チェックリスト

Phase 5Bを完成させるために必要な修正：

### 高優先度
- [ ] Decomposer: エンティティ抽出の改善（階乗・組み合わせパターン）
- [ ] ピース: requires条件の緩和（汎用ピース追加）
- [ ] Executor: 動作確認（単体テスト）

### 中優先度
- [ ] ピース選択: スコアリングロジックの改善
- [ ] エラーハンドリング: Noneが返る場合のデバッグ

### 低優先度
- [ ] ログ出力: デバッグ情報の追加

---

## 🎯 完了基準（再確認）

Phase 5Bを完了と判断する条件：

1. ✅ 単体テスト: 10問中7問以上正解（70%）
2. ✅ HLE検証: 126問中80問以上正解（63%）
3. ✅ PROGRESS.json更新

現状: 1/10問正解（10%） → **60%の改善が必要**

---

## 📝 次のセッションでの実装順序

1. **Decomposerのエンティティ抽出改善**（30分）
2. **単体テスト再実行**（10分）
3. **ピースrequires調整**（必要に応じて）（20分）
4. **単体テスト70%達成確認**（10分）
5. **HLE検証スクリプト実行**（30分）
6. **Phase 5B完了、Phase 5C開始**

**推定時間**: 2-3時間

---

## 🔧 デバッグスクリプト（即実行可能）

```bash
cd ~/.openclaw/workspace/verantyx_v6

# ステップ1: IR確認
python3 -c "
from decomposer.decomposer import RuleBasedDecomposer
d = RuleBasedDecomposer()
ir = d.decompose('Is 17 a prime number?')
print(f'Domain: {ir.domain}, Task: {ir.task}')
print(f'Entities: {ir.entities}')
"

# ステップ2: ピース選択確認
python3 -c "
from pieces.piece import PieceDB
db = PieceDB('pieces/piece_db.jsonl')
ir = {'domain': 'number_theory', 'task': 'decide', 'answer_schema': 'boolean'}
results = db.search(ir, top_k=3)
for p, s in results:
    print(f'{s:.2f} - {p.piece_id}')
"

# ステップ3: Executor確認
python3 -c "
from executors.number_theory import is_prime, factorial
print('is_prime(17):', is_prime(number=17))
print('factorial(5):', factorial(n=5))
"

# ステップ4: テスト再実行
python3 tests/test_phase_5b.py
```

---

**Status**: Phase 5B未完成、デバッグ必要  
**Next**: エンティティ抽出改善 → テスト再実行  
**Target**: 70%達成

---

*作成日: 2026-02-15 16:20 JST*  
*次のセッションで継続*
