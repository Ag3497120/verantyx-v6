# Agent E: Cross Simulator 小世界ライブラリ拡張

## 目標
Math HLE 問題で CEGIS が機能しない原因: worldgen が `number/matrix/graph/propositional/sequence/ring` の6種しかない。
Math 向け小世界を追加して CEGIS の証明カバレッジを拡大する。

## ワークスペース
`/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`

## 現状
- `cegis/worldgen.py` に WorldGenerator クラスあり
- 現在のドメイン: number, matrix, graph, propositional, set, function, permutation, sequence, ring, polynomial, group
- Math 1021問で27問しか解けていない (2.6%)
- 特に「代入で壊せる」「有限構造で確認できる」タイプが未カバー

## ChatGPT 推奨の優先小世界（ROI順）

### A. 有限代入世界（最優先）— `substitution` ドメイン
```python
# 恒等式・因数分解・多項式の同値を small integer / rational で検証
# world: {vars: {x: 3, y: -2, n: 7}} などの変数束縛
# verify: 式の評価値が一致するか

def _gen_substitution(self, params):
    """変数代入の小世界 — 多項式恒等式の反例探索用"""
    import random
    worlds = []
    seed_values = [-5,-4,-3,-2,-1,0,1,2,3,4,5,7,11,13]
    var_names = params.get('vars', ['x', 'n', 'k'])
    for _ in range(params.get('count', 20)):
        assignment = {v: random.choice(seed_values) for v in var_names}
        worlds.append(FiniteModel(
            domain='substitution',
            elements=list(assignment.values()),
            relations={},
            properties=assignment,
            size=len(assignment)
        ))
    return worlds
```

### B. 有限体世界 — `finite_field` ドメイン
```python
# GF(p), GF(p^k) の小有限体
# world: {p: 3, elements: [0,1,2], add_table: ..., mul_table: ...}
# verify: 体の公理チェック、次数計算など

def _gen_finite_field(self, params):
    p = params.get('p', 3)  # 素数
    elements = list(range(p))
    worlds = []
    for _ in range(params.get('count', 5)):
        worlds.append(FiniteModel(
            domain='finite_field',
            elements=elements,
            relations={'add': [(a, b, (a+b)%p) for a in elements for b in elements],
                       'mul': [(a, b, (a*b)%p) for a in elements for b in elements]},
            properties={'p': p, 'size': p, 'characteristic': p},
            size=p
        ))
    return worlds
```

### C. 有限群世界 — `finite_group` ドメイン（既存の `group` を強化）
```python
# 巡回群 Z/nZ, 対称群 S_n (小n)
# world: {n: 6, elements: [0..5], operation: 'addition_mod_n'}
# verify: 群の性質（結合律・単位元・逆元）

def _gen_finite_group(self, params):
    n = params.get('n', 6)
    elements = list(range(n))
    op_type = params.get('op', 'cyclic')
    worlds = []
    if op_type == 'cyclic':
        worlds.append(FiniteModel(
            domain='finite_group',
            elements=elements,
            relations={'op': [(a, b, (a+b)%n) for a in elements for b in elements]},
            properties={'order': n, 'type': 'cyclic', 'generator': 1},
            size=n
        ))
    return worlds
```

### D. mod p 世界 — `modular` ドメイン
```python
# 剰余演算の世界
# world: {mod: 7, elements: [0..6]}
# verify: 合同式、Fermat小定理、primitive rootなど

def _gen_modular(self, params):
    mod = params.get('mod', 7)
    worlds = []
    for m in [5, 7, 11, 13, 17]:  # 小素数
        worlds.append(FiniteModel(
            domain='modular',
            elements=list(range(m)),
            relations={},
            properties={'mod': m, 'elements': list(range(m))},
            size=m
        ))
    return worlds
```

## 実装手順

### 1. `cegis/worldgen.py` を編集
- `WorldGenerator.generate()` に新ドメインを追加
- 上記4つのドメインを実装
- `_DOMAIN_GENERATORS` マッピングに追加

### 2. `cegis/cegis_loop.py` のドメインマッピング更新
```python
_DOMAIN_TO_WORLD: Dict[str, str] = {
    ...
    # 追加
    "algebra":              "substitution",   # 代入検証
    "number_theory":        "modular",        # mod p
    "group_theory":         "finite_group",
    "field_theory":         "finite_field",
    "modular_arithmetic":   "modular",
    "polynomial":           "substitution",
}
```

### 3. verify ロジックも強化
`_find_counterexample` で substitution world の場合:
```python
if world.domain == 'substitution':
    # 候補の式を変数代入で評価して反例を探す
    assignment = world.properties
    try:
        import sympy
        # 候補値と期待値を比較
        result = eval_with_substitution(cand.value, assignment)
        expected = eval_with_substitution(ir_dict.get('expected_expr', ''), assignment)
        if result != expected:
            return {'type': 'substitution_fail', 'assignment': assignment}
    except:
        pass
```

### 4. テスト
```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6

python3 -c "
from cegis.worldgen import WorldGenerator
wg = WorldGenerator(max_worlds=10)

# substitution world テスト
worlds = wg.generate('substitution', {'vars': ['x', 'n'], 'count': 5})
print(f'substitution worlds: {len(worlds)}')
for w in worlds[:2]:
    print('  props:', w.properties)

# finite_field テスト
worlds = wg.generate('finite_field', {'p': 5, 'count': 3})
print(f'finite_field worlds: {len(worlds)}')

# modular テスト
worlds = wg.generate('modular', {'mod': 7})
print(f'modular worlds: {len(worlds)}')
"
```

### 5. 評価実行
```bash
python3 quick_eval_hle.py 2>&1 | grep -E "Math:|cegis_proved|Correct:|Accuracy:"
```

## 完了条件
- 4つの新世界ドメイン実装済み
- テスト通過
- Math スコアが改善（cegis_proved 数が増加）

## 結果報告
`AGENT_E_RESULTS.md` に書いて:
```bash
openclaw system event --text "Agent E完了: 小世界ライブラリ拡張。結果: AGENT_E_RESULTS.md" --mode now
```
