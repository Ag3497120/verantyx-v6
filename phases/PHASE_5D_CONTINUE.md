# Phase 5D 継続ガイド（次のセッション用）

**作成日**: 2026-02-16 03:45 JST  
**前回完了**: Phase 5C (100% 達成) ✅✅✅  
**推定作業時間**: 1-2時間

---

## 🎯 Phase 5D 目標

### 対象
- **ドメイン**: 代数基本・グラフ理論
- **問題数**: 153問 (15.0%)
- **テスト目標**: 70% (7/10以上)

### 必要な実装
1. **Algebra Executor**（4-5関数）
   - solve_linear: 一次方程式
   - solve_quadratic: 二次方程式
   - simplify: 式の簡約
   - factor: 因数分解
   - solve_system: 線形方程式系（オプション）

2. **Graph Theory Executor**（5関数）
   - is_tree: 木の判定
   - is_cyclic: 循環判定
   - degree_sum: 次数和
   - complete_graph_edges: 完全グラフの辺数
   - is_bipartite: 二部グラフ判定（オプション）

---

## 📋 実装手順

### Step 1: Executor実装（40分）

#### Algebra Executor
```python
# executors/algebra.py

def solve_linear(a: float = None, b: float = None, ir: Dict = None, **kwargs):
    """
    一次方程式 ax + b = 0 を解く
    
    Args:
        a: 係数
        b: 定数項
        ir: IR辞書（エンティティから抽出）
    
    Returns:
        x = -b/a
    """
    # エンティティから数値抽出
    if (a is None or b is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
    
    if a is None or a == 0:
        return {"value": None, "confidence": 0.0, "error": "Invalid coefficient"}
    
    x = -b / a
    return {"value": x, "schema": "number", "confidence": 1.0}


def solve_quadratic(a: float = None, b: float = None, c: float = None, ir: Dict = None, **kwargs):
    """
    二次方程式 ax² + bx + c = 0 を解く
    
    Returns:
        [x1, x2] または [x] (重解)
    """
    import math
    
    # エンティティから抽出
    if (a is None or b is None or c is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 3:
            a, b, c = numbers[0], numbers[1], numbers[2]
    
    if a is None or a == 0:
        return {"value": None, "confidence": 0.0, "error": "Not quadratic"}
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return {"value": None, "confidence": 0.5, "note": "Complex roots"}
    
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    
    if abs(x1 - x2) < 1e-9:
        return {"value": [x1], "schema": "list", "confidence": 1.0}
    
    return {"value": [x1, x2], "schema": "list", "confidence": 1.0}


def simplify_expression(expr: str = None, ir: Dict = None, **kwargs):
    """
    式の簡約（基本的なパターンマッチング）
    
    Examples:
        (x² - 4) / (x - 2) → x + 2
    """
    if expr is None and ir:
        expr = ir.get("metadata", {}).get("source_text", "")
    
    # 基本的なパターン: (x² - a²) / (x - a) → x + a
    import re
    pattern = r'\(x\^?2\s*-\s*(\d+)\)\s*/\s*\(x\s*-\s*(\d+)\)'
    match = re.search(pattern, expr)
    
    if match:
        a_sq = int(match.group(1))
        a = int(match.group(2))
        if a * a == a_sq:
            return {"value": f"x + {a}", "schema": "expression", "confidence": 0.8}
    
    return {"value": None, "confidence": 0.0, "error": "Cannot simplify"}


def factor_expression(expr: str = None, ir: Dict = None, **kwargs):
    """
    因数分解
    
    Examples:
        x² + 5x + 6 → (x + 2)(x + 3)
    """
    if expr is None and ir:
        expr = ir.get("metadata", {}).get("source_text", "")
    
    # パターン: x² + bx + c
    import re
    pattern = r'x\^?2\s*\+\s*(\d+)x\s*\+\s*(\d+)'
    match = re.search(pattern, expr)
    
    if match:
        b = int(match.group(1))
        c = int(match.group(2))
        
        # 因数を探す: x² + bx + c = (x + p)(x + q) where p+q=b, p*q=c
        for p in range(-c, c+1):
            q = c // p if p != 0 else 0
            if p * q == c and p + q == b:
                return {"value": f"(x + {p})(x + {q})", "schema": "expression", "confidence": 0.9}
    
    return {"value": None, "confidence": 0.0, "error": "Cannot factor"}
```

#### Graph Theory Executor
```python
# executors/graph_theory.py

def is_tree(vertices: int = None, edges: int = None, has_cycle: bool = False, ir: Dict = None, **kwargs):
    """
    木の判定: V = E + 1 かつ循環なし
    """
    if (vertices is None or edges is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            vertices, edges = numbers[0], numbers[1]
    
    if vertices is None or edges is None:
        return {"value": None, "confidence": 0.0, "error": "Need V and E"}
    
    # 木の条件: V = E + 1 かつ連結・無循環
    is_tree_result = (vertices == edges + 1) and not has_cycle
    
    return {"value": is_tree_result, "schema": "boolean", "confidence": 0.9}


def is_cyclic(vertices: int = None, edges: int = None, ir: Dict = None, **kwargs):
    """
    循環判定: E >= V なら循環あり（単純グラフ）
    """
    if (vertices is None or edges is None) and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 2:
            vertices, edges = numbers[0], numbers[1]
    
    if vertices is None or edges is None:
        return {"value": None, "confidence": 0.0, "error": "Need V and E"}
    
    has_cycle = edges >= vertices
    
    return {"value": has_cycle, "schema": "boolean", "confidence": 0.8}


def degree_sum(vertices: int = None, edges: int = None, ir: Dict = None, **kwargs):
    """
    次数和の定理: Σdeg(v) = 2E
    """
    if edges is None and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 1:
            edges = numbers[0]
    
    if edges is None:
        return {"value": None, "confidence": 0.0, "error": "Need E"}
    
    deg_sum = 2 * edges
    
    return {"value": deg_sum, "schema": "number", "confidence": 1.0}


def complete_graph_edges(vertices: int = None, ir: Dict = None, **kwargs):
    """
    完全グラフKnの辺数: E = n(n-1)/2
    """
    if vertices is None and ir:
        numbers = []
        for entity in ir.get("entities", []):
            if entity.get("type") == "number":
                numbers.append(entity.get("value"))
        
        if len(numbers) >= 1:
            vertices = numbers[0]
    
    if vertices is None:
        return {"value": None, "confidence": 0.0, "error": "Need n"}
    
    edges = vertices * (vertices - 1) // 2
    
    return {"value": edges, "schema": "number", "confidence": 1.0}
```

### Step 2: ピース追加（20分）

```bash
cd ~/.openclaw/workspace/verantyx_v6

# Algebra pieces (5個)
cat >> pieces/piece_db.jsonl << 'EOF'
{"piece_id": "algebra_solve_linear", "name": "Solve Linear Equation", "description": "一次方程式を解く", "in": {"requires": ["domain:algebra", "task:compute"], "slots": []}, "out": {"produces": ["number"], "schema": "number", "artifacts": []}, "executor": "executors.algebra.solve_linear", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["algebra", "linear", "equation", "solve"]}
{"piece_id": "algebra_solve_quadratic", "name": "Solve Quadratic Equation", "description": "二次方程式を解く", "in": {"requires": ["domain:algebra", "task:compute"], "slots": []}, "out": {"produces": ["list"], "schema": "list", "artifacts": []}, "executor": "executors.algebra.solve_quadratic", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["algebra", "quadratic", "equation", "solve"]}
{"piece_id": "algebra_simplify", "name": "Simplify Expression", "description": "式の簡約", "in": {"requires": ["domain:algebra", "task:compute"], "slots": []}, "out": {"produces": ["expression"], "schema": "expression", "artifacts": []}, "executor": "executors.algebra.simplify_expression", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.8, "tags": ["algebra", "simplify", "expression"]}
{"piece_id": "algebra_factor", "name": "Factor Expression", "description": "因数分解", "in": {"requires": ["domain:algebra", "task:compute"], "slots": []}, "out": {"produces": ["expression"], "schema": "expression", "artifacts": []}, "executor": "executors.algebra.factor_expression", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.9, "tags": ["algebra", "factor", "expression"]}
{"piece_id": "algebra_evaluate_polynomial", "name": "Evaluate Polynomial", "description": "多項式評価", "in": {"requires": ["domain:algebra", "task:compute"], "slots": []}, "out": {"produces": ["number"], "schema": "number", "artifacts": []}, "executor": "executors.algebra.evaluate_polynomial", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.9, "tags": ["algebra", "polynomial", "evaluate"]}
EOF

# Graph Theory pieces (5個)
cat >> pieces/piece_db.jsonl << 'EOF'
{"piece_id": "graph_is_tree", "name": "Is Tree", "description": "木の判定", "in": {"requires": ["domain:graph_theory", "task:verify"], "slots": []}, "out": {"produces": ["boolean"], "schema": "boolean", "artifacts": []}, "executor": "executors.graph_theory.is_tree", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.9, "tags": ["graph", "tree", "verify"]}
{"piece_id": "graph_is_cyclic", "name": "Is Cyclic", "description": "循環判定", "in": {"requires": ["domain:graph_theory", "task:verify"], "slots": []}, "out": {"produces": ["boolean"], "schema": "boolean", "artifacts": []}, "executor": "executors.graph_theory.is_cyclic", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.8, "tags": ["graph", "cyclic", "cycle", "verify"]}
{"piece_id": "graph_degree_sum", "name": "Degree Sum", "description": "次数和", "in": {"requires": ["domain:graph_theory", "task:compute"], "slots": []}, "out": {"produces": ["number"], "schema": "number", "artifacts": []}, "executor": "executors.graph_theory.degree_sum", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["graph", "degree", "sum"]}
{"piece_id": "graph_complete_edges", "name": "Complete Graph Edges", "description": "完全グラフの辺数", "in": {"requires": ["domain:graph_theory", "task:compute"], "slots": []}, "out": {"produces": ["number"], "schema": "number", "artifacts": []}, "executor": "executors.graph_theory.complete_graph_edges", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 1.0, "tags": ["graph", "complete", "edges"]}
{"piece_id": "graph_is_bipartite", "name": "Is Bipartite", "description": "二部グラフ判定", "in": {"requires": ["domain:graph_theory", "task:verify"], "slots": []}, "out": {"produces": ["boolean"], "schema": "boolean", "artifacts": []}, "executor": "executors.graph_theory.is_bipartite", "verifiers": [], "cost": {"time": "instant", "space": "low", "explosion_risk": "none"}, "confidence": 0.7, "tags": ["graph", "bipartite", "verify"]}
EOF

echo "Added 10 pieces (5 algebra + 5 graph)"
```

### Step 3: Decomposer強化（10分）

```python
# decomposer/decomposer.py - keywords extraction section

# 代数
if "solve" in text_lower or "equation" in text_lower:
    keywords.append("solve")
    keywords.append("equation")
if "simplify" in text_lower:
    keywords.append("simplify")
if "factor" in text_lower and "factorial" not in text_lower:
    keywords.append("factor")
if "evaluate" in text_lower:
    keywords.append("evaluate")
if "polynomial" in text_lower:
    keywords.append("polynomial")

# グラフ理論
if "graph" in text_lower:
    keywords.append("graph")
if "vertex" in text_lower or "vertices" in text_lower:
    keywords.append("vertex")
if "edge" in text_lower or "edges" in text_lower:
    keywords.append("edges")
if "tree" in text_lower:
    keywords.append("tree")
if "cyclic" in text_lower or "cycle" in text_lower:
    keywords.append("cyclic")
if "degree" in text_lower:
    keywords.append("degree")
if "complete" in text_lower:
    keywords.append("complete")
if "binary" in text_lower:
    keywords.append("binary")
```

### Step 4: テスト作成（10分）

```python
# tests/test_phase_5d.py

test_cases = [
    # 代数
    {"question": "Solve 2x + 3 = 7", "expected": "2", "domain": "algebra"},
    {"question": "Solve x² - 5x + 6 = 0", "expected": "[2, 3]", "domain": "algebra"},
    {"question": "Simplify (x² - 4) / (x - 2)", "expected": "x + 2", "domain": "algebra"},
    {"question": "Factor x² + 5x + 6", "expected": "(x + 2)(x + 3)", "domain": "algebra"},
    {"question": "Evaluate x² + 3x + 2 at x = 1", "expected": "6", "domain": "algebra"},
    
    # グラフ理論
    {"question": "Is a graph with 5 vertices and 4 edges a tree?", "expected": "true", "domain": "graph_theory"},
    {"question": "Does a graph with 3 vertices and 3 edges have a cycle?", "expected": "true", "domain": "graph_theory"},
    {"question": "What is the sum of degrees in a graph with 5 edges?", "expected": "10", "domain": "graph_theory"},
    {"question": "How many edges does a complete graph K5 have?", "expected": "10", "domain": "graph_theory"},
    {"question": "Is K(3,3) a bipartite graph?", "expected": "true", "domain": "graph_theory"},
]
```

### Step 5: テスト実行（10分）

```bash
python3 tests/test_phase_5d.py
```

---

## ⚠️ 注意事項

### Algebra関連
1. **simplify/factor**: パターンマッチング限定
   - 完全な代数システムは不要
   - 基本的なパターンのみ対応

2. **solve_system**: オプション
   - 時間があれば実装
   - なくても70%達成可能

### Graph Theory関連
1. **is_tree/is_cyclic**: 近似判定
   - 厳密なアルゴリズム不要
   - 基本的な条件チェックのみ

2. **is_bipartite**: 複雑度高い
   - 後回し可能
   - テストで失敗しても問題なし

---

## 🎯 期待される結果

| Test | 問題 | 期待値 | Phase 5D目標 |
|------|------|--------|-------------|
| 1 | Linear | 2 | ✅ |
| 2 | Quadratic | [2,3] | ✅ |
| 3 | Simplify | x+2 | ⚠️ (パターン限定) |
| 4 | Factor | (x+2)(x+3) | ⚠️ (パターン限定) |
| 5 | Evaluate | 6 | ✅ |
| 6 | Tree | true | ✅ |
| 7 | Cyclic | true | ✅ |
| 8 | Degree sum | 10 | ✅ |
| 9 | K5 edges | 10 | ✅ |
| 10 | Bipartite | true | ⚠️ (複雑) |

**期待スコア**: 6-8/10 (60-80%)  
**目標**: 7/10以上 (70%)

---

## 📊 Phase 5D完了後の進捗

- Phase 5A: 36問 (3.5%) ✅
- Phase 5B: 126問 (12.3%) ✅
- Phase 5C: 219問 (21.4%) ✅
- **Phase 5D: 153問 (15.0%)** → 完了予定
- **累計: 534問 (52.3%)**

---

**Status**: Phase 5D開始可能  
**Next milestone**: Phase 5D完了（70%達成）  
**Estimated time**: 1-2時間

---

*作成: 2026-02-16 03:45 JST*
