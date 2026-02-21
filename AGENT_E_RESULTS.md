# Agent E Results: Cross Simulator 小世界ライブラリ拡張

## 実装完了日時
2026-02-20

## 目標
Math HLE 問題で CEGIS が機能しない原因である小世界ドメインの不足を解消し、`substitution`, `finite_field`, `finite_group`, `modular` の4つの新しい小世界ドメインを追加して CEGIS の証明カバレッジを拡大する。

## 実装内容

### 1. 新しい小世界ドメインの追加 (`cegis/worldgen.py`)

#### A. 変数代入の小世界 (`substitution` ドメイン)
- **用途**: 多項式恒等式・因数分解の反例探索
- **実装**: 変数に具体的な値（整数・有理数）を代入して式の評価値を検証
- **生成戦略**:
  - Seed values: -5～5, 7, 11, 13, 17 + 有理数 (1/2, 1/3, 2/3, -1/2, 3/2)
  - 1〜4個の変数をランダムに選択
  - デフォルト30個のサンプル生成
- **検証例**:
  ```python
  世界: {x: 3, y: -2}
  候補式: x^2 + y^2 = (x+y)^2 を評価
  → 13 ≠ 1 → 反例発見
  ```

#### B. 有限体 (`finite_field` ドメイン)
- **用途**: 体の公理チェック、次数計算、primitive root検証
- **実装**: GF(p) (素数 p の有限体)
- **生成戦略**:
  - 素数: 2, 3, 5, 7, 11, 13
  - 加法・乗法テーブル、乗法逆元テーブルを生成
  - Primitive root を探索
- **性質**:
  - 全ての非零元が乗法逆元を持つ
  - Characteristic = p
  - Primitive root の存在

#### C. 有限群強化版 (`finite_group` ドメイン)
- **用途**: 群論の性質検証（巡回性、位数、単純性）
- **実装**: 巡回群 Z/nZ の拡張版
- **生成戦略**:
  - 位数: 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
  - 全ての生成元を列挙
  - Abel群、巡回群の判定
  - 素数位数 → 単純群

#### D. 剰余演算の世界 (`modular` ドメイン)
- **用途**: 合同式、Fermat小定理、Euler totient関数の検証
- **実装**: Z/mZ の剰余演算
- **生成戦略**:
  - 法: 2, 3, 5, 7, 11, 13, 17, 19, 23
  - ユニット群 (Z/mZ)* の計算
  - φ(m) (Euler totient) の計算
  - Fermat小定理の検証 (a^(p-1) ≡ 1 mod p)
  - Primitive root の探索（素数の場合）

### 2. ドメインマッピングの更新 (`cegis/cegis_loop.py`)

| 元のドメイン | 変更前 | 変更後 | 理由 |
|-------------|--------|--------|------|
| `algebra` | `number` | `substitution` | 代入で恒等式を検証 |
| `number_theory` | `number` | `modular` | mod p 演算で合同式を検証 |
| `advanced_number_theory` | `number` | `modular` | 高度な整数論も mod p で |
| `modular_arithmetic` | `ring` | `modular` | より特化したmod世界 |
| `polynomial` | (新規) | `substitution` | 多項式も代入検証 |
| `group_theory` | (新規) | `finite_group` | 群論専用 |
| `field_theory` | (新規) | `finite_field` | 体論専用 |

### 3. パラメータ生成の追加
新しいドメインに対応するパラメータ自動推定を実装:
- `substitution`: 変数リストを IR から抽出、count=30
- `finite_field`: primes=[2,3,5,7,11,13]
- `finite_group`: orders=[2,3,4,5,6,7,8,9,10,12]
- `modular`: moduli=[2,3,5,7,11,13,17,19,23]

## テスト結果

### 単体テスト
全ての新しい小世界ドメインが正常に動作:

```
=== Testing substitution world ===
Generated 5 substitution worlds
  World 1: {'x': 3}
  World 2: {'x': -1}
  World 3: {'x': Fraction(-1, 2)}

=== Testing finite_field ===
Generated 3 finite_field worlds
  World 1: p=3, size=3, primitive_root=2
  World 2: p=5, size=5, primitive_root=2
  World 3: p=7, size=7, primitive_root=3

=== Testing finite_group ===
Generated 3 finite_group worlds
  World 1: order=3, type=Z_3
  World 2: order=4, type=Z_4
  World 3: order=5, type=Z_5

=== Testing modular ===
Generated 3 modular worlds
  World 1: mod=5, prime=True, phi=4
  World 2: mod=7, prime=True, phi=6
  World 3: mod=11, prime=True, phi=10

=== All tests completed successfully! ===
```

### 評価結果 (HLE 2500)

**スコア**: 95/2500 = **3.80%** (変化なし)

#### カテゴリ別内訳:
- Biology/Medicine: 24/280 (8.6%)
- Humanities/Social Science: 11/219 (5.0%)
- Computer Science/AI: 11/241 (4.6%)
- Chemistry: 6/165 (3.6%)
- Physics: 8/230 (3.5%)
- **Math: 28/1021 (2.7%)** ← ターゲット
- Engineering: 3/111 (2.7%)
- Other: 4/233 (1.7%)

#### 手法別内訳:
- cegis_proved: 69
- unknown: 17
- math_cross_sim: 6
- puzzle_reasoning: 1
- propositional_simulation: 1
- hle_boost:detector:_detect_kb_known_value_mcq: 1

## 分析

### 実装は成功したが、スコア改善が見られなかった理由

1. **IR生成の問題**:
   - データセット (hle_2500_eval.jsonl) にはドメイン情報がない
   - IR生成時にドメインが正しく推定されていない可能性
   - Math問題の多くが "unknown" ドメインに分類され、デフォルトの "number" 世界に割り当てられている

2. **CEGIS適用前の段階で失敗**:
   - Math: 28/1021 (2.7%) のうち、cegis_proved は 12問のみ
   - 多くが "unknown" (8問) や "math_cross_sim" (6問) でカバー
   - CEGIS ループに到達する前の段階（IR生成、Piece抽出）で失敗している

3. **Verifier API の優先**:
   - CEGIS の _find_counterexample は Verifier API を優先する
   - SymPy/Z3 で判定できる場合、小世界テストは実行されない
   - 新しい小世界が使われるのは Verifier が "unknown" を返した場合のみ

4. **問題の難易度**:
   - HLE 2500 の Math 問題は高度な数学問題が多い
   - 例: "Compute the reduced 12-th dimensional Spin bordism of the classifying space of the Lie group G2"
   - 小世界では検証できないトポロジー・代数幾何などが多数含まれる

## 結論

### 成果
✅ 4つの新しい小世界ドメインを実装完了
✅ 単体テストは全て成功
✅ ドメインマッピングの更新完了
✅ パラメータ自動生成の実装完了

### 課題
❌ HLE 2500 スコアの改善なし (3.80% → 3.80%)
❌ Math カテゴリでのCEGIS活用が不十分 (12/1021 = 1.2%)

### 今後の推奨事項

1. **IR生成の改善（最優先）**:
   - Semantic パーサーでドメイン推定精度を向上
   - Math問題の分類精度向上（algebra, number_theory, etc.）

2. **CEGIS到達率の向上**:
   - Piece抽出の改善
   - HLE reasoning の前段階を強化

3. **小世界の活用測定**:
   - ログを追加して新しい小世界が実際に使われているか確認
   - Verifier API が "unknown" を返す問題の割合を測定

4. **特化型小世界の追加**:
   - 三角関数の世界（特殊角）
   - 組合せ論の世界（小さな n, k での検証）
   - 線形代数の世界（2×2, 3×3行列の性質）

## ファイル変更履歴

- `cegis/worldgen.py`: 4つの新ドメイン追加（+190行）
- `cegis/cegis_loop.py`: ドメインマッピング更新、パラメータ生成追加（+13行）

## 次のエージェントへの引き継ぎ事項

小世界ライブラリは拡張されたが、活用されていない。根本原因は：
1. IR生成でドメインが正しく分類されていない
2. Piece抽出で証明可能な候補が生成されていない
3. Verifier API が先に実行されるため小世界テストに到達しない

**推奨**: Agent F (IR生成改善) または Agent D (Piece抽出改善) を優先すべき。
