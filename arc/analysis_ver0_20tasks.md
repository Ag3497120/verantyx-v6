# ver=0 失敗タスク分析 (20問サンプル)

## 分類サマリー

| タグ | 件数 | タスクID |
|---|---|---|
| conditional_per_object | 5 | 320afe60, 6a1e5592, e4941b18, 40f6cd08, 140c817e |
| pattern_completion / gravity | 2 | 12422b43, 60a26a3e |
| spatial_reasoning (abstract) | 3 | 78e78cff, 2697da3f, 13713586 |
| extract_and_summarize | 5 | 642d658d, f8b3ba0a, b0722778, 6773b310, e872b94a |
| pattern_overlay / merge | 2 | 2ccd9fef, 15660dd6 |
| line_drawing / connect | 1 | d4a91cb9 |
| extract_subgrid_by_rule | 2 | 281123b4, fcb5c309 |

## 個別分析

### 320afe60 — conditional_per_object (recolor by shape)
- same size。背景4の中にcross/L字等のオブジェクトが散在
- 入力のcolor=1パターンが、形状に応じて異なる色(2,3)にrecolor
- **必要**: オブジェクト検出 → 形状分類 → 条件付きrecolor

### 12422b43 — gravity/pattern_completion
- same size。上部にパターンがあり、下部の空領域にコピー/反射される
- 非0セルが「落下」して空きスペースを埋める
- **必要**: gravity or pattern reflection

### 78e78cff — spatial_reasoning (symmetry expansion)
- same size。小さなパターンが対称的に拡張される
- 入力の非均一パターンを中心に十字/放射状に展開
- **必要**: 対称軸検出 → パターン展開

### 6a1e5592 — conditional_per_object (mapping between regions)
- same size。上半分のパターン(色5→色1に変換)を下半分に適用
- 2つの領域間の対応関係学習
- **必要**: 領域分割 → 対応付け → 色マッピング

### 642d658d — extract_and_summarize (count/classify → 1x1)
- 大きなグリッド → 1x1。何らかの集約関数
- ノイズの中から特定パターンを見つけて分類結果を1セルで返す
- **必要**: グリッド全体の統計量/パターン検出 → 単一値出力

### 40f6cd08 — conditional_per_object (nested rectangles)
- 30x30 same size。入れ子の矩形構造
- 一部の矩形の色が条件に基づいて変更される
- **必要**: 矩形/領域検出 → 条件判定 → recolor

### 2ccd9fef — pattern_overlay (fold/merge)
- 24x11→8x11。3倍の高さが1/3に。行を畳み込んでオーバーレイ
- **必要**: グリッド分割 → 畳み込み/マージ操作

### e4941b18 — conditional_per_object (swap/move single cell)
- same size。ごく少数のセルが変更。特定セルの位置をルールで移動
- 色8と色2の位置が入れ替わるようなパターン
- **必要**: 特定色のセル検出 → 位置ルール → 移動/swap

### 2697da3f — spatial_reasoning (scale + tile with gap)
- 小→大。パターンを拡大してタイリング
- 入力パターンが出力で間隔を空けて繰り返される
- **必要**: パターン認識 → スケールファクター計算 → tiling with gaps

### f8b3ba0a — extract_and_summarize (grid → small summary)
- 13xN → 3x1。グリッドの各行/列/領域を評価して3値のサマリー
- **必要**: 領域分割 → 各領域の特性判定 → サマリー生成

### b0722778 — extract_and_summarize (column reduction)
- 8x9 → 8x2。9列→2列。特定の列を選択 or 列のXOR/OR演算
- **必要**: 列間演算 or 選択ルール

### 15660dd6 — pattern_overlay (extract + recolor sub-regions)
- 19x19 → 5x17。大きなグリッドのサブ領域を抽出して合成
- **必要**: サブグリッド検出 → 抽出 → 再配置

### 140c817e — conditional_per_object (cross-projection from dots)
- same size, 49%変更。少数のドット(色1)から十字型に線を投影
- 線の交差でパターン生成
- **必要**: ドット検出 → 十字投影 → 交差処理

### 281123b4 — extract_subgrid_by_rule
- 4x19 → 4x4。長い行から特定の4x4サブグリッドを選択
- **必要**: サブグリッド候補列挙 → 選択ルール学習

### 60a26a3e — pattern_completion (connect/fill between)
- same size, 3%変更。パターンの間を接続/補完
- **必要**: 同色オブジェクト検出 → 間を接続

### 6773b310 — extract_and_summarize (grid → 3x3 boolean)
- 11x11 → 3x3。グリッドを3x3ブロックに分割、各ブロックの特性を0/1にエンコード
- **必要**: ブロック分割 → 特性判定 → boolean encoding

### d4a91cb9 — line_drawing (L字接続)
- same size。2つのドット間をL字型の線で接続
- **必要**: ドットペア検出 → L字パス生成

### fcb5c309 — extract_subgrid_by_rule (frame extraction)
- 大→小。グリッド内のフレーム構造を見つけてフレーム内を抽出、0→特定色に変換
- **必要**: フレーム検出 → 内容抽出 → 色置換

### 13713586 — spatial_reasoning (line extension to border)
- same size, 32%変更。短い線分を端まで延長
- **必要**: 線分検出 → 方向判定 → ボーダーまで延長

### e872b94a — extract_and_summarize (count connected components)
- NxM → Kx1。連結成分を数えて結果を返す
- **必要**: 連結成分分析 → カウント or 分類

## 戦略的優先順位

1. **line_drawing / connect** (d4a91cb9, 60a26a3e, 140c817e, 13713586) — 4タスク
   → 十字投影、L字接続、線延長の共通基盤
2. **extract_and_summarize** (642d658d, f8b3ba0a, b0722778, 6773b310, e872b94a) — 5タスク
   → grid-to-small-grid変換の汎用フレームワーク
3. **conditional_per_object** (320afe60, 6a1e5592, e4941b18, 40f6cd08) — 4タスク
   → オブジェクト条件分岐
4. **pattern_overlay/merge** (2ccd9fef, 15660dd6) — 2タスク
   → 畳み込み/マージ
