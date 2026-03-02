"""
arc/cross3d_projection.py — 立体十字 4方向射影エンジン

=== kofdaiの設計 ===
1. 各色の連結構造を3Dで「立てる」（Z軸に押し出し）
2. 4方向(N/S/E/W)から見た射影を記録
3. before(入力)とafter(出力)の射影を合成 → 移動/変換を把握
4. 1ブロック = 切り抜き or 色集合（特別扱い）
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from scipy.ndimage import label as scipy_label
from dataclasses import dataclass


@dataclass
class Block3D:
    """3Dに立てたブロック"""
    color: int
    cells_2d: List[Tuple[int, int]]  # 元の2D位置
    cells_3d: List[Tuple[int, int, int]]  # 3D位置 (r, c, z)
    bbox_2d: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    height: int  # Z方向の高さ
    
    # 4方向射影
    proj_north: Optional[np.ndarray] = None  # 北(上)から見た = Z-C平面
    proj_south: Optional[np.ndarray] = None  # 南(下)から見た
    proj_east: Optional[np.ndarray] = None   # 東(右)から見た = Z-R平面
    proj_west: Optional[np.ndarray] = None   # 西(左)から見た


def extract_blocks(grid, bg=0) -> List[Block3D]:
    """グリッドから色ブロックを抽出して3D化"""
    g = np.array(grid)
    h, w = g.shape
    struct = np.ones((3, 3), dtype=int)  # 8連結
    
    blocks = []
    for color in sorted(set(int(v) for v in g.flatten()) - {bg}):
        mask = (g == color).astype(int)
        labeled, n = scipy_label(mask, structure=struct)
        
        for i in range(1, n + 1):
            cells = list(zip(*np.where(labeled == i)))
            if not cells:
                continue
            
            r_min = min(r for r, c in cells)
            c_min = min(c for r, c in cells)
            r_max = max(r for r, c in cells)
            c_max = max(c for r, c in cells)
            
            # 3D化: 各セルを(r, c, 0)に配置、高さ=セル数の面積比
            bh = r_max - r_min + 1
            bw = c_max - c_min + 1
            area = bh * bw
            fill_ratio = len(cells) / area if area > 0 else 0
            
            # 高さ = セル数（各セルが1段のブロック）
            height = len(cells)
            
            # 3Dセル: Z軸方向に立てる（下辺を中心に立てる）
            cells_3d = []
            for z, (r, c) in enumerate(sorted(cells)):
                cells_3d.append((r, c, z))
            
            block = Block3D(
                color=color,
                cells_2d=cells,
                cells_3d=cells_3d,
                bbox_2d=(r_min, c_min, r_max, c_max),
                height=height,
            )
            
            # 4方向射影を計算
            _compute_projections(block, h, w)
            blocks.append(block)
    
    return blocks


def _compute_projections(block: Block3D, grid_h: int, grid_w: int):
    """4方向からの射影を計算"""
    r_min, c_min, r_max, c_max = block.bbox_2d
    bh = r_max - r_min + 1
    bw = c_max - c_min + 1
    
    # ブロックの2Dシルエット（上から見た形 = 元の2D形状）
    top_view = np.zeros((bh, bw), dtype=int)
    for r, c in block.cells_2d:
        top_view[r - r_min, c - c_min] = block.color
    
    # 北から見た射影 (Z-C平面): 各列のセルの高さプロファイル
    # → 列ごとに何セルあるか
    north = np.zeros((block.height, bw), dtype=int)
    col_counts = Counter(c - c_min for r, c in block.cells_2d)
    for c_rel, count in col_counts.items():
        for z in range(count):
            north[block.height - 1 - z, c_rel] = block.color
    block.proj_north = north
    
    # 南から見た = 北を左右反転
    block.proj_south = np.fliplr(north)
    
    # 東から見た射影 (Z-R平面): 各行のセルの高さプロファイル
    east = np.zeros((block.height, bh), dtype=int)
    row_counts = Counter(r - r_min for r, c in block.cells_2d)
    for r_rel, count in row_counts.items():
        for z in range(count):
            east[block.height - 1 - z, r_rel] = block.color
    block.proj_east = east
    
    # 西から見た = 東を左右反転
    block.proj_west = np.fliplr(east)


def projection_signature(blocks: List[Block3D]) -> Dict[str, list]:
    """全ブロックの射影を統合したシグネチャ"""
    sig = {
        'n_blocks': len(blocks),
        'colors': [b.color for b in blocks],
        'sizes': [len(b.cells_2d) for b in blocks],
        'heights': [b.height for b in blocks],
        'bboxes': [b.bbox_2d for b in blocks],
        'north_shapes': [b.proj_north.shape if b.proj_north is not None else None for b in blocks],
        'east_shapes': [b.proj_east.shape if b.proj_east is not None else None for b in blocks],
    }
    return sig


# ══════════════════════════════════════════════════════════════
# Before/After 合成（移動・変換検出）
# ══════════════════════════════════════════════════════════════

@dataclass
class Transform3D:
    """before→afterのブロック変換"""
    block_color: int
    action: str  # 'stay', 'move', 'clone', 'recolor', 'remove', 'appear'
    move_dr: int = 0
    move_dc: int = 0
    new_color: int = -1
    clone_positions: List[Tuple[int, int]] = None  # clone先の中心リスト
    template_rel: List[Tuple[int, int]] = None  # テンプレートの相対セル


def detect_transforms(grid_in, grid_out, bg=0) -> List[Transform3D]:
    """before/afterの射影差分から変換を検出"""
    blocks_in = extract_blocks(grid_in, bg)
    blocks_out = extract_blocks(grid_out, bg)
    
    transforms = []
    
    # 色でグループ化
    in_by_color = defaultdict(list)
    out_by_color = defaultdict(list)
    for b in blocks_in:
        in_by_color[b.color].append(b)
    for b in blocks_out:
        out_by_color[b.color].append(b)
    
    all_colors = set(in_by_color.keys()) | set(out_by_color.keys())
    
    for color in all_colors:
        ins = in_by_color.get(color, [])
        outs = out_by_color.get(color, [])
        
        if not ins and outs:
            # 新色出現 → appear or clone(色変え)
            for o in outs:
                transforms.append(Transform3D(color, 'appear'))
        
        elif ins and not outs:
            # 色消失 → remove
            for i in ins:
                transforms.append(Transform3D(color, 'remove'))
        
        elif len(ins) == 1 and len(outs) == 1:
            bi, bo = ins[0], outs[0]
            cells_i = set(bi.cells_2d)
            cells_o = set(bo.cells_2d)
            
            if cells_i == cells_o:
                transforms.append(Transform3D(color, 'stay'))
            else:
                # 移動検出
                ci = (sum(r for r,c in cells_i)/len(cells_i), 
                      sum(c for r,c in cells_i)/len(cells_i))
                co = (sum(r for r,c in cells_o)/len(cells_o),
                      sum(c for r,c in cells_o)/len(cells_o))
                dr = int(round(co[0] - ci[0]))
                dc = int(round(co[1] - ci[1]))
                transforms.append(Transform3D(color, 'move', dr, dc))
        
        elif len(ins) == 1 and len(outs) > 1:
            # 1→多 = clone
            bi = ins[0]
            positions = []
            for bo in outs:
                cr = sum(r for r,c in bo.cells_2d) / len(bo.cells_2d)
                cc = sum(c for r,c in bo.cells_2d) / len(bo.cells_2d)
                positions.append((int(round(cr)), int(round(cc))))
            
            tcr = int(round(sum(r for r,c in bi.cells_2d) / len(bi.cells_2d)))
            tcc = int(round(sum(c for r,c in bi.cells_2d) / len(bi.cells_2d)))
            template_rel = [(r-tcr, c-tcc) for r,c in bi.cells_2d]
            
            transforms.append(Transform3D(
                color, 'clone', 
                clone_positions=positions,
                template_rel=template_rel,
            ))
    
    return transforms


# ══════════════════════════════════════════════════════════════
# 砲台パターン特化ソルバー（045e512c型）
# ══════════════════════════════════════════════════════════════

def cannon_solve(train_pairs, test_input):
    """砲台パターン: テンプレートをドット方向に端まで繰り返しスタンプ"""
    from arc.grid import grid_eq
    
    gi = np.array(test_input)
    h, w = gi.shape
    bg = 0
    
    # 8連結でオブジェクト抽出
    struct = np.ones((3, 3), dtype=int)
    mask = (gi != bg).astype(int)
    labeled, n = scipy_label(mask, structure=struct)
    
    objs = []
    for i in range(1, n + 1):
        cells = list(zip(*np.where(labeled == i)))
        color = int(gi[cells[0]])
        cr = sum(r for r, c in cells) / len(cells)
        cc = sum(c for r, c in cells) / len(cells)
        objs.append({
            'cells': cells, 'size': len(cells), 'color': color,
            'center': (cr, cc)
        })
    
    if len(objs) < 2:
        return None
    
    # 色で分割（8連結で同色が1塊の場合と複数塊の場合）
    color_objs = defaultdict(list)
    for o in objs:
        color_objs[o['color']].append(o)
    
    # テンプレート = 最大オブジェクト
    template = max(objs, key=lambda o: o['size'])
    tcr = int(round(template['center'][0]))
    tcc = int(round(template['center'][1]))
    cross_rel = [(r - tcr, c - tcc) for r, c in template['cells']]
    
    rs = [r for r, c in template['cells']]
    cs = [c for r, c in template['cells']]
    bbox_h = max(rs) - min(rs) + 1
    bbox_w = max(cs) - min(cs) + 1
    
    pred = np.zeros_like(gi)
    for dr, dc in cross_rel:
        pred[tcr + dr, tcc + dc] = template['color']
    
    for o in objs:
        if o is template:
            continue
        
        dir_r = o['center'][0] - template['center'][0]
        dir_c = o['center'][1] - template['center'][1]
        
        if abs(dir_r) < 0.5:
            sign_r, sign_c = 0, (1 if dir_c > 0 else -1)
        elif abs(dir_c) < 0.5:
            sign_r, sign_c = (1 if dir_r > 0 else -1), 0
        elif abs(dir_r) > abs(dir_c) * 2:
            sign_r, sign_c = (1 if dir_r > 0 else -1), 0
        elif abs(dir_c) > abs(dir_r) * 2:
            sign_r, sign_c = 0, (1 if dir_c > 0 else -1)
        else:
            sign_r = 1 if dir_r > 0 else -1
            sign_c = 1 if dir_c > 0 else -1
        
        if sign_r != 0 and sign_c != 0:
            step = max(bbox_h, bbox_w) + 1
            step_r, step_c = sign_r * step, sign_c * step
        elif sign_r != 0:
            step_r, step_c = sign_r * (bbox_h + 1), 0
        else:
            step_r, step_c = 0, sign_c * (bbox_w + 1)
        
        nn = 1
        while True:
            cr_n = tcr + step_r * nn
            cc_n = tcc + step_c * nn
            if not (0 <= cr_n < h and 0 <= cc_n < w):
                break
            for cdr, cdc in cross_rel:
                nr, nc = cr_n + cdr, cc_n + cdc
                if 0 <= nr < h and 0 <= nc < w:
                    pred[nr, nc] = o['color']
            nn += 1
    
    return pred.tolist()


# ══════════════════════════════════════════════════════════════
# 汎用3D射影ソルバー
# ══════════════════════════════════════════════════════════════

def projection_solve(train_pairs, test_input):
    """3D射影ベースの汎用ソルバー"""
    from arc.grid import grid_eq
    
    # まず砲台パターンを試す
    result = cannon_solve(train_pairs, test_input)
    if result is not None:
        # train検証
        ok = True
        for inp, out in train_pairs:
            pred = cannon_solve(train_pairs, inp)
            if pred is None or not grid_eq(pred, out):
                ok = False; break
        if ok:
            return result
    
    # 他の3D射影パターンをここに追加...
    
    return None


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json, re
    from pathlib import Path
    from arc.grid import grid_eq
    
    split = 'evaluation' if '--eval' in sys.argv else 'training'
    data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
    
    solved = []
    for tf in sorted(data_dir.glob('*.json')):
        tid = tf.stem
        with open(tf) as f:
            task = json.load(f)
        tp = [(e['input'], e['output']) for e in task['train']]
        ti = task['test'][0]['input']
        to = task['test'][0].get('output')
        
        result = projection_solve(tp, ti)
        if result and to and grid_eq(result, to):
            solved.append(tid)
            print(f'  ✓ {tid}')
    
    total = len(list(data_dir.glob('*.json')))
    print(f'\n{split}: {len(solved)}/{total}')
