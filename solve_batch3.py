import json, os, copy, sys
from collections import defaultdict, deque, Counter

DATA_DIR = "/private/tmp/arc-agi-2/data/training"
OUT_DIR = os.path.expanduser("~/verantyx_v6/synth_results")

def load_task(tid):
    return json.load(open(f"{DATA_DIR}/{tid}.json"))

def test_fn(fn_code, examples):
    ns = {}
    exec(fn_code, ns)
    fn = ns['transform']
    for ex in examples:
        try:
            out = fn(copy.deepcopy(ex['input']))
            if out != ex['output']:
                return False
        except:
            return False
    return True

def save(tid, code):
    with open(os.path.join(OUT_DIR, f"{tid}.py"), 'w') as f:
        f.write(code)
    print(f"  SAVED {tid}")

def try_save(tid, code, task):
    if test_fn(code, task['train']):
        save(tid, code)
        return True
    return False

def debug(tid, code, task):
    ns = {}
    exec(code, ns)
    fn = ns['transform']
    for i, ex in enumerate(task['train'][:1]):
        try:
            out = fn(copy.deepcopy(ex['input']))
            print(f"    Ex{i}: exp={str(ex['output'])[:80]} got={str(out)[:80]}")
        except Exception as e:
            print(f"    Ex{i}: ERROR {e}")


# ===== SOLVERS =====

codes = {}

codes['b94a9452'] = '''
def transform(grid):
    R, C = len(grid), len(grid[0])
    cells = [(r,c,grid[r][c]) for r in range(R) for c in range(C) if grid[r][c]!=0]
    if not cells: return grid
    rows=[r for r,c,v in cells]; cols=[c for r,c,v in cells]
    r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
    border_color=grid[r1][c1]
    inner_color=None
    for r,c,v in cells:
        if r1<r<r2 and c1<c<c2 and v!=border_color:
            inner_color=v; break
    if inner_color is None:
        return [[grid[r][c] for c in range(c1,c2+1)] for r in range(r1,r2+1)]
    out=[]
    for r in range(r1,r2+1):
        row=[]
        for c in range(c1,c2+1):
            v=grid[r][c]
            if v==border_color: row.append(inner_color)
            elif v==inner_color: row.append(border_color)
            else: row.append(v)
        out.append(row)
    return out
'''

codes['beb8660c'] = '''
def transform(grid):
    R, C = len(grid), len(grid[0])
    # Find floor (uniform non-zero row)
    floor_row = None
    for r in range(R-1,-1,-1):
        if len(set(grid[r]))==1 and grid[r][0]!=0:
            floor_row=r; break
    if floor_row is None:
        floor_row=R-1
    floor_color=grid[floor_row][0]
    # Find objects by color
    obj_list=[]
    seen=set()
    for r in range(R):
        for c in range(C):
            v=grid[r][c]
            if v!=0 and v!=floor_color and v not in seen:
                seen.add(v)
                cells=[(rr,cc) for rr in range(R) for cc in range(C) if grid[rr][cc]==v]
                obj_list.append((len(cells),v,cells))
    obj_list.sort()
    result=[[0]*C for _ in range(R)]
    for c in range(C): result[floor_row][c]=floor_color
    cur=floor_row-1
    for size,v,cells in obj_list:
        if cur<0: break
        for i in range(size):
            col=C-1-i
            if 0<=col<C and 0<=cur<R:
                result[cur][col]=v
        cur-=1
    return result
'''

codes['c444b776'] = '''
def transform(grid):
    R, C = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    # Find divider row and col (all-same non-zero)
    div_row=None; div_col=None
    for r in range(R):
        vals=set(grid[r])
        if len(vals)==1 and list(vals)[0]!=0:
            if div_row is None: div_row=r
    for c in range(C):
        vals=set(grid[r][c] for r in range(R))
        if len(vals)==1 and list(vals)[0]!=0:
            if div_col is None: div_col=c
    if div_row is None or div_col is None: return grid
    # 4 quadrants
    quads=[
        (0,div_row-1,0,div_col-1),
        (0,div_row-1,div_col+1,C-1),
        (div_row+1,R-1,0,div_col-1),
        (div_row+1,R-1,div_col+1,C-1),
    ]
    def has_data(q):
        r1,r2,c1,c2=q
        return any(grid[r][c]!=0 for r in range(r1,r2+1) for c in range(c1,c2+1))
    src=None
    for q in quads:
        if has_data(q): src=q; break
    if src is None: return grid
    sr1,sr2,sc1,sc2=src
    src_data=[[grid[r][c] for c in range(sc1,sc2+1)] for r in range(sr1,sr2+1)]
    for q in quads:
        if not has_data(q):
            r1,r2,c1,c2=q
            for i in range(r2-r1+1):
                for j in range(c2-c1+1):
                    if i<len(src_data) and j<len(src_data[0]):
                        result[r1+i][c1+j]=src_data[i][j]
    return result
'''

codes['c97c0139'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    # Horizontal lines
    for r in range(R):
        c2s=[c for c in range(C) if grid[r][c]==2]
        if len(c2s)>=2:
            c1,c2=min(c2s),max(c2s)
            if all(grid[r][c]==2 for c in range(c1,c2+1)):
                L=c2-c1+1
                for d in range(1,L//2+1):
                    for sign in [-1,1]:
                        rr=r+sign*d
                        if 0<=rr<R:
                            for c in range(c1+d,c2-d+1):
                                if 0<=c<C and result[rr][c]==0:
                                    result[rr][c]=8
    # Vertical lines
    for c in range(C):
        r2s=[r for r in range(R) if grid[r][c]==2]
        if len(r2s)>=2:
            r1,r2=min(r2s),max(r2s)
            if all(grid[r][c]==2 for r in range(r1,r2+1)):
                L=r2-r1+1
                for d in range(1,L//2+1):
                    for sign in [-1,1]:
                        cc=c+sign*d
                        if 0<=cc<C:
                            for r in range(r1+d,r2-d+1):
                                if 0<=r<R and result[r][cc]==0:
                                    result[r][cc]=8
    return result
'''

codes['ca8de6ea'] = '''
def transform(grid):
    n=len(grid)
    out=[[0]*3 for _ in range(3)]
    h=n//2
    out[0][0]=grid[0][0]; out[0][2]=grid[0][n-1]
    out[2][0]=grid[n-1][0]; out[2][2]=grid[n-1][n-1]
    out[1][1]=grid[h][h]
    out[0][1]=grid[1][1]; out[1][2]=grid[1][n-2]
    out[1][0]=grid[n-2][1]; out[2][1]=grid[n-2][n-2]
    return out
'''

codes['d2acf2cb'] = '''
def transform(grid):
    result=[]
    for row in grid:
        result.append([6 if v==7 else 0 if v==8 else v for v in row])
    return result
'''

codes['c6e1b8da'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    from collections import defaultdict
    color_cells=defaultdict(list)
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0:
                color_cells[grid[r][c]].append((r,c))
    for v,cells in color_cells.items():
        rows=[r for r,c in cells]; cols=[c for r,c in cells]
        r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                result[r][c]=v
    return result
'''

codes['d06dbe63'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    r8=c8=None
    for r in range(R):
        for c in range(C):
            if grid[r][c]==8: r8,c8=r,c
    if r8 is None: return grid
    # Go UP
    r,c=r8-1,c8
    while 0<=r<R and 0<=c<C:
        result[r][c]=5
        r-=1
        if 0<=r<R:
            for dc in range(3):
                cc=c+dc
                if 0<=cc<C: result[r][cc]=5
            c+=2; r-=1
        else: break
    # Go DOWN
    r,c=r8+1,c8
    while 0<=r<R:
        if 0<=c<C: result[r][c]=5
        r+=1
        if 0<=r<R:
            for dc in range(3):
                cc=c-dc
                if 0<=cc<C: result[r][cc]=5
            c-=2; r+=1
        else: break
    return result
'''

codes['d37a1ef5'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    from collections import deque
    border_color=None
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0: border_color=grid[r][c]; break
        if border_color: break
    if border_color is None: return grid
    visited=[[False]*C for _ in range(R)]
    def find_rect(sr,sc):
        cells=[]
        q=deque([(sr,sc)]); visited[sr][sc]=True
        while q:
            r,c=q.popleft(); cells.append((r,c))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==border_color:
                    visited[nr][nc]=True; q.append((nr,nc))
        return cells
    for r in range(R):
        for c in range(C):
            if grid[r][c]==border_color and not visited[r][c]:
                cells=find_rect(r,c)
                rows=[p[0] for p in cells]; cols=[p[1] for p in cells]
                r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
                per=2*(r2-r1)+2*(c2-c1)
                if len(cells)==per and r2>r1 and c2>c1:
                    for rr in range(r1,r2+1):
                        for cc in range(c1,c2+1):
                            result[rr][cc]=border_color
    return result
'''

codes['d406998b'] = '''
def transform(grid):
    result=[]
    for row in grid:
        result.append([3 if v==5 and c%2==0 else v for c,v in enumerate(row)])
    return result
'''

codes['d4f3cd78'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    fives=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==5]
    if not fives: return grid
    r1=min(r for r,c in fives); r2=max(r for r,c in fives)
    c1=min(c for r,c in fives); c2=max(c for r,c in fives)
    for r in range(r1+1,r2):
        for c in range(c1+1,c2):
            if grid[r][c]==0: result[r][c]=8
    top_5s=set(c for c in range(c1,c2+1) if grid[r1][c]==5)
    gap_cols=sorted(set(range(c1,c2+1))-top_5s)
    if gap_cols:
        gc=gap_cols[len(gap_cols)//2]
        for r in range(0,r1+1):
            result[r][gc]=8
    return result
'''

codes['d5d6de2d'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[[0]*C for _ in range(R)]
    from collections import deque
    visited=[[False]*C for _ in range(R)]
    components=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0 and not visited[r][c]:
                v=grid[r][c]; cells=[]
                q=deque([(r,c)]); visited[r][c]=True
                while q:
                    rr,cc=q.popleft(); cells.append((rr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==v:
                            visited[nr][nc]=True; q.append((nr,nc))
                components.append((v,cells))
    if not components: return grid
    components.sort(key=lambda x:len(x[1]),reverse=True)
    v,cells=components[0]
    rows=[p[0] for p in cells]; cols=[p[1] for p in cells]
    r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
    for r in range(r1+1,r2):
        for c in range(c1+1,c2):
            result[r][c]=3
    return result
'''

codes['d687bc17'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    # Frame: top row 0, bottom R-1, left col 0, right col C-1
    top_c=next((grid[0][c] for c in range(C) if grid[0][c]!=0),None)
    bot_c=next((grid[R-1][c] for c in range(C) if grid[R-1][c]!=0),None)
    lft_c=next((grid[r][0] for r in range(R) if grid[r][0]!=0),None)
    rgt_c=next((grid[r][C-1] for r in range(R) if grid[r][C-1]!=0),None)
    cmap={}
    if top_c: cmap[top_c]="top"
    if bot_c: cmap[bot_c]="bot"
    if lft_c: cmap[lft_c]="lft"
    if rgt_c: cmap[rgt_c]="rgt"
    for r in range(1,R-1):
        for c in range(1,C-1):
            v=grid[r][c]
            if v!=0:
                if v in cmap:
                    s=cmap[v]
                    result[r][c]=0
                    if s=="top": result[0][c]=v
                    elif s=="bot": result[R-1][c]=v
                    elif s=="lft": result[r][0]=v
                    elif s=="rgt": result[r][C-1]=v
                else:
                    result[r][c]=0
    return result
'''

codes['d6ad076f'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    from collections import defaultdict
    cc=defaultdict(list)
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0: cc[grid[r][c]].append((r,c))
    colors=list(cc.keys())
    if len(colors)<2: return grid
    rects=[]
    for v in colors:
        cells=cc[v]
        rows=[r for r,c in cells]; cols=[c for r,c in cells]
        rects.append((v,min(rows),max(rows),min(cols),max(cols)))
    r0=rects[0]; r1=rects[1]
    ov_r1=max(r0[1],r1[1]); ov_r2=min(r0[2],r1[2])
    if r0[4]<r1[3]:
        gc1=r0[4]+1; gc2=r1[3]-1
    elif r1[4]<r0[3]:
        gc1=r1[4]+1; gc2=r0[3]-1
    else:
        return result
    if ov_r1<=ov_r2 and gc1<=gc2:
        for r in range(ov_r1,ov_r2+1):
            for c in range(gc1,gc2+1):
                result[r][c]=8
    return result
'''

codes['d6e50e54'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    # Find the colored region (non-7)
    from collections import defaultdict
    colors=defaultdict(list)
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=7: colors[grid[r][c]].append((r,c))
    # Find 1-region and 9s
    region_color=None
    nines=colors.get(9,[])
    for v in colors:
        if v!=7 and v!=9 and len(colors[v])>1:
            region_color=v; break
    if region_color is None: return grid
    reg_cells=colors[region_color]
    rows=[r for r,c in reg_cells]; cols=[c for r,c in reg_cells]
    r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
    # Replace region with 2
    for r,c in reg_cells:
        result[r][c]=2
    # Move 9s: reflect through nearest boundary
    for r9,c9 in nines:
        result[r9][c9]=0
        # Find closest boundary
        dist_top=r9-r2 if r9>r2 else None
        dist_bot=r1-r9 if r9<r1 else None
        dist_lft=c9-c2 if c9>c2 else None
        dist_rgt=c1-c9 if c9<c1 else None
        # Reflect
        if dist_top is not None and (dist_lft is None or dist_top<=dist_lft):
            new_r=r2-(dist_top-1)
            if r1<=new_r<=r2 and c1<=c9<=c2:
                result[new_r][c9]=9
        elif dist_lft is not None:
            new_c=c2-(dist_lft-1)
            if c1<=new_c<=c2 and r1<=r9<=r2:
                result[r9][new_c]=9
    return result
'''


codes['d22278a0'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[[0]*C for _ in range(R)]
    markers=[]
    for r in range(R):
        for c in range(C):
            if grid[r][c]!=0: markers.append((r,c,grid[r][c]))
    for r0,c0,v in markers:
        result[r0][c0]=v
        # Determine direction based on position
        mid_r,mid_c=R//2,C//2
        if r0<mid_r and c0<mid_c: dr,dc=-1,-1  # top-left corner
        elif r0<mid_r and c0>mid_c: dr,dc=-1,1  # top-right
        elif r0>mid_r and c0<mid_c: dr,dc=1,-1  # bot-left
        else: dr,dc=1,1  # bot-right
        # Invert: draw toward the other corners
        dr,dc=-dr,-dc
        # Staircase: row of width 1,3,5,... at each step
        r,c=r0,c0
        step=0
        while True:
            width=step*2+1
            # Draw bar
            for i in range(width):
                nc=c+(i if dc>0 else -i)
                if 0<=r<R and 0<=nc<C:
                    result[r][nc]=v
            r+=dr
            if r<0 or r>=R: break
            step+=1
    return result
'''

codes['c35c1b4c'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    threes=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==3]
    if not threes: return grid
    r1=min(r for r,c in threes); r2=max(r for r,c in threes)
    c1=min(c for r,c in threes); c2=max(c for r,c in threes)
    for r in range(r1,r2+1):
        for c in range(c1,c2+1):
            result[r][c]=3
    return result
'''

codes['cbded52d'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    sep_rows=[r for r in range(R) if all(grid[r][c]==0 for c in range(C))]
    sep_cols=[c for c in range(C) if all(grid[r][c]==0 for r in range(R))]
    if not sep_rows or not sep_cols: return grid
    row_sec=[]
    prev=0
    for sr in sep_rows:
        if sr>prev: row_sec.append((prev,sr-1))
        prev=sr+1
    if prev<R: row_sec.append((prev,R-1))
    col_sec=[]
    prev=0
    for sc in sep_cols:
        if sc>prev: col_sec.append((prev,sc-1))
        prev=sc+1
    if prev<C: col_sec.append((prev,C-1))
    nr,nc=len(row_sec),len(col_sec)
    # Find dominant bg
    from collections import Counter
    flat=[grid[r][c] for r in range(R) for c in range(C)]
    bg=Counter(flat).most_common(1)[0][0]
    # Find special values per cell
    from collections import defaultdict
    val_pos=defaultdict(list)
    for ri,(r1,r2) in enumerate(row_sec):
        for ci,(c1,c2) in enumerate(col_sec):
            for r in range(r1,r2+1):
                for c in range(c1,c2+1):
                    v=grid[r][c]
                    if v!=bg and v!=0:
                        ir,ic=r-r1,c-c1
                        val_pos[v].append((ri,ci,ir,ic))
    # Propagate
    for v,positions in val_pos.items():
        row_ct=Counter(ri for ri,ci,ir,ic in positions)
        col_ct=Counter(ci for ri,ci,ir,ic in positions)
        for ri,cnt in row_ct.items():
            if cnt>=2:
                pts=[(ir,ic) for pli,pci,ir,ic in positions if pli==ri]
                ir0,ic0=pts[0]
                for ci2 in range(nc):
                    c1,c2=col_sec[ci2]; r1,r2=row_sec[ri]
                    if r1+ir0<=r2 and c1+ic0<=c2:
                        result[r1+ir0][c1+ic0]=v
        for ci,cnt in col_ct.items():
            if cnt>=2:
                pts=[(ir,ic) for pli,pci,ir,ic in positions if pci==ci]
                ir0,ic0=pts[0]
                for ri2 in range(nr):
                    c1,c2=col_sec[ci]; r1,r2=row_sec[ri2]
                    if r1+ir0<=r2 and c1+ic0<=c2:
                        result[r1+ir0][c1+ic0]=v
    return result
'''

codes['cc9053aa'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    nines=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==9]
    eights=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==8]
    if not eights or not nines: return grid
    r1=min(r for r,c in eights); r2=max(r for r,c in eights)
    c1=min(c for r,c in eights); c2=max(c for r,c in eights)
    for r9,c9 in nines:
        if r9<r1 or r9>r2:
            if r9<r1:
                for c in range(c1,c2+1): result[r1][c]=9
                if c1<=c9<=c2:
                    for r in range(r9,r1): result[r][c9]=9
            else:
                for c in range(c1,c2+1): result[r2][c]=9
        if c9<c1 or c9>c2:
            if c9<c1:
                for r in range(r1,r2+1): result[r][c1]=9
            else:
                for r in range(r1,r2+1): result[r][c2]=9
    return result
'''

codes['c920a713'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    from collections import deque
    visited=[[False]*C for _ in range(R)]
    frames=[]
    for r in range(R):
        for c in range(C):
            v=grid[r][c]
            if v==0 or visited[r][c]: continue
            comp=[]
            q=deque([(r,c)]); visited[r][c]=True
            while q:
                rr,cc=q.popleft(); comp.append((rr,cc))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=rr+dr,cc+dc
                    if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==v:
                        visited[nr][nc]=True; q.append((nr,nc))
            rows=[p[0] for p in comp]; cols=[p[1] for p in comp]
            r1,r2,c1,c2=min(rows),max(rows),min(cols),max(cols)
            is_rect=all(p[0] in (r1,r2) or p[1] in (c1,c2) for p in comp)
            if is_rect and r2>r1 and c2>c1:
                frames.append((v,r1,c1,r2,c2))
    if not frames: return grid
    frames.sort(key=lambda f:(f[3]-f[1])*(f[4]-f[2]),reverse=True)
    n=len(frames)
    sz=2*n+1
    out=[[0]*sz for _ in range(sz)]
    for i,(v,r1,c1,r2,c2) in enumerate(frames):
        for c in range(i,sz-i): out[i][c]=v; out[sz-1-i][c]=v
        for r in range(i,sz-i): out[r][i]=v; out[r][sz-1-i]=v
    return out
'''

codes['baf41dbf'] = '''
def transform(grid):
    R,C=len(grid),len(grid[0])
    result=[row[:] for row in grid]
    sixes=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==6]
    box_color=None
    for r in range(R):
        for c in range(C):
            if grid[r][c] not in (0,6): box_color=grid[r][c]; break
        if box_color: break
    if box_color is None or not sixes: return grid
    box_cells=[(r,c) for r in range(R) for c in range(C) if grid[r][c]==box_color]
    br1=min(r for r,c in box_cells); br2=max(r for r,c in box_cells)
    bc1=min(c for r,c in box_cells); bc2=max(c for r,c in box_cells)
    top_full=all(grid[br1][c]==box_color for c in range(bc1,bc2+1))
    bot_full=all(grid[br2][c]==box_color for c in range(bc1,bc2+1))
    lft_full=all(grid[r][bc1]==box_color for r in range(br1,br2+1))
    rgt_full=all(grid[r][bc2]==box_color for r in range(br1,br2+1))
    for sr,sc in sixes:
        if not top_full and sr<br1:
            for c in range(bc1,bc2+1): result[sr][c]=box_color
            for r in range(sr+1,br1): result[r][bc1]=box_color; result[r][bc2]=box_color
        elif not bot_full and sr>br2:
            for c in range(bc1,bc2+1): result[sr][c]=box_color
            for r in range(br2+1,sr): result[r][bc1]=box_color; result[r][bc2]=box_color
        elif not lft_full and sc<bc1:
            for r in range(br1,br2+1): result[r][sc]=box_color
            for c in range(sc+1,bc1): result[br1][c]=box_color; result[br2][c]=box_color
        elif not rgt_full and sc>bc2:
            for r in range(br1,br2+1): result[r][sc]=box_color
            for c in range(bc2+1,sc): result[br1][c]=box_color; result[br2][c]=box_color
    return result
'''


missing = ["b94a9452", "baf41dbf", "bb52a14b", "bc93ec48", "bd14c3bf", "bd283c4a", "bd5af378", "beb8660c", "c3202e5a", "c35c1b4c", "c3fa4749", "c444b776", "c4d1a9ae", "c6141b15", "c61be7dc", "c62e2108", "c64f1187", "c658a4bd", "c6e1b8da", "c803e39c", "c87289bb", "c8b7cc0f", "c8cbb738", "c920a713", "c92b942c", "c97c0139", "ca8de6ea", "cbded52d", "cc9053aa", "cdecee7f", "ce039d91", "ce602527", "cf133acc", "cf98881b", "cfb2ce5a", "d017b73f", "d06dbe63", "d07ae81c", "d22278a0", "d23f8c26", "d255d7a7", "d2abd087", "d2acf2cb", "d304284e", "d37a1ef5", "d406998b", "d43fd935", "d4469b4b", "d47aa2ff", "d492a647", "d4c90558", "d4f3cd78", "d56f2372", "d5c634a2", "d5d6de2d", "d6542281", "d687bc17", "d6ad076f", "d6e50e54", "d749d46f", "d753a70b", "d89b689b", "d8c310e9", "d90796e8", "d931c21c", "d93c6891", "d94c3b52", "d968ffd4", "d9f24cd1", "da2b0fe3", "da6e95e5", "db118e2a"]

saved=0; failed=[]
for tid in missing:
    if tid not in codes: continue
    task=load_task(tid)
    code=codes[tid]
    print(f"Testing {tid}...")
    if test_fn(code, task['train']):
        save(tid, code); saved+=1
    else:
        failed.append(tid)
        print(f"  FAILED: {tid}")
        debug(tid, code, task)

print(f"\nSaved: {saved}, Failed: {failed}")
