def transform(grid):
    h = len(grid)
    w = len(grid[0])
    
    # Helper: BFS to get connected component
    def bfs(sr, sc, visited, color):
        component = []
        stack = [(sr, sc)]
        visited[sr][sc] = True
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
        return component
    
    # Find all connected components
    visited = [[False]*w for _ in range(h)]
    color_to_components = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = bfs(r, c, visited, color)
                if color not in color_to_components:
                    color_to_components[color] = []
                color_to_components[color].append(comp)
    
    # Find largest component for each color
    largest_comp = {}
    for color, comps in color_to_components.items():
        largest = max(comps, key=len)
        largest_comp[color] = largest
    
    # Check adjacency between largest components of different colors
    colors = list(largest_comp.keys())
    source_color = None
    target_color = None
    
    for i in range(len(colors)):
        for j in range(len(colors)):
            if i == j:
                continue
            color_a = colors[i]
            color_b = colors[j]
            comp_a = largest_comp[color_a]
            comp_b = largest_comp[color_b]
            
            # Check if any pixel in comp_a is adjacent to any pixel in comp_b
            adjacent = False
            for ra, ca in comp_a:
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = ra + dr, ca + dc
                    if (nr, nc) in comp_b:
                        adjacent = True
                        break
                if adjacent:
                    break
            
            if adjacent:
                source_color = color_a
                target_color = color_b
                break
        if source_color is not None:
            break
    
    # Create output grid (start as copy of input)
    output = [row[:] for row in grid]
    
    if source_color is not None and target_color is not None:
        # Remove source block (set to 0)
        for r, c in largest_comp[source_color]:
            output[r][c] = 0
        # Recolor target block to source_color
        for r, c in largest_comp[target_color]:
            output[r][c] = source_color
    
    return output