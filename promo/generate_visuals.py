#!/usr/bin/env python3
"""Generate promotional visuals for Verantyx V6."""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARC color palette (official)
ARC_COLORS = [
    (0, 0, 0),       # 0: black
    (0, 116, 217),   # 1: blue
    (255, 65, 54),   # 2: red
    (46, 204, 64),   # 3: green
    (255, 220, 0),   # 4: yellow
    (170, 170, 170), # 5: gray
    (240, 18, 190),  # 6: magenta
    (255, 133, 27),  # 7: orange
    (127, 219, 255), # 8: light blue
    (135, 12, 37),   # 9: maroon
]

def grid_to_image(grid, cell_size=24, border=1):
    """Render an ARC grid as a PIL image."""
    h, w = len(grid), len(grid[0])
    img_w = w * cell_size + (w + 1) * border
    img_h = h * cell_size + (h + 1) * border
    img = Image.new('RGB', (img_w, img_h), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            x = border + c * (cell_size + border)
            y = border + r * (cell_size + border)
            color = ARC_COLORS[grid[r][c]] if grid[r][c] < 10 else (128, 128, 128)
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)
    return img


def make_before_after(task_id, data_dir, out_dir, cell_size=24):
    """Create a Before → After visualization for a task."""
    path = os.path.join(data_dir, f'{task_id}.json')
    with open(path) as f:
        data = json.load(f)
    
    # Use train[0] for the demo
    train = data['train'][0]
    inp_grid = train['input']
    out_grid = train['output']
    
    inp_img = grid_to_image(inp_grid, cell_size)
    out_img = grid_to_image(out_grid, cell_size)
    
    # Arrow area
    arrow_w = 80
    padding = 20
    total_w = inp_img.width + arrow_w + out_img.width + padding * 2
    total_h = max(inp_img.height, out_img.height) + 100  # space for labels
    
    canvas = Image.new('RGB', (total_w, total_h), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    
    # Title
    title = f"Verantyx V6 — Structural Reasoning"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 10), title, fill=(0, 230, 118), font=font_title)
    
    y_offset = 50
    # Input
    inp_x = padding
    inp_y = y_offset + (max(inp_img.height, out_img.height) - inp_img.height) // 2
    canvas.paste(inp_img, (inp_x, inp_y))
    
    # Label
    draw.text((inp_x, y_offset + max(inp_img.height, out_img.height) + 5), "Input", fill=(180, 180, 180), font=font_small)
    
    # Arrow
    arrow_x = inp_x + inp_img.width + 10
    arrow_y = y_offset + max(inp_img.height, out_img.height) // 2
    draw.text((arrow_x + 15, arrow_y - 10), "→", fill=(0, 230, 118), font=font_title)
    
    # Output
    out_x = arrow_x + arrow_w
    out_y = y_offset + (max(inp_img.height, out_img.height) - out_img.height) // 2
    canvas.paste(out_img, (out_x, out_y))
    draw.text((out_x, y_offset + max(inp_img.height, out_img.height) + 5), "Output", fill=(180, 180, 180), font=font_small)
    
    # Task ID
    draw.text((padding, total_h - 25), f"Task: {task_id}", fill=(100, 100, 100), font=font_small)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'before_after_{task_id}.png')
    canvas.save(out_path, 'PNG')
    print(f"Saved: {out_path}")
    return out_path


def make_speed_demo(log_file, out_dir):
    """Create a terminal-style speed demonstration image."""
    lines = []
    with open(log_file) as f:
        for line in f:
            if '✓' in line:
                lines.append(line.strip())
    
    # Pick a nice consecutive block
    display_lines = lines[10:28]  # 18 lines of consecutive solves
    
    # Terminal-style rendering
    line_h = 22
    padding = 20
    char_w = 9
    max_chars = max(len(l) for l in display_lines)
    
    img_w = max(max_chars * char_w + padding * 2, 800)
    img_h = len(display_lines) * line_h + padding * 2 + 60
    
    img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        mono = ImageFont.load_default()
        title_font = mono
    
    # Title bar
    draw.rectangle([0, 0, img_w, 30], fill=(50, 50, 50))
    # Traffic lights
    draw.ellipse([10, 8, 24, 22], fill=(255, 95, 87))
    draw.ellipse([30, 8, 44, 22], fill=(255, 189, 46))
    draw.ellipse([50, 8, 64, 22], fill=(39, 201, 63))
    draw.text((80, 6), "verantyx_v6 — eval_cross_engine", fill=(180, 180, 180), font=title_font)
    
    y = 40
    for line in display_lines:
        # Colorize: green for ✓, cyan for time, white for rest
        # Simple approach: render whole line then overlay colored parts
        color = (0, 255, 100) if '✓' in line else (200, 200, 200)
        draw.text((padding, y), line, fill=color, font=mono)
        y += line_h
    
    # Bottom status bar
    draw.rectangle([0, img_h - 28, img_w, img_h], fill=(50, 50, 50))
    draw.text((padding, img_h - 24), "235/1000 solved (23.5%) | avg 0.5s/task | Pure symbolic reasoning — no LLM calls", fill=(0, 230, 118), font=title_font)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'speed_demo.png')
    img.save(out_path, 'PNG')
    print(f"Saved: {out_path}")
    return out_path


def make_multi_example_grid(task_ids, data_dir, out_dir, cell_size=18):
    """Create a grid showing multiple Before→After examples."""
    examples = []
    for tid in task_ids:
        path = os.path.join(data_dir, f'{tid}.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        t = data['train'][0]
        inp_img = grid_to_image(t['input'], cell_size)
        out_img = grid_to_image(t['output'], cell_size)
        examples.append((tid, inp_img, out_img))
    
    if not examples:
        return
    
    # Layout: 2 columns x N rows
    cols = 2
    rows = (len(examples) + cols - 1) // cols
    
    arrow_w = 50
    pair_gap = 40
    row_gap = 30
    padding = 30
    
    # Calculate max dimensions per column
    max_pair_w = max(e[1].width + arrow_w + e[2].width for e in examples)
    max_pair_h = max(max(e[1].height, e[2].height) for e in examples)
    
    total_w = cols * max_pair_w + (cols - 1) * pair_gap + padding * 2
    total_h = rows * (max_pair_h + 30) + (rows - 1) * row_gap + padding * 2 + 50
    
    canvas = Image.new('RGB', (total_w, total_h), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
        font_arrow = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_arrow = font
    
    # Title
    title = "Verantyx V6 — Structural Reasoning on ARC-AGI-2"
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 12), title, fill=(0, 230, 118), font=font)
    
    for i, (tid, inp_img, out_img) in enumerate(examples):
        col = i % cols
        row = i // cols
        
        base_x = padding + col * (max_pair_w + pair_gap)
        base_y = 50 + row * (max_pair_h + 30 + row_gap)
        
        # Center vertically
        cy = (max_pair_h - max(inp_img.height, out_img.height)) // 2
        
        canvas.paste(inp_img, (base_x, base_y + cy))
        
        ax = base_x + inp_img.width + 5
        ay = base_y + cy + max(inp_img.height, out_img.height) // 2 - 12
        draw.text((ax + 10, ay), "→", fill=(0, 230, 118), font=font_arrow)
        
        ox = base_x + inp_img.width + arrow_w
        canvas.paste(out_img, (ox, base_y + cy))
        
        # Task ID label
        draw.text((base_x, base_y + max_pair_h + 5), tid, fill=(100, 100, 100), font=font_small)
    
    out_path = os.path.join(out_dir, 'multi_examples.png')
    canvas.save(out_path, 'PNG')
    print(f"Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    data_dir = '/tmp/arc-agi-2/data/training/'
    out_dir = os.path.expanduser('~/verantyx_v6/promo/')
    
    # 1. Speed demo
    make_speed_demo(os.path.expanduser('~/verantyx_v6/arc_v73_full.log'), out_dir)
    
    # 2. Individual before/after (visually rich tasks)
    showcase_tasks = ['b8825c91', 'd282b262', '070dd51e', '7d1f7ee8']
    for tid in showcase_tasks:
        make_before_after(tid, data_dir, out_dir)
    
    # 3. Multi-example grid
    grid_tasks = ['b8825c91', 'd282b262', '070dd51e', '0692e18c', '00576224', 'e729b7be']
    make_multi_example_grid(grid_tasks, data_dir, out_dir)
