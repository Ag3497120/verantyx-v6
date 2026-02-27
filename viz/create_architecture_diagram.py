"""
Create Verantyx architecture diagram for technical promotion.
Shows the pure symbolic pipeline: Probe → Detect → Reason → Verify → Solve
"""

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 700
BG = (25, 25, 35)
BOX_COLORS = {
    'probe': (45, 120, 200),
    'detect': (200, 80, 45),
    'reason': (45, 180, 80),
    'verify': (180, 150, 40),
    'solve': (160, 50, 180),
}
TEXT_COLOR = (240, 240, 240)
DIM_TEXT = (160, 160, 170)
ARROW_COLOR = (100, 200, 100)


def draw_box(draw, x, y, w, h, color, title, items, font, font_sm):
    # Shadow
    draw.rounded_rectangle([x+3, y+3, x+w+3, y+h+3], radius=10, fill=(15, 15, 20))
    # Box
    draw.rounded_rectangle([x, y, x+w, y+h], radius=10, fill=color, outline=(255, 255, 255, 80), width=1)
    # Title
    draw.text((x + 12, y + 8), title, fill=TEXT_COLOR, font=font)
    # Items
    for i, item in enumerate(items):
        draw.text((x + 15, y + 35 + i * 18), f"• {item}", fill=(220, 220, 230), font=font_sm)


def draw_arrow(draw, x1, y1, x2, y2, label="", font=None):
    draw.line([(x1, y1), (x2, y2)], fill=ARROW_COLOR, width=2)
    # Arrowhead
    if x2 > x1:  # right arrow
        draw.polygon([(x2, y2), (x2-8, y2-5), (x2-8, y2+5)], fill=ARROW_COLOR)
    elif y2 > y1:  # down arrow
        draw.polygon([(x2, y2), (x2-5, y2-8), (x2+5, y2-8)], fill=ARROW_COLOR)
    if label and font:
        draw.text(((x1+x2)//2 - 20, min(y1, y2) - 18), label, fill=DIM_TEXT, font=font)


def create_diagram():
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except:
        font = font_lg = font_sm = font_title = ImageFont.load_default()
    
    # Title
    draw.text((30, 20), "Verantyx V6 -- Pure Symbolic Architecture", fill=TEXT_COLOR, font=font_title)
    draw.text((30, 60), "No LLM • No Neural Networks • No Training • Just Logic", fill=DIM_TEXT, font=font)
    
    # Pipeline boxes
    bw, bh = 190, 180
    y_top = 120
    gap = 25
    
    # Box 1: Cross Probe
    x1 = 30
    draw_box(draw, x1, y_top, bw, bh, BOX_COLORS['probe'],
            "1. Cross Probe",
            ["6-axis measurement", "Multi-scale scan", "z=0: pixel scale", "z=1: object scale", "z=2: panel scale",
             "Reach + hit detection"],
            font, font_sm)
    
    # Box 2: Object Detection
    x2 = x1 + bw + gap
    draw_box(draw, x2, y_top, bw, bh, BOX_COLORS['detect'],
            "2. Detect Objects",
            ["Connected components", "Bounding box fit", "Convex/concave faces", "Color + area match",
             "Boundary profiling", "Interlock detection"],
            font, font_sm)
    
    # Box 3: Rule Reasoning
    x3 = x2 + bw + gap
    draw_box(draw, x3, y_top, bw, bh, BOX_COLORS['reason'],
            "3. Rule Reasoning",
            ["65+ puzzle rules", "Neighborhood rules", "Cross3D transforms", "Corner stacking",
             "Gravity simulation", "Compositional DSL"],
            font, font_sm)
    
    # Box 4: CEGIS Verify
    x4 = x3 + bw + gap
    draw_box(draw, x4, y_top, bw, bh, BOX_COLORS['verify'],
            "4. CEGIS Verify",
            ["ALL train examples", "Exact match required", "Partial scoring", "Backtrack on fail",
             "Fixed-point iteration", "Cross-validation"],
            font, font_sm)
    
    # Box 5: Solve
    x5 = x4 + bw + gap
    draw_box(draw, x5, y_top, bw, bh, BOX_COLORS['solve'],
            "5. Output",
            ["Apply verified rule", "Generate output grid", "Multi-hypothesis", "Composition chains",
             "228/1000 solved", "Avg 0.65s / task"],
            font, font_sm)
    
    # Forward arrows
    for xa, xb in [(x1+bw, x2), (x2+bw, x3), (x3+bw, x4), (x4+bw, x5)]:
        mid_y = y_top + bh // 2
        draw_arrow(draw, xa+5, mid_y, xb-5, mid_y, font=font_sm)
    
    # CEGIS feedback loop (verify → reason)
    loop_y = y_top + bh + 10
    loop_color = (255, 100, 100)
    draw.line([(x4 + bw//2, y_top + bh), (x4 + bw//2, loop_y)], fill=loop_color, width=2)
    draw.line([(x4 + bw//2, loop_y), (x3 + bw//2, loop_y)], fill=loop_color, width=2)
    draw.line([(x3 + bw//2, loop_y), (x3 + bw//2, y_top + bh)], fill=loop_color, width=2)
    draw.polygon([(x3 + bw//2, y_top + bh), (x3 + bw//2 - 6, y_top + bh + 10), (x3 + bw//2 + 6, y_top + bh + 10)], fill=loop_color)
    draw.text(((x3 + x4)//2 + bw//4 - 30, loop_y + 2), "BACKTRACK ON FAIL", fill=loop_color, font=font)
    
    # Bottom section: Key insight
    y_bottom = y_top + bh + 40
    draw.line([(30, y_bottom), (W-30, y_bottom)], fill=(60, 60, 70), width=1)
    
    # Key stats
    draw.text((30, y_bottom + 15), "Key Insight:", fill=ARROW_COLOR, font=font)
    draw.text((30, y_bottom + 40), 
             "The 6-axis cross probe acts as a universal measurement tool — like inserting a ruler", 
             fill=DIM_TEXT, font=font)
    draw.text((30, y_bottom + 60), 
             "into the grid to detect object boundaries, movement patterns, and structural rules.",
             fill=DIM_TEXT, font=font)
    
    # Stats bar
    y_stats = y_bottom + 100
    stats = [
        ("228/1000", "Tasks Solved"),
        ("22.8%", "ARC-AGI-2 Score"),
        ("0.65s", "Avg (M4 MacBook)"),
        ("0", "LLMs Used"),
        ("65+", "Symbolic Rules"),
    ]
    
    stat_w = (W - 60) // len(stats)
    for i, (val, label) in enumerate(stats):
        sx = 30 + i * stat_w
        draw.text((sx, y_stats), val, fill=TEXT_COLOR, font=font_lg)
        draw.text((sx, y_stats + 35), label, fill=DIM_TEXT, font=font_sm)
    
    # Footer
    draw.text((30, H - 30), "github.com/Ag3497120/verantyx-v6 | Built by kofdai × OpenClaw", 
             fill=(80, 80, 90), font=font_sm)
    
    out_path = '/Users/motonishikoudai/verantyx_v6/viz/architecture_diagram.png'
    img.save(out_path, quality=95)
    print(f'Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    create_diagram()
