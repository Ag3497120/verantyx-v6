#!/usr/bin/env python3
"""Generate a GIF of the ARC eval output, frame by frame."""
from PIL import Image, ImageDraw, ImageFont
import subprocess, re

# Run eval and capture output
result = subprocess.run(
    ['python3', '-u', '-m', 'arc.eval_cross_engine', '--split', 'training', '--limit', '50'],
    capture_output=True, text=True, cwd='/Users/motonishikoudai/verantyx_v6'
)
lines = result.stdout.strip().split('\n')

# Settings
W, H = 900, 600
BG = (15, 15, 25)
GREEN = (80, 220, 100)
RED = (220, 80, 80)
CYAN = (14, 165, 233)
WHITE = (220, 220, 220)
GRAY = (100, 100, 120)

try:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    font_big = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    font_title = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
except:
    font = ImageFont.load_default()
    font_big = font
    font_title = font

frames = []
visible_lines = []
max_visible = 22

for i, line in enumerate(lines):
    visible_lines.append(line)
    if len(visible_lines) > max_visible:
        visible_lines = visible_lines[-max_visible:]
    
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    
    # Title bar
    draw.rectangle([0, 0, W, 45], fill=(20, 20, 35))
    draw.text((20, 10), "⚡ Verantyx V6 — ARC-AGI-2 Eval (84.0%)", fill=CYAN, font=font_title)
    
    # Draw lines
    y = 55
    for vl in visible_lines:
        if '✓' in vl:
            color = GREEN
        elif '✗' in vl:
            color = RED
        elif '✨' in vl:
            color = CYAN
        elif 'Score' in vl:
            color = CYAN
        else:
            color = GRAY
        
        # Clean up the line for display
        display = vl.strip()
        if len(display) > 85:
            display = display[:85] + '...'
        
        draw.text((20, y), display, fill=color, font=font)
        y += 22
    
    # Progress bar at bottom
    progress = (i + 1) / len(lines)
    draw.rectangle([20, H-40, W-20, H-20], outline=GRAY)
    draw.rectangle([20, H-40, 20 + int((W-40) * progress), H-20], fill=CYAN)
    draw.text((W//2 - 30, H-38), f"{progress*100:.0f}%", fill=WHITE, font=font)
    
    frames.append(img)

# Add a final "hold" frame
for _ in range(15):
    frames.append(frames[-1])

# Save GIF
frames[0].save(
    '/Users/motonishikoudai/verantyx_v6/demo.gif',
    save_all=True,
    append_images=frames[1:],
    duration=150,  # ms per frame
    loop=0
)
print(f"Created demo.gif with {len(frames)} frames")
