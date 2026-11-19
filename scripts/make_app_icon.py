"""Generate app icons for RiskGuard (layered background/foreground + startIcon).

Usage: python scripts/make_app_icon.py
Writes:
  harmony/AppScope/resources/base/media/background.png   (1024x1024 gradient)
  harmony/AppScope/resources/base/media/foreground.png   (1024x1024 shield, transparent)
  harmony/entry/src/main/resources/base/media/startIcon.png (512x512 composed)
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
APPSCOPE_MEDIA = ROOT / 'harmony/AppScope/resources/base/media'
ENTRY_MEDIA = ROOT / 'harmony/entry/src/main/resources/base/media'

SIZE = 1024
SS = 2  # supersampling


def vertical_gradient(size, top, bottom):
    img = Image.new('RGB', (1, size))
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        px[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return img.resize((size, size), Image.BILINEAR)


def shield_points(cx, cy, w, h):
    # Shield: rounded top, tapering to a point at the bottom.
    left, right = cx - w / 2, cx + w / 2
    top = cy - h / 2
    bottom = cy + h / 2
    return [
        (left, top + h * 0.12),
        (left + w * 0.08, top + h * 0.04),
        (cx, top),
        (right - w * 0.08, top + h * 0.04),
        (right, top + h * 0.12),
        (right - w * 0.02, top + h * 0.55),
        (cx, bottom),
        (left + w * 0.02, top + h * 0.55),
    ]


def draw_foreground(size):
    img = Image.new('RGBA', (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size * SS
    cx, cy = s / 2, s / 2
    w, h = s * 0.52, s * 0.62
    pts = shield_points(cx, cy, w, h)
    # soft shadow
    shadow = [(x + s * 0.012, y + s * 0.020) for (x, y) in pts]
    d.polygon(shadow, fill=(0, 0, 0, 70))
    # shield body: white with slight blue tint
    d.polygon(pts, fill=(255, 255, 255, 255))
    # inner shield
    inner = shield_points(cx, cy, w * 0.72, h * 0.72)
    d.polygon(inner, fill=(10, 61, 145, 255))
    # check mark inside inner shield
    lw = int(s * 0.055)
    check = [
        (cx - w * 0.20, cy - h * 0.02),
        (cx - w * 0.05, cy + h * 0.14),
        (cx + w * 0.24, cy - h * 0.20),
    ]
    d.line(check, fill=(255, 255, 255, 255), width=lw, joint='curve')
    # round the line ends
    for (x, y) in (check[0], check[-1]):
        r = lw / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
    return img.resize((size, size), Image.LANCZOS)


def main():
    APPSCOPE_MEDIA.mkdir(parents=True, exist_ok=True)
    ENTRY_MEDIA.mkdir(parents=True, exist_ok=True)

    bg = vertical_gradient(SIZE, (8, 32, 84), (10, 89, 247))
    # subtle diagonal highlight
    glow = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-SIZE * 0.25, -SIZE * 0.55, SIZE * 0.9, SIZE * 0.45],
               fill=(90, 150, 255, 80))
    bg = Image.alpha_composite(bg.convert('RGBA'), glow).convert('RGB')
    bg.save(APPSCOPE_MEDIA / 'background.png')

    fg = draw_foreground(SIZE)
    fg.save(APPSCOPE_MEDIA / 'foreground.png')

    # composed start icon (rounded-square canvas)
    icon = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    canvas = vertical_gradient(512, (8, 32, 84), (10, 89, 247)).convert('RGBA')
    mask = Image.new('L', (512, 512), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, 511, 511], radius=112, fill=255)
    icon.paste(canvas, (0, 0), mask)
    fg_small = fg.resize((330, 330), Image.LANCZOS)
    icon.alpha_composite(fg_small, ((512 - 330) // 2, (512 - 330) // 2))
    icon.save(ENTRY_MEDIA / 'startIcon.png')

    print('wrote icons:', APPSCOPE_MEDIA / 'background.png',
          APPSCOPE_MEDIA / 'foreground.png', ENTRY_MEDIA / 'startIcon.png')


if __name__ == '__main__':
    main()
