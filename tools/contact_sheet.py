"""Build a labelled contact sheet from rendered previews."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

src = Path(sys.argv[1] if len(sys.argv) > 1 else "out/gallery")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/gallery.jpg")
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cell = int(sys.argv[4]) if len(sys.argv) > 4 else 340

order = ["feluda", "portrait", "shonku", "scene", "linocut", "sandesh", "stipple",
         "chapter-head"]
files = []
for name in order:
    p = src / f"{name}.preview.jpg"
    if p.exists():
        files.append((name, p))
for p in sorted(src.glob("*.preview.jpg")):
    name = p.name.replace(".preview.jpg", "")
    if name not in [n for n, _ in files]:
        files.append((name, p))

if not files:
    raise SystemExit(f"no previews found in {src}")

thumbs = []
for name, p in files:
    im = Image.open(p).convert("RGB")
    im.thumbnail((cell, cell), Image.LANCZOS)
    thumbs.append((name, im))

label = 24
pad = 12
cw = max(t.width for _, t in thumbs)
ch = max(t.height for _, t in thumbs)
rows = (len(thumbs) + cols - 1) // cols

sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad + label) + pad),
                  (250, 248, 243))
drw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.load_default(15)
except TypeError:
    font = ImageFont.load_default()

for i, (name, im) in enumerate(thumbs):
    r, c = divmod(i, cols)
    x = pad + c * (cw + pad)
    y = pad + r * (ch + pad + label)
    sheet.paste(im, (x + (cw - im.width) // 2, y))
    drw.text((x + 2, y + ch + 5), name, fill=(70, 64, 58), font=font)

dst.parent.mkdir(parents=True, exist_ok=True)
sheet.save(dst, quality=86, optimize=True)
print(f"wrote {dst} ({sheet.width}x{sheet.height})")
