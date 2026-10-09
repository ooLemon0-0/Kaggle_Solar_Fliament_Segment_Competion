import numpy as np
from PIL import Image, ImageDraw


def colors(count):
    import colorsys
    return [tuple(int(255*c) for c in colorsys.hsv_to_rgb((i*.618033988749895)%1,.8,.95)) for i in range(count)]


def panel(gray, masks, title, annotations=None):
    rgb = np.repeat(np.asarray(gray.convert('L'))[..., None], 3, axis=2).astype(float)
    palette = colors(len(masks))
    for mask, color in zip(masks, palette):
        rgb[mask] = .55 * rgb[mask] + .45 * np.asarray(color)
    canvas = Image.fromarray(np.uint8(rgb))
    draw = ImageDraw.Draw(canvas)
    if annotations:
        for a, color in zip(annotations, palette):
            x, y, w, h = a['bbox']
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
            points = list(zip(a['spine'][::2], a['spine'][1::2]))
            if len(points) > 1:
                draw.line(points, fill=(255, 255, 255), width=2)
    canvas = canvas.resize((512, 512), Image.Resampling.LANCZOS)
    result = Image.new('RGB', (512, 560), 'white')
    result.paste(canvas, (0, 48))
    ImageDraw.Draw(result).text((8, 8), title, fill='black')
    return result


def join_panels(panels, path):
    from pathlib import Path
    image = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), 'white')
    x = 0
    for p in panels:
        image.paste(p, (x, 0))
        x += p.width
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
