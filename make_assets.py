"""
Генератор картинок для titanic_gui.py (нужен Pillow: pip install pillow).

    python make_assets.py путь/к/фото_титаника.jpg

Создаёт в папке assets/ три файла:
    titanic_splash.png        заставка (газетная вырезка с виньеткой и потёртостями)
    titanic_splash_hover.png  то же, но корабль подсвечен (показывается при наведении)
    paper_bg.png              состаренная бумага — фон окна расчёта

Силу эффектов можно менять в блоке НАСТРОЙКИ ниже и просто запустить скрипт снова.
Окно программы — 900x700, поэтому все картинки именно такого размера.
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

# ----------------------------- НАСТРОЙКИ -----------------------------
W, H = 900, 700

PAPER_COLOR = (222, 202, 160)   # основной цвет бумаги окна расчёта
BURN_COLOR = (46, 28, 14)       # цвет «подпалин» по краям и виньетки

VIGNETTE_SPLASH = 0.62          # сила виньетки на заставке   (0 — нет, 1 — сильная)
VIGNETTE_PAPER = 0.80           # сила виньетки на бумаге
GRAIN_SPLASH = 0.16             # зернистость заставки
GRAIN_PAPER = 0.10              # зернистость бумаги
DUST_SPECKS = 300               # количество пылинок и царапин на заставке
FOLD_STRENGTH = 0.55            # заметность сгибов (0 — нет)
# ---------------------------------------------------------------------

# Контур корабля на заставке (доли от ширины/высоты) — по нему определяется клик.
# Эти же числа должны совпадать с SHIP_OUTLINE в titanic_gui.py.
SHIP_OUTLINE = [
    (0.200, 0.589), (0.306, 0.493), (0.314, 0.243), (0.383, 0.226), (0.428, 0.307),
    (0.424, 0.200), (0.489, 0.183), (0.533, 0.264), (0.513, 0.146), (0.583, 0.143),
    (0.606, 0.214), (0.598, 0.137), (0.664, 0.134), (0.728, 0.214), (0.844, 0.179),
    (0.924, 0.179), (0.958, 0.271), (0.950, 0.300), (0.889, 0.346), (0.778, 0.436),
    (0.667, 0.531), (0.556, 0.646), (0.444, 0.717), (0.333, 0.717), (0.239, 0.650),
]


# ------------------------- вспомогательные слои ------------------------
def noise_layer(size, sigma=60, blur=0.0):
    layer = Image.effect_noise(size, sigma)
    return layer.filter(ImageFilter.GaussianBlur(blur)) if blur else layer


def darken_by(image, gain_l):
    """Умножает картинку на серый слой (255 — без изменений, меньше — темнее)."""
    return ImageChops.multiply(image, Image.merge("RGB", (gain_l,) * 3))


def grain(image, amount, seed):
    """Мелкое плёночное зерно: и светлые, и тёмные точки."""
    fine = Image.effect_noise(image.size, 38)
    fine = fine.point(lambda v: int(255 - amount * abs(v - 128) * 2.6))
    out = darken_by(image, fine)
    light = Image.effect_noise(image.size, 38).point(
        lambda v: int(max(0, v - 150) * amount * 5)
    )
    return ImageChops.add(out, Image.merge("RGB", (light,) * 3))


def mottling(image, strength):
    """Крупные неравномерные пятна — бумага выгорает местами (два масштаба шума)."""
    def octave(div, blur):
        small = Image.effect_noise((W // div + 2, H // div + 2), 26)
        return small.resize(image.size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(blur))

    big, mid = octave(70, 20), octave(24, 8)
    mixed = Image.blend(big, mid, 0.35)
    gain = mixed.point(lambda v: int(255 - strength * 5 * max(0, 132 - v)))
    return darken_by(image, gain)


def vignette(image, strength):
    radial = Image.radial_gradient("L").resize(image.size, Image.BICUBIC)
    mask = radial.point(lambda v: int(((v / 255) ** 2.3) * 255 * strength))
    burn = Image.new("RGB", image.size, BURN_COLOR)
    return Image.composite(burn, image, mask)


def edge_burn(image, strength, seed):
    """Потемнение по краям: плавный спад с неровной, «рваной» границей."""
    inset = 14
    mask = Image.new("L", image.size, 255)
    ImageDraw.Draw(mask).rectangle([inset, inset, W - inset, H - inset], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(30))
    wobble = Image.effect_noise((W // 14 + 2, H // 14 + 2), 34).resize(image.size, Image.BICUBIC)
    wobble = wobble.filter(ImageFilter.GaussianBlur(3))
    mask = ImageChops.add(mask, wobble, scale=1.0, offset=-128)
    mask = mask.point(lambda v: int(min(255, max(0, (v - 40) * 1.5)) * strength))
    return Image.composite(Image.new("RGB", image.size, BURN_COLOR), image, mask)


def creases(image, strength, seed):
    """Лёгкие сгибы: горизонтальный посередине и вертикальный — тёмная и светлая линии."""
    rnd = random.Random(seed)
    dark = Image.new("L", image.size, 255)
    light = Image.new("L", image.size, 0)
    dd, dl = ImageDraw.Draw(dark), ImageDraw.Draw(light)

    def wobbly(points_count, horizontal, pos):
        pts = []
        length = W if horizontal else H
        for i in range(points_count + 1):
            t = int(length * i / points_count)
            jitter = rnd.randint(-2, 2)
            pts.append((t, pos + jitter) if horizontal else (pos + jitter, t))
        return pts

    for horizontal, pos in ((True, H // 2 + 6), (False, W // 2 - 4)):
        base = wobbly(30, horizontal, pos)
        dd.line(base, fill=int(255 - 52 * strength), width=2)
        shifted = [(x + (0 if horizontal else 3), y + (3 if horizontal else 0)) for x, y in base]
        dl.line(shifted, fill=int(46 * strength), width=2)

    dark = dark.filter(ImageFilter.GaussianBlur(1.2))
    light = light.filter(ImageFilter.GaussianBlur(1.6))
    out = darken_by(image, dark)
    return ImageChops.add(out, Image.merge("RGB", (light,) * 3))


def dust_and_scratches(image, count, seed):
    rnd = random.Random(seed)
    layer = Image.new("L", image.size, 128)
    draw = ImageDraw.Draw(layer)
    for _ in range(count):
        x, y = rnd.randrange(W), rnd.randrange(H)
        if rnd.random() < 0.93:
            r = rnd.choice((0.6, 0.8, 1.0, 1.4, 2.0))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=rnd.choice((40, 60, 205, 225)))
        else:
            dx, dy = rnd.randint(-18, 18), rnd.randint(-18, 18)
            draw.line([x, y, x + dx, y + dy], fill=rnd.choice((70, 190)), width=1)
    layer = layer.filter(ImageFilter.GaussianBlur(0.5))
    dark = layer.point(lambda v: 255 if v >= 128 else int(255 - (128 - v) * 1.1))
    light = layer.point(lambda v: int((v - 128) * 1.1) if v > 128 else 0)
    out = darken_by(image, dark)
    return ImageChops.add(out, Image.merge("RGB", (light,) * 3))


# ------------------------------- картинки ------------------------------
def make_splash(source_path, seed=1912):
    src = Image.open(source_path).convert("RGB")
    sw, sh = src.size
    crop_h = round(sw * H / W)           # берём нижнюю часть кадра, где весь корабль
    top = max(0, sh - crop_h)
    img = src.crop((0, top, sw, sh)).resize((W, H), Image.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = mottling(img, 0.14)
    img = grain(img, GRAIN_SPLASH, seed)
    img = creases(img, FOLD_STRENGTH, seed)
    img = dust_and_scratches(img, DUST_SPECKS, seed)
    img = edge_burn(img, 0.60, seed)
    img = vignette(img, VIGNETTE_SPLASH)

    # дополнительное затемнение снизу — под подпись «нажмите на лайнер»
    shade = Image.linear_gradient("L").resize((W, H))      # чёрный сверху → белый снизу
    shade = shade.point(lambda v: int(max(0, v - 150) * 1.1))
    img = Image.composite(Image.new("RGB", (W, H), BURN_COLOR), img, shade)
    return img


def make_hover(splash):
    """Версия заставки с мягко подсвеченным кораблём (для наведения курсора)."""
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(x * W, y * H) for x, y in SHIP_OUTLINE], fill=255)
    soft = mask.filter(ImageFilter.GaussianBlur(14))

    lit = ImageEnhance.Brightness(splash).enhance(1.10)
    lit = ImageChops.add(lit, Image.new("RGB", (W, H), (34, 22, 10)))   # тёплый подъём теней
    out = Image.composite(lit, splash, soft)

    glow = mask.filter(ImageFilter.GaussianBlur(28)).point(lambda v: int(v * 0.30))
    glow_rgb = Image.merge("RGB", (
        glow.point(lambda v: int(v * 0.95)),
        glow.point(lambda v: int(v * 0.70)),
        glow.point(lambda v: int(v * 0.35)),
    ))
    return ImageChops.add(out, glow_rgb)


def make_paper(seed=1912):
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), PAPER_COLOR)
    img = mottling(img, 0.16)

    # бумажные волокна
    fibers = Image.new("L", (W, H), 255)
    fd = ImageDraw.Draw(fibers)
    for _ in range(1500):
        x, y = rnd.randrange(W), rnd.randrange(H)
        dx, dy = rnd.randint(-9, 9), rnd.randint(-5, 5)
        fd.line([x, y, x + dx, y + dy], fill=rnd.randint(236, 250), width=1)
    img = darken_by(img, fibers.filter(ImageFilter.GaussianBlur(0.4)))

    # пятна от влаги и чая
    stains = Image.new("L", (W, H), 255)
    sd = ImageDraw.Draw(stains)
    for _ in range(9):
        cx, cy = rnd.randrange(W), rnd.randrange(H)
        rx, ry = rnd.randint(40, 120), rnd.randint(30, 90)
        sd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rnd.randint(238, 248))
        sd.ellipse([cx - rx + 6, cy - ry + 6, cx + rx - 6, cy + ry - 6], fill=255 - (255 - rnd.randint(236, 248)) // 2)
    img = darken_by(img, stains.filter(ImageFilter.GaussianBlur(22)))

    img = grain(img, GRAIN_PAPER, seed)
    img = creases(img, FOLD_STRENGTH, seed)
    img = dust_and_scratches(img, 170, seed + 1)
    img = edge_burn(img, 0.72, seed)
    img = vignette(img, VIGNETTE_PAPER)
    return img


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Использование: python make_assets.py путь/к/фото_титаника.jpg")
    out_dir = Path(__file__).with_name("assets")
    out_dir.mkdir(exist_ok=True)

    splash = make_splash(sys.argv[1])
    splash.save(out_dir / "titanic_splash.png", optimize=True)
    make_hover(splash).save(out_dir / "titanic_splash_hover.png", optimize=True)
    make_paper().save(out_dir / "paper_bg.png", optimize=True)
    print("Готово:", ", ".join(p.name for p in sorted(out_dir.glob("*.png"))))


if __name__ == "__main__":
    main()
