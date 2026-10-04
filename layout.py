"""
Общая раскладка окна «газеты».

Этот файл читают и titanic_gui.py (куда ставить поля и кнопки), и make_assets.py
(где на фон «печатаются» заголовки и рамки). Поэтому они всегда совпадают.
Хотите сдвинуть блок — меняйте число здесь и перезапустите make_assets.py.
"""

WINDOW_W = 900
WINDOW_H = 700

LEFT = 48
RIGHT = WINDOW_W - 48

# «Уши» газеты по бокам от названия
EAR_LEFT = (50, 46, 158, 114)     # слева — растровая картинка корабля
EAR_RIGHT = (742, 46, 850, 114)   # справа — плашка EXTRA

# Горизонтальные линейки
MAST_RULE_Y = 130                 # толстая + тонкая под названием
DATELINE_RULE_Y = 157             # тонкая под строкой с датой
HEAD_RULE_Y = 237                 # толстая + тонкая под заголовком

# Форма «Пассажирское дело»
FORM_TOP = 252
FORM_BOTTOM = 503
TITLE_Y = 260
TITLE_RULE_Y = 284
COL_X = (64, 466)
COL_W = 370
FIELD_W_FULL = 772
ROW_Y0 = 292
ROW_PITCH = 47
WIDGET_DY = 17                    # поле ввода ниже подписи на столько пикселей
WIDGET_H = 28
NOTE_Y = 483

BUTTON = (48, 511, 852, 546)
RESULT = (48, 554, 852, 630)
FOOTER_RULE_Y = 642
FOOTER_Y = 649

BACK_RECT = (46, 28, 260, 46)     # область ссылки «К газетной обложке»

FIELDS = (
    # подпись, ключ, тип, значения, по умолчанию, колонка, строка, ширина в колонках
    ("Пол пассажира", "sex", "combo", ("Женский", "Мужской"), "Женский", 0, 0, 1),
    ("Класс каюты", "pclass", "combo", ("1", "2", "3"), "1", 1, 0, 1),
    ("Возраст, лет", "age", "entry", None, "30", 0, 1, 1),
    ("Стоимость билета", "fare", "entry", None, "14", 1, 1, 1),
    ("Братья / супруги", "siblings", "combo", tuple(str(v) for v in range(9)), "0", 0, 2, 1),
    ("Родители / дети", "parents", "combo", tuple(str(v) for v in range(7)), "0", 1, 2, 1),
    (
        "Порт отправления", "embarked", "combo",
        ("Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"), "Southampton (S)", 0, 3, 2,
    ),
)


def field_geometry(column, row, span):
    """(x, y подписи, y поля, ширина поля)"""
    y = ROW_Y0 + row * ROW_PITCH
    width = COL_W if span == 1 else FIELD_W_FULL
    return COL_X[column], y, y + WIDGET_DY, width


def static_texts():
    """Весь неизменяемый текст страницы. style → шрифт в make_assets.py."""
    items = [
        dict(text="The World", x=450, y=82, anchor="mm", style="masthead", size=86, fit=540),
        dict(id="tagline", text="THE NEWSPAPER TO THE WORLD", x=450, y=121, anchor="mm",
             style="caps", size=10, tracking=2.4),
        dict(text="VOL. 24  ·  NO. 50,000", x=50, y=146, anchor="lm", style="caps", size=10, tracking=1.2),
        dict(text="PRICE ONE CENT", x=450, y=146, anchor="mm", style="caps", size=10, tracking=1.6),
        dict(text="MONDAY, 15 APRIL 1912  ·  THE NORTH ATLANTIC", x=850, y=146, anchor="rm",
             style="caps", size=10, tracking=1.2),
        dict(text="ТИТАНИК ЗАТОНУЛ ПОСЛЕ СТОЛКНОВЕНИЯ С АЙСБЕРГОМ", x=50, y=164, anchor="lt",
             style="headline", size=34, fit=800),
        dict(text="705 спасены; более 1 500 пассажиров погибли", x=50, y=207, anchor="lt",
             style="deck", size=21, fit=800),
        dict(text="ПАССАЖИРСКОЕ ДЕЛО", x=COL_X[0], y=TITLE_Y, anchor="lt", style="title", size=19, tracking=1.6),
        dict(text="RECORD  No. 1912", x=RIGHT - 16, y=TITLE_Y + 6, anchor="rt", style="caps",
             size=10, tracking=1.6, color="red"),
        dict(text="* Если возраст или тариф неизвестны, оставьте поле пустым — будет использовано среднее значение.",
             x=COL_X[0], y=NOTE_Y, anchor="lt", style="note", size=12, fit=772),
        dict(text="Историческая статистика — не достоверный вывод о конкретном человеке.",
             x=50, y=FOOTER_Y, anchor="lt", style="note", size=12),
        dict(id="button", text="РАССЧИТАТЬ ВЕРОЯТНОСТЬ", x=(BUTTON[0] + BUTTON[2]) / 2,
             y=(BUTTON[1] + BUTTON[3]) / 2 + 1, anchor="mm", style="title", size=16,
             tracking=2.6, color="red"),
    ]
    for label, _key, _kind, _values, _default, column, row, span in FIELDS:
        x, y_label, _y_widget, _w = field_geometry(column, row, span)
        items.append(dict(text=label, x=x, y=y_label, anchor="lt", style="label", size=13, opacity=0.86))
    return items


# Контур корабля на заставке (доли от ширины/высоты картинки).
# Клик внутри него открывает расчёт; по нему же рисуется подсветка при наведении.
SHIP_OUTLINE = (
    (0.200, 0.589), (0.306, 0.493), (0.314, 0.243), (0.383, 0.226), (0.428, 0.307),
    (0.424, 0.200), (0.489, 0.183), (0.533, 0.264), (0.513, 0.146), (0.583, 0.143),
    (0.606, 0.214), (0.598, 0.137), (0.664, 0.134), (0.728, 0.214), (0.844, 0.179),
    (0.924, 0.179), (0.958, 0.271), (0.950, 0.300), (0.889, 0.346), (0.778, 0.436),
    (0.667, 0.531), (0.556, 0.646), (0.444, 0.717), (0.333, 0.717), (0.239, 0.650),
)
