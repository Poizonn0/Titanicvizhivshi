import csv
import ctypes
import math
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk


DATA_PATH = Path(__file__).with_name("titanic.csv")
ASSETS_DIR = Path(__file__).with_name("assets")
PORTS = ("S", "C", "Q")
FEATURE_COUNT = 9

# Окно фиксированного размера: фон и заставка — готовые картинки 900x700
# (создаются скриптом make_assets.py и лежат в папке assets).
WINDOW_W = 900
WINDOW_H = 700

# Палитра «газеты»
DESK = "#302219"
PAPER = "#ddc99f"
CARD = "#f0e4c8"
INK = "#2b1c12"
MUTED_INK = "#746044"
RULE = "#aa9068"
ACCENT = "#783324"
GOLD = "#9b7441"
CREAM = "#fffaf0"

APP_FONT_FAMILY = "Lora"
APP_FONT_FILE = ASSETS_DIR / "Lora-Bold.ttf"
SERIF = APP_FONT_FAMILY
SANS = APP_FONT_FAMILY

# Контур корабля на заставке (доли от ширины/высоты картинки).
# Клик внутри этого контура открывает расчёт. Должен совпадать с make_assets.py.
SHIP_OUTLINE = (
    (0.200, 0.589), (0.306, 0.493), (0.314, 0.243), (0.383, 0.226), (0.428, 0.307),
    (0.424, 0.200), (0.489, 0.183), (0.533, 0.264), (0.513, 0.146), (0.583, 0.143),
    (0.606, 0.214), (0.598, 0.137), (0.664, 0.134), (0.728, 0.214), (0.844, 0.179),
    (0.924, 0.179), (0.958, 0.271), (0.950, 0.300), (0.889, 0.346), (0.778, 0.436),
    (0.667, 0.531), (0.556, 0.646), (0.444, 0.717), (0.333, 0.717), (0.239, 0.650),
)


# ----------------------------------------------------------------------
# Модель (без изменений)
# ----------------------------------------------------------------------
def sigmoid(value):
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def median(values, fallback):
    ordered = sorted(values)
    if not ordered:
        return fallback
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def load_passengers(path):
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("В датасете не хватает столбцов: " + ", ".join(sorted(missing)))

        passengers = []
        for row in reader:
            try:
                passengers.append(
                    {
                        "survived": int(row["Survived"]),
                        "pclass": int(row["Pclass"]),
                        "sex": row["Sex"].strip().lower(),
                        "age": float(row["Age"]) if row["Age"].strip() else None,
                        "siblings": int(row["SibSp"] or 0),
                        "parents": int(row["Parch"] or 0),
                        "fare": float(row["Fare"]) if row["Fare"].strip() else None,
                        "embarked": row["Embarked"].strip().upper(),
                    }
                )
            except (TypeError, ValueError):
                continue

    if len(passengers) < 10:
        raise ValueError("В датасете слишком мало корректных строк для обучения.")
    return passengers


def feature_values(passenger, age_median, fare_median, common_port):
    port = passenger["embarked"]
    if port not in PORTS:
        port = common_port
    return [
        float(passenger["pclass"]),
        1.0 if passenger["sex"] == "male" else 0.0,
        passenger["age"] if passenger["age"] is not None else age_median,
        float(passenger["siblings"]),
        float(passenger["parents"]),
        passenger["fare"] if passenger["fare"] is not None else fare_median,
        1.0 if port == "C" else 0.0,
        1.0 if port == "Q" else 0.0,
        1.0 if port == "S" else 0.0,
    ]


class NeuralNetwork:
    def __init__(self, input_count=FEATURE_COUNT, hidden_count=12, seed=42):
        generator = random.Random(seed)
        self.hidden_weights = [
            [generator.uniform(-0.35, 0.35) for _ in range(input_count)]
            for _ in range(hidden_count)
        ]
        self.hidden_biases = [0.0] * hidden_count
        self.output_weights = [generator.uniform(-0.35, 0.35) for _ in range(hidden_count)]
        self.output_bias = 0.0
        self.means = []
        self.scales = []

    def fit(self, features, labels, epochs=140, learning_rate=0.045):
        self.means = [sum(row[column] for row in features) / len(features) for column in range(len(features[0]))]
        self.scales = []
        for column, mean in enumerate(self.means):
            variance = sum((row[column] - mean) ** 2 for row in features) / len(features)
            self.scales.append(math.sqrt(variance) or 1.0)

        normalized = [self._normalize(row) for row in features]
        order = list(range(len(normalized)))
        generator = random.Random(123)
        for _ in range(epochs):
            generator.shuffle(order)
            for index in order:
                values = normalized[index]
                hidden = [
                    sigmoid(sum(weight * value for weight, value in zip(weights, values)) + bias)
                    for weights, bias in zip(self.hidden_weights, self.hidden_biases)
                ]
                output = sigmoid(sum(weight * value for weight, value in zip(self.output_weights, hidden)) + self.output_bias)
                output_delta = output - labels[index]
                hidden_deltas = [
                    output_delta * self.output_weights[position] * activation * (1.0 - activation)
                    for position, activation in enumerate(hidden)
                ]

                for position, activation in enumerate(hidden):
                    self.output_weights[position] -= learning_rate * output_delta * activation
                self.output_bias -= learning_rate * output_delta
                for hidden_index, delta in enumerate(hidden_deltas):
                    self.hidden_biases[hidden_index] -= learning_rate * delta
                    for feature_index, value in enumerate(values):
                        self.hidden_weights[hidden_index][feature_index] -= learning_rate * delta * value

    def _normalize(self, values):
        return [(value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales)]

    def predict_probability(self, features):
        values = self._normalize(features)
        hidden = [
            sigmoid(sum(weight * value for weight, value in zip(weights, values)) + bias)
            for weights, bias in zip(self.hidden_weights, self.hidden_biases)
        ]
        return sigmoid(sum(weight * value for weight, value in zip(self.output_weights, hidden)) + self.output_bias)


def train_model(path=DATA_PATH):
    passengers = load_passengers(path)
    ages = [item["age"] for item in passengers if item["age"] is not None]
    fares = [item["fare"] for item in passengers if item["fare"] is not None]
    ports = [item["embarked"] for item in passengers if item["embarked"] in PORTS]
    age_median = median(ages, 30.0)
    fare_median = median(fares, 14.0)
    common_port = max(PORTS, key=ports.count) if ports else "S"

    generator = random.Random(2026)
    order = list(range(len(passengers)))
    generator.shuffle(order)
    split = max(1, int(len(order) * 0.8))
    training_rows = [passengers[index] for index in order[:split]]
    test_rows = [passengers[index] for index in order[split:]]

    def make_features(rows):
        return [feature_values(item, age_median, fare_median, common_port) for item in rows]

    validation_model = NeuralNetwork()
    validation_model.fit(
        make_features(training_rows),
        [item["survived"] for item in training_rows],
    )
    correct = sum(
        (validation_model.predict_probability(values) >= 0.5) == bool(item["survived"])
        for item, values in zip(test_rows, make_features(test_rows))
    )
    accuracy = correct / len(test_rows) if test_rows else 0.0

    model = NeuralNetwork()
    model.fit(make_features(passengers), [item["survived"] for item in passengers])
    return model, accuracy, len(passengers), age_median, fare_median, common_port


# ----------------------------------------------------------------------
# Вспомогательные функции интерфейса
# ----------------------------------------------------------------------
def px_font(family, pixels, weight="normal", slant="roman"):
    """Шрифт в пикселях (отрицательный размер) — вёрстка не зависит от масштаба Windows."""
    return tkfont.Font(family=family, size=-pixels, weight=weight, slant=slant)


def fit_font(text, family, pixels, weight, max_width, min_pixels=11):
    """Уменьшает шрифт, пока текст не поместится в max_width."""
    font = px_font(family, pixels, weight)
    while font.measure(text) > max_width and pixels > min_pixels:
        pixels -= 1
        font.configure(size=-pixels)
    return font


def register_app_font(root):
    """Загружает шрифт из assets только на время работы приложения в Windows."""
    if sys.platform != "win32":
        raise OSError("Загрузка шрифта из assets поддерживается только в Windows.")
    if not APP_FONT_FILE.is_file():
        raise FileNotFoundError(f"Не найден файл шрифта: {APP_FONT_FILE}")

    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    add_font = gdi32.AddFontResourceExW
    add_font.argtypes = (ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p)
    add_font.restype = ctypes.c_int
    remove_font = gdi32.RemoveFontResourceExW
    remove_font.argtypes = (ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p)
    remove_font.restype = ctypes.c_int

    font_path = str(APP_FONT_FILE)
    private_font = 0x10
    if not add_font(font_path, private_font, None):
        raise ctypes.WinError(ctypes.get_last_error())

    def unregister():
        if not remove_font(font_path, private_font, None):
            raise ctypes.WinError(ctypes.get_last_error())

    try:
        families = {family.casefold() for family in tkfont.families(root)}
    except tk.TclError:
        unregister()
        raise
    if APP_FONT_FAMILY.casefold() not in families:
        unregister()
        raise RuntimeError(f"Tk не обнаружил загруженный шрифт {APP_FONT_FAMILY}.")

    return unregister


def ease(t):
    return t * t * (3.0 - 2.0 * t)


def fade_window(root, start, end, duration_ms, done=None):
    """Плавно меняет прозрачность окна. Где прозрачность не поддерживается — просто пауза."""
    steps = max(1, duration_ms // 16)

    def step(index):
        value = start + (end - start) * ease(index / steps)
        try:
            root.attributes("-alpha", value)
        except tk.TclError:
            pass
        if index < steps:
            root.after(16, step, index + 1)
        elif done is not None:
            done()

    step(0)


def point_in_polygon(x, y, polygon):
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            crossing_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


# ----------------------------------------------------------------------
# Экран 1: газетная вырезка с Титаником
# ----------------------------------------------------------------------
class CoverScreen(tk.Canvas):
    def __init__(self, master, on_open):
        super().__init__(
            master, width=WINDOW_W, height=WINDOW_H, bg=DESK, highlightthickness=0, bd=0
        )
        self.on_open = on_open
        self.hovering = False
        self.normal_image = self._load("titanic_splash.png")
        self.hover_image = self._load("titanic_splash_hover.png")
        self.has_art = self.normal_image is not None

        self.image_id = None
        if self.has_art:
            self.image_id = self.create_image(0, 0, anchor="nw", image=self.normal_image)
            hint = "НАЖМИТЕ НА ЛАЙНЕР, ЧТОБЫ ОТКРЫТЬ АРХИВ ПАССАЖИРОВ"
        else:
            # Нет картинок — запасной вариант: тёмный экран и надпись
            self.create_text(
                WINDOW_W / 2, WINDOW_H / 2 - 40, text="The World",
                fill="#f4e4c1", font=px_font(SERIF, 64, "bold"),
            )
            hint = "НАЖМИТЕ, ЧТОБЫ ОТКРЫТЬ АРХИВ ПАССАЖИРОВ"

        self.hint_font = fit_font(hint, SANS, 16, "bold", WINDOW_W - 80)
        hint_y = WINDOW_H - 30
        self.create_text(
            WINDOW_W / 2 + 1, hint_y + 1, text=hint, fill="#24180f", font=self.hint_font
        )
        self.create_text(WINDOW_W / 2, hint_y, text=hint, fill="#f4e4c1", font=self.hint_font)

        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda _event: self._set_hover(False))
        self.bind("<Button-1>", self._on_click)

    @staticmethod
    def _load(name):
        try:
            return tk.PhotoImage(file=str(ASSETS_DIR / name))
        except tk.TclError:
            return None

    def _over_ship(self, x, y):
        if not self.has_art:
            return True
        return point_in_polygon(x / WINDOW_W, y / WINDOW_H, SHIP_OUTLINE)

    def _set_hover(self, hovering):
        if hovering == self.hovering:
            return
        self.hovering = hovering
        self.configure(cursor="hand2" if hovering else "")
        if self.has_art and self.hover_image is not None:
            self.itemconfigure(
                self.image_id,
                image=self.hover_image if hovering else self.normal_image,
            )

    def _on_motion(self, event):
        self._set_hover(self._over_ship(event.x, event.y))

    def _on_click(self, event):
        if self._over_ship(event.x, event.y):
            self.on_open()


# ----------------------------------------------------------------------
# Экран 2: «пассажирское дело» на состаренной газетной бумаге
# ----------------------------------------------------------------------
class RecordScreen(tk.Canvas):
    LEFT = 48
    RIGHT = WINDOW_W - 48
    CONTENT_W = RIGHT - LEFT
    COL_X = (64, 466)
    COL_W = 370
    FIELD_W_FULL = 772

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

    def __init__(
        self, master, model, accuracy, count, age_median, fare_median, common_port, on_back
    ):
        super().__init__(
            master, width=WINDOW_W, height=WINDOW_H, bg=PAPER, highlightthickness=0, bd=0
        )
        self.model = model
        self.accuracy = accuracy
        self.count = count
        self.age_median = age_median
        self.fare_median = fare_median
        self.common_port = common_port
        self.on_back = on_back
        self.inputs = {}
        self._shown = 0.0
        self._animation = None

        self._load_background()
        self._setup_styles()
        self._build_fonts()
        self._draw_frame()
        y = self._draw_masthead()
        y = self._draw_headline(y)
        y = self._draw_form(y)
        y = self._draw_button(y + 8)
        y = self._draw_result(y + 8)
        self._draw_footer(y + 10)

    # ----- подготовка ---------------------------------------------------
    def _load_background(self):
        self.background = None
        try:
            self.background = tk.PhotoImage(file=str(ASSETS_DIR / "paper_bg.png"))
        except tk.TclError:
            return  # без картинки остаётся плоский цвет бумаги
        self.create_image(0, 0, anchor="nw", image=self.background)

    def _setup_styles(self):
        self.field_font = px_font(SANS, 13)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "News.TCombobox",
            padding=(8, 3),
            fieldbackground=CARD,
            background=CARD,
            foreground=INK,
            arrowcolor=INK,
            bordercolor=RULE,
            lightcolor=RULE,
            darkcolor=RULE,
            font=self.field_font,
        )
        style.map(
            "News.TCombobox",
            fieldbackground=[("readonly", CARD)],
            background=[("readonly", CARD), ("active", CARD)],
            foreground=[("readonly", INK)],
            selectbackground=[("readonly", CARD)],
            selectforeground=[("readonly", INK)],
            bordercolor=[("focus", GOLD)],
        )
        style.configure(
            "News.TEntry",
            padding=(8, 3),
            fieldbackground=CARD,
            foreground=INK,
            bordercolor=RULE,
            lightcolor=RULE,
            darkcolor=RULE,
            font=self.field_font,
        )
        style.map(
            "News.TEntry",
            bordercolor=[("focus", GOLD)],
            lightcolor=[("focus", GOLD)],
            darkcolor=[("focus", GOLD)],
        )
        # выпадающий список комбобокса
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", CREAM)
        self.option_add("*TCombobox*Listbox.font", self.field_font)

    def _build_fonts(self):
        # ссылки на шрифты хранятся в объекте, иначе Tk удалит их
        self.f_back = px_font(SANS, 11, "bold")
        self.f_extra = px_font(SANS, 20, "bold")
        self.f_tagline = px_font(SANS, 11, "bold")
        self.f_dateline = px_font(SANS, 11)
        self.f_dateline_bold = px_font(SANS, 11, "bold")
        self.f_form_title = px_font(SERIF, 18, "bold")
        self.f_record = px_font(SANS, 11, "bold")
        self.f_label = px_font(SANS, 12, "bold")
        self.f_note = px_font(SERIF, 12, slant="italic")
        self.f_button = px_font(SANS, 14, "bold")
        self.f_caption = px_font(SANS, 11, "bold")
        self.f_outcome = px_font(SERIF, 22, "bold")
        self.f_percent = px_font(SERIF, 34, "bold")
        self.f_footer = px_font(SERIF, 12, slant="italic")

    # ----- рисование ------------------------------------------------------
    def _draw_frame(self):
        self.create_rectangle(18, 18, WINDOW_W - 18, WINDOW_H - 18, outline="#8c6c42", width=2)
        self.create_rectangle(24, 24, WINDOW_W - 24, WINDOW_H - 24, outline=RULE, width=1)

    def _draw_masthead(self):
        back = self.create_text(
            self.LEFT, 36, text="←  К ГАЗЕТНОЙ ОБЛОЖКЕ", anchor="w",
            fill=ACCENT, font=self.f_back, tags=("back",),
        )
        self.tag_bind("back", "<Enter>", lambda _e: (
            self.itemconfigure(back, fill=INK), self.configure(cursor="hand2")))
        self.tag_bind("back", "<Leave>", lambda _e: (
            self.itemconfigure(back, fill=ACCENT), self.configure(cursor="")))
        self.tag_bind("back", "<Button-1>", lambda _e: self.on_back())

        # плашки EXTRA слева и справа
        for x0 in (self.LEFT, self.RIGHT - 80):
            self.create_rectangle(x0, 52, x0 + 80, 84, outline=INK, width=1)
            self.create_text(x0 + 40, 68, text="EXTRA", fill=INK, font=self.f_extra)

        title_font = fit_font("The World", SERIF, 58, "bold", 560)
        self.create_text(WINDOW_W / 2, 72, text="The World", fill=INK, font=title_font)
        self.create_text(
            WINDOW_W / 2, 108, text="THE NEWSPAPER TO THE WORLD  ·  EXTRA EDITION",
            fill=MUTED_INK, font=self.f_tagline,
        )

        self.create_rectangle(self.LEFT, 121, self.RIGHT, 124, fill=INK, outline=INK)
        self.create_line(self.LEFT, 128, self.RIGHT, 128, fill=GOLD, width=1)
        self.create_text(
            self.LEFT, 140, text="VOL. 24  ·  NO. 50,000", anchor="w",
            fill=MUTED_INK, font=self.f_dateline,
        )
        self.create_text(
            self.RIGHT, 140, text="MONDAY, 15 APRIL 1912  ·  THE NORTH ATLANTIC", anchor="e",
            fill=MUTED_INK, font=self.f_dateline_bold,
        )
        self.create_line(self.LEFT, 152, self.RIGHT, 152, fill=RULE, width=1)
        return 152

    def _draw_headline(self, y):
        text = "ТИТАНИК ЗАТОНУЛ ПОСЛЕ СТОЛКНОВЕНИЯ С АЙСБЕРГОМ"
        self.f_headline = fit_font(text, SERIF, 29, "bold", self.CONTENT_W)
        self.create_text(self.LEFT, y + 6, text=text, anchor="nw", fill=INK, font=self.f_headline)
        y += 6 + self.f_headline.metrics("linespace")

        sub = "705 спасены; более 1 500 пассажиров погибли"
        self.f_subhead = fit_font(sub, SERIF, 22, "bold", self.CONTENT_W)
        self.create_text(self.LEFT, y + 1, text=sub, anchor="nw", fill=INK, font=self.f_subhead)
        y += 1 + self.f_subhead.metrics("linespace") + 6
        self.create_rectangle(self.LEFT, y, self.RIGHT, y + 2, fill=INK, outline=INK)
        return y + 2

    def _draw_form(self, y):
        top = y + 10
        rows = max(field[6] for field in self.FIELDS) + 1
        fields_top = top + 44
        row_pitch = 48
        bottom = fields_top + (rows - 1) * row_pitch + 17 + 28 + 28
        # рамка «колонки» без заливки — бумага просвечивает
        self.create_rectangle(self.LEFT, top, self.RIGHT, bottom, outline=RULE, width=1)

        self.create_text(
            self.COL_X[0], top + 10, text="ПАССАЖИРСКОЕ ДЕЛО", anchor="nw",
            fill=INK, font=self.f_form_title,
        )
        self.create_text(
            self.RIGHT - 16, top + 15, text="RECORD No.  1912", anchor="ne",
            fill=ACCENT, font=self.f_record,
        )
        self.create_line(
            self.COL_X[0], top + 36, self.RIGHT - 16, top + 36, fill=RULE, width=1
        )

        for label, key, kind, values, default, column, row, span in self.FIELDS:
            x = self.COL_X[column]
            y_row = fields_top + row * row_pitch
            width = self.COL_W if span == 1 else self.FIELD_W_FULL
            self.create_text(x, y_row, text=label, anchor="nw", fill=MUTED_INK, font=self.f_label)
            if kind == "combo":
                widget = ttk.Combobox(
                    self, values=values, state="readonly", style="News.TCombobox"
                )
                widget.set(default)
            else:
                widget = ttk.Entry(self, style="News.TEntry")
                widget.insert(0, default)
            widget.bind("<Return>", lambda _event: self.predict())
            self.create_window(x, y_row + 17, window=widget, anchor="nw", width=width, height=28)
            self.inputs[key] = widget

        note_y = fields_top + (rows - 1) * row_pitch + 17 + 28 + 6
        self.create_text(
            self.COL_X[0], note_y, anchor="nw", fill=MUTED_INK, font=self.f_note,
            text="* Если возраст или тариф неизвестны, оставьте поле пустым — будет использовано среднее значение.",
        )
        return bottom

    def _draw_button(self, y):
        x0, x1, y1 = self.LEFT, self.RIGHT, y + 36
        rect = self.create_rectangle(x0, y, x1, y1, fill=ACCENT, outline=ACCENT, tags=("btn",))
        self.create_rectangle(
            x0 + 3, y + 3, x1 - 3, y1 - 3, outline="#c9a46a", width=1, tags=("btn",)
        )
        self.create_text(
            (x0 + x1) / 2, (y + y1) / 2, text="РАССЧИТАТЬ ВЕРОЯТНОСТЬ",
            fill=CREAM, font=self.f_button, tags=("btn",),
        )
        self.tag_bind("btn", "<Enter>", lambda _e: (
            self.itemconfigure(rect, fill=INK, outline=INK), self.configure(cursor="hand2")))
        self.tag_bind("btn", "<Leave>", lambda _e: (
            self.itemconfigure(rect, fill=ACCENT, outline=ACCENT), self.configure(cursor="")))
        self.tag_bind("btn", "<Button-1>", lambda _e: self.predict())
        return y1

    def _draw_result(self, y):
        height = 80
        self.create_rectangle(self.LEFT, y, self.RIGHT, y + height, fill=INK, outline="#120b06")
        self.create_text(
            self.COL_X[0], y + 11, text="ИЗ РЕДАКЦИОННОЙ КОЛОНКИ", anchor="w",
            fill="#d9bf91", font=self.f_caption,
        )
        self.create_text(
            self.RIGHT - 16, y + 11, anchor="e", fill="#b7a991", font=self.f_caption,
            text=f"ОЦЕНКА МОДЕЛИ  ·  ТОЧНОСТЬ {self.accuracy:.0%}",
        )
        self.result_text = self.create_text(
            self.COL_X[0], y + 40, text="Ожидаем данные пассажира", anchor="w",
            fill="#fff8e9", font=self.f_outcome,
        )
        self.percent_text = self.create_text(
            self.RIGHT - 16, y + 38, text="—", anchor="e", fill="#d9bf91", font=self.f_percent,
        )
        self.bar_x0, self.bar_x1 = self.COL_X[0], self.RIGHT - 16
        self.bar_y = y + 65
        self.create_rectangle(
            self.bar_x0, self.bar_y, self.bar_x1, self.bar_y + 6, fill="#554b3d", outline=""
        )
        self.bar_fill = self.create_rectangle(
            self.bar_x0, self.bar_y, self.bar_x0, self.bar_y + 6, fill="#d9bf91", outline=""
        )
        return y + height

    def _draw_footer(self, y):
        self.create_line(self.LEFT, y, self.RIGHT, y, fill=RULE, width=1)
        self.create_text(
            self.LEFT, y + 7, anchor="nw", fill=MUTED_INK, font=self.f_footer,
            text="Историческая статистика — не достоверный вывод о конкретном человеке.",
        )

    # ----- логика ---------------------------------------------------------
    def predict(self):
        try:
            age_text = self.inputs["age"].get().strip()
            fare_text = self.inputs["fare"].get().strip()
            age = float(age_text) if age_text else self.age_median
            fare = float(fare_text) if fare_text else self.fare_median
            if not 0 <= age <= 100 or not 0 <= fare <= 1000:
                raise ValueError
            port_code = self.inputs["embarked"].get()[-2]
            passenger = {
                "pclass": int(self.inputs["pclass"].get()),
                "sex": "male" if self.inputs["sex"].get() == "Мужской" else "female",
                "age": age,
                "siblings": int(self.inputs["siblings"].get()),
                "parents": int(self.inputs["parents"].get()),
                "fare": fare,
                "embarked": port_code,
            }
        except (ValueError, IndexError):
            messagebox.showerror(
                "Проверьте данные",
                "Укажите возраст от 0 до 100 и стоимость билета от 0 до 1000.",
            )
            return

        probability = self.model.predict_probability(
            feature_values(passenger, self.age_median, self.fare_median, self.common_port)
        )
        outcome = "Вероятнее выжил" if probability >= 0.5 else "Вероятнее не выжил"
        self.itemconfigure(self.result_text, text=outcome)
        self.itemconfigure(
            self.bar_fill, fill="#d9bf91" if probability >= 0.5 else "#bc7463"
        )
        self._animate_to(probability)

    def _set_bar(self, value):
        self._shown = value
        self.itemconfigure(self.percent_text, text=f"{value:.0%}")
        self.coords(
            self.bar_fill,
            self.bar_x0, self.bar_y,
            self.bar_x0 + (self.bar_x1 - self.bar_x0) * value, self.bar_y + 6,
        )

    def _animate_to(self, target):
        """Плавно «доезжает» шкала и число до нового значения."""
        if self._animation is not None:
            self.after_cancel(self._animation)
        start = self._shown
        steps = 24

        def step(index):
            self._set_bar(start + (target - start) * ease(index / steps))
            if index < steps:
                self._animation = self.after(16, step, index + 1)
            else:
                self._animation = None

        step(0)


# ----------------------------------------------------------------------
# Приложение: два экрана и плавный переход между ними
# ----------------------------------------------------------------------
class TitanicApp:
    FADE_MS = 320

    def __init__(self, root, model, accuracy, count, age_median, fare_median, common_port):
        self.root = root
        self.busy = False
        root.title("The World — Titanic")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        root.resizable(False, False)
        root.configure(bg=DESK)

        self.cover = CoverScreen(root, on_open=lambda: self.switch_to(self.record))
        self.record = RecordScreen(
            root, model, accuracy, count, age_median, fare_median, common_port,
            on_back=lambda: self.switch_to(self.cover),
        )
        self.titles = {
            self.cover: "The World — Titanic",
            self.record: "The World — Passenger Record",
        }
        self.current = self.cover
        self.cover.pack()

        # появление программы из прозрачности
        try:
            root.attributes("-alpha", 0.0)
        except tk.TclError:
            return
        root.update_idletasks()
        fade_window(root, 0.0, 1.0, 450)

    def switch_to(self, target):
        if self.busy or target is self.current:
            return
        self.busy = True

        def swap():
            self.current.pack_forget()
            target.pack()
            self.current = target
            self.root.title(self.titles[target])
            self.root.update_idletasks()
            fade_window(self.root, 0.0, 1.0, self.FADE_MS, done=finish)

        def finish():
            self.busy = False

        fade_window(self.root, 1.0, 0.0, self.FADE_MS, done=swap)


def main():
    try:
        model, accuracy, count, age_median, fare_median, common_port = train_model()
    except (OSError, ValueError) as error:
        if "--self-test" in sys.argv:
            raise
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Не удалось обучить модель", str(error))
        root.destroy()
        return

    if "--self-test" in sys.argv:
        print(f"Обучение завершено: {count} пассажиров, точность на отложенной выборке {accuracy:.1%}.")
        if accuracy < 0.70:
            raise SystemExit("Точность ниже ожидаемого порога 70%.")
        return

    root = tk.Tk()
    try:
        unregister_font = register_app_font(root)
    except (OSError, RuntimeError, tk.TclError) as error:
        root.withdraw()
        messagebox.showerror("Не удалось загрузить шрифт", str(error), parent=root)
        root.destroy()
        return

    try:
        TitanicApp(root, model, accuracy, count, age_median, fare_median, common_port)
        root.mainloop()
    finally:
        try:
            unregister_font()
        finally:
            root.destroy()


if __name__ == "__main__":
    main()
