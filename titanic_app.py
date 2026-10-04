import csv
import math
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


DATA_PATH = Path(__file__).with_name("titanic.csv")
PORTS = ("S", "C", "Q")
FEATURE_COUNT = 9


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


class TitanicApp:
    def __init__(self, root, model, accuracy, count, age_median, fare_median, common_port):
        self.root = root
        self.model = model
        self.accuracy = accuracy
        self.count = count
        self.age_median = age_median
        self.fare_median = fare_median
        self.common_port = common_port
        self.root.title("Titanic | Прогноз выживания")
        self.root.geometry("560x590")
        self.root.minsize(500, 560)
        self.root.configure(bg="#f3f4ef")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=6, fieldbackground="white")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 9))

        page = tk.Frame(root, bg="#f3f4ef", padx=34, pady=26)
        page.pack(fill="both", expand=True)

        tk.Label(page, text="TITANIC / МОДЕЛЬ", bg="#f3f4ef", fg="#287a69", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(page, text="Оценка вероятности выживания", bg="#f3f4ef", fg="#172c2b", font=("Segoe UI", 21, "bold")).pack(anchor="w", pady=(5, 3))
        tk.Label(page, text="Заполните известные данные пассажира", bg="#f3f4ef", fg="#5e6b67", font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 18))

        form = tk.Frame(page, bg="#ffffff", padx=20, pady=16, highlightthickness=1, highlightbackground="#dfe4dc")
        form.pack(fill="x")
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(2, weight=1)

        self.inputs = {}
        fields = [
            ("Пол", "sex", "Женский", "Мужской", 0, 0),
            ("Возраст", "age", "", None, 0, 2),
            ("Класс", "pclass", "1", "2", 1, 0),
            ("Братья / супруги", "siblings", "0", "1", 1, 2),
            ("Родители / дети", "parents", "0", "1", 2, 0),
            ("Стоимость билета", "fare", "", None, 2, 2),
            ("Порт посадки", "embarked", "Southampton (S)", "Cherbourg (C)", 3, 0),
        ]
        for label, key, first, second, row, column in fields:
            cell = tk.Frame(form, bg="#ffffff")
            cell.grid(row=row, column=column, sticky="ew", padx=(0, 12) if column == 0 else (12, 0), pady=7)
            tk.Label(cell, text=label, bg="#ffffff", fg="#45534f", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
            if key == "sex":
                widget = ttk.Combobox(cell, values=("Женский", "Мужской"), state="readonly")
                widget.set(first)
            elif key == "pclass":
                widget = ttk.Combobox(cell, values=("1", "2", "3"), state="readonly")
                widget.set(first)
            elif key == "siblings":
                widget = ttk.Combobox(cell, values=tuple(str(value) for value in range(9)), state="readonly")
                widget.set(first)
            elif key == "parents":
                widget = ttk.Combobox(cell, values=tuple(str(value) for value in range(7)), state="readonly")
                widget.set(first)
            elif key == "embarked":
                widget = ttk.Combobox(cell, values=("Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"), state="readonly")
                widget.set(first)
            else:
                widget = ttk.Entry(cell)
                widget.insert(0, "30" if key == "age" else "14")
            widget.pack(fill="x")
            self.inputs[key] = widget

        button = ttk.Button(page, text="Рассчитать прогноз", command=self.predict)
        button.pack(fill="x", pady=(16, 10))

        self.result = tk.Label(page, text="", bg="#f3f4ef", fg="#172c2b", font=("Segoe UI", 17, "bold"), justify="left")
        self.result.pack(anchor="w", pady=(4, 0))
        accuracy_percent = round(self.accuracy * 100)
        tk.Label(
            page,
            text=f"Обучено на {self.count} пассажирах  ·  точность проверки: {accuracy_percent}%",
            bg="#f3f4ef",
            fg="#5e6b67",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(10, 3))
        tk.Label(
            page,
            text="Это статистическая оценка по историческим данным, а не достоверный вывод о конкретном человеке.",
            bg="#f3f4ef",
            fg="#73807b",
            font=("Segoe UI", 9),
            wraplength=470,
            justify="left",
        ).pack(anchor="w")

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
            messagebox.showerror("Проверьте данные", "Укажите возраст от 0 до 100 и стоимость билета от 0 до 1000.")
            return

        probability = self.model.predict_probability(
            feature_values(passenger, self.age_median, self.fare_median, self.common_port)
        )
        outcome = "Вероятнее выжил" if probability >= 0.5 else "Вероятнее не выжил"
        self.result.configure(text=f"{outcome}\nВероятность выживания: {probability:.0%}")


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
    app = TitanicApp(root, model, accuracy, count, age_median, fare_median, common_port)
    root.mainloop()


if __name__ == "__main__":
    main()