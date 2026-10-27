# ============================================================
# Распознавание рукописных цифр (0-9) с помощью нейросети
# Датасет: MNIST (встроен в Keras)
# Библиотека: TensorFlow / Keras
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# ============================================================
# ШАГ 1. Загрузка и подготовка данных
# ============================================================
print("Загружаем датасет MNIST...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Обучающая выборка: {x_train.shape}")
print(f"Тестовая выборка:  {x_test.shape}")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# ============================================================
# ШАГ 2. Создание простой модели нейросети
# ============================================================
print("Создаём модель...")

model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(28 * 28,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])

model.summary()

# ============================================================
# ШАГ 3. Компиляция и обучение модели
# ============================================================
print("Компилируем модель...")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Начинаем обучение...")

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# ============================================================
# ШАГ 4. Проверка точности на тестовых данных
# ============================================================
print("Проверяем точность на тестовой выборке...")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочность на тестовых данных: {test_acc * 100:.2f}%")
print(f"Значение функции потерь:    {test_loss:.4f}")

# ============================================================
# БОНУС. Пример предсказания для одной картинки
# ============================================================
print("\nПример предсказания для первой картинки из теста:")

sample = x_test[0:1]
true_label = y_test[0]

predictions = model.predict(sample, verbose=0)
predicted_label = np.argmax(predictions)

print(f"Правильный ответ: {true_label}")
print(f"Предсказание сети: {predicted_label}")
print(f"Уверенность: {predictions[0][predicted_label] * 100:.2f}%")
