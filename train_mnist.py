"""Train a small CNN on the real MNIST dataset (TensorFlow/Keras)."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Reproducibility
tf.keras.utils.set_random_seed(42)

print("Downloading the real MNIST dataset from Keras...")
(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()

# Split the original 60,000 training images into 50,000 train + 10,000 validation.
x_train, y_train = x_train_all[:50000], y_train_all[:50000]
x_val, y_val = x_train_all[50000:], y_train_all[50000:]

# The assignment uses digits 1-9. Remove zero AFTER the split; keep test untouched except zero removal.
def prepare(x, y):
    keep = y != 0
    x = x[keep].astype("float32") / 255.0
    y = y[keep].astype("int64") - 1  # map original labels 1..9 to 0..8
    return x[..., None], y

x_train, y_train = prepare(x_train, y_train)
x_val, y_val = prepare(x_val, y_val)
x_test, y_test = prepare(x_test_all, y_test_all)

print(f"Training images (digits 1-9): {len(x_train):,}")
print(f"Validation images (digits 1-9): {len(x_val):,}")
print(f"Test images (digits 1-9): {len(x_test):,}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(9, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
]
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=6,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
probabilities = model.predict(x_test, batch_size=256, verbose=0)
predictions = probabilities.argmax(axis=1)
cm = confusion_matrix(y_test, predictions, labels=list(range(9)))
report = classification_report(y_test, predictions, labels=list(range(9)), output_dict=True, zero_division=0)

model_path = MODEL_DIR / "mnist_digit_model.keras"
model.save(model_path)
np.savetxt(RESULTS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
metrics = {
    "dataset": "MNIST (real images downloaded using tensorflow.keras.datasets.mnist)",
    "original_train_count": 60000,
    "original_test_count": 10000,
    "split_before_zero_filter": {"train": 50000, "validation": 10000},
    "digits_used": "1-9 (zero removed)",
    "train_images_after_zero_filter": int(len(x_train)),
    "validation_images_after_zero_filter": int(len(x_val)),
    "test_images_after_zero_filter": int(len(x_test)),
    "test_accuracy": float(accuracy),
    "test_loss": float(loss),
    "classification_report": report,
    "epochs_completed": len(history.history["loss"]),
    "model_file": "models/mnist_digit_model.keras",
}
with open(RESULTS_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"\nTest accuracy on real MNIST digits 1-9: {accuracy:.4f}")
print(f"Saved model to: {model_path}")
print(f"Saved metrics and confusion matrix in: {RESULTS_DIR}")
