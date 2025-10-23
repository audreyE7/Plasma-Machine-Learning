import numpy as np, yaml, os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

cfg = yaml.safe_load(open("ml_seq/seq_config.yaml"))
data = np.load("data/seq_dataset.npz", allow_pickle=True)
X, y = data["X"], data["y"]
num_classes = len(cfg["labels"])

# train/val split
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# LSTM model
model = models.Sequential([
    layers.Input(shape=(cfg["seq_len"], X.shape[-1])),
    layers.Masking(mask_value=0.0),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
cb = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
]
hist = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=50, batch_size=64, callbacks=cb)
model.save("ml_seq/model_lstm.h5")
print("Saved ml_seq/model_lstm.h5")
