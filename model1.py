import tensorflow as tf
from tensorflow.keras import layers, Model

DEMO = False  # Set False for full model


def create_fl_vibration_cnn(input_shape=(1024, 3)):

    # Lighter model for fast FL demo
    if DEMO:
        f1 = 16   # small conv filters
        f2 = 32
        f3 = 64
        dense_units = 32
        dropout_rate = 0.4
    else:
        f1 = 32
        f2 = 64
        f3 = 128
        dense_units = 64
        dropout_rate = 0.3

    # -----------------------
    # Vibration input only
    # -----------------------
    inp = layers.Input(shape=input_shape, name="vibration")

    # Conv block 1
    x = layers.Conv1D(f1, kernel_size=5, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv block 2
    x = layers.Conv1D(f2, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Conv block 3
    x = layers.Conv1D(f3, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(dense_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    out = layers.Dense(1, activation="sigmoid", name="label")(x)

    return Model(inputs=inp, outputs=out)
