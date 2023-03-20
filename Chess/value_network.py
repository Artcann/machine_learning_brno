from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.optimizers import Adam
import numpy as np


def build_model(layer_size, layer_nb):
    model = Sequential()

    model.add(Input(shape=(14, 8, 8)))

    for _ in range(layer_nb):
        model.add(Conv2D(filters=layer_size, kernel_size=3, activation="sigmoid", data_format="channels_first", padding="same"))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="tanh"))

    return model

model = build_model(32, 4)
model.summary()
