from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.regularizers import L2
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import datetime

X = np.linspace(0.0 , 2.0 * np.pi, 10000).reshape(-1, 1)
Y = np.sin(X)

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)
Y = Y_scaler.fit_transform(Y)

regressor = Sequential()
regressor.add(Dense(units=50, activation="tanh", input_dim=1))
regressor.add(Dense(units=25, activation="tanh"))
regressor.add(Dense(1, activation="linear"))

regressor.compile(loss="mse", optimizer="SGD", metrics=["mean_squared_error"])

regressor.fit(X, Y, epochs=100, batch_size=32, verbose=0)

x = np.linspace(0.0 , 2.0 * np.pi, 10000).reshape(-1, 1)
x = X_scaler.fit_transform(x)

plt.figure()
plt.plot(X, Y)
plt.plot(x, regressor.predict(x))
plt.show()