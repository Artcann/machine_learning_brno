import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import keras_tuner as kt
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def model_builder(hypers):

    units = hypers.Int("units", min_value=16, max_value=64, step=16)

    model = Sequential()
    model.add(Dense(8, activation="relu"))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(8, activation="softmax"))

    model.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='project_assignement/Tuner',
                     project_name='project_assignement')

df_x = pd.read_csv('project_assignement/dataset/x_train.csv')
df_y = pd.read_csv('project_assignement/dataset/y_train.csv')


columns = [1, 2, 3, 4, 5, 7, 8, 9]

x_train, x_test, y_train, y_test = train_test_split(df_x[df_x.columns[columns]], df_y[df_y.columns[1]] - 1, train_size=0.8)

std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_norm = std_scale.transform(x_train)

std_scale_test = preprocessing.StandardScaler().fit(x_test)
x_test_norm = std_scale.transform(x_test)

stop_early = EarlyStopping(monitor='val_loss', patience=10)
tuner.search(x_train_norm, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=2)

best_hps = tuner.get_best_hyperparameters()[0]

classifier = tuner.hypermodel.build(best_hps)

history = classifier.fit(
    x_train_norm, 
    y_train, 
    epochs=15, batch_size=32, verbose=1,
    validation_data=(x_test_norm, y_test))

classifier.save("project_assignement/model_tuned")

loss, accuracy = classifier.evaluate(x_test_norm, y_test, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()