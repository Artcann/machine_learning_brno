import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

df_x = pd.read_csv('project_assignement/dataset/x_train.csv')
df_y = pd.read_csv('project_assignement/dataset/y_train.csv')


columns = [1, 2, 3, 4, 5, 7, 8, 9]

x_train, x_test, y_train, y_test = train_test_split(df_x[df_x.columns[columns]], df_y[df_y.columns[1]] - 1, train_size=0.8)

std_scale = preprocessing.StandardScaler().fit(x_train)
x_train_norm = std_scale.transform(x_train)

std_scale_test = preprocessing.StandardScaler().fit(x_test)
x_test_norm = std_scale.transform(x_test)

print(x_train_norm.shape)

classifier = Sequential()
classifier.add(Dense(8, activation="relu"))
classifier.add(Dense(16, activation="relu"))
classifier.add(Dense(8, activation="softmax"))

classifier.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])

history = classifier.fit(
    x_train_norm, 
    y_train, 
    epochs=15, batch_size=32, verbose=1,
    validation_data=(x_test_norm, y_test))

classifier.save("project_assignement/model")

loss, accuracy = classifier.evaluate(x_test_norm, y_test, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()