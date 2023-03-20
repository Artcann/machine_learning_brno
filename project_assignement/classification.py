from tensorflow import keras
from sklearn import preprocessing
import pandas as pd
import numpy as np

df = pd.read_csv("project_assignement/dataset/x_text.csv")

classifier = keras.models.load_model('project_assignement/model')

columns = [1, 2, 3, 4, 5, 7, 8, 9]

test_x = df[df.columns[columns]]

std_scale = preprocessing.StandardScaler().fit(test_x)
x_test_norm = std_scale.transform(test_x)

predictions = classifier.predict(x_test_norm)
columns = ["target"]

dataset = pd.DataFrame(data=np.argmax(predictions, axis=1) + 1, columns=columns)
dataset.index.name = "id"
dataset.to_csv("project_assignement/results.csv", index=True)

