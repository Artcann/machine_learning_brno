import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

df_x = pd.read_csv('project_assignement/dataset/x_train.csv')
df_y = pd.read_csv('project_assignement/dataset/y_train.csv')


columns = [1, 2, 3, 4, 5, 7, 8, 9]

x_train, x_test, y_train, y_test = train_test_split(df_x[df_x.columns[columns]], df_y[df_y.columns[1]] - 1, train_size=0.8)

SVMmodel = SVC(kernel='linear', C=10).fit(x_train,y_train)

print(f"The SVM accuracy is : {SVMmodel.score(x_test,y_test)}")