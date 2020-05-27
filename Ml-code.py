
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
import sys


dataset=pd.read_csv("Social_Network_Ads.csv")
dataset.info()
dataset.head()
dataset.columns


X = dataset[["Age","EstimatedSalary"]]
y = dataset["Purchased"]
X = X.values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

x = int(sys.argv[1])

if x==0:
    i=3
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
if x==1:
    for i in range(4,10):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        confusion_matrix(y_test,y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        if accuracy > 80:
            break