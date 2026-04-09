import numpy as np
import pandas as pd

df = pd.read_csv('customers.csv')

df = df.drop(columns=["Customer"])

df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df['Region'] = df['Region'].map({'Rural':0,'Urban':1})
df['Buys'] = df['Buys'].map({'No':0,'Yes':1})

X = df[['Age','Income','Gender','Region']].values
y = df['Buys'].values

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def euclidean_distance(train, test):
    return np.sqrt(np.sum((train - test) ** 2, axis=1))


def knn(X,y,test_point,k):
    test_point = (test_point-X.min(axis = 0))/ (X.max(axis = 0) - X.min(axis = 0))
    
    distance = euclidean_distance(X, test_point)

    k_indices = np.argsort(distance)[:k]

    k_nearest_labels = y[k_indices]

    counts = np.bincount(k_nearest_labels)
    return np.argmax(counts)

test = np.array([32,38000,1,1]) #Female , Rural

result = knn(X,y,test,3)

if result == 1:
    print("Prediction: Yes")
else:
    print("Prediction: No")