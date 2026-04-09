import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

file_path = "melb_data.csv"
file_data = pd.read_csv(file_path)
file_data.columns
print(file_data.columns)

file_data= file_data.dropna(axis=0)

y = file_data.Price

melb_features = ["Rooms","Bathroom","Landsize","Lattitude","Longtitude"]

X = file_data[melb_features]
mel_model = DecisionTreeRegressor(random_state=1)
mel_model.fit(X,y)

predictions = mel_model.predict(X)

predictions = mel_model.predict(X.head())
print(predictions)

# print(X.describe())

# print(X.head())