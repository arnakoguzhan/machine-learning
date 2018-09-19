import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.loc[:, ['Bedrooms', 'Bathrooms', 'Size']].values
Y = data.Price.values

# Model Intialization
reg = LinearRegression()

# Data Fitting to LinearRegression Model
reg = reg.fit(X, Y)

# Printing coefficients
print(f"Coefficients theta0 = {reg.intercept_}, other thetas = {reg.coef_}")

# Predicted Values
Y_pred = reg.predict(X)

# List 10 predict and their actual values
for i in range(10):
    predict = int(Y_pred[i])
    actual = int(Y[i])
    print(f"predict = {predict}, actual = {actual}")


# Model Evaluation
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = reg.score(X, Y)

print("MSE = ", mse)
print("RMSE = ", rmse)
print("R2 Score = ", r2)
