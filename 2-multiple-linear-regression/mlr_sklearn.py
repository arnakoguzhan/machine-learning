import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.loc[:, ['Bedrooms', 'Bathrooms', 'Size']].values
Y = data.Price.values.reshape(-1, 1)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=0)


# Model Intialization
reg = LinearRegression()

# Data Fitting to LinearRegression Model
reg = reg.fit(X_train, Y_train)

# Printing coefficients
print(f"Coefficients theta0 = {reg.intercept_}, other thetas = {reg.coef_}")

# Predicted Values
Y_pred = reg.predict(X_test)


# Model Evaluation
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / Y.shape[0])
    return rmse


def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# Print Scores
print("RMSE = ", rmse(Y_test, Y_pred))
print("R2 Score = ", r2_score(Y_test, Y_pred))


# Features are Bedrooms, Bathrooms and Size respectively
features = np.array([[3, 3, 2371]])

predict = reg.predict(features)

print("Predict = ", int(predict))
