import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.loc[:, ['Bedrooms', 'Bathrooms', 'Size']].values
Y = data.Price.values

m = X.shape[0]  #  number of samples

# bias_vector to add to X
bias_vector = np.ones((m, 1))
X = np.append(bias_vector, X, axis=1)


# Calculating coefficient
def normalEquation(X, Y):
    theta = []

    m = Y.shape[0]
    y = np.reshape(Y, (m, 1))

    # Now do the math :)
    # Normal Equation formula is :
    # inverse(X^T * X) * X^T * y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    return theta


theta = normalEquation(X, Y)

# Printing coefficients
print(f"Coefficients theta = {theta} ")

# Predicted Values
Y_pred = X.dot(theta)
m = Y_pred.shape[0]
Y_pred = np.reshape(Y_pred, (m,))

# List 10 predict and their actual values
for i in range(10):
    predict = int(Y_pred[i])
    actual = int(Y[i])
    print(f"predict = {predict}, actual = {actual}")


# Model Evaluation
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)


def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


print("MSE = ", mse)
print("RMSE = ", rmse)
print("R2 Score = ", r2_score(Y, Y_pred))
