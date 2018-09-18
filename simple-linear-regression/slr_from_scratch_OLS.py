import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading Data
data = pd.read_csv('RealEstate.csv')
# print(data.head())

# Collecting X and Y
X = data.Size.values
Y = data.Price.values

# Calculating coefficient
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
m = len(X)

# Using the OLS (Ordinary Least Squares)
# formula to calculate theta0 and theta1
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
theta1 = numer / denom
theta0 = mean_y - (theta1 * mean_x)

# Printing coefficients
print(f"Coefficients theta0 = {theta0}, theta1 = {theta1} ")


# Prediction and
def predict(theta0, theta1, X):
    return theta0 + theta1 * X


# Predicted Values
Y_pred = predict(theta0, theta1, X)

# Model Evaluation
# Calculating Root Mean Squares Error
rmse = 0
for i in range(m):
    y_pred = theta0 + theta1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print("RMSE = ", rmse)

# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(m):
    y_pred = theta0 + theta1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score = ", r2)


# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
