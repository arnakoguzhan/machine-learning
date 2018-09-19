import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Data preprocessing
data = pd.read_csv("RealEstate.csv")

# Converting Pandas dataframe to numpy array
X = data.Size.values
Y = data.Price.values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

m = X.shape[0]  #  number of samples


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


# Predicted Values
Y_pred = theta0 + theta1 * X


# Model Evaluation
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)

print("MSE Score = ", mse)
print("RMSE Score = ", rmse)

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
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
