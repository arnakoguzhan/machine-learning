import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Data preprocessing
data = pd.read_csv("data.csv")

# Converting Pandas dataframe to numpy array
X = data.x.values.reshape(-1, 1)
Y = data.y.values.reshape(-1, 1)

# Polynomial Features
# (simply add x^2 values to the array as a new column)
"""
# you can do that by yourself just add new column (x^2 values) to X
x2 = X**2 
X = np.append(X, x2, axis=1)
"""
poly_reg = PolynomialFeatures(degree=2)  #  Change degree and see what happens
X_poly = poly_reg.fit_transform(X)

# Model Intialization (RBF kernel)
reg = LinearRegression()
# Data Fitting to SVR Model
reg = reg.fit(X_poly, Y)
# Predicted Values
Y_pred = reg.predict(X_poly)


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
print("RMSE = ", rmse(Y, Y_pred))
print("R2 Score = ", r2_score(Y, Y_pred))


# Visualization
# Ploting Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
