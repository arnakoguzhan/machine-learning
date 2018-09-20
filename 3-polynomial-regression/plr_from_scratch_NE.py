import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data preprocessing
data = pd.read_csv("data.csv")

# Converting Pandas dataframe to numpy array
X = data.x.values.reshape(-1, 1)
Y = data.y.values.reshape(-1, 1)

# Add bias
m = X.shape[0]  # sample count
bias = np.ones((m, 1))
X_train = np.append(bias, X, axis=1)

# Add x^2 values
x2 = X**2
X_train = np.append(X_train, x2, axis=1)
Y_train = np.array(Y)


# Calculating theta with Normal Equation formula
def normalEquation(X, Y):
    theta = []

    # Normal Equation formula is :
    # inverse(X^T * X) * X^T * y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(Y)

    return theta


# Get theta
theta = normalEquation(X_train, Y_train)
print("Theta = ", theta)


# Predicted Values
Y_pred = X_train.dot(theta)


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
print("RMSE = ", rmse(Y_train, Y_pred))
print("R2 Score = ", r2_score(Y_train, Y_pred))


# Visualization
# Ploting Regression Line
plt.plot(X, Y_pred, color='#c93e4e', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#54a774', label='Scatter Plot')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
